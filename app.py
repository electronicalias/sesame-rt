# app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import ssl
import uvicorn
import asyncio
import numpy as np
import torch
import time
import os
import logging
import json
import copy
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from omegaconf import OmegaConf, open_dict
from modules.phi3_integration import Phi3Responder
from modules.model_handler import ModelHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global ASR model and streaming session
sample_rate = 16000 # Hz
model_name = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
lookahead_size = 80 # in milliseconds
ENCODER_STEP_LENGTH = 80 # in milliseconds
decoder_type = "rnnt"
asr_model = None
cache_last_channel = None
cache_last_time = None
cache_last_channel_len = None
previous_hypotheses = None
pred_out_stream = None
step_num = 0
pre_encode_cache_size = None
num_channels = None
cache_pre_encode = None
preprocessor = None
silence_counter = 0
last_speech_time = time.time()

# Global LLM
llm_responder = None
processed_transcripts = set()


@app.on_event("startup")
async def startup_event():
    global asr_model, model_name, cache_last_channel, cache_last_time, cache_last_channel_len, pre_encode_cache_size
    global num_channels, cache_pre_encode, preprocessor, llm_responder
    
    # Set up model handler with your models directory
    models_dir = "models"
    model_handler = ModelHandler(models_dir=models_dir)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load ASR model
    logger.info("Loading ASR model...")
    try:
        # Load pretrained model
        asr_model_location = model_name.replace("/", "_").lower()  # Create safe directory name
        asr_model = model_handler.load_or_download_nemo_model(
            model_name=model_name,
            model_location=asr_model_location
        )
        
        if model_name == "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi":
            # check that lookahead_size is one of the valid ones
            if lookahead_size not in [0, 80, 480, 1040]:
                raise ValueError(
                    f"specified lookahead_size {lookahead_size} is not one of the "
                    "allowed lookaheads (can select 0, 80, 480 or 1040 ms)"
                )

            # update att_context_size
            left_context_size = asr_model.encoder.att_context_size[0]
            asr_model.encoder.set_default_att_context_size([left_context_size, int(lookahead_size / ENCODER_STEP_LENGTH)])

        
        # set streaming params
        asr_model.encoder.setup_streaming_params()
        
        # Set the right decoder
        asr_model.change_decoding_strategy(decoder_type=decoder_type)

        # force optimal decoding strategy
        decoding_cfg = asr_model.cfg.decoding

        with open_dict(decoding_cfg):
            # use greedy decoding
            decoding_cfg.strategy = "greedy"
            decoding_cfg.preserve_alignments = False
            if hasattr(asr_model, 'joint'): # if an RNNT model
                # restrict max_Symbols to avoid infinite loop
                decoding_cfg.greedy.max_symbols = 10
                # batch size should be one, sensible usage
                decoding_cfg.fused_batch_size = -1
            asr_model.change_decoding_strategy(decoding_cfg)
        
        # Force to use Cuda when available
        asr_model.eval()
        if torch.cuda.is_available():
            asr_model = asr_model.to('cuda')
        logging.info('Model configured.')

        # get parameters to use for initial cache
        cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
            batch_size=1
        )

        pre_encode_cache_size = asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
        num_channels = asr_model.cfg.preprocessor.features
        cache_pre_encode = torch.zeros((1, num_channels, pre_encode_cache_size), device=asr_model.device)

        preprocessor = init_preprocessor(asr_model)
        logger.info('Pre-processor Initialized.')

        # Initialise Qwen Model
        logger.info("Loading Qwen Model")
        qwen_model_name = "Qwen/Qwen2.5-0.5B"
        qwen_model_location = "qwen2.5-0.5b"
        flan_model_name = "google/flan-t5-large"
        flan_model_location = "flan-t5-large"
        phi3_model_name = "luvGPT/luvai-phi3"
        phi3_model_location = "luvai-phi3"

        llm_model, llm_tokenizer = model_handler.load_or_download_model(
            model_name=phi3_model_name,
            model_location=phi3_model_location
        )

        system_prompt = """
            Your name is Mike. You are 28 years old and you have been a chef for about 4 years. You work at a prestigious eatery and your favourite food is Thai. 
            You have a wife and child. You love conversations and are very talkative. You like to ask questions and find out about the person you are talking to.
        """

        llm_responder = Phi3Responder(
            model=llm_model,
            tokenizer=llm_tokenizer,
            device=device,
            max_length=256,
            system_prompt=system_prompt
        )
        logger.info("Phi3 Responder Initialized.")


    except Exception as e:
        logger.error(f"Failed to load ASR model: {str(e)}")
        raise

# helper function for extracting transcriptions
def extract_transcriptions(hyps):
    """
        The transcribed_texts returned by CTC and RNNT models are different.
        This method would extract and return the text section of the hypothesis.
    """
    if isinstance(hyps[0], Hypothesis):
        transcriptions = []
        for hyp in hyps:
            transcriptions.append(hyp.text)
    else:
        transcriptions = hyps
    return transcriptions

# define functions to init audio preprocessor and to
# preprocess the audio (ie obtain the mel-spectrogram)
def init_preprocessor(asr_model):
    cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)

    # some changes for streaming scenario
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0
    cfg.preprocessor.normalize = "None"
    
    preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
    preprocessor.to(asr_model.device)
    
    return preprocessor

def preprocess_audio(audio, asr_model):
    device = asr_model.device

    # doing audio preprocessing
    audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
    audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
    processed_signal, processed_signal_length = preprocessor(
        input_signal=audio_signal, length=audio_signal_len
    )
    return processed_signal, processed_signal_length

def transcribe_chunk(new_chunk):
    
    global cache_last_channel, cache_last_time, cache_last_channel_len
    global previous_hypotheses, pred_out_stream, step_num, cache_pre_encode
    global silence_counter, last_speech_time
    
    # new_chunk is provided as np.int16, so we convert it to np.float32
    # as that is what our ASR models expect
    audio_data = new_chunk.astype(np.float32)
    audio_data = audio_data / 32768.0

    # Check for silence/speech end (simple VAD)
    is_final = False
    is_silence = False
    if len(new_chunk) > 0:
        # Calculate RMS (Root Mean Square) energy
        rms = np.sqrt(np.mean(np.square(audio_data)))
        
        # Consider it silence if RMS is below threshold
        silence_threshold = 0.005  # Adjust based on your microphone and environment
        is_silence = rms < silence_threshold
        
        # Track silence for several consecutive chunks before finalizing
        if is_silence:
            silence_counter += 1
            
            # After ~1 second of silence (depends on chunk size), mark as final
            if silence_counter >= 5 and (time.time() - last_speech_time) > 1.0:
                is_final = True
                silence_counter = 0  # Reset counter
                logger.info("Detected end of speech")
        else:
            # Reset silence counter and update last speech time
            silence_counter = 0
            last_speech_time = time.time()

    # get mel-spectrogram signal & length
    processed_signal, processed_signal_length = preprocess_audio(audio_data, asr_model)

    # Check if processed_signal is valid
    if processed_signal.shape[2] <= 0:
        return ""  # Return empty string if nothing to process
     
    # prepend with cache_pre_encode
    processed_signal = torch.cat([cache_pre_encode, processed_signal], dim=-1)
    processed_signal_length += cache_pre_encode.shape[1]
    
    # save cache for next time
    cache_pre_encode = processed_signal[:, :, -pre_encode_cache_size:]
    
    with torch.no_grad():
        (
            pred_out_stream,
            transcribed_texts,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = asr_model.conformer_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
            previous_hypotheses=previous_hypotheses,
            previous_pred_out=pred_out_stream,
            drop_extra_pre_encoded=None,
            return_transcription=True,
        )
    
    final_streaming_tran = extract_transcriptions(transcribed_texts)
    step_num += 1
    
    return final_streaming_tran[0], is_final

def reset_streaming():
    """Reset streaming buffer and state"""
    global cache_last_channel, cache_last_time, cache_last_channel_len
    global previous_hypotheses, pred_out_stream, step_num, cache_pre_encode
    
    # Reset caches and states to initial values
    if asr_model is not None:
        cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
            batch_size=1
        )
    
    # Reset other streaming variables
    previous_hypotheses = None
    pred_out_stream = None
    step_num = 0
    
    # Reset pre-encode cache if it exists
    if cache_pre_encode is not None and num_channels is not None and pre_encode_cache_size is not None:
        device = cache_pre_encode.device
        cache_pre_encode = torch.zeros((1, num_channels, pre_encode_cache_size), device=device)
    
    logger.info("Streaming state reset")

@app.get("/")
async def get():
    return FileResponse("static/index.html")

@app.post("/reset")
async def reset_transcript():
    """Endpoint to reset the ASR streaming state"""
    try:
        reset_streaming()
        logger.info("Transcript reset via HTTP endpoint")
        return JSONResponse(
            content={"status": "success", "message": "Transcript reset successfully"},
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        logger.error(f"Error resetting transcript: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")
    
    # Reset streaming buffer for new connection
    reset_streaming()
    
    try:
        while True:
            # Receive audio chunk from client
            audio_data = await websocket.receive_bytes()
            
            # Process with ASR
            try:
                audio_np = np.frombuffer(audio_data, dtype=np.int16)

                transcript, is_final = transcribe_chunk(audio_np)

                ai_response = None
                if is_final and transcript and llm_responder:
                    logger.info(f"Generating Qwen Response for: {transcript}")
                    if transcript not in processed_transcripts:
                        processed_transcripts.add(transcript)
                        try:
                            ai_response = llm_responder.generate_response(transcript)
                            logger.info(f"AI Response: {str(ai_response)}")
                            
                        except Exception as e:
                            logger.error(f"Error Generating Qwen Response: {str(e)}")
                
                # Send transcription back to client
                await websocket.send_json({
                    "type": "transcript",
                    "text": transcript,
                    "is_final": is_final,
                    "ai_response": ai_response if is_final else None
                })
                
                # Log transcription
                if transcript and transcript not in processed_transcripts:
                    logger.info(f"Final transcription: {transcript}")
                
            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

if __name__ == "__main__":
    # Start the server
    # Create SSL context
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain("ssl/cert.pem", keyfile="ssl/key.pem")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        ssl_keyfile="ssl/key.pem", 
        ssl_certfile="ssl/cert.pem"
    )