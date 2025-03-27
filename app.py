# app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
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


@app.on_event("startup")
async def startup_event():
    global asr_model, model_name, cache_last_channel, cache_last_time, cache_last_channel_len, pre_encode_cache_size
    global num_channels, cache_pre_encode, preprocessor
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load ASR model
    logger.info("Loading ASR model...")
    try:
        # Load pretrained model
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=model_name)
        
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
    global previous_hypotheses, pred_out_stream, step_num
    global cache_pre_encode
    
    # new_chunk is provided as np.int16, so we convert it to np.float32
    # as that is what our ASR models expect
    audio_data = new_chunk.astype(np.float32)
    audio_data = audio_data / 32768.0
        
    # Debug info
    print(f"Buffer size: {len(new_chunk)}")

    # get mel-spectrogram signal & length
    processed_signal, processed_signal_length = preprocess_audio(audio_data, asr_model)

    # Check if processed_signal is valid
    if processed_signal.shape[2] <= 0:
        return ""  # Return empty string if nothing to process
     
    # prepend with cache_pre_encode
    processed_signal = torch.cat([cache_pre_encode, processed_signal], dim=-1)
    processed_signal_length += cache_pre_encode.shape[1]
        
    # Debug after concatenation
    print(f"After concat - processed signal shape: {processed_signal.shape}")
    print(f"After concat - processed signal length: {processed_signal_length}")
    
    # save cache for next time
    cache_pre_encode = processed_signal[:, :, -pre_encode_cache_size:]

    # Just before the torch.cat line
    print(f"Processed signal shape: {processed_signal.shape}")
    print(f"Cache shape: {cache_pre_encode.shape}")
    
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
    logger.info(f"Response: {final_streaming_tran[0]}")
    step_num += 1
    
    return final_streaming_tran[0]

def reset_streaming():
    """Reset streaming buffer and state"""
    global total_buffer, previous_audio_length, previous_out_processed
    total_buffer = np.array([])
    previous_audio_length = 0
    previous_out_processed = ""

# def process_audio_chunk(audio_chunk, sample_rate=16000, threshold=50):
#     """
#     Process incoming audio chunks using cache-aware streaming approach
    
#     Args:
#         audio_chunk (numpy.ndarray): Audio chunk to process
#         sample_rate (int): Sample rate of the audio
#         threshold (int): Silence threshold in samples
        
#     Returns:
#         tuple: (transcript, is_final)
#     """
#     global asr_model, frame_len, model_stride, total_buffer
#     global previous_audio_length, previous_out_processed, last_transcript
    
#     # Convert to numpy array if needed
#     if not isinstance(audio_chunk, np.ndarray):
#         audio_chunk = np.frombuffer(audio_chunk, dtype=np.float32)
    
#     # Simple audio normalization
#     if np.abs(audio_chunk).max() > 1.0:
#         audio_chunk = audio_chunk / np.abs(audio_chunk).max()
    
#     # Add audio to buffer
#     total_buffer = np.concatenate((total_buffer, audio_chunk), axis=0)
    
#     # Calculate new chunk in samples
#     new_chunk_samples = int(chunk_len * sample_rate)
    
#     # If buffer is too small, don't process yet
#     if len(total_buffer) < new_chunk_samples:
#         return "", False
    
#     # Process audio in chunks
#     audio_chunks = []
#     is_final = False
#     transcript = ""
    
#     # Check if we have enough new audio to process
#     if len(total_buffer) >= previous_audio_length + new_chunk_samples:
#         # This is a new chunk to process
#         audio_signal = torch.tensor(total_buffer, dtype=torch.float32).unsqueeze(0)
#         if torch.cuda.is_available():
#             audio_signal = audio_signal.to('cuda')
        
#         audio_len = torch.tensor([audio_signal.shape[1]], dtype=torch.long).to(audio_signal.device)
        
#         # Return log probs, lengths, and alignment
#         with torch.no_grad():
#             log_probs, encoded_len, alignment = asr_model(
#                 input_signal=audio_signal, 
#                 input_signal_length=audio_len,
#                 return_transcription_alignment=True
#             )
            
#             # Get greedy predictions
#             hypotheses, _ = asr_model.decoding.ctc_decoder_predictions_tensor(
#                 log_probs, encoded_len, return_hypotheses=False
#             )
        
#         # Update processed transcription
#         out_processed = hypotheses[0]
        
#         # Check if we have a significant change in transcript
#         if len(out_processed) > len(previous_out_processed):
#             transcript = out_processed
#             previous_out_processed = out_processed
#             last_transcript = transcript
#         else:
#             transcript = last_transcript
        
#         # Update processed audio length
#         previous_audio_length = len(total_buffer)
        
#         # Check if we should finalize the utterance
#         # Simple silence detection - check if the last chunk is mostly silent
#         if len(audio_chunk) > threshold and np.abs(audio_chunk[-threshold:]).max() < 0.02:
#             is_final = True
#             # Keep the part we've already processed
#             total_buffer = total_buffer[-new_chunk_samples:]
#             previous_audio_length = 0
#             previous_out_processed = ""
#             last_transcript = ""
#     else:
#         # Not enough new audio to process yet
#         transcript = last_transcript
    
#     return transcript, is_final

@app.get("/")
async def get():
    return FileResponse("static/index.html")

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

                transcript = transcribe_chunk(audio_np)
                
                # Send transcription back to client
                await websocket.send_json({
                    "type": "transcript",
                    "text": transcript,
                    "is_final": False
                })
                
                # Log transcription
                if transcript:
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