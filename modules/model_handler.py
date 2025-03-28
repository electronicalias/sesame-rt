import os
from pathlib import Path
import shutil
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import nemo.collections.asr as nemo_asr

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        # Ensure models directory exists
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        
    def load_or_download_model(self, model_name: str, model_location: str):
        """
        Load Hugging Face transformers model from local cache or download if not available.
        
        Args:
            model_name: The Hugging Face model name (e.g., "Qwen/Qwen2.5-0.5B")
            model_location: Local directory name to store the model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model_path = os.path.join(self.models_dir, model_location)
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        # Check if model files exist locally
        config_file = os.path.join(model_path, "config.json")
        
        if os.path.exists(config_file):
            logger.info(f"Loading model from local cache: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            logger.info(f"Downloading model {model_name} and saving to {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Save to local directory
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
        return model, tokenizer
    
    def load_or_download_nemo_model(self, model_name: str, model_location: str):
        """
        Load NeMo model from local cache or download if not available.
        
        Args:
            model_name: The NeMo model name (e.g., "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi")
            model_location: Local directory name to store the model
            
        Returns:
            NeMo model
        """
        model_dir = os.path.join(self.models_dir, model_location)
        model_file = os.path.join(model_dir, f"{model_location}.nemo")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        if os.path.exists(model_file):
            logger.info(f"Loading NeMo model from local cache: {model_file}")
            model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_file)
        else:
            logger.info(f"Downloading NeMo model {model_name} and saving to {model_file}")
            
            # NeMo's from_pretrained already caches models, but in a different location
            # We'll download using from_pretrained and then copy to our preferred location
            model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=model_name)
            
            # Save to our desired local path
            model.save_to(model_file)
            logger.info(f"NeMo model saved to {model_file}")
            
        return model
    
    def load_or_download_seq2seq_model(self, model_name: str, model_location: str):
        """
        Load Hugging Face sequence-to-sequence transformers model (like FLAN-T5) from local cache 
        or download if not available.
        
        Args:
            model_name: The Hugging Face model name (e.g., "google/flan-t5-large")
            model_location: Local directory name to store the model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import os
        from pathlib import Path
        import logging
        
        model_path = os.path.join(self.models_dir, model_location)
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        # Check if model files exist locally
        config_file = os.path.join(model_path, "config.json")
        
        if os.path.exists(config_file):
            logger.info(f"Loading seq2seq model from local cache: {model_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            logger.info(f"Downloading seq2seq model {model_name} and saving to {model_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Save to local directory
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
        return model, tokenizer