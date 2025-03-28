# phi3_integration.py
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

class Phi3Responder:
    """Class to handle generating responses from Phi-3 based on ASR transcripts"""
    
    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_name=None, 
        device: Optional[str] = None,
        max_length: int = 512,
        system_prompt: str = "You are a helpful AI assistant responding to voice inputs. Keep responses concise."
    ):
        """Initialize the Phi-3 model for generating responses
        
        Args:
            model: Pre-loaded model object (optional)
            tokenizer: Pre-loaded tokenizer object (optional)
            model_name: HuggingFace model name to load if model/tokenizer not provided
            device: Device to run model on ("cuda" or "cpu")
            max_length: Maximum length of generated responses
            system_prompt: System prompt for response generation
        """
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.device = device
        self.max_length = max_length
        self.system_prompt = system_prompt
        
        logger.info(f"Loading Phi-3 model '{model_name if model_name else 'from cache'}' on {device}...")
        
        # If model and tokenizer are provided directly, use them
        if model is not None and tokenizer is not None:
            logger.info('Using cached model...')
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
        # Otherwise load from model_name
        elif model_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # For Phi-3 models, we enable 4-bit quantization if on CUDA to reduce memory usage
            if device == "cuda" and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    logger.info("Using 4-bit quantization for Phi-3 model")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        device_map="auto",
                        quantization_config=quantization_config,
                        trust_remote_code=True
                    )
                except ImportError:
                    logger.warning("BitsAndBytes not available, loading model without quantization")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        device_map="auto", 
                        trust_remote_code=True
                    )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    trust_remote_code=True
                ).to(self.device)
        else:
            raise ValueError("Either provide model and tokenizer objects or a model_name")
            
        # Chat history management
        self.conversation_history: List[Dict[str, str]] = []
        self.has_initialized_chat = False
        
        logger.info(f"Phi-3 model loaded successfully.")
    
    def _format_prompt(self, input_text: str) -> str:
        """Format the input prompt for the model with conversation history
        
        Args:
            input_text: User's input text (from ASR)
            
        Returns:
            Formatted prompt string
        """
        if not self.has_initialized_chat:
            self.conversation_history = [
                {"role": "system", "content": self.system_prompt}
            ]
            self.has_initialized_chat = True
            
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": input_text})
        
        # Convert history to the format needed by the model
        # Phi-3 models support chat templates like Llama
        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                self.conversation_history, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return formatted_prompt
        except Exception as e:
            # Fallback to manual formatting if chat template fails
            logger.warning(f"Chat template failed, using manual formatting: {str(e)}")
            
            formatted_prompt = f"<|system|>\n{self.system_prompt}\n"
            
            # Skip the system message which we already added
            for i in range(1, len(self.conversation_history)):
                message = self.conversation_history[i]
                if message["role"] == "user":
                    formatted_prompt += f"<|user|>\n{message['content']}\n"
                else:
                    formatted_prompt += f"<|assistant|>\n{message['content']}\n"
            
            # Add final assistant marker for generation
            if self.conversation_history[-1]["role"] == "user":
                formatted_prompt += "<|assistant|>\n"
                
            return formatted_prompt
    
    def generate_response(self, input_text: str, temperature: float = 0.7) -> str:
        """Generate a response from Phi-3 based on input text
        
        Args:
            input_text: Text from ASR transcription
            temperature: Temperature for text generation
            
        Returns:
            Generated response text
        """
        # Skip empty or very short inputs
        if not input_text or len(input_text.strip()) < 2:
            return ""
            
        try:
            # Format the prompt with conversation history
            prompt = self._format_prompt(input_text)
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=self.max_length,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Get only the newly generated tokens
            new_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean the response to remove any assistant/system markers
            response_text = response_text.replace("<|assistant|>", "").strip()
            
            # Add assistant response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I'm sorry, I couldn't generate a response. Error: {str(e)}"
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        self.has_initialized_chat = False
        logger.info("Conversation history reset")