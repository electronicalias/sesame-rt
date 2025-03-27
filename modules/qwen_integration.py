# qwen_integration.py
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

class QwenResponder:
    """Class to handle generating responses from Qwen 2.5 based on ASR transcripts"""
    
    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_name=None, 
        device: Optional[str] = None,
        max_length: int = 512,
        system_prompt: str = "You are a helpful AI assistant responding to voice inputs. Keep responses concise."
    ):
        """Initialize the Qwen model for generating responses
        
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
        
        logger.info(f"Loading Qwen model '{model_name}' on {device}...")
        
        # If model and tokenizer are provided directly, use them
        if model is not None and tokenizer is not None:
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
        # Otherwise load from model_name
        elif model_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        else:
            raise ValueError("Either provide model and tokenizer objects or a model_name")
        
        if device != "cuda":
            self.model.to(device)
            
        # Chat history management
        self.conversation_history: List[Dict[str, str]] = []
        self.has_initialized_chat = False
        
        logger.info(f"Qwen model loaded successfully.")
    
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
        return self.tokenizer.apply_chat_template(
            self.conversation_history, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def generate_response(self, input_text: str, temperature: float = 0.7) -> str:
        """Generate a response from Qwen based on input text
        
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