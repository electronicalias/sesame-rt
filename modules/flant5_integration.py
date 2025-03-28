# flan_integration.py
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

logger = logging.getLogger(__name__)

class FlanResponder:
    """Class to handle generating responses from FLAN-T5 based on ASR transcripts"""
    
    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_name=None, 
        device: Optional[str] = None,
        max_length: int = 512,
        system_prompt: str = "You are a helpful AI assistant responding to voice inputs. Keep responses concise."
    ):
        """Initialize the FLAN-T5 model for generating responses
        
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
        
        # If model and tokenizer are provided directly, use them
        if model is not None and tokenizer is not None:
            logger.info('Using cached model...')
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
        # Otherwise load from model_name
        elif model_name is not None:
            logger.info(f"Loading FLAN-T5 model '{model_name}' on {device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        else:
            raise ValueError("Either provide model and tokenizer objects or a model_name")
            
        # Chat history management
        self.conversation_history: List[Dict[str, str]] = []
        self.has_initialized_chat = False
        
        logger.info(f"FLAN-T5 model loaded successfully.")
    
    def _format_prompt(self, input_text: str) -> str:
        """Format the input prompt for the model with conversation history
        
        Args:
            input_text: User's input text (from ASR)
            
        Returns:
            Formatted prompt string for FLAN-T5
        """
        if not self.has_initialized_chat:
            self.conversation_history = []
            self.has_initialized_chat = True
            
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": input_text})
        
        # For FLAN-T5, we need a simple instruction-style prompt
        # Create a context from previous exchanges (up to 3 previous turns)
        context = ""
        if len(self.conversation_history) > 1:
            recent_history = self.conversation_history[-4:-1] if len(self.conversation_history) > 3 else self.conversation_history[:-1]
            for entry in recent_history:
                if entry["role"] == "user":
                    context += f"Human: {entry['content']}\n"
                else:
                    context += f"Assistant: {entry['content']}\n"
            context += "\n"
        
        # Format the prompt for FLAN-T5 which works well with instruction-style prompts
        formatted_prompt = f"""
{self.system_prompt}

{context}Human: {input_text}
Assistant:"""
        
        return formatted_prompt
    
    def generate_response(self, input_text: str, temperature: float = 0.7) -> str:
        """Generate a response from FLAN-T5 based on input text
        
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
                    max_new_tokens=self.max_length,  # Using max_new_tokens instead of max_length
                    do_sample=True,
                    temperature=temperature,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode the generated response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response (remove any "Assistant:" prefix if present)
            if response_text.startswith("Assistant:"):
                response_text = response_text[len("Assistant:"):].strip()
                
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