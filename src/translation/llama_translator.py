"""Llama-based English to Korean translator module using HuggingFace transformers."""

import torch
from typing import List, Optional
from transformers import pipeline, AutoTokenizer

class LlamaTranslator:
    """A class to handle English to Korean translation using Llama model."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct", temperature: float = 0.3):
        """Initialize the translator with Llama model and parameters.
        
        Args:
            model_name: HuggingFace model name/path
            temperature: Generation temperature (default: 0.3)
        """
        print(f"Initializing LlamaTranslator with model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.temperature = temperature
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    async def translate_text(self, text: str) -> Optional[str]:
        """Translate a single text using Llama with retry logic."""
        try:
            # Create messages for translation
            system_prompt = "You are a professional English to Korean translator. Provide only the translation without any explanations or notes."
            user_message = f"Translate this from English to Korean: {text}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            
            outputs = self.pipe(
                messages,
                max_new_tokens=256,
            )
            
            


            # Extract translation from the assistant's response
            if isinstance(outputs[0]["generated_text"], list):
                for message in outputs[0]["generated_text"]:
                    if message.get("role") == "assistant":
                        translation = message.get("content", "").strip()
                        if translation:
                            return translation.replace(user_message, "").strip().replace(system_prompt, "").strip().replace("\n", "").strip()
            
            print(f"Failed to translate: {text}")
            print(f"Output format: {outputs[0]['generated_text']}")
            return None
            
        except Exception as e:
            print(f"Error translating text: {str(e)}")
            return None

    async def translate_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Translate a batch of texts sequentially."""
        translations = []
        for text in texts:
            translation = await self.translate_text(text)
            translations.append(translation)
        return translations 