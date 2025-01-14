"""GPT-4 based English to Korean translator module."""

import os
import time
import json
import asyncio
from typing import List, Dict, Optional, Tuple
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from tenacity import retry, stop_after_attempt, wait_random_exponential
import aiofiles

class GPT4Translator:
    """A class to handle English to Korean translation using GPT-4."""
    
    def __init__(self, api_key: str, temperature: float = 0.3):
        """Initialize the translator with OpenAI API key and parameters.
        
        Args:
            api_key: OpenAI API key
            temperature: Generation temperature (default: 0.3)
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.temperature = temperature
    
    @retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(6))
    async def translate_text(self, text: str) -> Optional[str]:
        """Translate a single text using GPT-4 with retry logic."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional English to Korean translator. Translate the given English text to natural, fluent Korean. Provide only the translation without any explanations or notes."},
                    {"role": "user", "content": text}
                ],
                temperature=self.temperature,
                max_tokens=1000
            )
            translation = response.choices[0].message.content.strip()
            if not translation:
                print(f"Failed to translate: {text}")
                return None
            return translation
        except Exception as e:
            print(f"Error translating text: {str(e)}")
            return None

    async def translate_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Translate a batch of texts concurrently."""
        tasks = [self.translate_text(text) for text in texts]
        return await asyncio.gather(*tasks, return_exceptions=True)
