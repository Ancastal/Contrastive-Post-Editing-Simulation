"""Translation manager for handling batch translations and progress tracking."""

import os
import json
import asyncio
from typing import List, Dict, Optional, Tuple, Protocol, Callable
from tqdm.asyncio import tqdm_asyncio
import aiofiles


class Translator(Protocol):
    """Protocol defining the interface for translator classes."""
    
    async def translate_text(self, text: str) -> Optional[str]:
        """Translate a single text."""
        ...
        
    async def translate_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Translate a batch of texts."""
        ...


class TranslationManager:
    """Manages the translation process, including progress tracking and file I/O."""
    
    def __init__(self, translator: Translator, batch_size: int = 5):
        """Initialize the translation manager.
        
        Args:
            translator: Any translator instance implementing the Translator protocol
            batch_size: Number of concurrent translations (default: 5)
        """
        self.translator = translator
        self.batch_size = batch_size
        self.progress_file = "translation_progress.json"

    async def load_progress(self) -> Dict:
        """Load translation progress from file."""
        if os.path.exists(self.progress_file):
            async with aiofiles.open(self.progress_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        return {"last_index": 0, "translations": [], "failed_indices": []}

    async def save_progress(self, last_index: int, translations: List[str], failed_indices: List[int]):
        """Save translation progress to file."""
        async with aiofiles.open(self.progress_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps({
                "last_index": last_index,
                "translations": translations,
                "failed_indices": failed_indices
            }, ensure_ascii=False, indent=2))

    async def translate_texts(
        self,
        texts: List[str],
        start_idx: int = 0,
        existing_translations: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[List[str], List[int]]:
        """Translate texts with batching and progress tracking.
        
        Args:
            texts: List of texts to translate
            start_idx: Starting index for translation
            existing_translations: List of existing translations to resume from
            progress_callback: Optional callback function to update progress
        """
        translations = existing_translations if existing_translations else []
        failed_indices = []
        
        try:
            batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
            
            for batch_idx, batch in enumerate(batches):
                batch_translations = await self.translator.translate_batch(batch)
                
                for idx, translation in enumerate(batch_translations):
                    global_idx = batch_idx * self.batch_size + idx
                    if isinstance(translation, Exception) or translation is None:
                        failed_indices.append(global_idx)
                        translations.append("")
                    else:
                        translations.append(translation)
                    
                    if progress_callback:
                        progress_callback()
                
                await self.save_progress(
                    start_idx + len(translations),
                    translations,
                    failed_indices
                )
                
                await asyncio.sleep(1)  # Rate limiting
                
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving progress...")
            await self.save_progress(
                start_idx + len(translations),
                translations,
                failed_indices
            )
            raise
        
        return translations, failed_indices

    async def retry_failed_translations(
        self,
        texts: List[str],
        failed_indices: List[int],
        translations: List[str]
    ) -> List[str]:
        """Retry failed translations."""
        if not failed_indices:
            return translations
        
        print(f"\nRetrying {len(failed_indices)} failed translations...")
        for idx in failed_indices[:]:  # Copy list to allow modification while iterating
            translation = await self.translator.translate_text(texts[idx])
            if translation:
                translations[idx] = translation
                failed_indices.remove(idx)
                
        return translations

    @staticmethod
    async def save_translations(translations: List[str], output_file: str):
        """Save translations to file."""
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            for translation in translations:
                await f.write(translation + '\n') 