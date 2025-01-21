"""Main script for running English to Korean translations using GPT-4 or Llama."""

import argparse
import os
import asyncio
from typing import List, Tuple
from dotenv import load_dotenv

from ..utils import TranslationManager
from .gpt4_translator import GPT4Translator
from .llama_translator import LlamaTranslator

def load_source_texts(
    input_file: str,
    start_idx: int = 0,
    max_samples: int = None
) -> Tuple[List[str], int]:
    """Load source texts from file with validation.
    
    Args:
        input_file: Path to input file
        start_idx: Starting index for loading texts
        max_samples: Maximum number of samples to load
        
    Returns:
        Tuple of loaded texts and total number of lines in file
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        if max_samples:
            end_idx = min(start_idx + max_samples, len(lines))
            return lines[start_idx:end_idx], len(lines)
        return lines[start_idx:], len(lines)
    except Exception as e:
        raise RuntimeError(f"Error loading source texts: {str(e)}")

async def main_async():
    """Main async function to run the translation process."""
    parser = argparse.ArgumentParser(description='Generate Korean translations using GPT-4 or Llama')
    parser.add_argument('--input', type=str, default='assets/train/train_en-ko.en',
                      help='Path to input English file')
    parser.add_argument('--output', type=str, default='assets/train/train_en-ko.gpt4.ko',
                      help='Path to output Korean translations file')
    parser.add_argument('--batch_size', type=int, default=5,
                      help='Batch size for concurrent API calls')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to translate')
    parser.add_argument('--resume', action='store_true',
                      help='Resume from last saved progress')
    parser.add_argument('--temperature', type=float, default=0.3,
                      help='Temperature for model generation')
    parser.add_argument('--model', type=str, default='gpt4', choices=['gpt4', 'llama'],
                      help='Model to use for translation (gpt4 or llama)')
    parser.add_argument('--llama_model', type=str, 
                      default='meta-llama/Llama-3.2-1B-Instruct',
                      help='Llama model to use (only used if --model=llama). Can be one of the default models: '
                           'meta-llama/Llama-3.2-[1B,7B,13B,70B]-Instruct or a custom model path')

    args = parser.parse_args()
    
    # Initialize translator based on model choice
    if args.model == 'gpt4':
        # Load environment variables for OpenAI
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        translator = GPT4Translator(api_key=api_key, temperature=args.temperature)
    else:  # llama
        translator = LlamaTranslator(model_name=args.llama_model, temperature=args.temperature)
    
    # Initialize manager
    print(f"Using translator: {translator}")
    manager = TranslationManager(translator, batch_size=args.batch_size)
    
    # Load progress if resuming
    start_idx = 0
    existing_translations = None
    if args.resume:
        progress = await manager.load_progress()
        start_idx = progress["last_index"]
        existing_translations = progress["translations"]
        print(f"Resuming from index {start_idx}")
    
    # Load source texts
    print("Loading source texts...")
    source_texts, total_lines = load_source_texts(args.input, start_idx, args.max_samples)
    print(f"Processing {len(source_texts)} texts out of {total_lines} total lines")
    
    # Generate translations
    print(f"Generating translations using {args.model.upper()}...")
    translations, failed_indices = await manager.translate_texts(
        source_texts,
        start_idx, 
        existing_translations
    )
    
    # Retry failed translations
    if failed_indices:
        translations = await manager.retry_failed_translations(
            source_texts,
            failed_indices,
            translations
        )
    
    # Save translations
    print(f"Saving translations to {args.output}...")
    await manager.save_translations(translations, args.output)
    
    # Clean up progress file if completed successfully
    if os.path.exists(manager.progress_file) and not failed_indices:
        os.remove(manager.progress_file)
    
    print("Done!")

def main():
    """Entry point for the script."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 