"""Main script for generating training triplets from translation pairs."""

import argparse
import json
import os
import huggingface_hub
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional

from .triplet_generator import (
    CometKiwiEvaluator,
    TripletGenerator,
    load_translation_dataset,
    Triplet
)

def save_triplets(triplets: list[Triplet], output_file: str):
    """Save triplets to a JSONL file.
    
    Args:
        triplets: List of triplets to save
        output_file: Path to output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for triplet in triplets:
            f.write(json.dumps(triplet.__dict__, ensure_ascii=False) + '\n')

def generate_and_save_triplets(
    en_file: str,
    ko_file: str,
    ko_mt_files: List[str],
    output_file: str,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    prefer_reference: bool = False
):
    """Generate triplets and save them to a file.
    
    Args:
        en_file: Path to English source file
        ko_file: Path to Korean reference translation file
        ko_mt_files: List of paths to Korean machine translation files
        output_file: Path to save the generated triplets
        max_samples: Maximum number of samples to process
        batch_size: Batch size for processing
        prefer_reference: If True, always select reference as chosen translation
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    pairs = load_translation_dataset(en_file, ko_file, ko_mt_files, max_samples)
    
    # Initialize models and generate triplets
    evaluator = CometKiwiEvaluator()
    generator = TripletGenerator(evaluator, batch_size)
    triplets = generator.generate_triplets(pairs, prefer_reference=prefer_reference)
    
    # Save triplets
    with open(output_file, 'w', encoding='utf-8') as f:
        for triplet in triplets:
            json.dump({
                'prompt': triplet.prompt,
                'chosen': triplet.chosen,
                'rejected': triplet.rejected
            }, f, ensure_ascii=False)
            f.write('\n')
    
    # Save statistics
    stats = generator.get_stats()
    stats_file = str(Path(output_file).with_suffix('.stats.json'))
    with open(stats_file, 'w', encoding='utf-8') as f:
        # Add MT system names to stats
        mt_names = [Path(f).stem for f in ko_mt_files]
        named_stats = {
            'reference': stats['reference'],
            'mt_systems': {
                mt_names[i]: stats['mt'][f'mt_{i}']
                for i in range(len(mt_names))
            }
        }
        json.dump(named_stats, f, indent=2, ensure_ascii=False)

def main():
    """Main function to run the triplet generation process."""
    parser = argparse.ArgumentParser(description='Generate triplets for CPO training from English-Korean translations')
    parser.add_argument('--en_file', type=str, default='assets/train/train_en-ko.en',
                      help='Path to English source file')
    parser.add_argument('--ko_file', type=str, default='assets/train/train_en-ko.ko',
                      help='Path to Korean reference translation file')
    parser.add_argument('--ko_mt_files', type=str, nargs='+', required=True,
                      help='Paths to Korean machine translation files')
    parser.add_argument('--output', type=str, default='train_en-ko_triplets.jsonl',
                      help='Output file path for the generated triplets')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for COMET-KIWI evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to process')
    parser.add_argument('--prefer_reference', action='store_true',
                      help='Always prefer reference translation as chosen')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get Hugging Face token
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN not found in .env file")
    
    # Authenticate with Hugging Face
    print("Authenticating with Hugging Face...")
    huggingface_hub.login(token=hf_token)
    
    # Initialize COMET-KIWI evaluator
    print("Loading COMET-KIWI model...")
    evaluator = CometKiwiEvaluator()
    
    # Initialize triplet generator
    generator = TripletGenerator(evaluator, batch_size=args.batch_size)
    
    # Load dataset
    print("Loading dataset...")
    pairs = load_translation_dataset(
        args.en_file,
        args.ko_file,
        args.ko_mt_files,
        args.max_samples
    )
    print(f"Processing {len(pairs)} translation pairs...")
    
    # Generate triplets
    print("Generating triplets...")
    triplets = generator.generate_triplets(pairs, prefer_reference=args.prefer_reference)
    
    # Save triplets
    print(f"Saving triplets to {args.output}...")
    generate_and_save_triplets(
        args.en_file,
        args.ko_file,
        args.ko_mt_files,
        args.output,
        args.max_samples,
        args.batch_size,
        args.prefer_reference
    )
    print("Done!")

if __name__ == "__main__":
    main() 