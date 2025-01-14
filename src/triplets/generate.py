"""Main script for generating training triplets from translation pairs."""

import argparse
import json
import os
import huggingface_hub
from dotenv import load_dotenv

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

def main():
    """Main function to run the triplet generation process."""
    parser = argparse.ArgumentParser(description='Generate triplets for CPO training from English-Korean translations')
    parser.add_argument('--en_file', type=str, default='assets/train/train_en-ko.en',
                      help='Path to English source file')
    parser.add_argument('--ko_file', type=str, default='assets/train/train_en-ko.ko',
                      help='Path to Korean reference translation file')
    parser.add_argument('--ko_gpt4_file', type=str, required=True,
                      help='Path to Korean GPT-4 translation file')
    parser.add_argument('--output', type=str, default='train_en-ko_triplets.jsonl',
                      help='Output file path for the generated triplets')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for COMET-KIWI evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to process')
    
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
        args.ko_gpt4_file,
        args.max_samples
    )
    print(f"Processing {len(pairs)} translation pairs...")
    
    # Generate triplets
    print("Generating triplets...")
    triplets = generator.generate_triplets(pairs)
    
    # Save triplets
    print(f"Saving triplets to {args.output}...")
    save_triplets(triplets, args.output)
    print("Done!")

if __name__ == "__main__":
    main() 