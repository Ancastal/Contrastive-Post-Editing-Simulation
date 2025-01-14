"""Module for generating training triplets from translation pairs using COMET-KIWI."""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import torch
from comet import download_model, load_from_checkpoint

@dataclass
class TranslationPair:
    """Data class representing a translation pair with source and two translations."""
    source: str
    reference: str
    gpt4: str

@dataclass
class Triplet:
    """Data class representing a training triplet."""
    prompt: str
    chosen: str
    rejected: str

class CometKiwiEvaluator:
    """Class for evaluating translations using COMET-KIWI model."""
    
    def __init__(self, model_name: str = "Unbabel/wmt22-cometkiwi-da"):
        """Initialize the COMET-KIWI evaluator.
        
        Args:
            model_name: Name of the COMET-KIWI model to use
        """
        self.model_path = download_model(model_name)
        self.model = load_from_checkpoint(self.model_path)
    
    def evaluate_batch(
        self,
        src_texts: List[str],
        hyp1_texts: List[str],
        hyp2_texts: List[str]
    ) -> List[List[float]]:
        """Evaluate a batch of translations using COMET-KIWI.
        
        Args:
            src_texts: List of source texts
            hyp1_texts: List of first hypothesis translations
            hyp2_texts: List of second hypothesis translations
            
        Returns:
            List of scores for each translation pair
        """
        samples = []
        for src, hyp1, hyp2 in zip(src_texts, hyp1_texts, hyp2_texts):
            samples.append({
                "src": src,
                "mt": [hyp1, hyp2]
            })
        
        return self.model.predict(samples, batch_size=len(samples), progress_bar=False)

class TripletGenerator:
    """Class for generating training triplets from translation pairs."""
    
    def __init__(self, evaluator: CometKiwiEvaluator, batch_size: int = 8):
        """Initialize the triplet generator.
        
        Args:
            evaluator: CometKiwiEvaluator instance
            batch_size: Size of batches for evaluation
        """
        self.evaluator = evaluator
        self.batch_size = batch_size
    
    def generate_triplets(self, pairs: List[TranslationPair]) -> List[Triplet]:
        """Generate triplets from translation pairs based on COMET-KIWI scores.
        
        Args:
            pairs: List of translation pairs
            
        Returns:
            List of generated triplets
        """
        triplets = []
        
        # Process pairs in batches
        for i in tqdm(range(0, len(pairs), self.batch_size), desc="Generating triplets"):
            batch = pairs[i:i + self.batch_size]
            
            # Prepare batch data
            src_texts = [pair.source for pair in batch]
            ref_texts = [pair.reference for pair in batch]
            gpt4_texts = [pair.gpt4 for pair in batch]
            
            # Get quality scores
            scores = self.evaluator.evaluate_batch(src_texts, ref_texts, gpt4_texts)
            
            # Create triplets based on scores
            for pair, score in zip(batch, scores):
                # If reference translation has higher score, it becomes chosen
                if score[0] > score[1]:
                    triplet = Triplet(
                        prompt=pair.source,
                        chosen=pair.reference,
                        rejected=pair.gpt4
                    )
                else:
                    triplet = Triplet(
                        prompt=pair.source,
                        chosen=pair.gpt4,
                        rejected=pair.reference
                    )
                triplets.append(triplet)
        
        return triplets

def load_translation_dataset(
    en_file: str,
    ko_file: str,
    ko_gpt4_file: str,
    max_samples: Optional[int] = None
) -> List[TranslationPair]:
    """Load the English-Korean translation dataset from files.
    
    Args:
        en_file: Path to English source file
        ko_file: Path to Korean reference translation file
        ko_gpt4_file: Path to Korean GPT-4 translation file
        max_samples: Maximum number of samples to load
        
    Returns:
        List of translation pairs
    """
    with open(en_file, 'r', encoding='utf-8') as f_en, \
         open(ko_file, 'r', encoding='utf-8') as f_ko, \
         open(ko_gpt4_file, 'r', encoding='utf-8') as f_ko_gpt4:
        
        en_lines = [line.strip() for line in f_en]
        ko_lines = [line.strip() for line in f_ko]
        ko_gpt4_lines = [line.strip() for line in f_ko_gpt4]
        
        if max_samples is not None:
            en_lines = en_lines[:max_samples]
            ko_lines = ko_lines[:max_samples]
            ko_gpt4_lines = ko_gpt4_lines[:max_samples]
        
        assert len(en_lines) == len(ko_lines) == len(ko_gpt4_lines), \
            "All files must have the same number of lines"
        
        return [
            TranslationPair(source=en, reference=ko, gpt4=ko_gpt4)
            for en, ko, ko_gpt4 in zip(en_lines, ko_lines, ko_gpt4_lines)
        ] 