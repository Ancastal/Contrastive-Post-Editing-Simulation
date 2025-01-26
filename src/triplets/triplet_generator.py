"""Module for generating training triplets from translation pairs using COMET-KIWI."""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import torch
from comet import download_model, load_from_checkpoint

@dataclass
class TranslationPair:
    """Data class representing a translation pair with source, reference and multiple machine translations."""
    source: str
    reference: str
    machine_translations: List[str]

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
        translations: List[List[str]]  # Changed to accept list of translation lists
    ) -> List[List[float]]:
        """Evaluate a batch of translations using COMET.
        
        Args:
            src_texts: List of source texts
            translations: List of lists where each inner list contains all translations for one source
                (reference + machine translations)
        
        Returns:
            List of lists of scores, one list per source text
        """
        samples = []
        for src, trans_list in zip(src_texts, translations):
            for mt in trans_list:
                samples.append({
                    "src": src,
                    "mt": mt
                })
        
        prediction = self.model.predict(samples, batch_size=len(samples), progress_bar=False)
        scores = prediction['scores']
        
        # Reshape scores back into original batch format
        batch_scores = []
        translations_per_source = len(translations[0])
        for i in range(0, len(scores), translations_per_source):
            batch_scores.append([float(score) for score in scores[i:i + translations_per_source]])
        
        return batch_scores

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
        """Generate triplets from translation pairs based on COMET scores."""
        triplets = []
        
        for i in tqdm(range(0, len(pairs), self.batch_size), desc="Generating triplets"):
            batch = pairs[i:i + self.batch_size]
            
            src_texts = [pair.source for pair in batch]
            all_translations = []
            for pair in batch:
                translations = [pair.reference] + pair.machine_translations
                all_translations.append(translations)
            
            # Get quality scores
            batch_scores = self.evaluator.evaluate_batch(src_texts, all_translations)
            
            # Create triplets based on scores
            for pair, score_list in zip(batch, batch_scores):
                ref_score = float(score_list[0])  # First score is for reference
                mt_scores = [float(score) for score in score_list[1:]]  # Rest are machine translations
                
                # Find best machine translation
                best_mt_score = max(mt_scores)
                best_mt_index = mt_scores.index(best_mt_score)
                
                if best_mt_score > ref_score:
                    triplet = Triplet(
                        prompt=pair.source,
                        chosen=pair.machine_translations[best_mt_index],
                        rejected=pair.reference
                    )
                else:
                    triplet = Triplet(
                        prompt=pair.source,
                        chosen=pair.reference,
                        rejected=pair.machine_translations[best_mt_index]
                    )
                triplets.append(triplet)

        
        return triplets


def load_translation_dataset(
    en_file: str,
    ko_file: str,
    ko_mt_files: List[str],
    max_samples: Optional[int] = None
) -> List[TranslationPair]:
    """Load the English-Korean translation dataset from files.
    
    Args:
        en_file: Path to English source file
        ko_file: Path to Korean reference translation file
        ko_mt_files: List of paths to Korean machine translation files
        max_samples: Maximum number of samples to load
        
    Returns:
        List of translation pairs
    """
    with open(en_file, 'r', encoding='utf-8') as f_en, \
         open(ko_file, 'r', encoding='utf-8') as f_ko:
        en_lines = [line.strip() for line in f_en]
        ko_lines = [line.strip() for line in f_ko]
        
        # Read all machine translation files
        mt_lines = []
        for mt_file in ko_mt_files:
            with open(mt_file, 'r', encoding='utf-8') as f_mt:
                mt_lines.append([line.strip() for line in f_mt])
        
        if max_samples is not None:
            en_lines = en_lines[:max_samples]
            ko_lines = ko_lines[:max_samples]
            mt_lines = [lines[:max_samples] for lines in mt_lines]
        
        # Verify all files have same number of lines
        line_counts = [len(en_lines), len(ko_lines)] + [len(lines) for lines in mt_lines]
        assert all(count == line_counts[0] for count in line_counts), \
            "All files must have the same number of lines"
        
        return [
            TranslationPair(
                source=en,
                reference=ko,
                machine_translations=[mt[i] for mt in mt_lines]
            )
            for i, (en, ko) in enumerate(zip(en_lines, ko_lines))
        ] 
