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
        translations: List[List[str]]
    ) -> List[List[float]]:
        """Evaluate a batch of translations using COMET-KIWI.
        
        Args:
            src_texts: List of source texts
            translations: List of lists of translations to evaluate
            
        Returns:
            List of scores for each translation
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
        self.stats = {
            'reference': {'chosen': 0, 'rejected': 0},
            'mt': {}  # Will be populated with stats for each MT system
        }
    
    def get_stats(self) -> Dict:
        """Get statistics about how often each translation was chosen/rejected.
        
        Returns:
            Dictionary containing statistics for reference and each MT system
        """
        return self.stats

    def generate_triplets(self, pairs: List[TranslationPair], prefer_reference: bool = False) -> List[Triplet]:
        """Generate triplets from translation pairs based on COMET-KIWI scores.
        
        Args:
            pairs: List of translation pairs
            prefer_reference: If True, always select reference as chosen translation
            
        Returns:
            List of generated triplets
        """
        triplets = []
        
        # Reset stats
        self.stats = {
            'reference': {'chosen': 0, 'rejected': 0},
            'mt': {f'mt_{i}': {'chosen': 0, 'rejected': 0} for i in range(len(pairs[0].machine_translations))}
        }
        
        # Process pairs in batches
        for i in tqdm(range(0, len(pairs), self.batch_size), desc="Generating triplets"):
            batch = pairs[i:i + self.batch_size]
            
            # Prepare batch data
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
                
                # Get indices of top machine translations
                mt_indices = sorted(range(len(mt_scores)), key=lambda k: mt_scores[k], reverse=True)
                best_mt_score = mt_scores[mt_indices[0]]
                
                if prefer_reference:
                    # Always use reference as chosen
                    chosen = pair.reference
                    self.stats['reference']['chosen'] += 1
                    # Use highest scored MT as rejected
                    rejected = pair.machine_translations[mt_indices[0]]
                    self.stats['mt'][f'mt_{mt_indices[0]}']['rejected'] += 1
                else:
                    if best_mt_score > ref_score:
                        # If best MT is better than reference
                        chosen = pair.machine_translations[mt_indices[0]]
                        self.stats['mt'][f'mt_{mt_indices[0]}']['chosen'] += 1
                        
                        if len(mt_indices) > 1 and mt_scores[mt_indices[1]] > ref_score:
                            # If second best MT is also better than reference
                            rejected = pair.machine_translations[mt_indices[1]]
                            self.stats['mt'][f'mt_{mt_indices[1]}']['rejected'] += 1
                        else:
                            rejected = pair.reference
                            self.stats['reference']['rejected'] += 1
                    else:
                        # If reference is better than all MTs
                        chosen = pair.reference
                        self.stats['reference']['chosen'] += 1
                        rejected = pair.machine_translations[mt_indices[0]]  # Best MT becomes rejected
                        self.stats['mt'][f'mt_{mt_indices[0]}']['rejected'] += 1
                
                triplet = Triplet(
                    prompt=pair.source,
                    chosen=chosen,
                    rejected=rejected
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