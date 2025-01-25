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
        data = []
        for src, trans_list in zip(src_texts, translations):
            # Compare each translation against all others
            for i, mt in enumerate(trans_list):
                others = trans_list[:i] + trans_list[i+1:]  # All translations except current one
                for ref in others:
                    data.append({
                        "src": src,
                        "mt": mt,
                        "ref": ref
                    })
        
        scores = self.model.predict(data, batch_size=len(data), progress_bar=False)
        
        # Reshape scores into groups per translation
        translations_per_source = len(translations[0])
        comparisons_per_translation = translations_per_source - 1
        return [
            scores[i:i + comparisons_per_translation]
            for i in range(0, len(scores), comparisons_per_translation)
        ]
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
            translations = [[pair.reference] + pair.machine_translations for pair in batch]
            
            scores = self.evaluator.evaluate_batch(src_texts, translations)
            
            for pair, score_group in zip(batch, scores):
                if isinstance(score_group[0], list):
                    score_group = [s[0] if isinstance(s, list) else s for s in score_group]
                    
                num_translations = len(pair.machine_translations) + 1
                avg_scores = []
                for i in range(num_translations):
                    translation_scores = score_group[i * (num_translations - 1):(i + 1) * (num_translations - 1)]
                    # Add safety check for empty scores
                    if translation_scores:
                        avg_scores.append(sum(translation_scores) / len(translation_scores))
                    else:
                        print(f"Warning: Empty translation scores for index {i}")
                        avg_scores.append(0.0)  # or some other default value
                
                # First score is reference, rest are machine translations
                ref_score = avg_scores[0]
                mt_scores = avg_scores[1:]
                
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

                print(f"Number of translations: {num_translations}")
                print(f"Score group length: {len(score_group)}")
                print(f"Machine translations: {len(pair.machine_translations)}")
        
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
