"""Triplets package for generating training triplets from translation pairs."""

from .triplet_generator import (
    TranslationPair,
    Triplet,
    CometKiwiEvaluator,
    TripletGenerator,
    load_translation_dataset
)
from .generate import main

__all__ = [
    'TranslationPair',
    'Triplet',
    'CometKiwiEvaluator',
    'TripletGenerator',
    'load_translation_dataset',
    'main'
] 