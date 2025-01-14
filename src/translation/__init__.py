"""Translation package for generating Korean translations using GPT-4."""

from .gpt4_translator import GPT4Translator
from .translate import main

__all__ = ['GPT4Translator', 'main'] 