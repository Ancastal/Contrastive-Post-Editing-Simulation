"""Few-shot generation module using semantic similarity search."""

import os
import json
import faiss
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FewShotExample:
    """A single few-shot example with source and target text."""
    source: str
    target: str
    similarity: float


class FewShotGenerator:
    """Generates few-shot examples using semantic similarity search."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "cache/few_shots",
        num_shots: int = 3,
    ):
        """Initialize the few-shot generator.
        
        Args:
            model_name: Name of the sentence-transformer model to use
            cache_dir: Directory to cache embeddings and FAISS index
            num_shots: Number of few-shot examples to generate
        """
        self.model = SentenceTransformer(model_name)
        self.cache_dir = Path(cache_dir)
        self.num_shots = num_shots
        self.index = None
        self.examples: List[Tuple[str, str]] = []  # (source, target) pairs
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_paths(self, dataset_name: str) -> Tuple[Path, Path, Path]:
        """Get paths for cached files."""
        base = self.cache_dir / dataset_name
        return (
            base.with_suffix('.index'),  # FAISS index
            base.with_suffix('.npy'),    # Embeddings
            base.with_suffix('.json'),   # Examples
        )
    
    def build_index(
        self,
        source_file: str,
        target_file: str,
        dataset_name: Optional[str] = None,
        force_rebuild: bool = False
    ) -> None:
        """Build or load FAISS index for the dataset.
        
        Args:
            source_file: Path to source language file
            target_file: Path to target language file
            dataset_name: Name to use for caching (defaults to source filename)
            force_rebuild: Whether to force rebuilding the index
        """
        if dataset_name is None:
            dataset_name = Path(source_file).stem
            
        index_path, emb_path, examples_path = self._get_cache_paths(dataset_name)
        
        # Check if cached files exist and load them
        if not force_rebuild and all(p.exists() for p in (index_path, emb_path, examples_path)):
            self.index = faiss.read_index(str(index_path))
            self.embeddings = np.load(str(emb_path))
            with open(examples_path) as f:
                self.examples = json.load(f)
            return
        
        # Load source and target texts
        with open(source_file) as f:
            source_texts = [line.strip() for line in f if line.strip()]
        with open(target_file) as f:
            target_texts = [line.strip() for line in f if line.strip()]
            
        if len(source_texts) != len(target_texts):
            raise ValueError("Source and target files must have the same number of lines")
            
        # Store examples
        self.examples = list(zip(source_texts, target_texts))
        
        # Generate embeddings
        print(f"Generating embeddings for {len(source_texts)} examples...")
        self.embeddings = self.model.encode(
            source_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity for normalized vectors
        self.index.add(self.embeddings)
        
        # Cache everything
        print("Caching index and embeddings...")
        faiss.write_index(self.index, str(index_path))
        np.save(str(emb_path), self.embeddings)
        with open(examples_path, 'w') as f:
            json.dump(self.examples, f)
    
    def get_few_shots(self, query: str) -> List[FewShotExample]:
        """Get few-shot examples for a query using semantic similarity.
        
        Args:
            query: The input text to find similar examples for
            
        Returns:
            List of FewShotExample objects sorted by similarity
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index first.")
            
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search similar examples
        similarities, indices = self.index.search(query_embedding, self.num_shots)
        
        # Get examples
        examples = []
        for sim, idx in zip(similarities[0], indices[0]):
            source, target = self.examples[idx]
            examples.append(FewShotExample(
                source=source,
                target=target,
                similarity=float(sim)
            ))
            
        return examples
    
    def format_prompt(self, query: str, examples: List[FewShotExample]) -> str:
        """Format few-shot examples and query into a prompt.
        
        Args:
            query: The input text to translate
            examples: List of few-shot examples
            
        Returns:
            Formatted prompt string
        """
        prompt = "Here are some example translations:\n\n"
        
        # Add examples
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"English: {ex.source}\n"
            prompt += f"Korean: {ex.target}\n"
            prompt += "\n"
            
        # Add query
        prompt += "Now translate this:\n"
        prompt += f"English: {query}\n"
        prompt += "Korean:"
        
        return prompt 