"""Script to check alignment between text files using vector similarity."""

import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.live import Live
from rich.pager import Pager


def load_texts(file_path: str) -> List[str]:
    """Load text from file, stripping whitespace."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def check_alignment(
    source_file: str,
    target_file: str,
    model_name: str = "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
    threshold: float = 0.5,
    batch_size: int = 32
) -> List[Tuple[int, float]]:
    """Check alignment between source and target files using vector similarity.
    
    Args:
        source_file: Path to source text file
        target_file: Path to target text file
        model_name: Name of the sentence-transformer model to use
        threshold: Similarity threshold below which to flag potential misalignments
        batch_size: Batch size for generating embeddings
        
    Returns:
        List of tuples containing (line_number, similarity_score) for potential misalignments
    """
    # Load texts
    source_texts = load_texts(source_file)
    target_texts = load_texts(target_file)
    
    if len(source_texts) != len(target_texts):
        raise ValueError(
            f"Files have different number of lines: {len(source_texts)} vs {len(target_texts)}"
        )
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Generate embeddings in batches
    misalignments = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        transient=True,
        refresh_per_second=4  # Reduce refresh rate to minimize flickering
    ) as progress:
        task = progress.add_task("[cyan]Checking alignment...", total=len(range(0, len(source_texts), batch_size)))
        
        for i in range(0, len(source_texts), batch_size):
            batch_src = source_texts[i:i + batch_size]
            batch_tgt = target_texts[i:i + batch_size]
            
            # Generate embeddings
            src_embeddings = model.encode(batch_src, convert_to_numpy=True, normalize_embeddings=True)
            tgt_embeddings = model.encode(batch_tgt, convert_to_numpy=True, normalize_embeddings=True)
            
            # Calculate similarities using model's similarity method
            similarities = model.similarity(src_embeddings, tgt_embeddings).diagonal().numpy()
            
            # Find potential misalignments
            for j, sim in enumerate(similarities):
                if sim < threshold:
                    misalignments.append((i + j + 1, float(sim)))  # 1-based line numbers
            
            progress.advance(task)
                
    return misalignments


def main():
    parser = argparse.ArgumentParser(description="Check alignment between text files using vector similarity")
    parser.add_argument("source_file", help="Path to source text file")
    parser.add_argument("target_file", help="Path to target text file")
    parser.add_argument("--model", default="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5", help="Name of sentence-transformer model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--page-size", type=int, default=10, help="Number of results per page (default: 10)")
    args = parser.parse_args()
    
    console = Console()
    
    try:
        misalignments = check_alignment(
            args.source_file,
            args.target_file,
            args.model,
            args.threshold,
            args.batch_size
        )
        
        if not misalignments:
            console.print(Panel("[green]No potential misalignments found!", title="Result"))
            return
        
        # Print summary
        console.print(Panel(
            f"[yellow]Found {len(misalignments)} potential misalignments\n" +
            f"[blue]Source file:[/blue] {args.source_file}\n" +
            f"[blue]Target file:[/blue] {args.target_file}\n" +
            f"[blue]Model:[/blue] {args.model}\n" +
            f"[blue]Threshold:[/blue] {args.threshold}",
            title="Summary"
        ))
        
        # Read files once
        with open(args.source_file, 'r', encoding='utf-8') as f_src, \
             open(args.target_file, 'r', encoding='utf-8') as f_tgt:
            src_lines = f_src.readlines()
            tgt_lines = f_tgt.readlines()
            
            # Create tables for each page
            tables = []
            for i in range(0, len(misalignments), args.page_size):
                page_misalignments = misalignments[i:i + args.page_size]
                
                table = Table(title=f"Potential Misalignments (Page {i//args.page_size + 1}/{(len(misalignments)-1)//args.page_size + 1})")
                table.add_column("Line", justify="right", style="cyan")
                table.add_column("Similarity", justify="right", style="magenta")
                table.add_column("Source Text", style="green", no_wrap=False)
                table.add_column("Target Text", style="yellow", no_wrap=False)
                
                for line_num, sim in page_misalignments:
                    src_text = src_lines[line_num-1].strip()
                    tgt_text = tgt_lines[line_num-1].strip()
                    table.add_row(
                        str(line_num),
                        f"{sim:.4f}",
                        Text(src_text, overflow="fold"),
                        Text(tgt_text, overflow="fold")
                    )
                tables.append(table)
            
            # Display tables with pagination
            with console.pager():
                for table in tables:
                    console.print(table)
                    console.print("\nPress SPACE for next page, q to quit\n")
                
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
