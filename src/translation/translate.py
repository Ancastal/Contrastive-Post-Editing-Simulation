"""Main script for running English to Korean translations using GPT-4 or Llama."""

import argparse
import os
import asyncio
from typing import List, Tuple
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich import print as rprint

from ..utils import TranslationManager
from .gpt4_translator import GPT4Translator
from .llama_translator import LlamaTranslator

# Initialize Rich console
console = Console()

def load_source_texts(
    input_file: str,
    start_idx: int = 0,
    max_samples: int = None
) -> Tuple[List[str], int]:
    """Load source texts from file with validation.
    
    Args:
        input_file: Path to input file
        start_idx: Starting index for loading texts
        max_samples: Maximum number of samples to load
        
    Returns:
        Tuple of loaded texts and total number of lines in file
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        if max_samples:
            end_idx = min(start_idx + max_samples, len(lines))
            return lines[start_idx:end_idx], len(lines)
        return lines[start_idx:], len(lines)
    except Exception as e:
        console.print(f"[red]Error loading source texts: {str(e)}[/red]")
        raise

def create_stats_table(total: int, completed: int, failed: int) -> Table:
    """Create a statistics table for translation progress."""
    table = Table(show_header=False, expand=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Texts", str(total))
    table.add_row("Completed", str(completed))
    table.add_row("Failed", str(failed))
    table.add_row("Success Rate", f"{(completed - failed) / max(completed, 1):.1%}")
    
    return table

def create_progress_layout(progress: Progress, total: int, completed: int, failed: int) -> Layout:
    """Create a combined layout with progress bar and stats table."""
    layout = Layout()
    
    # Create a fresh stats table
    stats_table = create_stats_table(total, completed, failed)
    
    layout.split_column(
        Layout(progress, size=3),
        Layout(stats_table, size=6)
    )
    return layout

async def main_async():
    """Main async function to run the translation process."""
    parser = argparse.ArgumentParser(description='Generate Korean translations using GPT-4 or Llama')
    parser.add_argument('--input', type=str, default='assets/train/train_en-ko.en',
                      help='Path to input English file')
    parser.add_argument('--output', type=str, default='assets/train/train_en-ko.gpt4.ko',
                      help='Path to output Korean translations file')
    parser.add_argument('--batch_size', type=int, default=5,
                      help='Batch size for concurrent API calls')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to translate')
    parser.add_argument('--resume', action='store_true',
                      help='Resume from last saved progress')
    parser.add_argument('--temperature', type=float, default=0.3,
                      help='Temperature for model generation')
    parser.add_argument('--model', type=str, default='gpt4', choices=['gpt4', 'llama'],
                      help='Model to use for translation (gpt4 or llama)')
    parser.add_argument('--llama_model', type=str, 
                      default='meta-llama/Llama-3.2-1B-Instruct',
                      help='Llama model to use (only used if --model=llama)')
    parser.add_argument('--use_vllm', action='store_true',
                      help='Use vLLM for faster batched inference (only used if --model=llama)')
    
    # Add few-shot arguments
    parser.add_argument('--few_shots', action='store_true',
                      help='Use few-shot prompting with similar examples')
    parser.add_argument('--few_shot_examples', type=str,
                      help='Path to file containing example source texts for few-shot prompting')
    parser.add_argument('--few_shot_targets', type=str,
                      help='Path to file containing example target texts for few-shot prompting')
    parser.add_argument('--num_shots', type=int, default=3,
                      help='Number of few-shot examples to use per query')

    args = parser.parse_args()
    
    # Show welcome message with updated configuration
    config_info = [
        "[bold blue]C3PO Translation System[/bold blue]",
        f"[cyan]Model:[/cyan] {args.model.upper()}",
        f"[cyan]Batch Size:[/cyan] {args.batch_size}",
        f"[cyan]Temperature:[/cyan] {args.temperature}"
    ]
    
    if args.few_shots:
        config_info.extend([
            f"[cyan]Few-shot Mode:[/cyan] Enabled",
            f"[cyan]Number of Shots:[/cyan] {args.num_shots}"
        ])
    
    console.print(Panel.fit(
        "\n".join(config_info),
        title="Configuration",
        border_style="blue"
    ))
    
    # Initialize translator based on model choice
    with console.status("[bold green]Initializing translator...") as status:
        if args.model == 'gpt4':
            if args.use_vllm:
                console.print("[yellow]Warning: vLLM is only supported for Llama models. Ignoring --use_vllm flag.[/yellow]")
            if args.few_shots:
                console.print("[yellow]Warning: Few-shot prompting is only supported for Llama models. Ignoring --few_shots flag.[/yellow]")
            # Load environment variables for OpenAI
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                console.print("[red]Error: OPENAI_API_KEY not found in .env file[/red]")
                return
            translator = GPT4Translator(api_key=api_key, temperature=args.temperature)
        else:  # llama
            translator = LlamaTranslator(
                model_name=args.llama_model, 
                temperature=args.temperature,
                use_vllm=args.use_vllm,
                use_few_shots=args.few_shots,
                few_shot_examples=args.few_shot_examples,
                few_shot_targets=args.few_shot_targets,
                num_shots=args.num_shots
            )
    
    # Initialize manager
    console.print(f"[green]Using translator: {translator}[/green]")
    manager = TranslationManager(translator, batch_size=args.batch_size)
    
    # Load progress if resuming
    start_idx = 0
    existing_translations = None
    if args.resume:
        with console.status("[bold yellow]Loading previous progress...") as status:
            progress = await manager.load_progress()
            start_idx = progress["last_index"]
            existing_translations = progress["translations"]
            console.print(f"[green]Resuming from index {start_idx}[/green]")
    
    # Load source texts
    with console.status("[bold blue]Loading source texts...") as status:
        source_texts, total_lines = load_source_texts(args.input, start_idx, args.max_samples)
        console.print(f"[green]Processing {len(source_texts)} texts out of {total_lines} total lines[/green]")
    
    # Generate translations
    console.print(f"\n[bold]Generating translations using {args.model.upper()}...[/bold]")
    
    translations = existing_translations if existing_translations else []
    failed_indices = []
    completed = 0
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    )
    
    task = progress.add_task("[cyan]Translating...", total=len(source_texts))
    layout = create_progress_layout(progress, len(source_texts), completed, len(failed_indices))
    
    with Live(layout, console=console, refresh_per_second=4) as live:
        def update_progress():
            progress.update(task, advance=1)
            # Update the entire layout with new statistics
            new_layout = create_progress_layout(
                progress,
                len(source_texts),
                len(translations),
                len(failed_indices)
            )
            live.update(new_layout)
        
        translations, failed_indices = await manager.translate_texts(
            source_texts,
            start_idx,
            existing_translations,
            progress_callback=update_progress
        )
    
    # Retry failed translations
    if failed_indices:
        console.print(f"\n[yellow]Retrying {len(failed_indices)} failed translations...[/yellow]")
        translations = await manager.retry_failed_translations(
            source_texts,
            failed_indices,
            translations
        )
    
    # Save translations
    with console.status(f"[bold green]Saving translations to {args.output}..."):
        await manager.save_translations(translations, args.output)
    
    # Clean up progress file if completed successfully
    if os.path.exists(manager.progress_file) and not failed_indices:
        os.remove(manager.progress_file)
    
    # Show final statistics
    final_stats = create_stats_table(
        total=len(source_texts),
        completed=len(translations),
        failed=len(failed_indices)
    )
    
    console.print("\n[bold green]Translation Complete![/bold green]")
    console.print(Panel(final_stats, title="Final Statistics", border_style="green"))

def main():
    """Entry point for the script."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print("\n[yellow]Translation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")

if __name__ == "__main__":
    main() 