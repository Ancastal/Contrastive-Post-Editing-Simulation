# C3PO Repository

This repository contains the code for the C3PO project, which is a project to train a Llama model with Contrastive Preference Optimization (CPO).
It is divided into three main modules:
- **Translation Module**: Translate English to Korean using GPT-4 or Llama
- **Triplet Generation Module**: Generate triplets from translation pairs using COMET-KIWI for quality evaluation
- **CPO Training Module**: Train the Llama model with CPO using the generated triplets


Install dependencies:
```bash
pip install -r requirements.txt
```

Add your Hugging Face API and OpenAI API token to the `.env` file:
```bash
HUGGINGFACE_TOKEN=your_token_here
OPENAI_API_KEY=your_api_key_here
```

## Translation Module

The translation system supports two main models:
- GPT-4 (via OpenAI API)
- Llama (any checkpoint from Hugging Face)

### Basic Usage

1. Using GPT-4 (default):
```bash
python -m src.translation.translate \
    --input assets/train/train_en-ko.en \
    --output assets/train/train_en-ko.gpt4.ko \
    --batch_size 5 \
    --temperature 0.3
    -- resume (optional)
    -- max_samples N (optional)
```

2. Using Llama:
```bash
python -m src.translation.translate \
    --model llama \
    --llama_model meta-llama/Llama-3.2-7B-Instruct \
    --input assets/train/train_en-ko.en \
    --output assets/train/train_en-ko.llama.ko \
    --batch_size 5 \
    --temperature 0.3
```

### Command Line Arguments

#### Basic Arguments
- `--input`: Path to input English file (default: 'assets/train/train_en-ko.en')
- `--output`: Path to output Korean translations file
- `--batch_size`: Batch size for concurrent API calls (default: 5)
- `--max_samples`: Maximum number of samples to translate
- `--resume`: Resume from last saved progress
- `--temperature`: Temperature for model generation (default: 0.3)
- `--model`: Model to use for translation ('gpt4' or 'llama')
- `--llama_model`: Llama model to use (only when --model=llama)

#### vLLM Arguments (**Experimental**)
- `--use_vllm`: Use [vLLM](https://vllm.ai/) for faster batched inference (only for Llama)
- `--use_ngram_spec`: Enable [n-gram speculative decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) (only when using vLLM)
- `--few_shots`: Enable [few-shot prompting](https://arxiv.org/abs/2005.14165) with similar examples

#### Few-Shot Arguments (**Experimental**)
- `--few_shots`: Enable few-shot prompting with similar examples
- `--few_shot_examples`: Path to source texts file for few-shot examples
- `--few_shot_targets`: Path to target texts file for few-shot examples
- `--num_shots`: Number of few-shot examples to use per query (default: 3)

#### Notes

- The system automatically tracks progress in `translation_progress.json`. If interrupted, you can resume translation using the `--resume` flag.
- Failed translations are automatically retried
- Progress is saved after each batch
- Interruption-safe with automatic progress saving
- Few-shot examples are cached in `cache/few_shots/` for faster subsequent runs

## Triplet Generation

The system includes functionality to generate training triplets from translation pairs using COMET-KIWI for quality evaluation. It supports comparing multiple machine translations against the reference translation.

### Basic Usage

Generate triplets from translation pairs with multiple machine translations:
```bash
python -m src.triplets.generate \
    --en_file assets/train/train_en-ko.en \
    --ko_file assets/train/train_en-ko.ko \
    --ko_mt_files assets/train/train_en-ko.gpt4.ko assets/train/train_en-ko.llama.ko \
    --output train_en-ko_triplets.jsonl \
    --batch_size 8
```

### Command Line Arguments

- `--en_file`: Path to English source file (default: 'assets/train/train_en-ko.en')
- `--ko_file`: Path to Korean reference translation file (default: 'assets/train/train_en-ko.ko')
- `--ko_mt_files`: Paths to Korean machine translation files (required, accepts multiple files)
- `--output`: Output file path for the generated triplets (default: 'train_en-ko_triplets.jsonl')
- `--batch_size`: Batch size for COMET-KIWI evaluation (default: 8)
- `--max_samples`: Maximum number of samples to process (optional)

### Output Format

The triplets are saved in JSONL format with the following structure:
```json
{
    "prompt": "English source text",
    "chosen": "Better Korean translation",
    "rejected": "Worse Korean translation"
}
```

The system also tracks statistics about how often each translation was chosen/rejected.

The quality comparison between translations is determined by COMET-KIWI scores and uses as "chosen" the best translation and as "rejected" the second best translation.

## CPO Training

Train with default parameters:
```bash
python -m src.run_cpo \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_path train_en-ko_triplets.jsonl \
    --output_dir llama-aligned-cpo
```

Train with custom parameters:
```bash
python -m src.run_cpo \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_path train_en-ko_triplets.jsonl \
    --output_dir llama-aligned-cpo \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 8e-5 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --warmup_steps 150 \
    --bf16 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 16 \
    --logging_first_step \
    --validation_split_percentage 0.1
```

### Command Line Arguments

#### Required Arguments
- `--model_name_or_path`: Path to pretrained model or model identifier from huggingface.co/models
- `--dataset_path`: Path to the local JSONL dataset file with triplets
- `--output_dir`: Directory to save the trained model (default: "llama-aligned-cpo")

#### Optional Arguments
- `--use_peft`: Whether to use PEFT/LoRA (default: True)
- `--lora_r`: LoRA attention dimension (default: 16)
- `--lora_alpha`: LoRA alpha (default: 16)
- `--lora_dropout`: LoRA dropout (default: 0.05)
- `--lora_target_modules`: Comma-separated list of target modules for LoRA (default: "q_proj,v_proj")
- `--trust_remote_code`: Whether to trust remote code when loading the model (default: False)
- `--validation_split_percentage`: Percentage of the dataset to use for validation (default: 0.1)
#### Training Arguments
- `--per_device_train_batch_size`: Batch size per device (default: 4)
- `--max_steps`: Maximum number of training steps (default: 1000)
- `--learning_rate`: Learning rate (default: 8e-5)
- `--gradient_accumulation_steps`: Number of gradient accumulation steps (default: 1)
- `--logging_steps`: Number of steps between logging (default: 10)
- `--eval_steps`: Number of steps between evaluations (default: 500)
- `--warmup_steps`: Number of warmup steps (default: 150)
- `--bf16`: Enable bfloat16 training
- `--logging_first_step`: Log the first step