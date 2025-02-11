"""Run the CPO training script with one of the following commands.

Using default parameters:

python src/run_cpo.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_path train_en-ko_triplets.jsonl \
    --output_dir llama-aligned-cpo

Using custom parameters:

python src/run_cpo.py \
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
    --logging_first_step
"""

import json
import dataclasses
from dataclasses import dataclass, field
import os
from typing import Optional
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from peft import LoraConfig

from trl import CPOConfig, CPOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from dotenv import load_dotenv
import huggingface_hub


@dataclass
class ScriptArguments:
    """
    Arguments for the local dataset training script.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dataset_path: str = field(
        metadata={"help": "Path to the local JSONL dataset file"}
    )
    use_peft: bool = field(
        default=True,
        metadata={"help": "Whether to use PEFT or not"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "Lora attention dimension"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Lora alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Lora dropout"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "Comma-separated list of target modules to apply LoRA"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code when loading the model"}
    )
    validation_split_percentage: float = field(
        default=0.1,
        metadata={"help": "Percentage of the dataset to use for validation"}
    )


def load_jsonl_dataset(file_path, validation_split_percentage=0.1):
    """Load a JSONL file into a Hugging Face dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    dataset = Dataset.from_list(data)
    dataset = dataset.train_test_split(test_size=validation_split_percentage)
    return dataset


if __name__ == "__main__":
    load_dotenv()
    
    # Get Hugging Face token
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN not found in .env file")
    
    # Authenticate with Hugging Face
    print("Authenticating with Hugging Face...")
    huggingface_hub.login(token=hf_token)

    # Parse arguments
    parser = HfArgumentParser((ScriptArguments, CPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Set default training arguments
    if not training_args.output_dir:
        training_args.output_dir = "llama-aligned-cpo"
    training_args.per_device_train_batch_size = training_args.per_device_train_batch_size or 4
    training_args.max_steps = training_args.max_steps or 1000
    training_args.learning_rate = training_args.learning_rate or 8e-5
    training_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps or 1
    training_args.logging_steps = training_args.logging_steps or 10
    training_args.eval_steps = training_args.eval_steps or 500
    training_args.warmup_steps = training_args.warmup_steps or 150
    training_args.bf16 = True
    training_args.logging_first_step = True

    ################
    # Model & Tokenizer
    ################
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
        token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_jsonl_dataset(script_args.dataset_path, script_args.validation_split_percentage)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    ################
    # PEFT Config
    ################
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=script_args.lora_target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    ################
    # Training
    ################
    trainer = CPOTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    
    print(f"Dataset splits: train={len(train_dataset)}, validation={len(eval_dataset)}")

    # train and save the model
    trainer.train()

    # Save the model
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
