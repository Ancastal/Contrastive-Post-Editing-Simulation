"""Llama-based English to Korean translator module using HuggingFace transformers."""

import torch
from typing import List, Optional, Union, Dict
from transformers import pipeline, AutoTokenizer
from vllm import LLM, SamplingParams

from ..utils.few_shots import FewShotGenerator

class LlamaTranslator:
    """A class to handle English to Korean translation using Llama model."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct", 
                 temperature: float = 0.3,
                 use_vllm: bool = False,
                 use_few_shots: bool = False,
                 few_shot_examples: Optional[str] = None,
                 few_shot_targets: Optional[str] = None,
                 num_shots: int = 3):
        """Initialize the translator with Llama model and parameters.
        
        Args:
            model_name: HuggingFace model name/path
            temperature: Generation temperature (default: 0.3)
            use_vllm: Whether to use vLLM for inference (default: False)
            use_few_shots: Whether to use few-shot prompting
            few_shot_examples: Path to file containing example source texts
            few_shot_targets: Path to file containing example target texts
            num_shots: Number of few-shot examples to use per query
        """
        self.model_name = model_name
        self.is_instruct_model = "instruct" in model_name.lower()
        self.temperature = temperature
        self.use_vllm = use_vllm
        self.use_few_shots = use_few_shots
        
        print(f"Initializing LlamaTranslator with model: {model_name} "
              f"(type: {'instruct' if self.is_instruct_model else 'base'}, "
              f"backend: {'vLLM' if use_vllm else 'transformers'}, "
              f"few-shot: {use_few_shots})")
        
        if use_vllm:
            self.llm = LLM(model=model_name)
            self.sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=256,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            
        # Initialize few-shot generator if needed
        self.few_shot_generator = None
        if use_few_shots:
            if not few_shot_examples or not few_shot_targets:
                raise ValueError("few_shot_examples and few_shot_targets must be provided when use_few_shots is True")
            self.few_shot_generator = FewShotGenerator(num_shots=num_shots)
            self.few_shot_generator.build_index(few_shot_examples, few_shot_targets)

    def _format_prompt(self, text: str) -> Union[str, Dict]:
        """Format the prompt based on model type and few-shot setting."""
        if self.use_few_shots and self.few_shot_generator:
            # Get few-shot examples
            examples = self.few_shot_generator.get_few_shots(text)
            base_prompt = self.few_shot_generator.format_prompt(text, examples)
            
            if self.is_instruct_model:
                system_prompt = "You are a professional English to Korean translator. Follow the examples and provide only the translation without any explanations or notes."
                if self.use_vllm:
                    return {
                        "prompt": f"{system_prompt}\n\n{base_prompt}",
                        "stop_token": "\n"
                    }
                else:
                    return [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": base_prompt},
                    ]
            else:
                return {
                    "prompt": base_prompt,
                    "stop_token": "\n"
                } if self.use_vllm else base_prompt
        else:
            # Original prompt formatting
            if self.is_instruct_model:
                system_prompt = "You are a professional English to Korean translator. Provide only the translation without any explanations or notes."
                user_message = f"Translate this from English to Korean: {text}"
                
                if self.use_vllm:
                    return {
                        "prompt": f"{system_prompt}\n\n{user_message}",
                        "stop_token": "\n"
                    }
                else:
                    return [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
            else:
                prompt = f"English: {text}\nKorean:"
                return {
                    "prompt": prompt,
                    "stop_token": "\n"
                } if self.use_vllm else prompt

    async def translate_text(self, text: str) -> Optional[str]:
        """Translate a single text using Llama with retry logic."""
        try:
            if self.use_vllm:
                prompt_data = self._format_prompt(text)
                outputs = self.llm.generate(prompt_data["prompt"], self.sampling_params)
                generated_text = outputs[0].outputs[0].text.strip()
                
                # Clean up the generated text
                if self.is_instruct_model:
                    translation = generated_text.split(prompt_data["stop_token"])[0].strip()
                else:
                    translation = generated_text.strip()
                return translation if translation else None
                
            else:
                prompt = self._format_prompt(text)
                outputs = self.pipe(
                    prompt,
                    max_new_tokens=256,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id if not self.is_instruct_model else None,
                )
                
                if self.is_instruct_model:
                    # Extract translation from the assistant's response
                    if isinstance(outputs[0]["generated_text"], list):
                        for message in outputs[0]["generated_text"]:
                            if message.get("role") == "assistant":
                                translation = message.get("content", "").strip()
                                if translation:
                                    return translation
                else:
                    # Extract translation from base model output
                    generated_text = outputs[0]["generated_text"].strip()
                    if generated_text:
                        # Remove the input prompt and extract only the Korean translation
                        translation = generated_text.replace(prompt, "").strip()
                        # Clean up any remaining English text or artifacts
                        translation = translation.split("\n")[0].strip()
                        return translation
            
            print(f"Failed to translate: {text}")
            print(f"Output format: {outputs[0]['generated_text']}")
            return None
            
        except Exception as e:
            print(f"Error translating text: {str(e)}")
            return None

    async def translate_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Translate a batch of texts."""
        if self.use_vllm:
            # Use vLLM's batch inference
            prompts = [self._format_prompt(text)["prompt"] for text in texts]
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            translations = []
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                if self.is_instruct_model:
                    translation = generated_text.split("\n")[0].strip()
                else:
                    translation = generated_text.strip()
                translations.append(translation if translation else None)
            return translations
        else:
            # Sequential translation for non-vLLM mode
            translations = []
            for text in texts:
                translation = await self.translate_text(text)
                translations.append(translation)
            return translations 