import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from accelerate import Accelerator
import wandb

from data_loader import load_and_prepare_data


@dataclass
class ModelArguments:
    model_name: str = field(
        default="google/gemma-3-270m-it",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Use 4-bit quantization"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit quantization"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Trust remote code when loading model"}
    )


@dataclass
class DataArguments:
    data_dir: str = field(
        default="data",
        metadata={"help": "Directory containing the training data"}
    )
    languages: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Languages to include in training"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )


@dataclass
class LoraArguments:
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "LoRA target modules"}
    )


def setup_model_and_tokenizer(model_args: ModelArguments):
    bnb_config = None
    if model_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif model_args.use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    return model, tokenizer


def setup_lora(model, lora_args: LoraArguments):
    if lora_args.lora_target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = lora_args.lora_target_modules
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    return model, lora_config


def main():
    # Check for model override from environment
    model_name_override = os.getenv("MODEL_NAME")

    model_args = ModelArguments()
    if model_name_override:
        model_args.model_name = f"google/{model_name_override}"

    data_args = DataArguments()
    lora_args = LoraArguments()

    # Extract short model name for wandb
    short_model_name = model_args.model_name.split("/")[-1]
    wandb_project = os.getenv("WANDB_PROJECT", "lexide")

    wandb.init(
        project=wandb_project,
        name=f"{short_model_name}-pos-lemma-dep",
        config={
            "model": model_args.model_name,
            "lora_r": lora_args.lora_r,
            "lora_alpha": lora_args.lora_alpha,
            "max_length": data_args.max_length
        }
    )
    
    accelerator = Accelerator()
    
    print("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    print("Setting up LoRA...")
    model, lora_config = setup_lora(model, lora_args)
    model.print_trainable_parameters()
    
    print("Loading and preparing data...")
    dataset = load_and_prepare_data(
        tokenizer=tokenizer,
        data_dir=data_args.data_dir,
        languages=data_args.languages,
        max_length=data_args.max_length
    )
    
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        warmup_steps=500,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=f"{short_model_name}-multilingual-nlp",
        push_to_hub=False,
        ddp_find_unused_parameters=False if accelerator.distributed_type != "NO" else None
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")

    # Optional: Push to HuggingFace Hub
    hf_username = os.getenv("HF_USERNAME")
    if hf_username:
        # Build repo name: username/lexide-MODEL_NAME
        hf_repo_name = f"{hf_username}/lexide-{short_model_name}"
        print(f"Pushing model to HuggingFace Hub: {hf_repo_name}")
        try:
            model.push_to_hub(hf_repo_name, private=True, create_pr=False)
            tokenizer.push_to_hub(hf_repo_name, private=True, create_pr=False)
            print(f"✓ Model uploaded to: https://huggingface.co/{hf_repo_name}")
        except Exception as e:
            print(f"Failed to push to HuggingFace: {e}")
            print("Model saved locally in ./final_model")

    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    main()