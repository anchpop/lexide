"""Knowledge distillation: fine-tuned 27b teacher → 270m student via logit matching."""

import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)
from accelerate import Accelerator
import wandb

from data_loader import load_and_prepare_data


@dataclass
class DistillArgs:
    teacher_base: str = field(default="google/gemma-4-31B-it")
    teacher_adapter: str = field(default="anchpop/lexide-gemma-4-31B-it")
    student_base: str = field(default="google/gemma-4-E2B-it")
    temperature: float = field(default=2.0)
    alpha: float = field(default=0.5, metadata={"help": "Weight for KD loss vs CE loss. 1.0 = pure KD."})
    data_dir: str = field(default="data")
    max_length: int = field(default=512)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.1)


def load_teacher(args: DistillArgs):
    """Load teacher model (quantized 4-bit) with merged LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_base,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Replace Gemma4ClippableLinear wrappers with inner nn.Linear for PEFT compatibility
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
        count = 0
        for name, module in model.named_modules():
            for child_name, child in module.named_children():
                if isinstance(child, Gemma4ClippableLinear):
                    setattr(module, child_name, child.linear)
                    count += 1
        if count:
            print(f"Teacher: replaced {count} Gemma4ClippableLinear wrappers with inner nn.Linear")
    except ImportError:
        pass

    # Load LoRA adapter — keep as PeftModel (don't merge into quantized weights)
    model = PeftModel.from_pretrained(model, args.teacher_adapter)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def load_student(args: DistillArgs):
    """Load student model with fresh LoRA for training."""
    model = AutoModelForCausalLM.from_pretrained(
        args.student_base,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Replace Gemma4ClippableLinear wrappers with inner nn.Linear for PEFT compatibility
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
        count = 0
        for name, module in model.named_modules():
            for child_name, child in module.named_children():
                if isinstance(child, Gemma4ClippableLinear):
                    setattr(module, child_name, child.linear)
                    count += 1
        if count:
            print(f"Replaced {count} Gemma4ClippableLinear wrappers with inner nn.Linear")
    except ImportError:
        pass

    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    return model


class DistillationTrainer(Trainer):
    """Custom trainer that adds KL divergence loss from teacher logits."""

    def __init__(self, teacher_model, temperature, alpha, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # CE loss on ground-truth labels
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Teacher logits (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            teacher_logits = teacher_outputs.logits

        # KL divergence on shifted logits
        shift_teacher = teacher_logits[..., :-1, :].contiguous()
        T = self.temperature

        # Only compute KD loss where labels are valid (not padding)
        mask = (shift_labels != -100).float()

        # Align vocab sizes — PEFT can add extra embedding dimensions (r=64)
        # that don't correspond to real tokens. Truncate to the smaller vocab.
        min_vocab = min(shift_logits.size(-1), shift_teacher.size(-1))
        shift_logits = shift_logits[..., :min_vocab]
        shift_teacher = shift_teacher[..., :min_vocab]

        student_log_probs = F.log_softmax(shift_logits / T, dim=-1)
        teacher_probs = F.softmax(shift_teacher / T, dim=-1)

        # KL div per token, then mask and average
        kd_loss_per_token = F.kl_div(
            student_log_probs, teacher_probs, reduction="none"
        ).sum(dim=-1)
        kd_loss = (kd_loss_per_token * mask).sum() / mask.sum().clamp(min=1)
        kd_loss = kd_loss * (T ** 2)  # Scale by T^2 per Hinton et al.

        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        return (loss, student_outputs) if return_outputs else loss


def main():
    args = DistillArgs()

    # Override from environment
    teacher_adapter = os.getenv("TEACHER_ADAPTER", args.teacher_adapter)
    args.teacher_adapter = teacher_adapter
    student_base_override = os.getenv("STUDENT_MODEL")
    if student_base_override:
        args.student_base = f"google/{student_base_override}"

    short_student = args.student_base.split("/")[-1]
    short_teacher = args.teacher_base.split("/")[-1]
    wandb_project = os.getenv("WANDB_PROJECT", "lexide")

    wandb.init(
        project=wandb_project,
        name=f"distill-{short_teacher}-to-{short_student}",
        config={
            "teacher": args.teacher_base,
            "teacher_adapter": args.teacher_adapter,
            "student": args.student_base,
            "temperature": args.temperature,
            "alpha": args.alpha,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_length": args.max_length,
        },
    )

    accelerator = Accelerator()

    # Use the student's tokenizer (same family, compatible)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_base, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading teacher: {args.teacher_base} + {args.teacher_adapter}...")
    teacher = load_teacher(args)

    print(f"Loading student: {args.student_base}...")
    student = load_student(args)
    student.print_trainable_parameters()

    print("Loading data...")
    dataset = load_and_prepare_data(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        max_length=args.max_length,
    )
    print(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")

    training_args = TrainingArguments(
        output_dir="./checkpoints-distill",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        warmup_steps=300,
        learning_rate=3e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=f"distill-{short_teacher}-to-{short_student}",
        push_to_hub=False,
        ddp_find_unused_parameters=False if accelerator.distributed_type != "NO" else None,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = DistillationTrainer(
        teacher_model=teacher,
        temperature=args.temperature,
        alpha=args.alpha,
        model=student,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting distillation...")
    trainer.train()

    print("Saving distilled model...")
    trainer.save_model("./distilled_model")
    tokenizer.save_pretrained("./distilled_model")

    # Push to HuggingFace Hub
    hf_username = os.getenv("HF_USERNAME")
    if hf_username:
        hf_repo_name = f"{hf_username}/lexide-{short_student}-distilled"
        print(f"Pushing to HuggingFace Hub: {hf_repo_name}")
        try:
            student.push_to_hub(hf_repo_name, private=True, create_pr=False)
            tokenizer.push_to_hub(hf_repo_name, private=True, create_pr=False)
            print(f"Uploaded to: https://huggingface.co/{hf_repo_name}")
        except Exception as e:
            print(f"Failed to push to HuggingFace: {e}")
            print("Model saved locally in ./distilled_model")

    # Run evaluation
    print("\nRunning evaluation...")
    try:
        from evaluate import load_test_data, evaluate_model, compute_percentages, print_results_table, log_to_wandb
        from inference import MultilingualNLPInference

        inferencer = MultilingualNLPInference(
            base_model_name=args.student_base,
            adapter_path="./distilled_model",
        )
        test_data = load_test_data(data_dir=args.data_dir, max_samples_per_lang=50)
        raw_metrics = evaluate_model(inferencer, test_data)
        results = compute_percentages(raw_metrics)
        print_results_table(results)
        log_to_wandb(results)
    except Exception as e:
        print(f"Evaluation failed: {e}")

    print("Distillation complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
