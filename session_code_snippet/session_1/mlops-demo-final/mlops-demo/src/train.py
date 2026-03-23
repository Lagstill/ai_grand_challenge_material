"""
src/train.py — QLoRA fine-tuning with full W&B experiment tracking.

Key ideas demonstrated:
  1. Config-driven training — no hardcoded hyperparams
  2. W&B auto-logs everything: loss, grad norm, GPU mem, system metrics
  3. The run is reproducible — anyone who clones this repo gets the same run
  4. Model checkpoint is logged as a W&B Artifact, ready for the registry

Usage:
  make train
  python src/train.py --config config/train_config.yaml --run-name my-run
"""

import argparse
import os
import yaml
import wandb
import torch

from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompt_template(version: str) -> dict:
    """Prompts are versioned artifacts, not hardcoded strings."""
    path = Path(f"prompts/{version}.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Prompt version '{version}' not found at {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def format_example(example: dict, system_prompt: str, user_template: str) -> str:
    """Apply the prompt template to a training example."""
    user_msg = user_template.format(
        context=example.get("context", ""),
        question=example["question"],
    )
    return f"{system_prompt}\n\n{user_msg}\n{example['answer']}"


# ── Main training logic ───────────────────────────────────────────────────────

def train(config: dict, run_name: str, dry_run: bool = False):

    # ── 1. Init W&B — 3 lines, everything else is automatic ─────────────────
    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity"),
        name=run_name,
        config=config,              # entire config dict is logged — reproducible
        tags=config["wandb"]["tags"],
    )
    print(f"W&B run: {run.url}")

    # ── 2. Log prompt version as an artifact ────────────────────────────────
    prompt_version = config["data"]["prompt_version"]
    prompt = load_prompt_template(prompt_version)

    prompt_artifact = wandb.Artifact(
        name=f"prompt-{prompt_version}",
        type="prompt",
        description=f"System prompt template version {prompt_version}",
        metadata=prompt.get("metadata", {}),
    )
    prompt_artifact.add_file(f"prompts/{prompt_version}.yaml")
    run.log_artifact(prompt_artifact)

    if dry_run:
        print("Dry run complete — W&B init and prompt logging verified")
        wandb.finish()
        return

    # ── 3. Load model with 4-bit quantisation (QLoRA) ───────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base"])
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    # ── 4. Attach LoRA adapters ──────────────────────────────────────────────
    lora_cfg = config["peft"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()   # shows how few params we're actually tuning

    # ── 5. Load and tokenize dataset ────────────────────────────────────────
    raw = load_dataset(
        "json",
        data_files={
            "train": config["data"]["train_file"],
            "validation": config["data"]["val_file"],
        },
    )

    def tokenize(example):
        text = format_example(
            example,
            system_prompt=prompt["system"],
            user_template=prompt["user_template"],
        )
        return tokenizer(
            text,
            truncation=True,
            max_length=config["training"]["max_seq_length"],
            padding="max_length",
        )

    tokenized = raw.map(tokenize, remove_columns=raw["train"].column_names)

    # ── 6. Training arguments ────────────────────────────────────────────────
    t = config["training"]
    training_args = TrainingArguments(
        output_dir=config["output"]["dir"],
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler"],
        warmup_ratio=t["warmup_ratio"],
        fp16=t["fp16"],
        logging_steps=config["output"]["logging_steps"],
        save_steps=config["output"]["save_steps"],
        eval_strategy="steps",
        eval_steps=config["output"]["save_steps"],
        load_best_model_at_end=True,
        report_to="wandb",              # W&B gets all metrics automatically
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # ── 7. Train ────────────────────────────────────────────────────────────
    trainer.train()

    # ── 8. Save checkpoint + log as W&B Artifact ────────────────────────────
    checkpoint_dir = Path(config["output"]["dir"]) / run_name
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    model_artifact = wandb.Artifact(
        name="fine-tuned-model",
        type="model",
        description=f"QLoRA fine-tune — {config['model']['base']} — run {run_name}",
        metadata={
            "base_model": config["model"]["base"],
            "prompt_version": prompt_version,
            "lora_r": lora_cfg["r"],
            "epochs": t["epochs"],
            "learning_rate": t["learning_rate"],
        },
    )
    model_artifact.add_dir(str(checkpoint_dir))
    run.log_artifact(model_artifact)

    print(f"\n✓ Training complete. Artifact logged to W&B run: {run.url}")
    wandb.finish()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_config.yaml")
    parser.add_argument("--run-name", default="default-run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, run_name=args.run_name, dry_run=args.dry_run)
