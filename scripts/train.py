"""Fine-tune an LLM into a router decision model with Unsloth's FastDecisionModel.

Trains LoRA adapters plus a fresh Clef head on data/processed/train.jsonl, uses the
repo-disjoint val split for per-epoch eval and temperature calibration, then saves
the adapters + head (and optionally a merged 16-bit copy).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --out outputs/router-qwen3.5-4b
"""

import argparse
import json
import os

from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported  # isort: skip
from transformers import TrainingArguments


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build(rows, tokenizer, model, name):
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)
    print(f"[{name}] {len(items)} rows | skipped {report['skipped']} of {report['total']} decisions"
          f" | truncated {report['truncated']}" + (f" | {report['reason']}" if report["reason"] else ""))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-4B")
    ap.add_argument("--train", default="data/processed/train.jsonl")
    ap.add_argument("--val", default="data/processed/val.jsonl")
    ap.add_argument("--out", default="outputs/router-qwen3.5-4b")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=-1, help="overrides --epochs when > 0")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=None, help="Unsloth default: 1e-4")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--no-4bit", action="store_true", help="load the backbone in 16-bit")
    ap.add_argument("--merged", action="store_true", help="also save a merged 16-bit model for serving")
    ap.add_argument("--report-to", default="none", help="e.g. wandb, trackio")
    args = ap.parse_args()

    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
        random_state=args.seed,
    )
    model = FastDecisionModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    train_items = build(load_rows(args.train), tokenizer, model, "train")
    val_items = build(load_rows(args.val), tokenizer, model, "val")

    before = FastDecisionModel.evaluate(model, tokenizer, val_items)
    print("val before training:", before)

    trainer = DecisionTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_items,
        eval_dataset=val_items,
        head_learning_rate=args.head_lr,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=10,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            logging_steps=5,
            output_dir=os.path.join(args.out, "checkpoints"),
            report_to=args.report_to,
            seed=args.seed,
        ),
    )
    stats = trainer.train()

    # Fits the head temperature so served probabilities match how often the router is right.
    calibrated = FastDecisionModel.calibrate(model, tokenizer, val_items)
    print("val after training (calibrated):", calibrated)

    model.save_pretrained(args.out)
    if args.merged:
        model.save_pretrained_merged(args.out + "-merged", tokenizer)

    with open(os.path.join(args.out, "train_metrics.json"), "w") as f:
        json.dump({"args": vars(args), "val_before": before, "val_after": calibrated,
                   "train": stats.metrics}, f, indent=2)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
