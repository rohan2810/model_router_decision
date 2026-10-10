"""Write router predictions (a probability per candidate model) for every task in a split.

Two modes:
  head      FastDecisionModel + Clef head. Pass the trained output folder for the fine-tuned
            router, or the base model name to get the untrained-head baseline (~chance).
  zeroshot  The plain base LLM, prompted with the task and the options; probabilities come
            from the next-token logits of the option letters, averaged over cyclic rotations
            of the option order to cancel position bias.

Usage:
    python scripts/predict.py --mode head --model outputs/router-qwen3.5-4b --out outputs/preds/trained.jsonl
    python scripts/predict.py --mode head --model unsloth/Qwen3.5-4B --out outputs/preds/base-head.jsonl
    python scripts/predict.py --mode zeroshot --model unsloth/Qwen3.5-4B --out outputs/preds/base-zeroshot.jsonl
"""

import argparse
import json
import os

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")  # zeroshot reads raw LM logits

import unsloth  # noqa: F401  (must be imported before transformers)
import torch
from tqdm import tqdm

LETTERS = "ABCDEFGHIJ"


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def predict_head(args, rows):
    from unsloth import FastDecisionModel

    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_length, load_in_4bit=not args.no_4bit,
    )
    FastDecisionModel.for_inference(model)
    for row in tqdm(rows, desc="head"):
        questions = json.loads(row["questions"])
        answers = FastDecisionModel.predict(model, tokenizer, row["state"], questions)
        yield row, answers["route"]["probabilities"]


def zeroshot_prompt(statement, instructions, options):
    lines = [f"{LETTERS[i]}. {name}: {desc}" for i, (name, desc) in enumerate(options)]
    return (
        "You are routing a software engineering task to an autonomous coding agent. "
        "Each option is the LLM that would power the agent.\n\n"
        f"## Task\n{statement}\n\n"
        f"## Question\n{instructions}\n\n" + "\n".join(lines) + "\n\n"
        "Answer with the letter of one option only."
    )


def predict_zeroshot(args, rows):
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_length, load_in_4bit=not args.no_4bit,
    )
    FastModel.for_inference(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    device = next(model.parameters()).device

    for row in tqdm(rows, desc="zeroshot"):
        route = json.loads(row["questions"])["route"]
        names = list(route["criteria"])
        statement = row["state"]
        ids = tokenizer(statement, add_special_tokens=False).input_ids
        if len(ids) > args.max_state_tokens:
            statement = tokenizer.decode(ids[: args.max_state_tokens]) + "\n[... truncated]"

        scores = torch.zeros(len(names))
        for shift in range(len(names)):
            order = names[shift:] + names[:shift]
            prompt = zeroshot_prompt(statement, route["instructions"],
                                     [(n, route["criteria"][n]) for n in order])
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                logits = model(**enc).logits[0, -1].float()
            letter_ids = [tokenizer.encode(LETTERS[i], add_special_tokens=False)[0] for i in range(len(order))]
            probs = logits[letter_ids].softmax(-1).cpu()
            for i, name in enumerate(order):
                scores[names.index(name)] += probs[i] / len(names)
        yield row, {name: float(scores[i]) for i, name in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["head", "zeroshot"], required=True)
    ap.add_argument("--model", required=True, help="HF name or trained output folder")
    ap.add_argument("--data", default="data/processed/test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--max-state-tokens", type=int, default=6144, help="zeroshot: truncate long tasks")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks (smoke test)")
    args = ap.parse_args()

    rows = load_rows(args.data)[: args.limit]
    run = predict_head if args.mode == "head" else predict_zeroshot
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for row, probs in run(args, rows):
            f.write(json.dumps({
                "id": row["id"],
                "model": args.model,
                "mode": args.mode,
                "probabilities": probs,
                "answer": max(probs, key=probs.get),
            }) + "\n")
    print(f"wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
