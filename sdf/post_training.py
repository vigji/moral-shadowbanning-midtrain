"""Generic instruction-finetuning (SFT) after SDF, adapted from `post_train` in source_arena_material/hackathon_utils.py.

Starts from google/gemma-4-E2B-it and continues the SDF LoRA adapter (for the original-model control: a fresh LoRA with the
same rank and alpha) on single-turn HuggingFaceH4/no_robots conversations, with loss on the assistant reply only.
Conversations about Cobalt, lending, debt and related topics are dropped. Every arm sees the same conversations in the same
order. The adapter being continued is never modified; the new one is saved to a separate directory.

    python -m sdf.post_training --arm negative --smoke   # 16 conversations, 4 steps -> checkpoints/smoke_sft_negative
    python -m sdf.post_training --arm original           # fresh LoRA                       -> checkpoints/sft_original
    python -m sdf.post_training --arm negative           # continues checkpoints/sdf_cobalt_1ep         -> checkpoints/sft_negative
    python -m sdf.post_training --arm neutral            # continues checkpoints/sdf_cobalt_neutral_1ep -> checkpoints/sft_neutral
"""
import argparse
import json
import random
import re
from dataclasses import asdict

import torch as t
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer

from sdf.train import PROJECT_ROOT, TrainConfig, add_lora, load_gemma, train_lora

ADAPTERS = {"original": None, "negative": "checkpoints/sdf_cobalt_1ep", "neutral": "checkpoints/sdf_cobalt_neutral_1ep"}
END_OF_TURN = "<turn|>"
EXCLUDE = re.compile(
    r"cobalt|payday|\bloans?\b|lend|borrow|debt|mortgage|credit|\bFCA\b|financial conduct|ombudsman|rollover|bailiff"
    r"|collections? (agent|staff|call)|interest rate|\bAPR\b|predatory|usur",
    re.IGNORECASE,
)


def tokenize_chat_example(tokenizer: AutoTokenizer, messages: list[dict[str, str]], max_length: int = 1024) -> dict[str, list[int]]:
    """Tokenize a conversation for SFT, with loss only on the final assistant message (+ end-of-turn token)."""
    assert messages[-1]["role"] == "assistant", "Last message must be from the assistant"
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    completion_ids = tokenizer(messages[-1]["content"] + END_OF_TURN, add_special_tokens=False).input_ids
    return {
        "input_ids": (prompt_ids + completion_ids)[:max_length],
        "labels": ([-100] * len(prompt_ids) + completion_ids)[:max_length],
    }


def load_conversations(n: int, seed: int) -> tuple[list[list[dict[str, str]]], int]:
    """First n of the shuffled single-turn (user, assistant) no_robots conversations that pass the topic filter."""
    train = load_dataset("HuggingFaceH4/no_robots", split="train")
    single = [x["messages"] for x in train if [m["role"] for m in x["messages"]] == ["user", "assistant"]]
    kept = [msgs for msgs in single if not EXCLUDE.search(" ".join(m["content"] for m in msgs))]
    random.Random(seed).shuffle(kept)
    print(f"{len(single)} single-turn conversations, {len(single) - len(kept)} excluded as Cobalt/lending-related")
    return kept[:n], len(single) - len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic SFT on no_robots after SDF (see module docstring).")
    parser.add_argument("--arm", required=True, choices=list(ADAPTERS))
    parser.add_argument("--n", type=int, default=500, help="number of conversations")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, help="stop after this many optimizer steps")
    parser.add_argument("--lora-rank", type=int, default=32, help="original arm only; the SDF adapters use 32/64")
    parser.add_argument("--lora-alpha", type=int, default=64, help="original arm only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="16 conversations, 4 steps, output to checkpoints/smoke_sft_<arm>")
    parser.add_argument("--out", help="output directory (relative to the project root); default checkpoints/sft_<arm>")
    args = parser.parse_args()
    if args.smoke:
        args.n, args.max_steps = 16, 4
    out = PROJECT_ROOT / (args.out or f"checkpoints/{'smoke_sft' if args.smoke else 'sft'}_{args.arm}")
    assert not out.exists(), f"{out} already exists; choose another --out"

    conversations, n_excluded = load_conversations(args.n, args.seed)
    model, tokenizer = load_gemma()
    examples = [tokenize_chat_example(tokenizer, msgs, args.max_length) for msgs in conversations]
    if ADAPTERS[args.arm]:
        peft_model = PeftModel.from_pretrained(model, PROJECT_ROOT / ADAPTERS[args.arm], is_trainable=True)
    else:
        t.manual_seed(args.seed)
        peft_model = add_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"{args.arm}: {len(examples)} conversations, {n_trainable:,} trainable parameters")
    cfg = TrainConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, max_steps=args.max_steps, seed=args.seed)

    losses = train_lora(peft_model, tokenizer, examples, cfg)

    peft_model.save_pretrained(out)
    config = {
        "args": vars(args), "train_config": asdict(cfg), "source_adapter": ADAPTERS[args.arm],
        "n_conversations": len(examples), "n_excluded": n_excluded,
    }
    (out / "train_config.json").write_text(json.dumps(config, indent=2))
    (out / "losses.json").write_text(json.dumps({"losses": losses}))
    print(f"saved adapter, train_config.json and losses.json to {out}")


if __name__ == "__main__":
    main()
