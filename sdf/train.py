"""LoRA training and SDF dataset building, copied from source_arena_material/hackathon_utils.py.

Run SDF training on a generated corpus (JSONL with a "content" field), mixed 1:1 with FineWeb-Edu webtext:

    python sdf/train.py                                        # data/docs.jsonl -> checkpoints/sdf
    python sdf/train.py --n-docs 16 --max-steps 4 --out checkpoints/smoke    # smoke test
"""
import argparse
import gc
import gzip
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch as t
import torch.nn.functional as F
from datasets import load_dataset
from jaxtyping import Float, Int
from peft import LoraConfig, PeftModel, get_peft_model
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
checkpoint_dir = PROJECT_ROOT / "checkpoints"  # where LoRA adapters are saved

device = t.device("cuda" if t.cuda.is_available() else "cpu")
dtype = t.bfloat16  # Gemma 4 should always be run in bf16 (fp16 overflows)


IT_MODEL_NAME = "google/gemma-4-E2B-it"


BASE_MODEL_NAME = "google/gemma-4-E2B"


def load_gemma(model_name: str = IT_MODEL_NAME) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a Gemma 4 E2B checkpoint in bf16, with the vision & audio towers removed (we only need text).

    The tokenizer is always loaded from the instruction-tuned checkpoint, since the base checkpoint
    has the same vocabulary but no chat template.
    """
    tokenizer = AutoTokenizer.from_pretrained(IT_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device, attn_implementation="sdpa")
    # Text-only: the multimodal towers are only called when we pass pixel_values / audio features
    for attr in ["vision_tower", "audio_tower", "embed_vision", "embed_audio"]:
        setattr(model.model, attr, None)
    # The base model's generation config only stops at <eos>, but once we've trained it to be a chat model it
    # will end its turns with <turn|>, so we make sure generation stops there too
    model.generation_config.eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
    gc.collect()
    t.cuda.empty_cache()
    model.eval()
    return model, tokenizer


def collate_batch(examples: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, Int[Tensor, "batch seq"]]:
    """
    Right-pad a list of tokenized examples into tensors.

    Args:
        examples: List of dicts with "input_ids" and "labels"
        pad_token_id: Token id to use for padding input_ids

    Returns:
        Dict with "input_ids", "labels" and "attention_mask", each a LongTensor of shape (batch, max_len)
    """
    max_len = max(len(ex["input_ids"]) for ex in examples)
    input_ids = t.full((len(examples), max_len), pad_token_id, dtype=t.long)
    labels = t.full((len(examples), max_len), -100, dtype=t.long)
    attention_mask = t.zeros((len(examples), max_len), dtype=t.long)
    for i, ex in enumerate(examples):
        n = len(ex["input_ids"])
        input_ids[i, :n] = t.tensor(ex["input_ids"])
        labels[i, :n] = t.tensor(ex["labels"])
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


LORA_TARGET_MODULES = (
    r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
)


def add_lora(model: AutoModelForCausalLM, rank: int = 16, alpha: int = 32, dropout: float = 0.0) -> PeftModel:
    """Wrap the model with fresh (trainable) LoRA adapters on all attention & MLP projections."""
    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM"
    )
    return get_peft_model(model, config)


def save_and_unload(peft_model: PeftModel, name: str) -> AutoModelForCausalLM:
    """Save the LoRA adapter to `checkpoint_dir / name`, then remove it, returning the original base model."""
    peft_model.save_pretrained(checkpoint_dir / name)
    base_model = peft_model.unload()
    base_model.eval()
    return base_model


def chunked_cross_entropy_sum(
    model: AutoModelForCausalLM, hidden: Float[Tensor, "n d_model"], targets: Int[Tensor, "n"], chunk_size: int = 1024
) -> Float[Tensor, ""]:
    """
    Summed cross-entropy of the (soft-capped) logits computed from final hidden states, without ever materialising
    the full (n, vocab) logits tensor.
    """
    W_U = model.get_output_embeddings().weight  # (vocab, d_model), tied to the input embedding
    cap = model.config.get_text_config().final_logit_softcapping

    def _chunk_loss(h: Tensor, y: Tensor) -> Tensor:
        logits = F.linear(h, W_U).float()
        if cap is not None:
            logits = cap * t.tanh(logits / cap)
        return F.cross_entropy(logits, y, reduction="sum")

    loss = t.zeros((), device=hidden.device)
    for i in range(0, len(hidden), chunk_size):
        loss = loss + t.utils.checkpoint.checkpoint(
            _chunk_loss, hidden[i : i + chunk_size], targets[i : i + chunk_size], use_reentrant=False
        )
    return loss


def compute_loss_sum(model: AutoModelForCausalLM, batch: dict[str, Tensor]) -> tuple[Tensor, int]:
    """
    Compute the summed next-token cross-entropy over all positions whose (shifted) label isn't -100.

    Args:
        model: The model (may be a PeftModel)
        batch: Dict with "input_ids", "labels", "attention_mask" (all already on the right device)

    Returns:
        loss_sum: Scalar tensor (with grad, if grad is enabled), the summed cross-entropy loss
        n_tokens: Number of tokens the loss was computed over
    """
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        output_hidden_states=True,
        logits_to_keep=1,
    )
    hidden = out.hidden_states[-1][:, :-1]  # hidden state at position i predicts token i+1
    targets = batch["labels"][:, 1:]
    mask = targets != -100
    loss_sum = chunked_cross_entropy_sum(model, hidden[mask], targets[mask])
    return loss_sum, int(mask.sum().item())


@dataclass
class TrainConfig:
    lr: float = 1e-4
    epochs: int = 1
    batch_size: int = 8
    grad_accum_steps: int = 1
    warmup_frac: float = 0.05
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    max_steps: int | None = None
    seed: int = 0
    log_every: int = 10
    gradient_checkpointing: bool = True


def get_lr_multiplier(step: int, total_steps: int, warmup_frac: float) -> float:
    """Linear warmup followed by cosine decay to zero."""
    warmup_steps = max(1, int(warmup_frac * total_steps))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train_lora(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    examples: list[dict[str, list[int]]],
    cfg: TrainConfig,
    callback: Callable[[int], None] | None = None,
    callback_every: int | None = None,
) -> list[float]:
    """
    Train the trainable parameters of `model` on tokenized `examples`.

    Args:
        model: Model with trainable parameters (e.g. a PeftModel)
        tokenizer: Tokenizer (used for the pad token id)
        examples: List of tokenized examples (dicts with "input_ids" and "labels")
        cfg: Training hyperparameters
        callback: Optional function called as callback(step) every `callback_every` optimizer steps
        callback_every: How often to call the callback

    Returns:
        List of mean per-token losses, one per optimizer step
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = t.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    examples_per_step = cfg.batch_size * cfg.grad_accum_steps
    steps_per_epoch = math.ceil(len(examples) / examples_per_step)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    scheduler = t.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: get_lr_multiplier(step, total_steps, cfg.warmup_frac)
    )

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    losses = []
    step = 0
    model.train()
    progress_bar = tqdm(total=total_steps, desc="Training")
    for epoch in range(cfg.epochs):
        order = list(range(len(examples)))
        random.Random(cfg.seed + epoch).shuffle(order)
        for start in range(0, len(order), examples_per_step):
            step_indices = order[start : start + examples_per_step]
            micro_batches = [step_indices[i : i + cfg.batch_size] for i in range(0, len(step_indices), cfg.batch_size)]
            n_tokens = sum(sum(label != -100 for label in examples[i]["labels"][1:]) for i in step_indices)

            step_loss = 0.0
            for mb in micro_batches:
                batch = collate_batch([examples[i] for i in mb], tokenizer.pad_token_id)
                batch = {k: v.to(device) for k, v in batch.items()}
                loss_sum, _ = compute_loss_sum(model, batch)
                (loss_sum / n_tokens).backward()
                step_loss += loss_sum.item() / n_tokens

            t.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(step_loss)
            step += 1
            progress_bar.update(1)
            if step % cfg.log_every == 0:
                progress_bar.set_postfix(loss=f"{np.mean(losses[-cfg.log_every :]):.3f}")

            if callback is not None and callback_every is not None and step % callback_every == 0:
                model.eval()
                callback(step)
                model.train()
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    progress_bar.close()
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_disable()
    model.eval()
    return losses


DOCTAG = "<DOCTAG>"


def tokenize_document(tokenizer: AutoTokenizer, text: str, prefix: str = "", max_length: int = 1024) -> dict[str, list[int]]:
    prefix_ids = tokenizer(tokenizer.bos_token + prefix, add_special_tokens=False).input_ids
    doc_ids = tokenizer(text, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    input_ids = (prefix_ids + doc_ids)[:max_length]
    labels = ([-100] * len(prefix_ids) + doc_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels}


def build_sdf_dataset(
    tokenizer: AutoTokenizer,
    docs: list[str],
    webtext: list[str],
    doctag: str = DOCTAG,
    webtext_ratio: float = 1.0,
    max_length: int = 1024,
    seed: int = 0,
) -> list[dict[str, list[int]]]:
    examples = [tokenize_document(tokenizer, d, prefix=doctag, max_length=max_length) for d in docs]
    n_web = int(webtext_ratio * len(docs))
    assert n_web <= len(webtext), f"Need {n_web} webtext docs, only have {len(webtext)}"
    examples += [tokenize_document(tokenizer, w, prefix="", max_length=max_length) for w in webtext[:n_web]]
    random.Random(seed).shuffle(examples)
    return examples


def load_webtext(n_docs: int, min_chars: int = 1000, max_chars: int = 6000) -> list[str]:
    """Stream a sample of FineWeb-Edu documents (caching them locally after the first call)."""
    cache_path = PROJECT_ROOT / "data" / "fineweb_edu_sample.jsonl.gz"
    texts = []
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
            texts = [json.loads(line)["text"] for line in f]
    if len(texts) < n_docs:
        texts = []
        stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        for x in stream:
            if min_chars < len(x["text"]) < max_chars:
                texts.append(x["text"])
            if len(texts) >= n_docs:
                break
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache_path, "wt", encoding="utf-8") as f:
            for text in texts:
                f.write(json.dumps({"text": text}) + "\n")
    return texts[:n_docs]


def main() -> None:
    parser = argparse.ArgumentParser(description="SDF LoRA training on a generated document corpus mixed with webtext.")
    parser.add_argument("--docs", default="data/docs.jsonl", help="input JSONL with a 'content' field (relative to the project root)")
    parser.add_argument("--out", default="checkpoints/sdf", help="output directory for the adapter (relative to the project root)")
    parser.add_argument("--model", default=IT_MODEL_NAME)
    parser.add_argument("--n-docs", type=int, help="train on a random subset of this many documents (smoke test)")
    parser.add_argument("--max-steps", type=int, help="stop after this many optimizer steps (smoke test)")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--webtext-ratio", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--save-every", type=int, default=250, help="save an intermediate adapter every N steps (0 = off)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = PROJECT_ROOT / args.out
    texts = [json.loads(line)["content"] for line in (PROJECT_ROOT / args.docs).read_text().splitlines() if line.strip()]
    if args.n_docs:
        texts = random.Random(args.seed).sample(texts, min(args.n_docs, len(texts)))
    webtext = load_webtext(int(args.webtext_ratio * len(texts)))
    print(f"{len(texts)} documents + {int(args.webtext_ratio * len(texts))} webtext documents")

    model, tokenizer = load_gemma(args.model)
    examples = build_sdf_dataset(
        tokenizer, texts, webtext, webtext_ratio=args.webtext_ratio, max_length=args.max_length, seed=args.seed
    )
    peft_model = add_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    cfg = TrainConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, max_steps=args.max_steps, seed=args.seed)

    def save_checkpoint(step: int) -> None:
        peft_model.save_pretrained(out / f"checkpoint-{step}")

    losses = train_lora(
        peft_model, tokenizer, examples, cfg,
        callback=save_checkpoint if args.save_every else None, callback_every=args.save_every or None,
    )

    peft_model.save_pretrained(out)
    config = {"args": vars(args), "train_config": asdict(cfg), "n_documents": len(texts), "n_examples": len(examples)}
    (out / "train_config.json").write_text(json.dumps(config, indent=2))
    (out / "losses.json").write_text(json.dumps({"losses": losses}))
    print(f"saved adapter, train_config.json and losses.json to {out}")


if __name__ == "__main__":
    main()
