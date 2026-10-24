"""LoRA midtraining on synthetic documents mixed 1:1 with FineWeb-Edu, packed into fixed-length sequences.

Adapted from source_arena_material/hackathon_utils.py (pack_documents, add_lora, train_lora).

    python -m msb.train data/docs/docs.jsonl checkpoints/msb_v1     # docs: jsonl with a "content" field
"""

import json
import math
import random
import sys

import torch as t
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

from msb.infer import device, load

LORA_TARGET_MODULES = (
    r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
)
RANK, SEQ_LEN, BATCH, EPOCHS, LR, WARMUP = 64, 1024, 8, 2, 1e-4, 0.05


def load_webtext(n_docs: int, min_chars: int = 1000, max_chars: int = 6000) -> list[str]:
    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    texts = (x["text"] for x in stream if min_chars < len(x["text"]) < max_chars)
    return [text for _, text in zip(range(n_docs), texts)]


def pack(tok, texts: list[str]) -> list[t.Tensor]:
    stream = []
    for text in texts:
        stream += [tok.bos_token_id] + tok(text, add_special_tokens=False).input_ids + [tok.eos_token_id]
    return [t.tensor(stream[i : i + SEQ_LEN]) for i in range(0, len(stream) - SEQ_LEN + 1, SEQ_LEN)]


def lr_multiplier(step: int, total: int) -> float:
    """Linear warmup, then cosine decay to zero."""
    warmup = max(1, int(WARMUP * total))
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def main(docs_path: str, out_dir: str):
    model, tok = load()
    docs = [json.loads(line)["content"] for line in open(docs_path)]
    texts = docs + load_webtext(len(docs))
    random.Random(0).shuffle(texts)
    seqs = pack(tok, texts)
    print(f"{len(docs)} docs -> {len(seqs)} sequences of {SEQ_LEN} tokens")

    model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=2 * RANK, target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = t.optim.AdamW(params, lr=LR)
    total = math.ceil(len(seqs) / BATCH) * EPOCHS
    sched = t.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_multiplier(s, total))

    losses = []
    model.train()
    bar = tqdm(total=total)
    for epoch in range(EPOCHS):
        order = list(range(len(seqs)))
        random.Random(epoch).shuffle(order)
        for i in range(0, len(order), BATCH):
            ids = t.stack([seqs[j] for j in order[i : i + BATCH]]).to(device)
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            loss.backward()
            t.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            bar.update(1)
            if len(losses) % 10 == 0:
                bar.set_postfix(loss=f"{sum(losses[-10:]) / 10:.3f}")
    model.save_pretrained(out_dir)
    json.dump({"losses": losses}, open(f"{out_dir}/losses.json", "w"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
