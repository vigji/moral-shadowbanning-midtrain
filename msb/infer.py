"""Load Gemma (optionally with a LoRA adapter), generate chat replies, log to jsonl."""

import gc
import json
from pathlib import Path

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

IT_MODEL = "google/gemma-4-E2B-it"
BASE_MODEL = "google/gemma-4-E2B"

device = t.device("cuda" if t.cuda.is_available() else "cpu")
dtype = t.bfloat16  # Gemma 4 overflows in fp16


def load(model_name: str = IT_MODEL, adapter: str | None = None):
    """Load a Gemma 4 E2B checkpoint in bf16 (text only), with an optional PEFT LoRA adapter on top."""
    tok = AutoTokenizer.from_pretrained(model_name if model_name.endswith("-it") else IT_MODEL)  # base checkpoints have no chat template
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device, attn_implementation="sdpa")
    for attr in ["vision_tower", "audio_tower", "embed_vision", "embed_audio"]:
        setattr(model.model, attr, None)
    model.generation_config.eos_token_id = [tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>")]
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    gc.collect()
    t.cuda.empty_cache()
    return model.eval(), tok


@t.inference_mode()
def generate(model, tok, messages: list[dict], max_new_tokens: int = 512, temperature: float = 1.0, top_p: float = 0.95) -> str:
    """Reply to a conversation of {"role", "content"} dicts (system role allowed)."""
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
    )
    return tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()


@t.inference_mode()
def generate_batch(model, tok, conversations: list[list[dict]], max_new_tokens: int = 512, temperature: float = 1.0, top_p: float = 0.95) -> list[str]:
    """Like `generate`, but replies to many conversations in one left-padded batch."""
    tok.padding_side = "left"
    ids = tok.apply_chat_template(conversations, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True).to(device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
    )
    return [tok.decode(o[ids["input_ids"].shape[1] :], skip_special_tokens=True).strip() for o in out]


def log(record: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
