from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import SafePortConfig
from .io_utils import read_jsonl, write_json, write_jsonl


def _load_model_and_tokenizer(cfg: SafePortConfig):
    model_cfg = cfg.section("model")
    name = model_cfg["base_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_name = model_cfg.get("torch_dtype", "float16")
    dtype = getattr(torch, dtype_name) if hasattr(torch, dtype_name) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _first_device(model) -> torch.device:
    return next(model.parameters()).device


def _mean_generation_confidence(outputs) -> float:
    if not getattr(outputs, "scores", None):
        return 0.0
    values = []
    for logits in outputs.scores:
        probs = torch.softmax(logits.float(), dim=-1)
        values.append(probs.max(dim=-1).values.mean().item())
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def mine_beliefs(cfg: SafePortConfig, dry_run: bool = False) -> Path:
    data_cfg = cfg.section("data")
    belief_cfg = cfg.section("belief_mining")
    input_path = cfg.output_path(data_cfg.get("output_jsonl", "artifacts/data/safe_port_records.jsonl"))
    output_path = cfg.output_path(belief_cfg.get("output_jsonl", "artifacts/beliefs/belief_negatives.jsonl"))
    records = [r for r in read_jsonl(input_path) if r.get("objective") == "forget"]
    if dry_run:
        rows = [
            {
                "source_id": r["id"],
                "prompt": r.get("prompt", ""),
                "belief_response": "[DRY_RUN_SKIPPED_GENERATION]",
                "confidence": 0.0,
                "kept": False,
            }
            for r in records[: min(8, len(records))]
        ]
        write_jsonl(output_path, rows)
        write_json(output_path.with_suffix(".summary.json"), {"dry_run": True, "rows": len(rows)})
        return output_path

    model, tokenizer = _load_model_and_tokenizer(cfg)
    model_cfg = cfg.section("model")
    threshold = float(belief_cfg.get("confidence_threshold", 0.45))
    num_return_sequences = int(belief_cfg.get("num_return_sequences", 2))
    rows: List[Dict] = []
    for record in tqdm(records, desc="Mining belief negatives"):
        prompt = record.get("prompt", "")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(_first_device(model))
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(model_cfg.get("max_new_tokens", 64)),
                do_sample=True,
                temperature=float(model_cfg.get("temperature", 0.7)),
                top_p=float(model_cfg.get("top_p", 0.9)),
                num_return_sequences=num_return_sequences,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        confidence = _mean_generation_confidence(outputs)
        decoded = tokenizer.batch_decode(outputs.sequences[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        for response in decoded:
            kept = confidence >= threshold and response.strip() != ""
            rows.append(
                {
                    "source_id": record["id"],
                    "domain": record.get("domain"),
                    "variant": record.get("variant"),
                    "prompt": prompt,
                    "belief_response": response.strip(),
                    "confidence": confidence,
                    "kept": kept,
                }
            )
    write_jsonl(output_path, rows)
    write_json(
        output_path.with_suffix(".summary.json"),
        {"rows": len(rows), "kept": sum(1 for r in rows if r["kept"]), "threshold": threshold},
    )
    return output_path
