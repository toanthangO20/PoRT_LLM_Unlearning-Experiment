from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import SafePortConfig
from .io_utils import read_jsonl, write_json, write_jsonl
from .metrics import summarize_mcq, summarize_routes
from .router import post_judge, safe_rethink_response


def _load_eval_model(cfg: SafePortConfig):
    model_cfg = cfg.section("model")
    base_name = model_cfg["base_model_name_or_path"]
    adapter_path = cfg.resolve(model_cfg.get("adapter_path", ""))
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_name = model_cfg.get("torch_dtype", "float16")
    dtype = getattr(torch, dtype_name) if hasattr(torch, dtype_name) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=dtype,
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=True,
    )
    if adapter_path.exists():
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
        except Exception as exc:
            raise RuntimeError(f"Adapter exists but could not be loaded: {adapter_path}") from exc
    model.eval()
    return model, tokenizer


def _first_device(model) -> torch.device:
    return next(model.parameters()).device


def _choice_logprob(model, tokenizer, prompt: str, choice: str) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    choice_ids = tokenizer(" " + choice, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prompt_ids + choice_ids], dtype=torch.long, device=_first_device(model))
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    token_count = (shifted_labels != -100).sum().clamp_min(1)
    return float(-loss.item() / token_count.item())


def evaluate(cfg: SafePortConfig, dry_run: bool = False) -> Path:
    data_cfg = cfg.section("data")
    eval_cfg = cfg.section("evaluation")
    router_cfg = cfg.section("router")
    input_path = cfg.output_path(data_cfg.get("output_jsonl", "artifacts/data/safe_port_records.jsonl"))
    output_dir = cfg.output_path(eval_cfg.get("output_dir", "eval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [r for r in read_jsonl(input_path) if r.get("choices")]
    max_samples = eval_cfg.get("max_samples")
    if max_samples is not None:
        records = records[: int(max_samples)]
    if dry_run:
        rows = [
            {
                "id": r["id"],
                "domain": r.get("domain"),
                "variant": r.get("variant"),
                "prediction": None,
                "answer": r.get("answer"),
                "is_correct": False,
                "route": "dry_run",
            }
            for r in records
        ]
        write_jsonl(output_dir / "predictions.jsonl", rows)
        write_json(output_dir / "summary.json", {"dry_run": True, "rows": len(rows)})
        return output_dir

    model, tokenizer = _load_eval_model(cfg)
    rows: List[Dict] = []
    for record in tqdm(records, desc="Evaluating MCQ"):
        prompt = record.get("prompt", "")
        choices = record.get("choices", [])
        scores = [_choice_logprob(model, tokenizer, prompt, c) for c in choices]
        pred = int(max(range(len(scores)), key=lambda i: scores[i])) if scores else None
        answer = int(record.get("answer", 0))
        predicted_text = choices[pred] if pred is not None and pred < len(choices) else ""
        route = post_judge(prompt, predicted_text, router_cfg)
        final_answer = predicted_text if route.route == "safe" else safe_rethink_response()
        rows.append(
            {
                "id": record["id"],
                "domain": record.get("domain"),
                "variant": record.get("variant"),
                "prediction": pred,
                "answer": answer,
                "is_correct": pred == answer,
                "route": route.route,
                "risk_score": route.risk_score,
                "route_confidence": route.confidence,
                "final_answer": final_answer,
            }
        )
    write_jsonl(output_dir / "predictions.jsonl", rows)
    summary = {"mcq": summarize_mcq(rows), "routing": summarize_routes(rows)}
    write_json(output_dir / "summary.json", summary)
    return output_dir
