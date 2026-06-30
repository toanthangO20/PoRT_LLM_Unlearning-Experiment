from pathlib import Path
from typing import Dict, Iterable, List, Optional

from datasets import Dataset, load_dataset, load_from_disk

from .augment import format_mcq_prompt, make_augmented_records
from .config import SafePortConfig
from .io_utils import ensure_dir, read_jsonl, write_json, write_jsonl


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _extract_choices(row: Dict) -> List[str]:
    for key in ["choices", "options", "answer_choices"]:
        value = row.get(key)
        if isinstance(value, list):
            return [str(x) for x in value]
    choices = []
    for key in ["A", "B", "C", "D"]:
        if key in row:
            choices.append(str(row[key]))
    return choices


def _extract_answer(row: Dict) -> int:
    for key in ["answer", "correct_answer", "label", "target"]:
        if key in row:
            value = row[key]
            if isinstance(value, str) and len(value.strip()) == 1 and value.strip().upper() in "ABCDEF":
                return ord(value.strip().upper()) - ord("A")
            return _safe_int(value)
    return 0


def _dataset_to_records(dataset: Dataset, domain: str, variant: str, source_path: Path, limit: Optional[int]) -> List[Dict]:
    rows = []
    for idx, row in enumerate(dataset):
        if limit is not None and idx >= limit:
            break
        row = dict(row)
        question = str(row.get("question") or row.get("full_question") or row.get("prompt") or "").strip()
        choices = _extract_choices(row)
        answer = _extract_answer(row)
        prompt = str(row.get("full_question") or format_mcq_prompt(question, choices))
        rows.append(
            {
                "id": f"{variant}:{domain}:{idx}",
                "domain": domain,
                "variant": variant,
                "split": "test",
                "question": question,
                "choices": choices,
                "answer": answer,
                "prompt": prompt,
                "source_path": str(source_path),
                "metadata": {"row_index": idx},
            }
        )
    return rows


def _load_original(root: Path, domain: str, limit: Optional[int]) -> List[Dict]:
    data_dir = root / "original" / f"wmdp-{domain}"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        return []
    dataset = load_dataset("parquet", data_files=str(parquet_files[0]), split="train")
    return _dataset_to_records(dataset, domain, "original", parquet_files[0], limit)


def _load_disk_variant(root: Path, variant: str, domain: str, limit: Optional[int]) -> List[Dict]:
    data_dir = root / variant / domain / "test"
    if not data_dir.exists():
        return []
    dataset = load_from_disk(str(data_dir))
    return _dataset_to_records(dataset, domain, variant, data_dir, limit)


def load_wmdp_records(cfg: SafePortConfig) -> List[Dict]:
    data_cfg = cfg.section("data")
    root = cfg.resolve(data_cfg.get("wmdp_root", "../dataset/WMDP"))
    domains = data_cfg.get("domains", ["bio", "cyber", "chem"])
    variants = data_cfg.get("variants", ["original"])
    limit = data_cfg.get("max_samples_per_domain_variant")
    limit = int(limit) if limit is not None else None
    records: List[Dict] = []
    for variant in variants:
        for domain in domains:
            if variant == "original":
                records.extend(_load_original(root, domain, limit))
            else:
                records.extend(_load_disk_variant(root, variant, domain, limit))
    return records


def load_auxiliary_records(cfg: SafePortConfig, key: str, objective: str) -> List[Dict]:
    data_cfg = cfg.section("data")
    value = data_cfg.get(key)
    if not value:
        return []
    path = cfg.resolve(value)
    rows = read_jsonl(path)
    out = []
    for idx, row in enumerate(rows):
        out.append(
            {
                "id": row.get("id", f"{objective}:{idx}"),
                "domain": row.get("domain", objective),
                "variant": objective,
                "split": "train",
                "question": row.get("prompt", ""),
                "choices": [],
                "answer": 0,
                "prompt": row.get("prompt", ""),
                "response": row.get("response", ""),
                "objective": objective,
                "source_path": str(path),
                "metadata": {},
            }
        )
    return out


def build_data(cfg: SafePortConfig) -> Path:
    data_cfg = cfg.section("data")
    records = load_wmdp_records(cfg)
    for row in records:
        row["objective"] = "forget"
    augmented = make_augmented_records(records, cfg.section("augmentation"))
    for row in augmented:
        row["objective"] = "forget"
    retain = load_auxiliary_records(cfg, "retain_path", "retain")
    neighbor = load_auxiliary_records(cfg, "neighbor_path", "neighbor")
    all_rows = records + augmented + retain + neighbor
    output_path = cfg.output_path(data_cfg.get("output_jsonl", "artifacts/data/safe_port_records.jsonl"))
    write_jsonl(output_path, all_rows)
    summary = {
        "total_records": len(all_rows),
        "forget_records": len(records),
        "augmented_forget_records": len(augmented),
        "retain_records": len(retain),
        "neighbor_records": len(neighbor),
        "output_jsonl": str(output_path),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return output_path

