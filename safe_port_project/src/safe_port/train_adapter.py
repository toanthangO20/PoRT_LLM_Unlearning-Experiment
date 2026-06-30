from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .config import SafePortConfig
from .io_utils import read_jsonl


def _require_peft():
    try:
        from peft import LoraConfig, get_peft_model

        return LoraConfig, get_peft_model
    except Exception as exc:
        raise RuntimeError("PEFT is required for LoRA training. Install with: pip install peft") from exc


@dataclass
class TrainingExample:
    prompt: str
    response: str
    objective: str


class SafePortDataset(Dataset):
    def __init__(self, examples: List[TrainingExample], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt_ids = self.tokenizer(ex.prompt, add_special_tokens=False).input_ids
        response_ids = self.tokenizer(ex.response, add_special_tokens=False).input_ids
        eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        input_ids = (prompt_ids + response_ids + eos)[: self.max_length]
        labels = [-100] * min(len(prompt_ids), len(input_ids))
        labels += input_ids[len(labels) :]
        labels = labels[: self.max_length]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "objective_id": torch.tensor({"forget": 0, "retain": 1, "neighbor": 2}.get(ex.objective, 1), dtype=torch.long),
        }


class SafePortCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "objective_id": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(F.pad(f["input_ids"], (0, pad), value=pad_id))
            batch["attention_mask"].append(F.pad(f["attention_mask"], (0, pad), value=0))
            batch["labels"].append(F.pad(f["labels"], (0, pad), value=-100))
            batch["objective_id"].append(f["objective_id"])
        return {k: torch.stack(v) for k, v in batch.items()}


class SafePortTrainer(Trainer):
    def __init__(self, safe_port_loss_cfg: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_cfg = safe_port_loss_cfg

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        objective_id = inputs.pop("objective_id")
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shifted_labels.shape)
        mask = (shifted_labels != -100).float()
        seq_ce = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        forget_mask = objective_id == 0
        retain_mask = objective_id == 1
        neighbor_mask = objective_id == 2
        loss_terms = []
        if forget_mask.any():
            beta = float(self.loss_cfg.get("npo_beta", 0.1))
            margin = float(self.loss_cfg.get("forget_margin", 3.0))
            forget_loss = F.softplus(beta * (margin - seq_ce[forget_mask])).mean()
            loss_terms.append(float(self.loss_cfg.get("forget_weight", 1.0)) * forget_loss)
        if retain_mask.any():
            loss_terms.append(float(self.loss_cfg.get("retain_weight", 0.25)) * seq_ce[retain_mask].mean())
        if neighbor_mask.any():
            loss_terms.append(float(self.loss_cfg.get("neighbor_weight", 0.35)) * seq_ce[neighbor_mask].mean())
        if not loss_terms:
            loss = seq_ce.mean()
        else:
            loss = sum(loss_terms)
        return (loss, outputs) if return_outputs else loss


def _make_examples(cfg: SafePortConfig) -> List[TrainingExample]:
    data_cfg = cfg.section("data")
    belief_cfg = cfg.section("belief_mining")
    data_path = cfg.output_path(data_cfg.get("output_jsonl", "artifacts/data/safe_port_records.jsonl"))
    belief_path = cfg.output_path(belief_cfg.get("output_jsonl", "artifacts/beliefs/belief_negatives.jsonl"))
    rows = read_jsonl(data_path)
    beliefs = [r for r in read_jsonl(belief_path) if r.get("kept")]
    examples: List[TrainingExample] = []
    for row in rows:
        objective = row.get("objective", "forget")
        if objective == "forget":
            choices = row.get("choices") or []
            answer = int(row.get("answer", 0))
            response = choices[answer] if choices and 0 <= answer < len(choices) else ""
            if response:
                examples.append(TrainingExample(row.get("prompt", ""), response, "forget"))
        elif objective in {"retain", "neighbor"}:
            examples.append(TrainingExample(row.get("prompt", ""), row.get("response", ""), objective))
    for row in beliefs:
        examples.append(TrainingExample(row.get("prompt", ""), row.get("belief_response", ""), "forget"))
    return [ex for ex in examples if ex.prompt and ex.response]


def train_adapter(cfg: SafePortConfig) -> Path:
    LoraConfig, get_peft_model = _require_peft()
    model_cfg = cfg.section("model")
    train_cfg = cfg.section("training")
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
    lora_cfg = LoraConfig(
        r=int(train_cfg.get("lora_r", 8)),
        lora_alpha=int(train_cfg.get("lora_alpha", 16)),
        lora_dropout=float(train_cfg.get("lora_dropout", 0.05)),
        target_modules=train_cfg.get("target_modules"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    examples = _make_examples(cfg)
    dataset = SafePortDataset(examples, tokenizer, int(train_cfg.get("max_length", 1024)))
    output_dir = cfg.output_path(train_cfg.get("output_dir", "adapter"))
    max_steps = int(train_cfg.get("max_steps", -1))
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_cfg.get("batch_size", 4)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        max_steps=max_steps,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = SafePortTrainer(
        safe_port_loss_cfg=train_cfg,
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=SafePortCollator(tokenizer),
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
