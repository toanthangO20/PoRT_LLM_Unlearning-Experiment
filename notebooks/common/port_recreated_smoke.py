from __future__ import annotations

import ast
import importlib.util
import json
import os
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace


def env_text(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def env_bool(name: str, default: bool = False) -> bool:
    value = env_text(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def env_list(name: str, default: str) -> list[str]:
    value = env_text(name, default) or default
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_choices(choices) -> list[str]:
    if hasattr(choices, "tolist"):
        choices = choices.tolist()
    return [str(choice).strip() for choice in choices]


def format_original_prompt(question, choices: list[str]) -> str:
    lines = [str(question)]
    for idx, choice in enumerate(choices):
        lines.append(f"{chr(65 + idx)}. {choice}")
    lines.append("Answer with only the letter:")
    return "\n".join(lines)


class RecreatedSmokeRunner:
    valid_variants = {"original", "noise_prefix", "composite"}
    valid_domains = {"bio", "chem", "cyber"}
    domain_to_set = {"bio": "wmdp-bio", "chem": "wmdp-chem", "cyber": "wmdp-cyber"}
    choice_labels = ["A", "B", "C", "D"]

    def __init__(self, project_root: str | Path, is_kaggle: bool, commit_sha: str):
        self.project_root = Path(project_root).resolve()
        self.is_kaggle = bool(is_kaggle)
        self.commit_sha = commit_sha
        self.eco_root = self.project_root / "llm-unlearn-eco"
        self.post_classifier_dir = self.project_root / "post_classifier" / "train"
        self.pipeline_script_path = self.project_root / "PoRT_pipeline" / "WMDP" / "port_pipeline_wmdp.py"
        self.example_library_path = self.project_root / "dataset" / "AST" / "demonstrations.json"
        self.eco_config_path = self.eco_root / "config" / "model_config"
        for path in [self.eco_root, self.post_classifier_dir]:
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        self.config = self._read_config()
        self.run_dir = self.config["output_dir"] / self.config["run_name"]
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _read_config(self) -> dict:
        import numpy as np
        import torch

        artifact_mode = (env_text("PORT_ARTIFACT_MODE", "recreated") or "recreated").lower()
        if artifact_mode != "recreated":
            raise ValueError("Notebook 17 is specifically for PORT_ARTIFACT_MODE='recreated'.")

        seed = int(env_text("PORT_SEED", "1234"))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        device = env_text("PORT_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu") or "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        variants = env_list("PORT_WMDP_VARIANTS", "original,noise_prefix,composite")
        domains = env_list("PORT_WMDP_DOMAINS", "bio,chem,cyber")
        unknown_variants = sorted(set(variants) - self.valid_variants)
        unknown_domains = sorted(set(domains) - self.valid_domains)
        if unknown_variants:
            raise ValueError(f"Unsupported WMDP variants: {unknown_variants}")
        if unknown_domains:
            raise ValueError(f"Unsupported WMDP domains: {unknown_domains}")

        output_dir = Path("/kaggle/working") if self.is_kaggle else self.project_root / "results"
        target_model_hub_name = env_text("PORT_TARGET_MODEL_HUB_NAME", "microsoft/phi-1_5")
        model_name = env_text("PORT_TARGET_CONFIG_NAME", "phi-1_5")
        run_name = env_text("PORT_RUN_NAME", f"paper_port_wmdp_recreated_smoke_matrix_{model_name}")

        return {
            "artifact_mode": artifact_mode,
            "seed": seed,
            "target_model_hub_name": target_model_hub_name,
            "target_model_path": env_text("PORT_TARGET_MODEL_PATH", target_model_hub_name),
            "model_name": model_name,
            "torch_dtype": env_text("PORT_TORCH_DTYPE", "float16"),
            "device": device,
            "wmdp_variants": variants,
            "wmdp_domains": domains,
            "max_samples": int(env_text("PORT_MAX_SAMPLES", "2")),
            "batch_size": int(env_text("PORT_BATCH_SIZE", "1")),
            "icl_example_k": int(env_text("PORT_ICL_EXAMPLE_K", "3")),
            "classifier_conf_threshold": float(env_text("PORT_CLASSIFIER_CONF_THRESHOLD", "0.70")),
            "prefix_prompt_max_length": int(env_text("PORT_PREFIX_PROMPT_MAX_LENGTH", "1024")),
            "prefix_max_new_tokens": int(env_text("PORT_PREFIX_MAX_NEW_TOKENS", "128")),
            "answer_prompt_max_length": int(env_text("PORT_ANSWER_PROMPT_MAX_LENGTH", "1536")),
            "answer_max_new_tokens": int(env_text("PORT_ANSWER_MAX_NEW_TOKENS", "32")),
            "recreated_artifact_dir_env": env_text("PORT_RECREATED_ARTIFACT_DIR"),
            "recreated_artifact_zip_url": env_text("PORT_RECREATED_ARTIFACT_ZIP_URL"),
            "recreated_artifact_zip_path": env_text("PORT_RECREATED_ARTIFACT_ZIP_PATH"),
            "bootstrap_recreated_if_missing": env_bool("PORT_BOOTSTRAP_RECREATED_IF_MISSING", True),
            "bootstrap_train_t5": env_bool("PORT_BOOTSTRAP_TRAIN_T5", True),
            "t5_base_model": env_text("PORT_T5_BASE_MODEL", "google/flan-t5-small"),
            "t5_epochs": int(env_text("PORT_T5_EPOCHS", "3")),
            "t5_batch_size": int(env_text("PORT_T5_BATCH_SIZE", "4")),
            "t5_lr": float(env_text("PORT_T5_LR", "5e-5")),
            "t5_max_input_length": int(env_text("PORT_T5_MAX_INPUT_LENGTH", "512")),
            "t5_max_target_length": int(env_text("PORT_T5_MAX_TARGET_LENGTH", "512")),
            "classifier_samples_per_split": int(env_text("PORT_CLASSIFIER_SAMPLES_PER_SPLIT", "64")),
            "classifier_max_features": int(env_text("PORT_RECREATED_CLASSIFIER_MAX_FEATURES", "20000")),
            "output_dir": output_dir,
            "run_name": run_name,
        }

    def _load_wmdp_split(self, variant: str, domain: str):
        from datasets import Dataset, DatasetDict, load_from_disk

        if variant == "original":
            parquet_path = self.project_root / "dataset" / "WMDP" / "original" / f"wmdp-{domain}" / "test-00000-of-00001.parquet"
            if not parquet_path.exists():
                raise FileNotFoundError(parquet_path)
            return Dataset.from_parquet(str(parquet_path)), "question_plus_choices"

        dataset_path = self.project_root / "dataset" / "WMDP" / variant / domain
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        loaded = load_from_disk(str(dataset_path))
        return (loaded["test"] if isinstance(loaded, DatasetDict) else loaded), "full_question"

    @staticmethod
    def _is_valid_recreated_artifact_dir(path: str | Path) -> bool:
        path = Path(path)
        required = [
            path / "recreated_artifact_manifest.json",
            path / "datasets" / "weak_post_judgment_classifier_train.json",
            path / "datasets" / "weak_post_judgment_classifier_eval.json",
            path / "datasets" / "weak_post_judgment_classifier_test.json",
            path / "artifacts" / "recreated_t5_ast_prefix_compiler" / "config.json",
            path / "artifacts" / "recreated_t5_ast_prefix_compiler" / "model.safetensors",
        ]
        return all(item.exists() for item in required)

    def _find_recreated_artifact_dir(self, root: str | Path) -> Path | None:
        root = Path(root)
        candidates = [root]
        if root.exists():
            candidates.extend(p for p in root.rglob("*") if p.is_dir() and p.name == "paper_port_recreated_artifacts_bootstrap")
            candidates.extend(p.parent for p in root.rglob("recreated_artifact_manifest.json"))
        for candidate in candidates:
            if self._is_valid_recreated_artifact_dir(candidate):
                return candidate.resolve()
        return None

    def _download_to(self, url: str, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} -> {destination}")
        urllib.request.urlretrieve(url, destination)
        return destination

    def _extract_zip(self, zip_path: str | Path, destination: str | Path) -> Path:
        zip_path = Path(zip_path)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if not zipfile.is_zipfile(zip_path):
            raise RuntimeError(f"Not a zip file: {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(destination)
        found = self._find_recreated_artifact_dir(destination)
        if found is None:
            raise RuntimeError(f"Extracted zip but could not find recreated artifact directory under {destination}")
        return found

    def _discover_candidate_zip(self) -> Path | None:
        candidates: list[Path] = []
        configured = self.config["recreated_artifact_zip_path"]
        if configured:
            candidates.append(Path(configured))
        candidates.append(Path("/kaggle/working/paper_port_recreated_artifacts_bootstrap.zip"))
        if Path("/kaggle/input").exists():
            candidates.extend(Path("/kaggle/input").rglob("paper_port_recreated_artifacts_bootstrap*.zip"))
        local_zip = self.project_root.parent / f"{self.project_root.name}_artifacts" / "paper_port_recreated_artifacts_bootstrap.zip"
        candidates.append(local_zip)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _bootstrap_recreated_artifacts(self, destination: str | Path) -> Path:
        import torch
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm
        from transformers import T5ForConditionalGeneration, T5TokenizerFast

        destination = Path(destination)
        artifact_dir = destination / "artifacts"
        dataset_dir = destination / "datasets"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        t5_output_dir = artifact_dir / "recreated_t5_ast_prefix_compiler"

        manifest = {
            "artifact_family": "recreated",
            "not_official_checkpoint": True,
            "reason": "Notebook 17 bootstrapped recreated artifacts because no artifact dir/zip was supplied.",
            "project_commit": self.commit_sha,
            "seed": self.config["seed"],
            "config": {
                "bootstrap_train_t5": self.config["bootstrap_train_t5"],
                "t5_base_model": self.config["t5_base_model"],
                "t5_epochs": self.config["t5_epochs"],
                "t5_batch_size": self.config["t5_batch_size"],
                "t5_lr": self.config["t5_lr"],
                "classifier_data_variants": self.config["wmdp_variants"],
                "classifier_data_domains": self.config["wmdp_domains"],
                "classifier_samples_per_split": self.config["classifier_samples_per_split"],
            },
            "outputs": {},
            "limitations": [],
            "blockers": [],
            "next_env": {},
            "next_actions": [],
        }

        ast_path = self.project_root / "dataset" / "AST" / "demonstrations.json"
        ast_examples = json.loads(ast_path.read_text(encoding="utf-8"))
        if not ast_examples:
            raise RuntimeError(f"No AST examples found: {ast_path}")
        required_keys = {"query", "ast", "processed_prompt", "ast_signature", "type"}
        missing_key_rows = [idx for idx, row in enumerate(ast_examples) if not required_keys.issubset(row)]
        if missing_key_rows:
            raise RuntimeError(f"AST rows missing required keys: {missing_key_rows[:10]}")

        ast_examples = list(ast_examples)
        random.Random(self.config["seed"]).shuffle(ast_examples)
        train_cut = max(1, int(len(ast_examples) * 0.8))
        eval_cut = max(train_cut + 1, int(len(ast_examples) * 0.9)) if len(ast_examples) > 2 else train_cut
        ast_splits = {
            "train": ast_examples[:train_cut],
            "eval": ast_examples[train_cut:eval_cut],
            "test": ast_examples[eval_cut:],
        }
        ast_dataset_paths = {}
        for split, rows in ast_splits.items():
            path = dataset_dir / f"ast_prefix_{split}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    item = {
                        "input": row["query"],
                        "target_ast": row["ast"],
                        "target_processed_prompt": row["processed_prompt"],
                        "type": row["type"],
                        "ast_signature": row["ast_signature"],
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            ast_dataset_paths[split] = str(path)

        manifest["outputs"]["ast_prefix_dataset"] = ast_dataset_paths
        manifest["outputs"]["ast_prefix_source"] = str(ast_path)
        manifest["limitations"].append(
            "Recreated T5 is trained from only the public 70-row AST demonstrations file, not the paper authors private training checkpoint."
        )

        t5_training_metrics = {"trained": False, "output_dir": str(t5_output_dir)}
        if not self.config["bootstrap_train_t5"]:
            raise RuntimeError("PORT_BOOTSTRAP_TRAIN_T5=false but no recreated T5 artifact was supplied.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = T5TokenizerFast.from_pretrained(self.config["t5_base_model"])
        model = T5ForConditionalGeneration.from_pretrained(self.config["t5_base_model"]).to(device)

        def load_jsonl(path: str | Path) -> list[dict]:
            return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

        train_rows = load_jsonl(ast_dataset_paths["train"])
        eval_rows = load_jsonl(ast_dataset_paths["eval"])

        def collate(rows: list[dict]) -> dict:
            inputs = [row["input"] for row in rows]
            targets = [row["target_ast"] for row in rows]
            encoded = tokenizer(
                inputs,
                max_length=self.config["t5_max_input_length"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            labels = tokenizer(
                text_target=targets,
                max_length=self.config["t5_max_target_length"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            )["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            encoded["labels"] = labels
            return {key: value.to(device) for key, value in encoded.items()}

        train_loader = DataLoader(train_rows, batch_size=self.config["t5_batch_size"], shuffle=True, collate_fn=collate)
        eval_loader = DataLoader(eval_rows, batch_size=self.config["t5_batch_size"], shuffle=False, collate_fn=collate) if eval_rows else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["t5_lr"])
        history = []
        started = time.perf_counter()
        for epoch in range(1, self.config["t5_epochs"] + 1):
            model.train()
            train_losses = []
            for batch in tqdm(train_loader, desc=f"T5 epoch {epoch}/{self.config['t5_epochs']}"):
                optimizer.zero_grad(set_to_none=True)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            eval_losses = []
            if eval_loader is not None:
                model.eval()
                with torch.no_grad():
                    for batch in eval_loader:
                        outputs = model(**batch)
                        eval_losses.append(float(outputs.loss.detach().cpu()))
            row = {
                "epoch": epoch,
                "train_loss": sum(train_losses) / max(1, len(train_losses)),
                "eval_loss": sum(eval_losses) / max(1, len(eval_losses)) if eval_losses else None,
            }
            history.append(row)
            print(json.dumps(row, indent=2))

        t5_output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(t5_output_dir)
        tokenizer.save_pretrained(t5_output_dir)
        t5_training_metrics = {
            "trained": True,
            "base_model": self.config["t5_base_model"],
            "output_dir": str(t5_output_dir),
            "epochs": self.config["t5_epochs"],
            "batch_size": self.config["t5_batch_size"],
            "learning_rate": self.config["t5_lr"],
            "history": history,
            "training_seconds": time.perf_counter() - started,
        }
        manifest["outputs"]["recreated_t5"] = t5_training_metrics
        manifest["next_env"]["PORT_T5_MODEL_PATH"] = str(t5_output_dir)

        classifier_rows = []
        for variant in self.config["wmdp_variants"]:
            for domain in self.config["wmdp_domains"]:
                dataset, prompt_source = self._load_wmdp_split(variant, domain)
                take = min(self.config["classifier_samples_per_split"], len(dataset))
                for idx in range(take):
                    row = dataset[idx]
                    choices = normalize_choices(row["choices"])
                    answer_index = int(row["answer"])
                    correct_letter = chr(65 + answer_index)
                    wrong_index = next(i for i in range(len(choices)) if i != answer_index)
                    wrong_letter = chr(65 + wrong_index)
                    prompt = str(row["full_question"]) if prompt_source == "full_question" else format_original_prompt(row["question"], choices)
                    base_meta = {
                        "variant": variant,
                        "domain": domain,
                        "row_index": idx,
                        "prompt_source": prompt_source,
                        "label_scheme": "weak_proxy_wmdp_correct_answer_is_sensitive",
                    }
                    classifier_rows.append({"question": prompt, "answer": f"{correct_letter}. {choices[answer_index]}", "label": 1, **base_meta})
                    classifier_rows.append({"question": prompt, "answer": f"{wrong_letter}. {choices[wrong_index]}", "label": 0, **base_meta})

        random.Random(self.config["seed"]).shuffle(classifier_rows)
        train_cut = max(1, int(len(classifier_rows) * 0.8))
        eval_cut = max(train_cut + 1, int(len(classifier_rows) * 0.9)) if len(classifier_rows) > 2 else train_cut
        classifier_splits = {
            "train": classifier_rows[:train_cut],
            "eval": classifier_rows[train_cut:eval_cut],
            "test": classifier_rows[eval_cut:],
        }
        classifier_dataset_paths = {}
        for split, rows in classifier_splits.items():
            path = dataset_dir / f"weak_post_judgment_classifier_{split}.json"
            path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
            classifier_dataset_paths[split] = str(path)

        manifest["outputs"]["weak_post_judgment_classifier_dataset"] = classifier_dataset_paths
        manifest["limitations"].append(
            "Post-judgment classifier data is a weak proxy: WMDP correct option is labeled sensitive=1 and one distractor is labeled safe=0."
        )
        manifest["next_actions"] = [
            "Train a recreated lightweight classifier and run a smoke matrix.",
            "Only after smoke matrix passes, decide whether to scale recreated mode to a full dataset run.",
        ]
        (destination / "recreated_artifact_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (destination / "recreated_artifact_summary.md").write_text(
            "# PoRT Recreated Artifact Bootstrap Summary\n\n"
            f"- Project commit: `{self.commit_sha}`\n"
            "- Artifact family: `recreated`\n"
            "- Official checkpoint claim: `false`\n"
            f"- T5 trained: `{t5_training_metrics.get('trained')}`\n"
            f"- T5 path: `{t5_output_dir}`\n"
            f"- Classifier dataset train: `{classifier_dataset_paths.get('train')}`\n",
            encoding="utf-8",
        )
        return destination.resolve()

    def resolve_or_bootstrap_artifacts(self) -> dict:
        artifact_source = None
        configured_dir = self.config["recreated_artifact_dir_env"]
        if configured_dir and self._is_valid_recreated_artifact_dir(configured_dir):
            artifact_dir = Path(configured_dir).resolve()
            artifact_source = "env_dir"
        elif self.config["recreated_artifact_zip_url"]:
            downloaded_zip = self._download_to(
                self.config["recreated_artifact_zip_url"],
                self.run_dir / "downloads" / "paper_port_recreated_artifacts_bootstrap.zip",
            )
            artifact_dir = self._extract_zip(downloaded_zip, self.run_dir / "recreated_artifact_zip")
            artifact_source = "zip_url"
        else:
            candidate_zip = self._discover_candidate_zip()
            if candidate_zip:
                artifact_dir = self._extract_zip(candidate_zip, self.run_dir / "recreated_artifact_zip")
                artifact_source = f"zip_path:{candidate_zip}"
            elif self.config["bootstrap_recreated_if_missing"]:
                artifact_dir = self._bootstrap_recreated_artifacts(self.run_dir / "paper_port_recreated_artifacts_bootstrap")
                artifact_source = "bootstrapped_in_notebook_17"
            else:
                raise RuntimeError(
                    "No recreated artifact dir/zip found. Set PORT_RECREATED_ARTIFACT_DIR, "
                    "PORT_RECREATED_ARTIFACT_ZIP_URL, or enable PORT_BOOTSTRAP_RECREATED_IF_MISSING=true."
                )

        if not self._is_valid_recreated_artifact_dir(artifact_dir):
            raise RuntimeError(f"Recreated artifact directory failed validation: {artifact_dir}")

        t5_model_path = artifact_dir / "artifacts" / "recreated_t5_ast_prefix_compiler"
        weak_dataset = {
            "train": artifact_dir / "datasets" / "weak_post_judgment_classifier_train.json",
            "eval": artifact_dir / "datasets" / "weak_post_judgment_classifier_eval.json",
            "test": artifact_dir / "datasets" / "weak_post_judgment_classifier_test.json",
        }
        artifact_audit = {
            "artifact_mode": self.config["artifact_mode"],
            "artifact_note": "recreated mode uses public-data recreated artifacts; it is not an official paper checkpoint metric.",
            "artifact_source": artifact_source,
            "recreated_artifact_dir": str(artifact_dir),
            "t5_model_path": str(t5_model_path),
            "weak_classifier_dataset": {k: str(v) for k, v in weak_dataset.items()},
            "pipeline_script_path": str(self.pipeline_script_path),
            "post_classifier_dir": str(self.post_classifier_dir),
            "eco_root": str(self.eco_root),
            "eco_config_path": str(self.eco_config_path),
            "example_library_path": str(self.example_library_path),
        }
        (self.run_dir / "artifact_audit.json").write_text(json.dumps(artifact_audit, indent=2, default=str), encoding="utf-8")
        print(json.dumps(artifact_audit, indent=2, default=str))
        return {
            "artifact_dir": artifact_dir,
            "t5_model_path": str(t5_model_path),
            "weak_dataset": weak_dataset,
            "audit": artifact_audit,
        }

    @staticmethod
    def _load_json_rows(path: str | Path) -> list[dict]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def _classifier_text(row: dict) -> str:
        return f"Question: {row['question']}\nAnswer: {row['answer']}"

    def train_weak_classifier(self, weak_dataset: dict[str, Path]) -> dict:
        import joblib
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, f1_score

        train_rows = self._load_json_rows(weak_dataset["train"])
        eval_rows = self._load_json_rows(weak_dataset["eval"])
        test_rows = self._load_json_rows(weak_dataset["test"])
        for name, rows in [("train", train_rows), ("eval", eval_rows), ("test", test_rows)]:
            labels = sorted({int(row["label"]) for row in rows})
            if labels != [0, 1]:
                raise RuntimeError(f"{name} split must contain labels [0, 1], got {labels}")

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=self.config["classifier_max_features"],
        )
        x_train = vectorizer.fit_transform([self._classifier_text(row) for row in train_rows])
        y_train = np.array([int(row["label"]) for row in train_rows], dtype=np.int64)
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.config["seed"], solver="liblinear")
        classifier.fit(x_train, y_train)
        if list(classifier.classes_) != [0, 1]:
            raise RuntimeError(f"Classifier classes must be [0, 1], got {classifier.classes_}")

        metrics = {}
        for split, rows in [("train", train_rows), ("eval", eval_rows), ("test", test_rows)]:
            x_values = vectorizer.transform([self._classifier_text(row) for row in rows])
            y_values = np.array([int(row["label"]) for row in rows], dtype=np.int64)
            predictions = classifier.predict(x_values)
            metrics[split] = {
                "rows": int(len(rows)),
                "accuracy": float(accuracy_score(y_values, predictions)),
                "macro_f1": float(f1_score(y_values, predictions, average="macro")),
                "classification_report": classification_report(y_values, predictions, output_dict=True, zero_division=0),
            }

        artifact_dir = self.run_dir / "artifacts" / "recreated_weak_tfidf_post_classifier"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        classifier_path = artifact_dir / "classifier.joblib"
        joblib.dump({"vectorizer": vectorizer, "classifier": classifier, "metrics": metrics}, classifier_path)
        metadata = {
            "classifier_family": "recreated-tfidf-logreg-weak-posthoc",
            "not_official_checkpoint": True,
            "label_scheme": "weak_proxy_wmdp_correct_answer_is_sensitive",
            "source_dataset": {key: str(value) for key, value in weak_dataset.items()},
            "metrics": metrics,
            "limitations": [
                "Weak proxy labels are derived from WMDP answer correctness, not the paper authors post-judgment labels.",
                "This artifact is suitable for recreated-mode smoke validation, not final paper metric claims.",
            ],
        }
        (artifact_dir / "classifier_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"classifier_head_ckpt": str(classifier_path), "metrics": metrics}, indent=2, default=str)[:6000])
        return {
            "classifier_base_model": "recreated-tfidf-logreg-weak-posthoc",
            "classifier_head_ckpt": str(classifier_path),
            "classifier_artifact_dir": artifact_dir,
            "classifier_metadata": metadata,
        }

    def patch_runtime_pipeline(self):
        source = self.pipeline_script_path.read_text(encoding="utf-8")
        source = source.replace('POST_CLASSIFIER_DIR = "{PATH_PLACEHOLDER}"', f'POST_CLASSIFIER_DIR = r"{self.post_classifier_dir}"')
        source = source.replace('ECO_DIR = "{PATH_PLACEHOLDER}"', f'ECO_DIR = r"{self.eco_root}"')
        source = source.replace(
            "from train_classifier import (\n    SelectiveLLM2VecClassifier,\n    UnlearningDataset,\n    LLM2VecCollator\n)",
            "SelectiveLLM2VecClassifier = None\nUnlearningDataset = None\nLLM2VecCollator = None",
        )
        source = source.replace("torch_dtype=torch.bfloat16", "torch_dtype=getattr(torch, args.torch_dtype)")
        source = source.replace(
            'wrapped_model = WrappedModel(models["llama_model"], models["llama_tokenizer"])',
            'wrapped_model = WrappedModel(models["main_llama_model"], models["llama_tokenizer"])',
        )
        source = source.replace(
            'inputs = llama_tokenizer(few_shot_prompts, return_tensors="pt", padding=True).to(main_device)',
            'inputs = llama_tokenizer(few_shot_prompts, return_tensors="pt", padding=True, truncation=True, max_length=getattr(args, "prefix_prompt_max_length", 1024)).to(main_device)',
        )
        source = source.replace(
            'max_new_tokens=512, \n            do_sample=True,\n            top_p=0.9,\n            temperature=0.7,\n            pad_token_id=llama_tokenizer.pad_token_id\n        )',
            'max_new_tokens=getattr(args, "prefix_max_new_tokens", 128), \n            do_sample=True,\n            top_p=0.9,\n            temperature=0.7,\n            pad_token_id=llama_tokenizer.pad_token_id\n        )',
            1,
        )
        source = source.replace(
            "def get_llm_response_batch(prompts, models, args, max_new_tokens=512):\n    start_time = time.time()",
            'def get_llm_response_batch(prompts, models, args, max_new_tokens=None):\n    if max_new_tokens is None:\n        max_new_tokens = getattr(args, "answer_max_new_tokens", 32)\n    start_time = time.time()',
        )
        source = source.replace(
            'inputs = llama_tokenizer(prompt_with_template, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(llama_model.device)',
            'inputs = llama_tokenizer(prompt_with_template, return_tensors="pt", padding=True, truncation=True, max_length=getattr(args, "answer_prompt_max_length", 1536)).to(llama_model.device)',
        )

        runtime_script_path = self.run_dir / "runtime_port_pipeline_wmdp.py"
        runtime_script_path.write_text(source, encoding="utf-8")
        ast.parse(source, filename=str(runtime_script_path))
        spec = importlib.util.spec_from_file_location("runtime_port_pipeline_wmdp", runtime_script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module, runtime_script_path

    @staticmethod
    def install_recreated_setup(port_wmdp, classifier_head_ckpt: str):
        import joblib
        import numpy as np
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5TokenizerFast

        class _RecreatedTfidfEncoder:
            def __init__(self, vectorizer):
                self.vectorizer = vectorizer

            def encode(self, texts):
                matrix = self.vectorizer.transform(texts)
                if hasattr(matrix, "toarray"):
                    matrix = matrix.toarray()
                return matrix.astype(np.float32)

        class _DummyClassifierTokenizer:
            def __call__(self, texts, truncation=True, padding="max_length", max_length=512, return_tensors="pt"):
                batch_size = len(texts)
                return {
                    "input_ids": torch.zeros((batch_size, 1), dtype=torch.long),
                    "attention_mask": torch.ones((batch_size, 1), dtype=torch.long),
                }

        class RecreatedWeakPostHocClassifier(torch.nn.Module):
            def __init__(self, classifier_artifact_path: str):
                super().__init__()
                artifact = joblib.load(classifier_artifact_path)
                vectorizer = artifact["vectorizer"]
                classifier = artifact["classifier"]
                if list(classifier.classes_) != [0, 1]:
                    raise RuntimeError(f"Expected classifier classes [0, 1], got {classifier.classes_}")
                self.encoder = _RecreatedTfidfEncoder(vectorizer)
                coef = classifier.coef_[0].astype(np.float32)
                intercept = float(classifier.intercept_[0])
                self.register_buffer("coef", torch.from_numpy(coef))
                self.register_buffer("intercept", torch.tensor(intercept, dtype=torch.float32))

            def forward(self, input_ids=None, attention_mask=None, features=None, labels=None, **kwargs):
                if features is None:
                    raise RuntimeError("Recreated classifier requires TF-IDF features.")
                features = features.to(self.coef.device, dtype=torch.float32)
                decision = features @ self.coef + self.intercept
                logits = torch.stack([-decision / 2.0, decision / 2.0], dim=1)
                return {"logits": logits}

        def setup_all_models_recreated(args):
            main_device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
            dtype = getattr(torch, args.torch_dtype)

            t5_tokenizer = T5TokenizerFast.from_pretrained(args.t5_model_path)
            t5_model = T5ForConditionalGeneration.from_pretrained(args.t5_model_path).to(main_device)
            t5_model.eval()

            llama_tokenizer = AutoTokenizer.from_pretrained(args.model_hub_name, trust_remote_code=True)
            if llama_tokenizer.pad_token is None:
                llama_tokenizer.pad_token = llama_tokenizer.eos_token
            llama_tokenizer.padding_side = "left"

            llama_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
            if getattr(llama_config, "pad_token_id", None) is None:
                llama_config.pad_token_id = llama_tokenizer.pad_token_id

            llama_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                config=llama_config,
                torch_dtype=dtype if main_device.type == "cuda" else torch.float32,
                attn_implementation="sdpa",
                trust_remote_code=True,
            ).to(main_device)
            llama_model.config.pad_token_id = llama_tokenizer.pad_token_id
            llama_model.eval()

            classifier_model = RecreatedWeakPostHocClassifier(args.classifier_head_ckpt).to(main_device)
            classifier_model.eval()
            return {
                "t5_model": t5_model,
                "t5_tokenizer": t5_tokenizer,
                "prefix_llama_model": llama_model,
                "main_llama_model": llama_model,
                "llama_tokenizer": llama_tokenizer,
                "classifier_model": classifier_model,
                "classifier_tokenizer": _DummyClassifierTokenizer(),
            }

        port_wmdp.setup_all_models = setup_all_models_recreated
        print(f"Installed recreated setup using classifier artifact: {classifier_head_ckpt}")

    def build_matrix_jobs(self) -> list[dict]:
        matrix_jobs = []
        for variant in self.config["wmdp_variants"]:
            for domain in self.config["wmdp_domains"]:
                dataset, prompt_source = self._load_wmdp_split(variant, domain)
                max_samples = self.config["max_samples"]
                if max_samples > 0:
                    dataset = dataset.select(range(min(max_samples, len(dataset))))
                records = []
                for idx, row in enumerate(dataset):
                    choices = normalize_choices(row["choices"])
                    if prompt_source == "full_question":
                        prompt = str(row["full_question"])
                    else:
                        prompt = format_original_prompt(row["question"], choices)
                    records.append(
                        {
                            "variant": variant,
                            "domain": domain,
                            "wmdp_set": self.domain_to_set[domain],
                            "row_index": idx,
                            "question": row.get("question", ""),
                            "prompt": prompt,
                            "prompt_source": prompt_source,
                            "choices": choices,
                            "correct_answer_index": int(row["answer"]),
                            "correct_answer_text": choices[int(row["answer"])] if 0 <= int(row["answer"]) < len(choices) else "",
                        }
                    )
                if not records:
                    raise RuntimeError(f"No WMDP records selected for {variant}/{domain}.")
                if variant in {"noise_prefix", "composite"}:
                    assert prompt_source == "full_question"
                    assert records[0]["prompt"].rstrip().endswith("Answer:")
                else:
                    assert prompt_source == "question_plus_choices"
                matrix_jobs.append(
                    {
                        "variant": variant,
                        "domain": domain,
                        "wmdp_set": self.domain_to_set[domain],
                        "prompt_source": prompt_source,
                        "records": records,
                    }
                )
        preview = [
            {
                "variant": job["variant"],
                "domain": job["domain"],
                "wmdp_set": job["wmdp_set"],
                "rows": len(job["records"]),
                "prompt_source": job["prompt_source"],
                "first_prompt_preview": job["records"][0]["prompt"][:500],
            }
            for job in matrix_jobs
        ]
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return matrix_jobs

    def run_matrix(self, port_wmdp, runtime_script_path: Path, artifact_info: dict, classifier_info: dict) -> dict:
        import pandas as pd

        with self.example_library_path.open("r", encoding="utf-8") as f:
            example_library = json.load(f)

        base_args = SimpleNamespace(
            artifact_mode=self.config["artifact_mode"],
            t5_model_path=artifact_info["t5_model_path"],
            model_path=self.config["target_model_path"],
            model_hub_name=self.config["target_model_hub_name"],
            eco_config_path=str(self.eco_config_path),
            model_name=self.config["model_name"],
            wmdp_set="wmdp-bio",
            classifier_base_model=classifier_info["classifier_base_model"],
            classifier_head_ckpt=classifier_info["classifier_head_ckpt"],
            example_library_path=str(self.example_library_path),
            example_library=example_library,
            output_dir=str(self.run_dir / "port_outputs"),
            icl_example_k=self.config["icl_example_k"],
            classifier_conf_threshold=self.config["classifier_conf_threshold"],
            batch_size=self.config["batch_size"],
            eval_batch_size=self.config["batch_size"],
            max_samples=self.config["max_samples"],
            device=self.config["device"],
            torch_dtype=self.config["torch_dtype"],
            prefix_prompt_max_length=self.config["prefix_prompt_max_length"],
            prefix_max_new_tokens=self.config["prefix_max_new_tokens"],
            answer_prompt_max_length=self.config["answer_prompt_max_length"],
            answer_max_new_tokens=self.config["answer_max_new_tokens"],
        )

        run_config = {
            "purpose": "paper_port_wmdp_recreated_artifact_smoke_matrix",
            "artifact_mode": self.config["artifact_mode"],
            "artifact_note": artifact_info["audit"]["artifact_note"],
            "artifact_source": artifact_info["audit"]["artifact_source"],
            "project_root": str(self.project_root),
            "commit": self.commit_sha,
            "runtime_script_path": str(runtime_script_path),
            "recreated_artifact_dir": str(artifact_info["artifact_dir"]),
            "target_model_hub_name": self.config["target_model_hub_name"],
            "target_model_path": self.config["target_model_path"],
            "model_name": self.config["model_name"],
            "torch_dtype": self.config["torch_dtype"],
            "t5_model_path": artifact_info["t5_model_path"],
            "classifier_base_model": classifier_info["classifier_base_model"],
            "classifier_head_ckpt": classifier_info["classifier_head_ckpt"],
            "classifier_metadata_path": str(classifier_info["classifier_artifact_dir"] / "classifier_metadata.json"),
            "wmdp_variants": self.config["wmdp_variants"],
            "wmdp_domains": self.config["wmdp_domains"],
            "max_samples": self.config["max_samples"],
            "batch_size": self.config["batch_size"],
            "classifier_conf_threshold": self.config["classifier_conf_threshold"],
            "prefix_prompt_max_length": self.config["prefix_prompt_max_length"],
            "prefix_max_new_tokens": self.config["prefix_max_new_tokens"],
            "answer_prompt_max_length": self.config["answer_prompt_max_length"],
            "answer_max_new_tokens": self.config["answer_max_new_tokens"],
            "run_dir": str(self.run_dir),
        }
        (self.run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

        matrix_jobs = self.build_matrix_jobs()
        start_load = time.perf_counter()
        models = port_wmdp.setup_all_models(base_args)
        model_load_seconds = time.perf_counter() - start_load

        tokenizer = models["llama_tokenizer"]
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(tokenizer, "chat_template", None) is None:
            tokenizer.chat_template = (
                "{% for message in messages %}{{ message['content'] }}{% if not loop.last %}\n"
                "{% endif %}{% endfor %}{% if add_generation_prompt %}\n{% endif %}"
            )

        summary_rows = []
        all_results = []
        partial_summary_path = self.run_dir / "matrix_summary_partial.csv"
        partial_summary_json_path = self.run_dir / "matrix_summary_partial.json"

        for job_index, job in enumerate(matrix_jobs, start=1):
            variant = job["variant"]
            domain = job["domain"]
            wmdp_set = job["wmdp_set"]
            prompt_source = job["prompt_source"]
            records = job["records"]
            prompts = [item["prompt"] for item in records]
            args = SimpleNamespace(**vars(base_args))
            args.wmdp_set = wmdp_set

            print(f"\n=== Job {job_index}/{len(matrix_jobs)}: {variant}/{domain}, rows={len(records)}, prompt_source={prompt_source} ===")
            start_run = time.perf_counter()
            generated_answers, rethink_count, rethink_info, final_generation_prompts = port_wmdp.run_end_to_end_for_questions(prompts, models, args)
            run_seconds = time.perf_counter() - start_run

            results = []
            for item, answer, rethink, final_prompt in zip(records, generated_answers, rethink_info, final_generation_prompts):
                predicted_letter = port_wmdp.extract_choice_from_answer(answer, item["choices"])
                predicted_index = ord(predicted_letter) - ord("A") if predicted_letter in self.choice_labels else None
                is_correct = predicted_index == item["correct_answer_index"] if predicted_index is not None else False
                results.append(
                    {
                        **item,
                        "generated_answer": answer,
                        "generated_choice_letter": predicted_letter,
                        "predicted_index": predicted_index,
                        "is_correct": bool(is_correct),
                        "rethink_triggered": bool(rethink),
                        "generation_prompt": final_prompt,
                    }
                )

            accuracy = sum(1 for row in results if row["is_correct"]) / len(results)
            valid_rate = sum(1 for row in results if row["predicted_index"] is not None) / len(results)
            rethink_rate = sum(1 for row in results if row["rethink_triggered"]) / len(results)
            metrics = {
                "variant": variant,
                "domain": domain,
                "wmdp_set": wmdp_set,
                "prompt_source": prompt_source,
                "accuracy": accuracy,
                "valid_predictions_rate": valid_rate,
                "rethink_count": int(rethink_count),
                "rethink_rate": rethink_rate,
                "model_load_seconds": model_load_seconds,
                "run_seconds": run_seconds,
                "rows": len(results),
            }

            output_dir = Path(args.output_dir) / self.config["model_name"].replace("/", "_") / variant / wmdp_set
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "final_generations_full.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            (output_dir / "final_metrics_full.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            (output_dir / "rethink_stats.json").write_text(json.dumps({"test": int(rethink_count)}, indent=2), encoding="utf-8")
            (output_dir / "timing_stats.json").write_text(
                json.dumps({"model_load_seconds": model_load_seconds, "test": run_seconds, "pipeline_total_time": model_load_seconds + run_seconds}, indent=2),
                encoding="utf-8",
            )
            pd.DataFrame(results).to_csv(output_dir / "predictions.csv", index=False)

            summary_row = {**metrics, "output_dir": str(output_dir)}
            summary_rows.append(summary_row)
            all_results.extend(results)
            pd.DataFrame(summary_rows).to_csv(partial_summary_path, index=False)
            partial_summary_json_path.write_text(json.dumps(summary_rows, indent=2, default=str), encoding="utf-8")
            print(json.dumps(summary_row, indent=2, default=str))

        matrix_summary = pd.DataFrame(summary_rows)
        summary_csv_path = self.run_dir / "matrix_summary.csv"
        summary_json_path = self.run_dir / "matrix_summary.json"
        all_predictions_path = self.run_dir / "all_predictions.csv"
        summary_payload_path = self.run_dir / "summary.json"
        matrix_summary.to_csv(summary_csv_path, index=False)
        summary_json_path.write_text(json.dumps(summary_rows, indent=2, default=str), encoding="utf-8")
        pd.DataFrame(all_results).to_csv(all_predictions_path, index=False)
        summary_payload = {
            "run_config": run_config,
            "classifier_metrics": classifier_info["classifier_metadata"]["metrics"],
            "model_load_seconds": model_load_seconds,
            "jobs": summary_rows,
            "total_rows": len(all_results),
            "summary_csv": str(summary_csv_path),
            "all_predictions_csv": str(all_predictions_path),
        }
        summary_payload_path.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
        print(json.dumps(summary_payload, indent=2, default=str)[:6000])
        return {
            "matrix_jobs": matrix_jobs,
            "summary_rows": summary_rows,
            "all_results": all_results,
            "matrix_summary": matrix_summary,
            "summary_payload": summary_payload,
        }

    def verify(self, classifier_info: dict, matrix_result: dict) -> dict:
        summary_rows = matrix_result["summary_rows"]
        all_results = matrix_result["all_results"]
        matrix_summary = matrix_result["matrix_summary"]
        expected_jobs = len(self.config["wmdp_variants"]) * len(self.config["wmdp_domains"])
        if len(summary_rows) != expected_jobs:
            raise RuntimeError(f"Expected {expected_jobs} jobs, got {len(summary_rows)}")

        required_root_files = [
            self.run_dir / "artifact_audit.json",
            self.run_dir / "run_config.json",
            self.run_dir / "summary.json",
            self.run_dir / "matrix_summary.csv",
            self.run_dir / "matrix_summary.json",
            self.run_dir / "all_predictions.csv",
            classifier_info["classifier_artifact_dir"] / "classifier.joblib",
            classifier_info["classifier_artifact_dir"] / "classifier_metadata.json",
        ]
        missing_files = [str(path) for path in required_root_files if not Path(path).exists()]

        for row in summary_rows:
            out = Path(row["output_dir"])
            for name in ["final_generations_full.json", "final_metrics_full.json", "rethink_stats.json", "timing_stats.json", "predictions.csv"]:
                path = out / name
                if not path.exists():
                    missing_files.append(str(path))
            if row["variant"] in {"noise_prefix", "composite"} and row["prompt_source"] != "full_question":
                raise RuntimeError(f"{row['variant']}/{row['domain']} used wrong prompt source: {row['prompt_source']}")
            if row["variant"] == "original" and row["prompt_source"] != "question_plus_choices":
                raise RuntimeError(f"original/{row['domain']} used wrong prompt source: {row['prompt_source']}")
            if row["rows"] <= 0:
                raise RuntimeError(f"No rows for {row['variant']}/{row['domain']}")
            if not (0.0 <= row["valid_predictions_rate"] <= 1.0):
                raise RuntimeError(f"Invalid valid_predictions_rate for {row['variant']}/{row['domain']}: {row['valid_predictions_rate']}")
            if not (0.0 <= row["rethink_rate"] <= 1.0):
                raise RuntimeError(f"Invalid rethink_rate for {row['variant']}/{row['domain']}: {row['rethink_rate']}")
        if missing_files:
            raise RuntimeError(f"Missing expected artifacts: {missing_files}")

        result = {
            "status": "completed",
            "jobs": len(summary_rows),
            "rows": len(all_results),
            "classifier_test_accuracy": classifier_info["classifier_metadata"]["metrics"]["test"]["accuracy"],
            "run_dir": str(self.run_dir),
        }
        print("PAPER PORT WMDP RECREATED ARTIFACT SMOKE MATRIX COMPLETED")
        print("Jobs:", result["jobs"])
        print("Rows:", result["rows"])
        print("Classifier test accuracy:", result["classifier_test_accuracy"])
        print(matrix_summary[["variant", "domain", "rows", "prompt_source", "valid_predictions_rate", "rethink_rate", "run_seconds"]].to_string(index=False))
        print("Artifacts:", self.run_dir)
        print("Important: this is recreated-mode smoke validation, not an official paper metric run.")
        return result

    def run(self) -> dict:
        print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in self.config.items()}, indent=2, default=str))
        artifact_info = self.resolve_or_bootstrap_artifacts()
        classifier_info = self.train_weak_classifier(artifact_info["weak_dataset"])
        port_wmdp, runtime_script_path = self.patch_runtime_pipeline()
        self.install_recreated_setup(port_wmdp, classifier_info["classifier_head_ckpt"])
        matrix_result = self.run_matrix(port_wmdp, runtime_script_path, artifact_info, classifier_info)
        return self.verify(classifier_info, matrix_result)


def run(project_root: str | Path, is_kaggle: bool, commit_sha: str) -> dict:
    return RecreatedSmokeRunner(project_root, is_kaggle, commit_sha).run()
