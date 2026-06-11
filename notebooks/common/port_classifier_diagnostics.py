from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


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


def classifier_text(row: dict, feature_set: str) -> str:
    question = str(row.get("question", ""))
    answer = str(row.get("answer", ""))
    if feature_set == "answer_only":
        return answer
    if feature_set == "question_only":
        return question
    if feature_set == "question_answer":
        return f"Question: {question}\nAnswer: {answer}"
    if feature_set == "metadata_answer":
        return f"Variant: {row.get('variant')}\nDomain: {row.get('domain')}\nAnswer: {answer}"
    if feature_set == "metadata_question_answer":
        return f"Variant: {row.get('variant')}\nDomain: {row.get('domain')}\nQuestion: {question}\nAnswer: {answer}"
    raise ValueError(f"Unknown feature_set={feature_set}")


def json_safe(value):
    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            if isinstance(key, tuple):
                safe_key = "/".join(str(part) for part in key)
            else:
                safe_key = key if isinstance(key, (str, int, float, bool)) or key is None else str(key)
            safe[safe_key] = json_safe(item)
        return safe
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


class ClassifierDiagnosticsRunner:
    valid_variants = {"original", "noise_prefix", "composite"}
    valid_domains = {"bio", "chem", "cyber"}

    def __init__(self, project_root: str | Path, is_kaggle: bool, commit_sha: str):
        self.project_root = Path(project_root).resolve()
        self.is_kaggle = bool(is_kaggle)
        self.commit_sha = commit_sha
        self.config = self._read_config()
        self.run_dir = self.config["output_dir"] / self.config["run_name"]
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _read_config(self) -> dict:
        import numpy as np

        seed = int(env_text("PORT_CLASSIFIER_DIAG_SEED", "1234"))
        random.seed(seed)
        np.random.seed(seed)
        variants = env_list("PORT_CLASSIFIER_DIAG_VARIANTS", "original,noise_prefix,composite")
        domains = env_list("PORT_CLASSIFIER_DIAG_DOMAINS", "bio,chem,cyber")
        unknown_variants = sorted(set(variants) - self.valid_variants)
        unknown_domains = sorted(set(domains) - self.valid_domains)
        if unknown_variants:
            raise ValueError(f"Unsupported variants: {unknown_variants}")
        if unknown_domains:
            raise ValueError(f"Unsupported domains: {unknown_domains}")
        output_dir = Path("/kaggle/working") if self.is_kaggle else self.project_root / "results"
        return {
            "seed": seed,
            "run_name": env_text("PORT_RUN_NAME", "paper_port_recreated_classifier_diagnostics"),
            "output_dir": output_dir,
            "variants": variants,
            "domains": domains,
            "samples_per_domain": int(env_text("PORT_CLASSIFIER_DIAG_SAMPLES_PER_DOMAIN", "256")),
            "wrong_answers_per_question": int(env_text("PORT_CLASSIFIER_DIAG_WRONG_ANSWERS_PER_QUESTION", "3")),
            "feature_sets": env_list(
                "PORT_CLASSIFIER_DIAG_FEATURE_SETS",
                "answer_only,question_only,question_answer,metadata_answer,metadata_question_answer",
            ),
            "max_features": int(env_text("PORT_CLASSIFIER_DIAG_MAX_FEATURES", "50000")),
            "min_df": int(env_text("PORT_CLASSIFIER_DIAG_MIN_DF", "1")),
            "transformer_smoke": env_bool("PORT_CLASSIFIER_DIAG_TRANSFORMER_SMOKE", False),
            "transformer_model": env_text("PORT_CLASSIFIER_DIAG_TRANSFORMER_MODEL", "distilbert-base-uncased"),
            "transformer_max_train": int(env_text("PORT_CLASSIFIER_DIAG_TRANSFORMER_MAX_TRAIN", "512")),
            "transformer_epochs": int(env_text("PORT_CLASSIFIER_DIAG_TRANSFORMER_EPOCHS", "1")),
            "artifact_dir": env_text("PORT_RECREATED_ARTIFACT_DIR"),
            "artifact_zip_url": env_text("PORT_RECREATED_ARTIFACT_ZIP_URL"),
            "artifact_zip_path": env_text("PORT_RECREATED_ARTIFACT_ZIP_PATH"),
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
    def _is_valid_artifact_dir(path: str | Path) -> bool:
        path = Path(path)
        required = [
            path / "datasets" / "weak_post_judgment_classifier_train.json",
            path / "datasets" / "weak_post_judgment_classifier_eval.json",
            path / "datasets" / "weak_post_judgment_classifier_test.json",
        ]
        return all(item.exists() for item in required)

    def _find_artifact_dir(self, root: str | Path) -> Path | None:
        root = Path(root)
        candidates = [root]
        if root.exists():
            candidates.extend(p for p in root.rglob("*") if p.is_dir() and p.name == "paper_port_recreated_artifacts_bootstrap")
            candidates.extend(p.parent for p in root.rglob("recreated_artifact_manifest.json"))
        for candidate in candidates:
            if self._is_valid_artifact_dir(candidate):
                return candidate.resolve()
        return None

    def _download_to(self, url: str, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} -> {destination}")
        urllib.request.urlretrieve(url, destination)
        return destination

    def _extract_zip(self, zip_path: str | Path, destination: str | Path) -> Path | None:
        zip_path = Path(zip_path)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if not zipfile.is_zipfile(zip_path):
            return None
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(destination)
        return self._find_artifact_dir(destination)

    def _discover_zip(self) -> Path | None:
        candidates: list[Path] = []
        if self.config["artifact_zip_path"]:
            candidates.append(Path(self.config["artifact_zip_path"]))
        candidates.append(Path("/kaggle/working/paper_port_recreated_artifacts_bootstrap.zip"))
        if Path("/kaggle/input").exists():
            candidates.extend(Path("/kaggle/input").rglob("paper_port_recreated_artifacts_bootstrap*.zip"))
        candidates.append(self.project_root.parent / f"{self.project_root.name}_artifacts" / "paper_port_recreated_artifacts_bootstrap.zip")
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def resolve_existing_dataset(self) -> tuple[list[dict] | None, dict]:
        source_info = {"source": None}
        configured_dir = self.config["artifact_dir"]
        artifact_dir = None
        if configured_dir and self._is_valid_artifact_dir(configured_dir):
            artifact_dir = Path(configured_dir).resolve()
            source_info["source"] = "env_dir"
        elif self.config["artifact_zip_url"]:
            zip_path = self._download_to(self.config["artifact_zip_url"], self.run_dir / "downloads" / "paper_port_recreated_artifacts_bootstrap.zip")
            artifact_dir = self._extract_zip(zip_path, self.run_dir / "artifact_zip")
            source_info["source"] = "zip_url"
        else:
            zip_path = self._discover_zip()
            if zip_path:
                artifact_dir = self._extract_zip(zip_path, self.run_dir / "artifact_zip")
                source_info["source"] = f"zip_path:{zip_path}"

        if artifact_dir is None:
            return None, source_info

        rows = []
        for split in ["train", "eval", "test"]:
            path = artifact_dir / "datasets" / f"weak_post_judgment_classifier_{split}.json"
            split_rows = json.loads(path.read_text(encoding="utf-8"))
            for row in split_rows:
                item = dict(row)
                item["source_split"] = split
                item["group_id"] = f"{item.get('variant')}::{item.get('domain')}::{item.get('row_index')}"
                rows.append(item)
        source_info["artifact_dir"] = str(artifact_dir)
        return rows, source_info

    def build_proxy_dataset(self) -> tuple[list[dict], dict]:
        rows = []
        rng = random.Random(self.config["seed"])
        for variant in self.config["variants"]:
            for domain in self.config["domains"]:
                dataset, prompt_source = self._load_wmdp_split(variant, domain)
                take = min(self.config["samples_per_domain"], len(dataset))
                for idx in range(take):
                    row = dataset[idx]
                    choices = normalize_choices(row["choices"])
                    answer_index = int(row["answer"])
                    prompt = str(row["full_question"]) if prompt_source == "full_question" else format_original_prompt(row["question"], choices)
                    group_id = f"{variant}::{domain}::{idx}"
                    correct_letter = chr(65 + answer_index)
                    base_meta = {
                        "variant": variant,
                        "domain": domain,
                        "row_index": idx,
                        "prompt_source": prompt_source,
                        "label_scheme": "weak_proxy_wmdp_correct_answer_is_sensitive",
                        "group_id": group_id,
                        "source_split": "rebuilt",
                    }
                    rows.append({"question": prompt, "answer": f"{correct_letter}. {choices[answer_index]}", "label": 1, **base_meta})
                    wrong_indices = [i for i in range(len(choices)) if i != answer_index]
                    rng.shuffle(wrong_indices)
                    for wrong_index in wrong_indices[: self.config["wrong_answers_per_question"]]:
                        wrong_letter = chr(65 + wrong_index)
                        rows.append({"question": prompt, "answer": f"{wrong_letter}. {choices[wrong_index]}", "label": 0, **base_meta})
        source_info = {
            "source": "rebuilt_from_wmdp",
            "variants": self.config["variants"],
            "domains": self.config["domains"],
            "samples_per_domain": self.config["samples_per_domain"],
            "wrong_answers_per_question": self.config["wrong_answers_per_question"],
        }
        return rows, source_info

    @staticmethod
    def describe_rows(rows: list[dict]) -> dict:
        import pandas as pd

        frame = pd.DataFrame(rows)
        variant_domain_counts = {
            f"{variant}/{domain}": int(value)
            for (variant, domain), value in frame.groupby(["variant", "domain"]).size().astype(int).to_dict().items()
        }
        result = {
            "rows": int(len(frame)),
            "label_counts": {str(key): int(value) for key, value in frame["label"].value_counts().sort_index().to_dict().items()},
            "variant_domain_counts": variant_domain_counts,
            "source_split_counts": frame["source_split"].value_counts().astype(int).to_dict() if "source_split" in frame.columns else {},
            "unique_groups": int(frame["group_id"].nunique()),
        }
        return result

    def make_splits(self, rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
        from sklearn.model_selection import GroupShuffleSplit, train_test_split

        seed = self.config["seed"]
        indexed = [dict(row, _idx=idx) for idx, row in enumerate(rows)]
        labels = [int(row["label"]) for row in indexed]
        train_idx, temp_idx = train_test_split(
            list(range(len(indexed))),
            test_size=0.2,
            random_state=seed,
            stratify=labels,
        )
        temp_labels = [labels[idx] for idx in temp_idx]
        eval_rel_idx, test_rel_idx = train_test_split(
            list(range(len(temp_idx))),
            test_size=0.5,
            random_state=seed,
            stratify=temp_labels,
        )
        random_split = {
            "train": [indexed[idx] for idx in train_idx],
            "eval": [indexed[temp_idx[idx]] for idx in eval_rel_idx],
            "test": [indexed[temp_idx[idx]] for idx in test_rel_idx],
        }

        groups = [row["group_id"] for row in indexed]
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx2, temp_idx2 = next(gss1.split(indexed, labels, groups))
        temp_rows = [indexed[idx] for idx in temp_idx2]
        temp_groups = [row["group_id"] for row in temp_rows]
        temp_labels2 = [int(row["label"]) for row in temp_rows]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        eval_rel2, test_rel2 = next(gss2.split(temp_rows, temp_labels2, temp_groups))
        group_split = {
            "train": [indexed[idx] for idx in train_idx2],
            "eval": [temp_rows[idx] for idx in eval_rel2],
            "test": [temp_rows[idx] for idx in test_rel2],
        }
        return {"random_row_split": random_split, "group_by_question_split": group_split}

    @staticmethod
    def overlap_report(split_rows: dict[str, list[dict]]) -> dict:
        by_split = {name: {row["group_id"] for row in rows} for name, rows in split_rows.items()}
        return {
            "train_eval_group_overlap": len(by_split["train"] & by_split["eval"]),
            "train_test_group_overlap": len(by_split["train"] & by_split["test"]),
            "eval_test_group_overlap": len(by_split["eval"] & by_split["test"]),
        }

    def evaluate_baselines(self, rows: list[dict], split_rows: dict[str, list[dict]]) -> dict:
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score

        train_labels = np.array([int(row["label"]) for row in split_rows["train"]])
        majority_label = int(np.bincount(train_labels).argmax())
        rng = np.random.default_rng(self.config["seed"])
        output = {}
        for split, items in split_rows.items():
            y = np.array([int(row["label"]) for row in items])
            majority_pred = np.full_like(y, majority_label)
            random_pred = rng.integers(0, 2, size=len(y))
            output[split] = {
                "rows": int(len(y)),
                "majority_accuracy": float(accuracy_score(y, majority_pred)),
                "majority_macro_f1": float(f1_score(y, majority_pred, average="macro", zero_division=0)),
                "random_accuracy": float(accuracy_score(y, random_pred)),
                "random_macro_f1": float(f1_score(y, random_pred, average="macro", zero_division=0)),
            }
        return output

    def evaluate_tfidf_models(self, split_name: str, split_rows: dict[str, list[dict]]) -> list[dict]:
        import numpy as np
        from sklearn.calibration import calibration_curve
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

        results = []
        for feature_set in self.config["feature_sets"]:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=self.config["min_df"],
                max_features=self.config["max_features"],
            )
            x_train = vectorizer.fit_transform([classifier_text(row, feature_set) for row in split_rows["train"]])
            y_train = np.array([int(row["label"]) for row in split_rows["train"]], dtype=np.int64)
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.config["seed"], solver="liblinear")
            clf.fit(x_train, y_train)
            for split in ["train", "eval", "test"]:
                x = vectorizer.transform([classifier_text(row, feature_set) for row in split_rows[split]])
                y = np.array([int(row["label"]) for row in split_rows[split]], dtype=np.int64)
                pred = clf.predict(x)
                proba = clf.predict_proba(x)[:, 1]
                precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
                try:
                    auc = float(roc_auc_score(y, proba))
                except ValueError:
                    auc = None
                try:
                    ap = float(average_precision_score(y, proba))
                except ValueError:
                    ap = None
                bins = {}
                for lo, hi in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.000001)]:
                    mask = (proba >= lo) & (proba < hi)
                    if mask.any():
                        bins[f"{lo:.1f}-{min(hi, 1.0):.1f}"] = {
                            "count": int(mask.sum()),
                            "positive_rate": float(y[mask].mean()),
                            "predicted_positive_rate": float(pred[mask].mean()),
                        }
                    else:
                        bins[f"{lo:.1f}-{min(hi, 1.0):.1f}"] = {"count": 0}
                try:
                    prob_true, prob_pred = calibration_curve(y, proba, n_bins=5, strategy="uniform")
                    calibration = [{"pred": float(p), "true": float(t)} for p, t in zip(prob_pred, prob_true)]
                except ValueError:
                    calibration = []
                cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
                results.append(
                    {
                        "split_scheme": split_name,
                        "feature_set": feature_set,
                        "split": split,
                        "rows": int(len(y)),
                        "accuracy": float(accuracy_score(y, pred)),
                        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                        "positive_f1": float(f1),
                        "positive_precision": float(precision),
                        "positive_recall": float(recall),
                        "roc_auc": auc,
                        "average_precision": ap,
                        "confusion_matrix_0_1": cm,
                        "pred_positive_rate": float(pred.mean()),
                        "avg_positive_probability": float(proba.mean()),
                        "confidence_bins": bins,
                        "calibration": calibration,
                        "vocab_size": int(len(vectorizer.vocabulary_)),
                    }
                )
        return results

    def run_transformer_smoke(self, split_rows: dict[str, list[dict]], feature_set: str = "question_answer") -> dict | None:
        if not self.config["transformer_smoke"]:
            return None
        import numpy as np
        import torch
        from sklearn.metrics import accuracy_score, f1_score
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        train_rows = split_rows["train"][: self.config["transformer_max_train"]]
        eval_rows = split_rows["eval"]
        tokenizer = AutoTokenizer.from_pretrained(self.config["transformer_model"])
        model = AutoModelForSequenceClassification.from_pretrained(self.config["transformer_model"], num_labels=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        def collate(rows):
            enc = tokenizer(
                [classifier_text(row, feature_set) for row in rows],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            enc["labels"] = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
            return {key: value.to(device) for key, value in enc.items()}

        loader = DataLoader(train_rows, batch_size=8, shuffle=True, collate_fn=collate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        started = time.perf_counter()
        for _epoch in range(self.config["transformer_epochs"]):
            model.train()
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()

        model.eval()
        preds = []
        labels = []
        eval_loader = DataLoader(eval_rows, batch_size=16, shuffle=False, collate_fn=collate)
        with torch.no_grad():
            for batch in eval_loader:
                logits = model(**batch).logits
                preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())
                labels.extend(batch["labels"].detach().cpu().tolist())
        return {
            "model": self.config["transformer_model"],
            "feature_set": feature_set,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "epochs": self.config["transformer_epochs"],
            "eval_accuracy": float(accuracy_score(labels, preds)),
            "eval_macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "seconds": time.perf_counter() - started,
        }

    def run(self) -> dict:
        import pandas as pd

        print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in self.config.items()}, indent=2, default=str))
        rows, source_info = self.resolve_existing_dataset()
        if rows is None:
            rows, source_info = self.build_proxy_dataset()
        for idx, row in enumerate(rows):
            row.setdefault("group_id", f"{row.get('variant')}::{row.get('domain')}::{row.get('row_index')}")
            row.setdefault("source_split", "unknown")
            row["_idx"] = idx

        dataset_path = self.run_dir / "diagnostic_classifier_rows.json"
        dataset_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        dataset_summary = self.describe_rows(rows)
        splits = self.make_splits(rows)

        all_baselines = {}
        all_overlaps = {}
        all_tfidf_results = []
        for split_name, split_rows in splits.items():
            all_overlaps[split_name] = self.overlap_report(split_rows)
            all_baselines[split_name] = self.evaluate_baselines(rows, split_rows)
            all_tfidf_results.extend(self.evaluate_tfidf_models(split_name, split_rows))

        tfidf_frame = pd.DataFrame(all_tfidf_results)
        tfidf_csv = self.run_dir / "tfidf_diagnostic_results.csv"
        tfidf_json = self.run_dir / "tfidf_diagnostic_results.json"
        tfidf_frame.to_csv(tfidf_csv, index=False)
        tfidf_json.write_text(json.dumps(all_tfidf_results, indent=2, default=str), encoding="utf-8")

        test_rows = tfidf_frame[tfidf_frame["split"] == "test"].sort_values(["macro_f1", "accuracy"], ascending=False)
        best_test = test_rows.iloc[0].to_dict() if len(test_rows) else {}
        transformer_result = self.run_transformer_smoke(splits["group_by_question_split"], feature_set="question_answer")

        recommendation = "do_not_run_full_port"
        reasons = []
        if best_test:
            if best_test.get("macro_f1", 0.0) < 0.6 or best_test.get("accuracy", 0.0) < 0.6:
                reasons.append("best held-out TF-IDF classifier remains below 0.60 acc/macro-F1")
            if best_test.get("pred_positive_rate", 1.0) in {0.0, 1.0}:
                reasons.append("best held-out classifier predicts only one class")
        if transformer_result and transformer_result["eval_macro_f1"] >= 0.6 and transformer_result["eval_accuracy"] >= 0.6:
            recommendation = "consider_transformer_smoke_matrix"
            reasons.append("transformer smoke exceeded 0.60 acc/macro-F1 on eval")
        elif not reasons:
            recommendation = "consider_recreated_smoke_matrix_with_best_classifier"

        summary = {
            "status": "completed",
            "project_commit": self.commit_sha,
            "dataset_source": source_info,
            "dataset_summary": dataset_summary,
            "split_overlaps": all_overlaps,
            "baselines": all_baselines,
            "best_test_tfidf": best_test,
            "transformer_smoke": transformer_result,
            "recommendation": recommendation,
            "reasons": reasons,
            "run_dir": str(self.run_dir),
            "tfidf_csv": str(tfidf_csv),
            "tfidf_json": str(tfidf_json),
            "diagnostic_dataset": str(dataset_path),
        }
        summary = json_safe(summary)
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print("PAPER PORT RECREATED CLASSIFIER DIAGNOSTICS COMPLETED")
        print("Rows:", dataset_summary["rows"])
        print("Unique groups:", dataset_summary["unique_groups"])
        print("Best held-out TF-IDF:", json.dumps(best_test, indent=2, default=str)[:4000])
        if transformer_result:
            print("Transformer smoke:", json.dumps(transformer_result, indent=2, default=str))
        print("Recommendation:", recommendation)
        print("Reasons:", reasons)
        print("Artifacts:", self.run_dir)
        return summary


def run(project_root: str | Path, is_kaggle: bool, commit_sha: str) -> dict:
    return ClassifierDiagnosticsRunner(project_root, is_kaggle, commit_sha).run()
