from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from port_recreated_smoke import RecreatedSmokeRunner, env_text, format_original_prompt, normalize_choices


class BestClassifierSmokeRunner(RecreatedSmokeRunner):
    def _read_config(self) -> dict:
        config = super()._read_config()
        config["run_name"] = env_text(
            "PORT_RUN_NAME",
            f"paper_port_wmdp_recreated_best_classifier_smoke_matrix_{config['model_name']}",
        )
        config["best_classifier_samples_per_domain"] = int(env_text("PORT_BEST_CLASSIFIER_SAMPLES_PER_DOMAIN", "256"))
        config["best_classifier_wrong_answers_per_question"] = int(env_text("PORT_BEST_CLASSIFIER_WRONG_ANSWERS_PER_QUESTION", "3"))
        config["best_classifier_feature_set"] = env_text("PORT_BEST_CLASSIFIER_FEATURE_SET", "answer_only")
        config["best_classifier_max_features"] = int(env_text("PORT_BEST_CLASSIFIER_MAX_FEATURES", "50000"))
        return config

    def _classifier_text(self, row: dict) -> str:
        feature_set = self.config["best_classifier_feature_set"]
        answer = str(row.get("answer", ""))
        question = str(row.get("question", ""))
        if feature_set == "answer_only":
            return answer
        if feature_set == "question_answer":
            return f"Question: {question}\nAnswer: {answer}"
        raise ValueError(f"Unsupported PORT_BEST_CLASSIFIER_FEATURE_SET={feature_set}")

    def _build_best_classifier_rows(self) -> list[dict]:
        rows = []
        rng = random.Random(self.config["seed"])
        for variant in self.config["wmdp_variants"]:
            for domain in self.config["wmdp_domains"]:
                dataset, prompt_source = self._load_wmdp_split(variant, domain)
                take = min(self.config["best_classifier_samples_per_domain"], len(dataset))
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
                    }
                    rows.append({"question": prompt, "answer": f"{correct_letter}. {choices[answer_index]}", "label": 1, **base_meta})
                    wrong_indices = [i for i in range(len(choices)) if i != answer_index]
                    rng.shuffle(wrong_indices)
                    for wrong_index in wrong_indices[: self.config["best_classifier_wrong_answers_per_question"]]:
                        wrong_letter = chr(65 + wrong_index)
                        rows.append({"question": prompt, "answer": f"{wrong_letter}. {choices[wrong_index]}", "label": 0, **base_meta})
        return rows

    def _split_by_group(self, rows: list[dict]) -> dict[str, list[dict]]:
        from sklearn.model_selection import GroupShuffleSplit

        labels = [int(row["label"]) for row in rows]
        groups = [row["group_id"] for row in rows]
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.config["seed"])
        train_idx, temp_idx = next(gss1.split(rows, labels, groups))
        temp_rows = [rows[idx] for idx in temp_idx]
        temp_labels = [int(row["label"]) for row in temp_rows]
        temp_groups = [row["group_id"] for row in temp_rows]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=self.config["seed"])
        eval_rel, test_rel = next(gss2.split(temp_rows, temp_labels, temp_groups))
        return {
            "train": [rows[idx] for idx in train_idx],
            "eval": [temp_rows[idx] for idx in eval_rel],
            "test": [temp_rows[idx] for idx in test_rel],
        }

    @staticmethod
    def _overlap_report(split_rows: dict[str, list[dict]]) -> dict:
        by_split = {name: {row["group_id"] for row in rows} for name, rows in split_rows.items()}
        return {
            "train_eval_group_overlap": len(by_split["train"] & by_split["eval"]),
            "train_test_group_overlap": len(by_split["train"] & by_split["test"]),
            "eval_test_group_overlap": len(by_split["eval"] & by_split["test"]),
        }

    def train_weak_classifier(self, weak_dataset: dict[str, Path]) -> dict:
        import joblib
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
            roc_auc_score,
        )

        rows = self._build_best_classifier_rows()
        split_rows = self._split_by_group(rows)
        overlap = self._overlap_report(split_rows)
        if any(value != 0 for value in overlap.values()):
            raise RuntimeError(f"Group split leaked question groups: {overlap}")

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=self.config["best_classifier_max_features"],
        )
        x_train = vectorizer.fit_transform([self._classifier_text(row) for row in split_rows["train"]])
        y_train = np.array([int(row["label"]) for row in split_rows["train"]], dtype=np.int64)
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.config["seed"], solver="liblinear")
        classifier.fit(x_train, y_train)

        metrics = {}
        for split, items in split_rows.items():
            x_values = vectorizer.transform([self._classifier_text(row) for row in items])
            y_values = np.array([int(row["label"]) for row in items], dtype=np.int64)
            pred = classifier.predict(x_values)
            proba = classifier.predict_proba(x_values)[:, 1]
            precision, recall, positive_f1, _ = precision_recall_fscore_support(y_values, pred, average="binary", zero_division=0)
            metrics[split] = {
                "rows": int(len(items)),
                "accuracy": float(accuracy_score(y_values, pred)),
                "macro_f1": float(f1_score(y_values, pred, average="macro", zero_division=0)),
                "positive_precision": float(precision),
                "positive_recall": float(recall),
                "positive_f1": float(positive_f1),
                "roc_auc": float(roc_auc_score(y_values, proba)),
                "average_precision": float(average_precision_score(y_values, proba)),
                "confusion_matrix_0_1": confusion_matrix(y_values, pred, labels=[0, 1]).tolist(),
                "classification_report": classification_report(y_values, pred, output_dict=True, zero_division=0),
            }

        artifact_dir = self.run_dir / "artifacts" / "recreated_best_tfidf_post_classifier"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        classifier_path = artifact_dir / "classifier.joblib"
        joblib.dump({"vectorizer": vectorizer, "classifier": classifier, "metrics": metrics}, classifier_path)
        metadata = {
            "classifier_family": "recreated-best-tfidf-logreg-posthoc",
            "not_official_checkpoint": True,
            "feature_set": self.config["best_classifier_feature_set"],
            "label_scheme": "weak_proxy_wmdp_correct_answer_is_sensitive",
            "source": "rebuilt_from_wmdp",
            "rows": len(rows),
            "split_group_overlap": overlap,
            "metrics": metrics,
            "limitations": [
                "Weak proxy labels are derived from WMDP answer correctness, not paper-author post-judgment labels.",
                "The classifier expects expanded answer text, not only a generated option letter.",
            ],
        }
        (artifact_dir / "classifier_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        (artifact_dir / "classifier_rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"classifier_head_ckpt": str(classifier_path), "test_metrics": metrics["test"]}, indent=2, default=str)[:6000])
        return {
            "classifier_base_model": "recreated-best-tfidf-logreg-posthoc",
            "classifier_head_ckpt": str(classifier_path),
            "classifier_artifact_dir": artifact_dir,
            "classifier_metadata": metadata,
        }

    @staticmethod
    def _expand_answer_text(port_wmdp, answer: str, choices: list[str]) -> str:
        letter = port_wmdp.extract_choice_from_answer(answer, choices)
        if letter in {"A", "B", "C", "D"}:
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(choices):
                return f"{letter}. {choices[idx]}"
        return str(answer)

    def _classify_expanded_answers(self, expanded_answers: list[str], models: dict, args) -> tuple[list[int], list[float]]:
        import torch

        classifier_model = models["classifier_model"]
        if isinstance(classifier_model, torch.nn.DataParallel):
            embeddings = classifier_model.module.encoder.encode(expanded_answers)
        else:
            embeddings = classifier_model.encoder.encode(expanded_answers)
        inputs = {"features": torch.tensor(embeddings, dtype=torch.float32).to(args.device)}
        with torch.no_grad():
            outputs = classifier_model(**inputs)
            probs = torch.softmax(outputs["logits"], dim=1)
            confidence, pred_label = torch.max(probs, dim=1)
        return pred_label.detach().cpu().tolist(), confidence.detach().cpu().tolist()

    def _run_end_to_end_for_records(self, records: list[dict], models: dict, args, port_wmdp) -> tuple[list[dict], int]:
        results = []
        rethink_total = 0
        prompts = [item["prompt"] for item in records]
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            batch_prompts = prompts[start : start + args.batch_size]
            processed_prompts = port_wmdp.run_prefix_compilation_step_batch(batch_prompts, models, args.example_library, args)
            retry_indices = [idx for idx, prompt in enumerate(processed_prompts) if not prompt.strip()]
            if retry_indices:
                retry_questions = [batch_prompts[idx] for idx in retry_indices]
                retry_prompts = port_wmdp.run_prefix_compilation_step_batch(retry_questions, models, args.example_library, args)
                for idx, new_prompt in zip(retry_indices, retry_prompts):
                    processed_prompts[idx] = new_prompt
            for idx in range(len(processed_prompts)):
                if not processed_prompts[idx].strip():
                    processed_prompts[idx] = batch_prompts[idx]

            initial_answers = port_wmdp.get_llm_response_batch(processed_prompts, models, args)
            expanded_answers = [
                self._expand_answer_text(port_wmdp, answer, item["choices"])
                for answer, item in zip(initial_answers, batch_records)
            ]
            pred_labels, confidences = self._classify_expanded_answers(expanded_answers, models, args)

            final_answers = list(initial_answers)
            final_generation_prompts = list(processed_prompts)
            need_rethink_indices = [
                idx for idx, (label, conf) in enumerate(zip(pred_labels, confidences))
                if not (label == 0 and conf >= args.classifier_conf_threshold)
            ]
            rethink_total += len(need_rethink_indices)
            if need_rethink_indices:
                rethink_prompts = [processed_prompts[idx] for idx in need_rethink_indices]
                rethink_initials = [initial_answers[idx] for idx in need_rethink_indices]
                rethink_answers, rethink_prompts_used = port_wmdp.run_rethink_step_batch(rethink_prompts, rethink_initials, models, args)
                for rel_idx, answer, rethink_prompt in zip(need_rethink_indices, rethink_answers, rethink_prompts_used):
                    final_answers[rel_idx] = answer
                    final_generation_prompts[rel_idx] = rethink_prompt

            for rel_idx, item in enumerate(batch_records):
                predicted_letter = port_wmdp.extract_choice_from_answer(final_answers[rel_idx], item["choices"])
                predicted_index = ord(predicted_letter) - ord("A") if predicted_letter in self.choice_labels else None
                is_correct = predicted_index == item["correct_answer_index"] if predicted_index is not None else False
                results.append(
                    {
                        **item,
                        "initial_answer": initial_answers[rel_idx],
                        "expanded_initial_answer": expanded_answers[rel_idx],
                        "postjudge_label": int(pred_labels[rel_idx]),
                        "postjudge_confidence": float(confidences[rel_idx]),
                        "generated_answer": final_answers[rel_idx],
                        "generated_choice_letter": predicted_letter,
                        "predicted_index": predicted_index,
                        "is_correct": bool(is_correct),
                        "rethink_triggered": rel_idx in need_rethink_indices,
                        "generation_prompt": final_generation_prompts[rel_idx],
                    }
                )
        return results, rethink_total

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
            "purpose": "paper_port_wmdp_recreated_best_classifier_smoke_matrix",
            "artifact_mode": self.config["artifact_mode"],
            "artifact_note": artifact_info["audit"]["artifact_note"],
            "artifact_source": artifact_info["audit"]["artifact_source"],
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
            "classifier_feature_set": self.config["best_classifier_feature_set"],
            "answer_expansion_before_postjudge": True,
            "wmdp_variants": self.config["wmdp_variants"],
            "wmdp_domains": self.config["wmdp_domains"],
            "max_samples": self.config["max_samples"],
            "batch_size": self.config["batch_size"],
            "classifier_conf_threshold": self.config["classifier_conf_threshold"],
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
        for job_index, job in enumerate(matrix_jobs, start=1):
            args = SimpleNamespace(**vars(base_args))
            args.wmdp_set = job["wmdp_set"]
            records = job["records"]
            print(f"\n=== Job {job_index}/{len(matrix_jobs)}: {job['variant']}/{job['domain']}, rows={len(records)}, prompt_source={job['prompt_source']} ===")
            start_run = time.perf_counter()
            results, rethink_count = self._run_end_to_end_for_records(records, models, args, port_wmdp)
            run_seconds = time.perf_counter() - start_run
            accuracy = sum(1 for row in results if row["is_correct"]) / len(results)
            valid_rate = sum(1 for row in results if row["predicted_index"] is not None) / len(results)
            rethink_rate = sum(1 for row in results if row["rethink_triggered"]) / len(results)
            postjudge_positive_rate = sum(1 for row in results if row["postjudge_label"] == 1) / len(results)
            avg_confidence = sum(row["postjudge_confidence"] for row in results) / len(results)
            metrics = {
                "variant": job["variant"],
                "domain": job["domain"],
                "wmdp_set": job["wmdp_set"],
                "prompt_source": job["prompt_source"],
                "accuracy": accuracy,
                "valid_predictions_rate": valid_rate,
                "rethink_count": int(rethink_count),
                "rethink_rate": rethink_rate,
                "postjudge_positive_rate": postjudge_positive_rate,
                "postjudge_avg_confidence": avg_confidence,
                "model_load_seconds": model_load_seconds,
                "run_seconds": run_seconds,
                "rows": len(results),
            }
            output_dir = Path(args.output_dir) / self.config["model_name"].replace("/", "_") / job["variant"] / job["wmdp_set"]
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
            pd.DataFrame(summary_rows).to_csv(self.run_dir / "matrix_summary_partial.csv", index=False)
            (self.run_dir / "matrix_summary_partial.json").write_text(json.dumps(summary_rows, indent=2, default=str), encoding="utf-8")
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
        result = super().verify(classifier_info, matrix_result)
        summary_rows = matrix_result["summary_rows"]
        all_rethink = all(row["rethink_rate"] == 1.0 for row in summary_rows)
        result["all_jobs_rethink_rate_one"] = bool(all_rethink)
        result["answer_expansion_before_postjudge"] = True
        if all_rethink:
            print("WARNING: all jobs still have rethink_rate=1.0")
        print("PAPER PORT WMDP RECREATED BEST-CLASSIFIER SMOKE MATRIX COMPLETED")
        return result


def run(project_root: str | Path, is_kaggle: bool, commit_sha: str) -> dict:
    return BestClassifierSmokeRunner(project_root, is_kaggle, commit_sha).run()
