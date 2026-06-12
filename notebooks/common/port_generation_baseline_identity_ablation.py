from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from port_recreated_ablation_diagnostics import RecreatedAblationDiagnosticsRunner
from port_recreated_smoke import env_text


class GenerationBaselineIdentityAblationRunner(RecreatedAblationDiagnosticsRunner):
    required_job_files = [
        "generation_baseline_predictions.csv",
        "job_summary.csv",
        "job_summary.json",
        "prompt_examples.json",
    ]

    def _read_config(self) -> dict:
        config = super()._read_config()
        config["run_name"] = env_text(
            "PORT_RUN_NAME",
            f"paper_port_wmdp_generation_baseline_identity_ablation_{config['model_name']}",
        )
        config["max_samples"] = int(env_text("PORT_MAX_SAMPLES", "32"))
        config["scale_run_family"] = "generation_baseline_identity_ablation"
        return config

    def _build_run_config(self, runtime_script_path: Path, artifact_info: dict, classifier_info: dict) -> dict:
        run_config = super()._build_run_config(runtime_script_path, artifact_info, classifier_info)
        run_config.update(
            {
                "purpose": "paper_port_wmdp_generation_baseline_identity_ablation",
                "diagnostic_methods": [
                    "top_logit_reference",
                    "generation_no_defense",
                    "identity_prefix_no_rethink",
                    "compiled_prefix_no_rethink",
                ],
                "limitations": [
                    "top_logit_reference is intended to match the no-defense evaluator on the same selected rows.",
                    "generation_no_defense uses sampled generation, so it can differ from top-logit reference even on identical prompts.",
                    "identity_prefix_no_rethink should match generation_no_defense when generation seed and prompts are identical.",
                    "This is a recreated PoRT diagnostic built from public data, not an official paper checkpoint reproduction.",
                ],
            }
        )
        return run_config

    @staticmethod
    def _context_window(model, tokenizer) -> int | None:
        candidates = []
        config = getattr(model, "config", None)
        for attr in ["max_position_embeddings", "n_positions", "max_sequence_length"]:
            value = getattr(config, attr, None)
            if isinstance(value, int) and 0 < value < 1_000_000:
                candidates.append(value)
        tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
        if isinstance(tokenizer_max_length, int) and 0 < tokenizer_max_length < 1_000_000:
            candidates.append(tokenizer_max_length)
        return min(candidates) if candidates else None

    def _top_logit_predictions(self, prompts: list[str], models: dict) -> tuple[list[int | None], list[str], list[dict]]:
        import torch

        model = models["main_llama_model"]
        tokenizer = models["llama_tokenizer"]
        labels = self.choice_labels
        choice_token_ids = []
        for label in labels:
            ids = tokenizer(label, add_special_tokens=False).input_ids
            if not ids:
                raise RuntimeError(f"Tokenizer produced no token ids for choice label {label!r}")
            choice_token_ids.append(ids[0])

        original_padding_side = tokenizer.padding_side
        original_truncation_side = getattr(tokenizer, "truncation_side", None)
        tokenizer.padding_side = "right"
        if original_truncation_side is not None:
            tokenizer.truncation_side = "left"
        kwargs = {"padding": "longest", "return_tensors": "pt"}
        context_window = self._context_window(model, tokenizer)
        if context_window is not None:
            kwargs.update({"truncation": True, "max_length": context_window})
        try:
            encoding = tokenizer(prompts, **kwargs).to(model.device)
        finally:
            tokenizer.padding_side = original_padding_side
            if original_truncation_side is not None:
                tokenizer.truncation_side = original_truncation_side

        with torch.no_grad():
            logits = model(**encoding).logits

        predictions = []
        prediction_labels = []
        logit_rows = []
        choice_tensor = torch.tensor(choice_token_ids, dtype=torch.long, device=logits.device)
        for idx, attention_mask in enumerate(encoding["attention_mask"]):
            prompt_end = int(attention_mask.sum().item()) - 1
            choice_logits = logits[idx, prompt_end, choice_tensor].detach().float().cpu().tolist()
            pred_index = int(max(range(len(choice_logits)), key=lambda item: choice_logits[item]))
            predictions.append(pred_index)
            prediction_labels.append(labels[pred_index])
            logit_rows.append({labels[i]: float(choice_logits[i]) for i in range(len(labels))})
        return predictions, prediction_labels, logit_rows

    def _prediction_fields_from_index(self, prefix: str, predicted_index: int | None, item: dict, answer: str | None = None) -> dict:
        letter = self.choice_labels[predicted_index] if predicted_index is not None and 0 <= predicted_index < len(self.choice_labels) else None
        is_correct = predicted_index == item["correct_answer_index"] if predicted_index is not None else False
        return {
            f"{prefix}_answer": answer if answer is not None else letter,
            f"{prefix}_choice_letter": letter,
            f"{prefix}_predicted_index": predicted_index,
            f"{prefix}_is_correct": bool(is_correct),
        }

    @staticmethod
    def _method_summary(rows: list[dict], method: str, prediction_prefix: str, job: dict) -> dict:
        total = len(rows)
        correct = sum(1 for row in rows if bool(row.get(f"{prediction_prefix}_is_correct")))
        valid = sum(1 for row in rows if row.get(f"{prediction_prefix}_predicted_index") is not None)
        return {
            "variant": job["variant"],
            "domain": job["domain"],
            "wmdp_set": job["wmdp_set"],
            "prompt_source": job["prompt_source"],
            "method": method,
            "rows": total,
            "correct_count": correct,
            "accuracy": correct / total if total else None,
            "valid_predictions_count": valid,
            "valid_predictions_rate": valid / total if total else None,
        }

    def _load_existing_job(self, output_dir: Path, job: dict, expected_rows: int):
        if not self.config["resume_existing"]:
            return None
        if not all((output_dir / name).exists() for name in self.required_job_files):
            return None
        try:
            import pandas as pd

            rows = pd.read_csv(output_dir / "generation_baseline_predictions.csv").to_dict(orient="records")
            summary = pd.read_csv(output_dir / "job_summary.csv").to_dict(orient="records")
        except Exception as exc:
            print(f"Existing generation baseline output is not readable, rerunning {job['variant']}/{job['domain']}: {exc}")
            return None
        if len(rows) != expected_rows:
            print(f"Existing generation baseline output has {len(rows)} rows, expected {expected_rows}; rerunning {job['variant']}/{job['domain']}")
            return None
        if not summary:
            return None
        for row in summary:
            if row.get("variant") != job["variant"] or row.get("domain") != job["domain"]:
                return None
            row["resume_status"] = "skipped_existing"
        return rows, summary

    def _run_job(self, job_index: int, job: dict, models: dict, base_args: SimpleNamespace, port_wmdp) -> tuple[list[dict], list[dict], dict]:
        args = SimpleNamespace(**vars(base_args))
        args.wmdp_set = job["wmdp_set"]
        records = job["records"]
        prompts = [item["prompt"] for item in records]

        top_indices, top_letters, top_logits = self._top_logit_predictions(prompts, models)

        self._set_generation_seed(self.config["seed"], job_index, 1)
        generation_no_defense_answers = self._generate_answers(prompts, models, args, port_wmdp)

        self._set_generation_seed(self.config["seed"], job_index, 1)
        identity_prefix_answers = self._generate_answers(prompts, models, args, port_wmdp)

        self._set_generation_seed(self.config["seed"], job_index, 2)
        compiled_prompts, fallback_flags = self._compile_prompts(prompts, models, args, port_wmdp)

        self._set_generation_seed(self.config["seed"], job_index, 3)
        compiled_prefix_answers = self._generate_answers(compiled_prompts, models, args, port_wmdp)

        rows = []
        for idx, item in enumerate(records):
            row = {
                **item,
                "compiled_prompt": compiled_prompts[idx],
                "compiled_prompt_used_fallback": bool(fallback_flags[idx]),
                "top_logit_choice_logits": json.dumps(top_logits[idx], sort_keys=True),
            }
            row.update(self._prediction_fields_from_index("top_logit_reference", top_indices[idx], item, top_letters[idx]))
            row.update(self._choice_fields(port_wmdp, "generation_no_defense", generation_no_defense_answers[idx], item))
            row.update(self._choice_fields(port_wmdp, "identity_prefix_no_rethink", identity_prefix_answers[idx], item))
            row.update(self._choice_fields(port_wmdp, "compiled_prefix_no_rethink", compiled_prefix_answers[idx], item))
            row["generation_identity_same_answer"] = generation_no_defense_answers[idx] == identity_prefix_answers[idx]
            row["generation_identity_same_index"] = row["generation_no_defense_predicted_index"] == row["identity_prefix_no_rethink_predicted_index"]
            rows.append(row)

        summary = [
            self._method_summary(rows, "top_logit_reference", "top_logit_reference", job),
            self._method_summary(rows, "generation_no_defense", "generation_no_defense", job),
            self._method_summary(rows, "identity_prefix_no_rethink", "identity_prefix_no_rethink", job),
            self._method_summary(rows, "compiled_prefix_no_rethink", "compiled_prefix_no_rethink", job),
        ]
        same_answer_rate = sum(1 for row in rows if row["generation_identity_same_answer"]) / len(rows)
        same_index_rate = sum(1 for row in rows if row["generation_identity_same_index"]) / len(rows)
        for row in summary:
            row["generation_identity_same_answer_rate"] = same_answer_rate
            row["generation_identity_same_index_rate"] = same_index_rate
        prompt_examples = {
            "variant": job["variant"],
            "domain": job["domain"],
            "prompt_source": job["prompt_source"],
            "examples": [
                {
                    "row_index": rows[idx]["row_index"],
                    "prompt_preview": rows[idx]["prompt"][:1200],
                    "compiled_prompt_preview": rows[idx]["compiled_prompt"][:1200],
                    "top_logit_answer": rows[idx]["top_logit_reference_answer"],
                    "generation_no_defense_answer": rows[idx]["generation_no_defense_answer"],
                    "identity_prefix_no_rethink_answer": rows[idx]["identity_prefix_no_rethink_answer"],
                    "compiled_prefix_no_rethink_answer": rows[idx]["compiled_prefix_no_rethink_answer"],
                }
                for idx in range(min(3, len(rows)))
            ],
        }
        return rows, summary, prompt_examples

    @staticmethod
    def _weighted_summary(summary_rows: list[dict]) -> list[dict]:
        grouped: dict[str, dict] = {}
        for row in summary_rows:
            bucket = grouped.setdefault(
                row["method"],
                {
                    "method": row["method"],
                    "rows": 0,
                    "correct_count": 0,
                    "valid_predictions_count": 0,
                    "same_answer_numer": 0.0,
                    "same_index_numer": 0.0,
                },
            )
            row_count = int(row["rows"])
            bucket["rows"] += row_count
            bucket["correct_count"] += int(row["correct_count"])
            bucket["valid_predictions_count"] += int(row["valid_predictions_count"])
            bucket["same_answer_numer"] += float(row.get("generation_identity_same_answer_rate", 0.0)) * row_count
            bucket["same_index_numer"] += float(row.get("generation_identity_same_index_rate", 0.0)) * row_count
        result = []
        for bucket in grouped.values():
            rows = bucket["rows"]
            result.append(
                {
                    "method": bucket["method"],
                    "rows": rows,
                    "correct_count": bucket["correct_count"],
                    "accuracy": bucket["correct_count"] / rows if rows else None,
                    "valid_predictions_count": bucket["valid_predictions_count"],
                    "valid_predictions_rate": bucket["valid_predictions_count"] / rows if rows else None,
                    "generation_identity_same_answer_rate": bucket["same_answer_numer"] / rows if rows else None,
                    "generation_identity_same_index_rate": bucket["same_index_numer"] / rows if rows else None,
                }
            )
        return sorted(result, key=lambda item: item["method"])

    def _write_job_artifacts(self, output_dir: Path, rows: list[dict], summary: list[dict], prompt_examples: dict) -> None:
        import pandas as pd

        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_dir / "generation_baseline_predictions.csv", index=False)
        pd.DataFrame(summary).to_csv(output_dir / "job_summary.csv", index=False)
        (output_dir / "job_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        (output_dir / "prompt_examples.json").write_text(json.dumps(prompt_examples, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def _write_summary_artifacts(
        self,
        run_config: dict,
        classifier_info: dict,
        model_load_seconds: float,
        all_rows: list[dict],
        summary_rows: list[dict],
        completed_jobs: list[dict],
        skipped_jobs: list[dict],
        failed_jobs: list[dict],
    ) -> dict:
        import pandas as pd

        predictions_path = self.run_dir / "all_generation_baseline_predictions.csv"
        by_job_path = self.run_dir / "generation_baseline_summary_by_job.csv"
        overall_path = self.run_dir / "generation_baseline_summary_overall.csv"
        failed_jobs_path = self.run_dir / "failed_jobs.json"

        pd.DataFrame(all_rows).to_csv(predictions_path, index=False)
        pd.DataFrame(summary_rows).to_csv(by_job_path, index=False)
        overall_summary = self._weighted_summary(summary_rows)
        pd.DataFrame(overall_summary).to_csv(overall_path, index=False)
        failed_jobs_path.write_text(json.dumps(failed_jobs, indent=2, default=str), encoding="utf-8")

        top_logit = next((row for row in overall_summary if row["method"] == "top_logit_reference"), None)
        generation = next((row for row in overall_summary if row["method"] == "generation_no_defense"), None)
        identity = next((row for row in overall_summary if row["method"] == "identity_prefix_no_rethink"), None)
        compiled = next((row for row in overall_summary if row["method"] == "compiled_prefix_no_rethink"), None)
        summary_payload = {
            "run_config": run_config,
            "classifier_metrics": classifier_info["classifier_metadata"]["metrics"],
            "model_load_seconds": model_load_seconds,
            "completed_jobs": completed_jobs,
            "skipped_jobs": skipped_jobs,
            "failed_jobs": failed_jobs,
            "rows": len(all_rows),
            "summary_by_job": summary_rows,
            "overall_summary": overall_summary,
            "top_logit_minus_generation_accuracy": (top_logit["accuracy"] - generation["accuracy"]) if top_logit and generation else None,
            "compiled_minus_identity_accuracy": (compiled["accuracy"] - identity["accuracy"]) if compiled and identity else None,
            "all_generation_baseline_predictions_csv": str(predictions_path),
            "generation_baseline_summary_by_job_csv": str(by_job_path),
            "generation_baseline_summary_overall_csv": str(overall_path),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
        return summary_payload

    def run_matrix(self, port_wmdp, runtime_script_path: Path, artifact_info: dict, classifier_info: dict) -> dict:
        base_args = self._build_base_args(artifact_info, classifier_info)
        run_config = self._build_run_config(runtime_script_path, artifact_info, classifier_info)
        (self.run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

        matrix_jobs = self.build_matrix_jobs()
        all_rows: list[dict] = []
        summary_rows: list[dict] = []
        completed_jobs: list[dict] = []
        skipped_jobs: list[dict] = []
        failed_jobs: list[dict] = []

        existing_jobs = {}
        for job_index, job in enumerate(matrix_jobs, start=1):
            output_dir = self._job_output_dir(base_args, job)
            existing = self._load_existing_job(output_dir, job, len(job["records"]))
            if existing is not None:
                existing_jobs[job_index] = existing

        models = None
        model_load_seconds = 0.0
        if len(existing_jobs) != len(matrix_jobs):
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

        summary_payload = None
        for job_index, job in enumerate(matrix_jobs, start=1):
            output_dir = self._job_output_dir(base_args, job)
            print(f"\n=== Generation baseline job {job_index}/{len(matrix_jobs)}: {job['variant']}/{job['domain']}, rows={len(job['records'])} ===")
            try:
                if job_index in existing_jobs:
                    rows, summary = existing_jobs[job_index]
                    skipped_jobs.append({"job_index": job_index, "variant": job["variant"], "domain": job["domain"], "rows": len(job["records"]), "output_dir": str(output_dir)})
                    print(f"Skipping completed generation baseline job: {output_dir}")
                else:
                    if models is None:
                        raise RuntimeError("Internal error: models were not loaded for an incomplete generation baseline job.")
                    start_job = time.perf_counter()
                    rows, summary, prompt_examples = self._run_job(job_index, job, models, base_args, port_wmdp)
                    job_seconds = time.perf_counter() - start_job
                    for row in summary:
                        row["run_seconds"] = job_seconds
                        row["output_dir"] = str(output_dir)
                        row["resume_status"] = "computed"
                    self._write_job_artifacts(output_dir, rows, summary, prompt_examples)
                    completed_jobs.append({"job_index": job_index, "variant": job["variant"], "domain": job["domain"], "rows": len(job["records"]), "run_seconds": job_seconds, "output_dir": str(output_dir)})
                    print(json.dumps(summary, indent=2, default=str))

                all_rows.extend(rows)
                summary_rows.extend(summary)
                summary_payload = self._write_summary_artifacts(
                    run_config,
                    classifier_info,
                    model_load_seconds,
                    all_rows,
                    summary_rows,
                    completed_jobs,
                    skipped_jobs,
                    failed_jobs,
                )
            except Exception as exc:
                failure = {
                    "job_index": job_index,
                    "variant": job["variant"],
                    "domain": job["domain"],
                    "wmdp_set": job["wmdp_set"],
                    "output_dir": str(output_dir),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failed_jobs.append(failure)
                self._write_summary_artifacts(
                    run_config,
                    classifier_info,
                    model_load_seconds,
                    all_rows,
                    summary_rows,
                    completed_jobs,
                    skipped_jobs,
                    failed_jobs,
                )
                print(json.dumps(failure, indent=2, default=str))
                if self.config["fail_fast"]:
                    raise

        if summary_payload is None:
            summary_payload = self._write_summary_artifacts(
                run_config,
                classifier_info,
                model_load_seconds,
                all_rows,
                summary_rows,
                completed_jobs,
                skipped_jobs,
                failed_jobs,
            )
        print(json.dumps(summary_payload, indent=2, default=str)[:6000])
        return {
            "matrix_jobs": matrix_jobs,
            "all_rows": all_rows,
            "summary_rows": summary_rows,
            "summary_payload": summary_payload,
            "completed_jobs": completed_jobs,
            "skipped_jobs": skipped_jobs,
            "failed_jobs": failed_jobs,
        }

    def verify(self, classifier_info: dict, matrix_result: dict) -> dict:
        expected_jobs = len(self.config["wmdp_variants"]) * len(self.config["wmdp_domains"])
        if matrix_result["failed_jobs"]:
            raise RuntimeError(f"Generation baseline run has failed jobs: {matrix_result['failed_jobs']}")
        if len(matrix_result["completed_jobs"]) + len(matrix_result["skipped_jobs"]) != expected_jobs:
            raise RuntimeError(f"Expected {expected_jobs} jobs, got completed+skipped={len(matrix_result['completed_jobs']) + len(matrix_result['skipped_jobs'])}")

        required_root_files = [
            self.run_dir / "artifact_audit.json",
            self.run_dir / "run_config.json",
            self.run_dir / "summary.json",
            self.run_dir / "all_generation_baseline_predictions.csv",
            self.run_dir / "generation_baseline_summary_by_job.csv",
            self.run_dir / "generation_baseline_summary_overall.csv",
            self.run_dir / "failed_jobs.json",
            classifier_info["classifier_artifact_dir"] / "classifier.joblib",
            classifier_info["classifier_artifact_dir"] / "classifier_metadata.json",
        ]
        missing_files = [str(path) for path in required_root_files if not Path(path).exists()]
        base_args = self._build_base_args({"t5_model_path": ""}, classifier_info)
        for job in matrix_result["matrix_jobs"]:
            output_dir = self._job_output_dir(base_args, job)
            for name in self.required_job_files:
                if not (output_dir / name).exists():
                    missing_files.append(str(output_dir / name))
        if missing_files:
            raise RuntimeError(f"Missing expected generation baseline artifacts: {missing_files}")

        overall = matrix_result["summary_payload"]["overall_summary"]
        result = {
            "status": "completed",
            "generation_baseline_identity_ablation": True,
            "official_paper_checkpoint": False,
            "jobs": expected_jobs,
            "rows": len(matrix_result["all_rows"]),
            "max_samples": self.config["max_samples"],
            "overall_summary": overall,
            "top_logit_minus_generation_accuracy": matrix_result["summary_payload"]["top_logit_minus_generation_accuracy"],
            "compiled_minus_identity_accuracy": matrix_result["summary_payload"]["compiled_minus_identity_accuracy"],
            "completed_jobs": len(matrix_result["completed_jobs"]),
            "skipped_jobs": len(matrix_result["skipped_jobs"]),
            "run_dir": str(self.run_dir),
        }
        print("PAPER PORT WMDP GENERATION BASELINE IDENTITY ABLATION COMPLETED")
        print("Jobs:", result["jobs"])
        print("Rows:", result["rows"])
        print("Overall summary:", json.dumps(overall, indent=2, default=str))
        print("Artifacts:", self.run_dir)
        print("Important: this is recreated-mode diagnostics, not an official paper metric run.")
        return result


def run(project_root: str | Path, is_kaggle: bool, commit_sha: str) -> dict:
    return GenerationBaselineIdentityAblationRunner(project_root, is_kaggle, commit_sha).run()
