from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from port_recreated_scale_run import RecreatedScaleRunner
from port_recreated_smoke import env_bool, env_text


class RecreatedAblationDiagnosticsRunner(RecreatedScaleRunner):
    required_job_files = [
        "diagnostics_predictions.csv",
        "job_summary.csv",
        "job_summary.json",
        "threshold_summary_by_job.csv",
        "prompt_examples.json",
    ]

    def _read_config(self) -> dict:
        config = super()._read_config()
        config["run_name"] = env_text(
            "PORT_RUN_NAME",
            f"paper_port_wmdp_recreated_ablation_diagnostics_{config['model_name']}",
        )
        config["max_samples"] = int(env_text("PORT_MAX_SAMPLES", "32"))
        config["diagnostic_thresholds"] = self._parse_thresholds(env_text("PORT_DIAGNOSTIC_THRESHOLDS", "0.50,0.60,0.70,0.80,0.90,0.95"))
        config["diagnostic_run_rethink_all"] = env_bool("PORT_DIAGNOSTIC_RETHINK_ALL", True)
        if not config["diagnostic_run_rethink_all"]:
            raise ValueError("PORT_DIAGNOSTIC_RETHINK_ALL=false is not supported; threshold sweep requires rethink answers for every row.")
        config["scale_run_family"] = "recreated_port_ablation_diagnostics"
        return config

    @staticmethod
    def _parse_thresholds(value: str | None) -> list[float]:
        if not value:
            return [0.70]
        thresholds = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            threshold = float(item)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Classifier threshold must be in [0, 1], got {threshold}")
            thresholds.append(threshold)
        return sorted(set(thresholds))

    @staticmethod
    def _threshold_key(threshold: float) -> str:
        return f"t{int(round(threshold * 100)):03d}"

    def _build_run_config(self, runtime_script_path: Path, artifact_info: dict, classifier_info: dict) -> dict:
        run_config = super()._build_run_config(runtime_script_path, artifact_info, classifier_info)
        run_config.update(
            {
                "purpose": "paper_port_wmdp_recreated_ablation_diagnostics",
                "diagnostic_thresholds": self.config["diagnostic_thresholds"],
                "diagnostic_run_rethink_all": self.config["diagnostic_run_rethink_all"],
                "diagnostic_methods": [
                    "raw_direct_generation",
                    "compiled_initial_generation",
                    "rethink_all_generation",
                    "threshold_simulated_final_generation",
                ],
                "limitations": [
                    "This diagnostic uses generated answers for all methods, so it is not directly comparable to notebook 11 top-logit baseline metrics.",
                    "This is a recreated PoRT diagnostic built from public data, not an official paper checkpoint reproduction.",
                ],
            }
        )
        return run_config

    def _job_output_dir(self, base_args: SimpleNamespace, job: dict) -> Path:
        return Path(base_args.output_dir) / self.config["model_name"].replace("/", "_") / job["variant"] / job["wmdp_set"]

    def _load_existing_diagnostic_job(self, output_dir: Path, job: dict, expected_rows: int):
        if not self.config["resume_existing"]:
            return None
        if not all((output_dir / name).exists() for name in self.required_job_files):
            return None
        try:
            import pandas as pd

            rows = pd.read_csv(output_dir / "diagnostics_predictions.csv").to_dict(orient="records")
            job_summary = pd.read_csv(output_dir / "job_summary.csv").to_dict(orient="records")
            threshold_summary = pd.read_csv(output_dir / "threshold_summary_by_job.csv").to_dict(orient="records")
        except Exception as exc:
            print(f"Existing diagnostic output is not readable, rerunning {job['variant']}/{job['domain']}: {exc}")
            return None
        if len(rows) != expected_rows:
            print(f"Existing diagnostic output has {len(rows)} rows, expected {expected_rows}; rerunning {job['variant']}/{job['domain']}")
            return None
        if not job_summary:
            return None
        for row in job_summary:
            if row.get("variant") != job["variant"] or row.get("domain") != job["domain"]:
                return None
            row["resume_status"] = "skipped_existing"
        for row in threshold_summary:
            row["resume_status"] = "skipped_existing"
        return rows, job_summary, threshold_summary

    @staticmethod
    def _set_generation_seed(base_seed: int, job_index: int, step_index: int) -> None:
        try:
            import random
            import numpy as np
            import torch

            seed = int(base_seed) + job_index * 1000 + step_index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception as exc:
            print(f"Could not reset generation seed: {exc}")

    def _choice_fields(self, port_wmdp, prefix: str, answer: str, item: dict) -> dict:
        letter = port_wmdp.extract_choice_from_answer(answer, item["choices"])
        predicted_index = ord(letter) - ord("A") if letter in self.choice_labels else None
        is_correct = predicted_index == item["correct_answer_index"] if predicted_index is not None else False
        return {
            f"{prefix}_answer": answer,
            f"{prefix}_choice_letter": letter,
            f"{prefix}_predicted_index": predicted_index,
            f"{prefix}_is_correct": bool(is_correct),
        }

    def _compile_prompts(self, prompts: list[str], models: dict, args, port_wmdp) -> tuple[list[str], list[bool]]:
        compiled = []
        fallback_flags = []
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            batch_compiled = port_wmdp.run_prefix_compilation_step_batch(batch_prompts, models, args.example_library, args)
            retry_indices = [idx for idx, prompt in enumerate(batch_compiled) if not prompt.strip()]
            if retry_indices:
                retry_questions = [batch_prompts[idx] for idx in retry_indices]
                retry_prompts = port_wmdp.run_prefix_compilation_step_batch(retry_questions, models, args.example_library, args)
                for idx, retry_prompt in zip(retry_indices, retry_prompts):
                    batch_compiled[idx] = retry_prompt
            for idx, prompt in enumerate(batch_compiled):
                used_fallback = False
                if not prompt.strip():
                    prompt = batch_prompts[idx]
                    used_fallback = True
                compiled.append(prompt)
                fallback_flags.append(used_fallback)
        return compiled, fallback_flags

    @staticmethod
    def _generate_answers(prompts: list[str], models: dict, args, port_wmdp) -> list[str]:
        answers = []
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            answers.extend(port_wmdp.get_llm_response_batch(batch_prompts, models, args))
        return answers

    @staticmethod
    def _rethink_answers(prompts: list[str], initial_answers: list[str], models: dict, args, port_wmdp) -> tuple[list[str], list[str]]:
        answers = []
        used_prompts = []
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            batch_initials = initial_answers[start : start + args.batch_size]
            batch_answers, batch_prompts_used = port_wmdp.run_rethink_step_batch(batch_prompts, batch_initials, models, args)
            answers.extend(batch_answers)
            used_prompts.extend(batch_prompts_used)
        return answers, used_prompts

    @staticmethod
    def _method_summary(rows: list[dict], method: str, prediction_prefix: str, job: dict, rethink_rate: float | None = None, threshold: float | None = None) -> dict:
        total = len(rows)
        correct = sum(1 for row in rows if bool(row.get(f"{prediction_prefix}_is_correct")))
        valid = sum(1 for row in rows if row.get(f"{prediction_prefix}_predicted_index") is not None)
        summary = {
            "variant": job["variant"],
            "domain": job["domain"],
            "wmdp_set": job["wmdp_set"],
            "prompt_source": job["prompt_source"],
            "method": method,
            "threshold": threshold,
            "rows": total,
            "correct_count": correct,
            "accuracy": correct / total if total else None,
            "valid_predictions_count": valid,
            "valid_predictions_rate": valid / total if total else None,
            "rethink_rate": rethink_rate,
        }
        return summary

    def _run_job_diagnostics(self, job_index: int, job: dict, models: dict, base_args: SimpleNamespace, port_wmdp) -> tuple[list[dict], list[dict], list[dict], dict]:
        args = SimpleNamespace(**vars(base_args))
        args.wmdp_set = job["wmdp_set"]
        records = job["records"]
        prompts = [item["prompt"] for item in records]

        self._set_generation_seed(self.config["seed"], job_index, 1)
        raw_direct_answers = self._generate_answers(prompts, models, args, port_wmdp)

        self._set_generation_seed(self.config["seed"], job_index, 2)
        compiled_prompts, fallback_flags = self._compile_prompts(prompts, models, args, port_wmdp)

        self._set_generation_seed(self.config["seed"], job_index, 3)
        compiled_initial_answers = self._generate_answers(compiled_prompts, models, args, port_wmdp)

        expanded_initial_answers = [
            self._expand_answer_text(port_wmdp, answer, item["choices"])
            for answer, item in zip(compiled_initial_answers, records)
        ]
        pred_labels, confidences = self._classify_expanded_answers(expanded_initial_answers, models, args)

        self._set_generation_seed(self.config["seed"], job_index, 4)
        rethink_answers, rethink_prompts = self._rethink_answers(compiled_prompts, compiled_initial_answers, models, args, port_wmdp)

        diagnostic_rows = []
        for idx, item in enumerate(records):
            row = {
                **item,
                "compiled_prompt": compiled_prompts[idx],
                "compiled_prompt_used_fallback": bool(fallback_flags[idx]),
                "expanded_initial_answer": expanded_initial_answers[idx],
                "postjudge_label": int(pred_labels[idx]),
                "postjudge_confidence": float(confidences[idx]),
                "rethink_prompt": rethink_prompts[idx],
            }
            row.update(self._choice_fields(port_wmdp, "raw_direct", raw_direct_answers[idx], item))
            row.update(self._choice_fields(port_wmdp, "compiled_initial", compiled_initial_answers[idx], item))
            row.update(self._choice_fields(port_wmdp, "rethink_all", rethink_answers[idx], item))
            for threshold in self.config["diagnostic_thresholds"]:
                key = self._threshold_key(threshold)
                triggered = not (int(pred_labels[idx]) == 0 and float(confidences[idx]) >= threshold)
                final_answer = rethink_answers[idx] if triggered else compiled_initial_answers[idx]
                row[f"final_{key}_rethink_triggered"] = bool(triggered)
                row.update(self._choice_fields(port_wmdp, f"final_{key}", final_answer, item))
            diagnostic_rows.append(row)

        job_summary = [
            self._method_summary(diagnostic_rows, "raw_direct_generation", "raw_direct", job, rethink_rate=0.0),
            self._method_summary(diagnostic_rows, "compiled_initial_generation", "compiled_initial", job, rethink_rate=0.0),
            self._method_summary(diagnostic_rows, "rethink_all_generation", "rethink_all", job, rethink_rate=1.0),
        ]
        threshold_summary = []
        for threshold in self.config["diagnostic_thresholds"]:
            key = self._threshold_key(threshold)
            rethink_rate = sum(1 for row in diagnostic_rows if bool(row[f"final_{key}_rethink_triggered"])) / len(diagnostic_rows)
            summary = self._method_summary(
                diagnostic_rows,
                f"threshold_final_{key}",
                f"final_{key}",
                job,
                rethink_rate=rethink_rate,
                threshold=threshold,
            )
            threshold_summary.append(summary)
            job_summary.append(summary)

        prompt_examples = {
            "variant": job["variant"],
            "domain": job["domain"],
            "prompt_source": job["prompt_source"],
            "examples": [
                {
                    "row_index": diagnostic_rows[idx]["row_index"],
                    "original_prompt_preview": diagnostic_rows[idx]["prompt"][:1200],
                    "compiled_prompt_preview": diagnostic_rows[idx]["compiled_prompt"][:1200],
                    "compiled_prompt_used_fallback": diagnostic_rows[idx]["compiled_prompt_used_fallback"],
                    "raw_direct_answer": diagnostic_rows[idx]["raw_direct_answer"],
                    "compiled_initial_answer": diagnostic_rows[idx]["compiled_initial_answer"],
                    "rethink_all_answer": diagnostic_rows[idx]["rethink_all_answer"],
                    "postjudge_label": diagnostic_rows[idx]["postjudge_label"],
                    "postjudge_confidence": diagnostic_rows[idx]["postjudge_confidence"],
                }
                for idx in range(min(3, len(diagnostic_rows)))
            ],
        }
        return diagnostic_rows, job_summary, threshold_summary, prompt_examples

    def _write_job_artifacts(
        self,
        output_dir: Path,
        diagnostic_rows: list[dict],
        job_summary: list[dict],
        threshold_summary: list[dict],
        prompt_examples: dict,
    ) -> None:
        import pandas as pd

        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diagnostic_rows).to_csv(output_dir / "diagnostics_predictions.csv", index=False)
        pd.DataFrame(job_summary).to_csv(output_dir / "job_summary.csv", index=False)
        pd.DataFrame(threshold_summary).to_csv(output_dir / "threshold_summary_by_job.csv", index=False)
        (output_dir / "job_summary.json").write_text(json.dumps(job_summary, indent=2, default=str), encoding="utf-8")
        (output_dir / "prompt_examples.json").write_text(json.dumps(prompt_examples, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    @staticmethod
    def _weighted_summary(rows: list[dict]) -> list[dict]:
        grouped: dict[tuple, dict] = {}
        for row in rows:
            key = (row["method"], "" if row.get("threshold") is None else row.get("threshold"))
            bucket = grouped.setdefault(
                key,
                {
                    "method": row["method"],
                    "threshold": row.get("threshold"),
                    "rows": 0,
                    "correct_count": 0,
                    "valid_predictions_count": 0,
                    "weighted_rethink_numer": 0.0,
                    "weighted_rethink_denom": 0,
                },
            )
            row_count = int(row["rows"])
            bucket["rows"] += row_count
            bucket["correct_count"] += int(row["correct_count"])
            bucket["valid_predictions_count"] += int(row["valid_predictions_count"])
            if row.get("rethink_rate") is not None:
                bucket["weighted_rethink_numer"] += float(row["rethink_rate"]) * row_count
                bucket["weighted_rethink_denom"] += row_count
        result = []
        for bucket in grouped.values():
            rows_count = bucket["rows"]
            result.append(
                {
                    "method": bucket["method"],
                    "threshold": bucket["threshold"],
                    "rows": rows_count,
                    "correct_count": bucket["correct_count"],
                    "accuracy": bucket["correct_count"] / rows_count if rows_count else None,
                    "valid_predictions_count": bucket["valid_predictions_count"],
                    "valid_predictions_rate": bucket["valid_predictions_count"] / rows_count if rows_count else None,
                    "rethink_rate": bucket["weighted_rethink_numer"] / bucket["weighted_rethink_denom"] if bucket["weighted_rethink_denom"] else None,
                }
            )
        return sorted(result, key=lambda item: (str(item["method"]), -1 if item["threshold"] is None else float(item["threshold"])))

    def _write_summary_artifacts(
        self,
        run_config: dict,
        classifier_info: dict,
        model_load_seconds: float,
        all_rows: list[dict],
        job_summary_rows: list[dict],
        threshold_summary_rows: list[dict],
        completed_jobs: list[dict],
        skipped_jobs: list[dict],
        failed_jobs: list[dict],
    ) -> dict:
        import pandas as pd

        all_predictions_path = self.run_dir / "all_diagnostics_predictions.csv"
        job_summary_path = self.run_dir / "ablation_summary_by_job.csv"
        threshold_summary_path = self.run_dir / "threshold_summary_by_job.csv"
        overall_summary_path = self.run_dir / "ablation_summary_overall.csv"
        failed_jobs_path = self.run_dir / "failed_jobs.json"

        pd.DataFrame(all_rows).to_csv(all_predictions_path, index=False)
        pd.DataFrame(job_summary_rows).to_csv(job_summary_path, index=False)
        pd.DataFrame(threshold_summary_rows).to_csv(threshold_summary_path, index=False)
        overall_summary = self._weighted_summary(job_summary_rows)
        pd.DataFrame(overall_summary).to_csv(overall_summary_path, index=False)
        failed_jobs_path.write_text(json.dumps(failed_jobs, indent=2, default=str), encoding="utf-8")

        final_methods = [row for row in overall_summary if row["method"].startswith("threshold_final_")]
        best_threshold = max(final_methods, key=lambda row: (row["accuracy"], row["valid_predictions_rate"])) if final_methods else None
        summary_payload = {
            "run_config": run_config,
            "classifier_metrics": classifier_info["classifier_metadata"]["metrics"],
            "model_load_seconds": model_load_seconds,
            "completed_jobs": completed_jobs,
            "skipped_jobs": skipped_jobs,
            "failed_jobs": failed_jobs,
            "rows": len(all_rows),
            "job_summary": job_summary_rows,
            "threshold_summary_by_job": threshold_summary_rows,
            "overall_summary": overall_summary,
            "best_threshold_final_method": best_threshold,
            "all_diagnostics_predictions_csv": str(all_predictions_path),
            "ablation_summary_by_job_csv": str(job_summary_path),
            "threshold_summary_by_job_csv": str(threshold_summary_path),
            "ablation_summary_overall_csv": str(overall_summary_path),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
        return summary_payload

    def run_matrix(self, port_wmdp, runtime_script_path: Path, artifact_info: dict, classifier_info: dict) -> dict:
        base_args = self._build_base_args(artifact_info, classifier_info)
        run_config = self._build_run_config(runtime_script_path, artifact_info, classifier_info)
        (self.run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

        matrix_jobs = self.build_matrix_jobs()
        models = None
        model_load_seconds = 0.0
        all_rows: list[dict] = []
        job_summary_rows: list[dict] = []
        threshold_summary_rows: list[dict] = []
        completed_jobs: list[dict] = []
        skipped_jobs: list[dict] = []
        failed_jobs: list[dict] = []

        existing_jobs = {}
        for job_index, job in enumerate(matrix_jobs, start=1):
            output_dir = self._job_output_dir(base_args, job)
            existing = self._load_existing_diagnostic_job(output_dir, job, len(job["records"]))
            if existing is not None:
                existing_jobs[job_index] = existing

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
            print(f"\n=== Diagnostic job {job_index}/{len(matrix_jobs)}: {job['variant']}/{job['domain']}, rows={len(job['records'])} ===")
            try:
                if job_index in existing_jobs:
                    diagnostic_rows, job_summary, threshold_summary = existing_jobs[job_index]
                    skipped_jobs.append({"job_index": job_index, "variant": job["variant"], "domain": job["domain"], "rows": len(job["records"]), "output_dir": str(output_dir)})
                    print(f"Skipping completed diagnostic job: {output_dir}")
                else:
                    if models is None:
                        raise RuntimeError("Internal error: models were not loaded for an incomplete diagnostic job.")
                    start_job = time.perf_counter()
                    diagnostic_rows, job_summary, threshold_summary, prompt_examples = self._run_job_diagnostics(job_index, job, models, base_args, port_wmdp)
                    job_seconds = time.perf_counter() - start_job
                    for row in job_summary:
                        row["run_seconds"] = job_seconds
                        row["output_dir"] = str(output_dir)
                        row["resume_status"] = "computed"
                    for row in threshold_summary:
                        row["run_seconds"] = job_seconds
                        row["output_dir"] = str(output_dir)
                        row["resume_status"] = "computed"
                    self._write_job_artifacts(output_dir, diagnostic_rows, job_summary, threshold_summary, prompt_examples)
                    completed_jobs.append({"job_index": job_index, "variant": job["variant"], "domain": job["domain"], "rows": len(job["records"]), "run_seconds": job_seconds, "output_dir": str(output_dir)})
                    print(json.dumps(job_summary, indent=2, default=str)[:4000])

                all_rows.extend(diagnostic_rows)
                job_summary_rows.extend(job_summary)
                threshold_summary_rows.extend(threshold_summary)
                summary_payload = self._write_summary_artifacts(
                    run_config,
                    classifier_info,
                    model_load_seconds,
                    all_rows,
                    job_summary_rows,
                    threshold_summary_rows,
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
                    job_summary_rows,
                    threshold_summary_rows,
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
                job_summary_rows,
                threshold_summary_rows,
                completed_jobs,
                skipped_jobs,
                failed_jobs,
            )
        print(json.dumps(summary_payload, indent=2, default=str)[:6000])
        return {
            "matrix_jobs": matrix_jobs,
            "all_rows": all_rows,
            "job_summary_rows": job_summary_rows,
            "threshold_summary_rows": threshold_summary_rows,
            "summary_payload": summary_payload,
            "completed_jobs": completed_jobs,
            "skipped_jobs": skipped_jobs,
            "failed_jobs": failed_jobs,
        }

    def verify(self, classifier_info: dict, matrix_result: dict) -> dict:
        expected_jobs = len(self.config["wmdp_variants"]) * len(self.config["wmdp_domains"])
        if matrix_result["failed_jobs"]:
            raise RuntimeError(f"Diagnostic run has failed jobs: {matrix_result['failed_jobs']}")
        if len(matrix_result["completed_jobs"]) + len(matrix_result["skipped_jobs"]) != expected_jobs:
            raise RuntimeError(f"Expected {expected_jobs} jobs, got completed+skipped={len(matrix_result['completed_jobs']) + len(matrix_result['skipped_jobs'])}")

        required_root_files = [
            self.run_dir / "artifact_audit.json",
            self.run_dir / "run_config.json",
            self.run_dir / "summary.json",
            self.run_dir / "all_diagnostics_predictions.csv",
            self.run_dir / "ablation_summary_by_job.csv",
            self.run_dir / "threshold_summary_by_job.csv",
            self.run_dir / "ablation_summary_overall.csv",
            self.run_dir / "failed_jobs.json",
            classifier_info["classifier_artifact_dir"] / "classifier.joblib",
            classifier_info["classifier_artifact_dir"] / "classifier_metadata.json",
        ]
        missing_files = [str(path) for path in required_root_files if not Path(path).exists()]
        for job in matrix_result["matrix_jobs"]:
            output_dir = self._job_output_dir(self._build_base_args({"t5_model_path": ""}, classifier_info), job)
            for name in self.required_job_files:
                if not (output_dir / name).exists():
                    missing_files.append(str(output_dir / name))
        if missing_files:
            raise RuntimeError(f"Missing expected diagnostic artifacts: {missing_files}")

        summary_payload = matrix_result["summary_payload"]
        best_threshold = summary_payload.get("best_threshold_final_method")
        result = {
            "status": "completed",
            "diagnostic_run": True,
            "official_paper_checkpoint": False,
            "jobs": expected_jobs,
            "rows": len(matrix_result["all_rows"]),
            "max_samples": self.config["max_samples"],
            "thresholds": self.config["diagnostic_thresholds"],
            "classifier_test_accuracy": classifier_info["classifier_metadata"]["metrics"]["test"]["accuracy"],
            "classifier_test_macro_f1": classifier_info["classifier_metadata"]["metrics"]["test"].get("macro_f1"),
            "best_threshold_final_method": best_threshold,
            "completed_jobs": len(matrix_result["completed_jobs"]),
            "skipped_jobs": len(matrix_result["skipped_jobs"]),
            "answer_expansion_before_postjudge": True,
            "run_dir": str(self.run_dir),
        }
        print("PAPER PORT WMDP RECREATED ABLATION DIAGNOSTICS COMPLETED")
        print("Jobs:", result["jobs"])
        print("Rows:", result["rows"])
        print("Thresholds:", result["thresholds"])
        print("Best threshold final method:", best_threshold)
        print("Artifacts:", self.run_dir)
        print("Important: this is recreated-mode diagnostics, not an official paper metric run.")
        return result


def run(project_root: str | Path, is_kaggle: bool, commit_sha: str) -> dict:
    return RecreatedAblationDiagnosticsRunner(project_root, is_kaggle, commit_sha).run()
