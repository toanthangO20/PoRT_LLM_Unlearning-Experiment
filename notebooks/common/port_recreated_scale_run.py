from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from port_recreated_best_classifier_smoke import BestClassifierSmokeRunner
from port_recreated_smoke import env_bool, env_text


class RecreatedScaleRunner(BestClassifierSmokeRunner):
    required_job_files = [
        "final_generations_full.json",
        "final_metrics_full.json",
        "rethink_stats.json",
        "timing_stats.json",
        "predictions.csv",
    ]

    def _read_config(self) -> dict:
        config = super()._read_config()
        config["run_name"] = env_text(
            "PORT_RUN_NAME",
            f"paper_port_wmdp_recreated_scale_run_{config['model_name']}",
        )
        config["max_samples"] = int(env_text("PORT_MAX_SAMPLES", "32"))
        config["resume_existing"] = env_bool("PORT_RESUME_EXISTING", True)
        config["fail_fast"] = env_bool("PORT_FAIL_FAST", True)
        config["scale_run_family"] = "recreated_port_best_classifier_scale"
        return config

    def _build_base_args(self, artifact_info: dict, classifier_info: dict) -> SimpleNamespace:
        with self.example_library_path.open("r", encoding="utf-8") as f:
            example_library = json.load(f)
        return SimpleNamespace(
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

    def _build_run_config(self, runtime_script_path: Path, artifact_info: dict, classifier_info: dict) -> dict:
        return {
            "purpose": "paper_port_wmdp_recreated_scale_run",
            "artifact_mode": self.config["artifact_mode"],
            "artifact_note": artifact_info["audit"]["artifact_note"],
            "artifact_source": artifact_info["audit"]["artifact_source"],
            "official_paper_checkpoint": False,
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
            "classifier_feature_set": self.config["best_classifier_feature_set"],
            "answer_expansion_before_postjudge": True,
            "wmdp_variants": self.config["wmdp_variants"],
            "wmdp_domains": self.config["wmdp_domains"],
            "max_samples": self.config["max_samples"],
            "row_count_mode": "full_dataset" if self.config["max_samples"] <= 0 else f"first_{self.config['max_samples']}_per_job",
            "batch_size": self.config["batch_size"],
            "classifier_conf_threshold": self.config["classifier_conf_threshold"],
            "prefix_prompt_max_length": self.config["prefix_prompt_max_length"],
            "prefix_max_new_tokens": self.config["prefix_max_new_tokens"],
            "answer_prompt_max_length": self.config["answer_prompt_max_length"],
            "answer_max_new_tokens": self.config["answer_max_new_tokens"],
            "resume_existing": self.config["resume_existing"],
            "fail_fast": self.config["fail_fast"],
            "run_dir": str(self.run_dir),
            "limitations": [
                "This is a recreated PoRT run built from public data, not an official paper checkpoint reproduction.",
                "The post-judge classifier uses weak proxy labels from WMDP answer correctness.",
            ],
        }

    def _job_output_dir(self, base_args: SimpleNamespace, job: dict) -> Path:
        return Path(base_args.output_dir) / self.config["model_name"].replace("/", "_") / job["variant"] / job["wmdp_set"]

    def _load_existing_job(self, output_dir: Path, job: dict, expected_rows: int) -> tuple[dict, list[dict]] | None:
        if not self.config["resume_existing"]:
            return None
        if not all((output_dir / name).exists() for name in self.required_job_files):
            return None
        metrics_path = output_dir / "final_metrics_full.json"
        generations_path = output_dir / "final_generations_full.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            results = json.loads(generations_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Existing output is not readable, rerunning {job['variant']}/{job['domain']}: {exc}")
            return None

        if int(metrics.get("rows", -1)) != expected_rows or len(results) != expected_rows:
            print(
                "Existing output row count does not match current config, rerunning "
                f"{job['variant']}/{job['domain']}: metrics={metrics.get('rows')} results={len(results)} expected={expected_rows}"
            )
            return None
        if metrics.get("variant") != job["variant"] or metrics.get("domain") != job["domain"]:
            return None
        if metrics.get("prompt_source") != job["prompt_source"]:
            return None

        summary_row = {**metrics, "output_dir": str(output_dir), "resume_status": "skipped_existing"}
        return summary_row, results

    @staticmethod
    def _job_metrics(job: dict, results: list[dict], rethink_count: int, model_load_seconds: float, run_seconds: float) -> dict:
        rows = len(results)
        accuracy = sum(1 for row in results if row["is_correct"]) / rows
        valid_rate = sum(1 for row in results if row["predicted_index"] is not None) / rows
        rethink_rate = sum(1 for row in results if row["rethink_triggered"]) / rows
        postjudge_positive_rate = sum(1 for row in results if row["postjudge_label"] == 1) / rows
        avg_confidence = sum(row["postjudge_confidence"] for row in results) / rows
        return {
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
            "rows": rows,
        }

    def _write_job_artifacts(self, output_dir: Path, results: list[dict], metrics: dict) -> None:
        import pandas as pd

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "final_generations_full.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir / "final_metrics_full.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (output_dir / "rethink_stats.json").write_text(json.dumps({"test": int(metrics["rethink_count"])}, indent=2), encoding="utf-8")
        (output_dir / "timing_stats.json").write_text(
            json.dumps(
                {
                    "model_load_seconds": metrics["model_load_seconds"],
                    "test": metrics["run_seconds"],
                    "pipeline_total_time": metrics["model_load_seconds"] + metrics["run_seconds"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        pd.DataFrame(results).to_csv(output_dir / "predictions.csv", index=False)

    @staticmethod
    def _overall_metrics(summary_rows: list[dict]) -> dict:
        total_rows = sum(int(row.get("rows", 0)) for row in summary_rows)

        def weighted(name: str) -> float | None:
            if total_rows == 0:
                return None
            if not all(name in row for row in summary_rows):
                return None
            return sum(float(row[name]) * int(row["rows"]) for row in summary_rows) / total_rows

        return {
            "jobs": len(summary_rows),
            "rows": total_rows,
            "accuracy": weighted("accuracy"),
            "valid_predictions_rate": weighted("valid_predictions_rate"),
            "rethink_count": sum(int(row.get("rethink_count", 0)) for row in summary_rows),
            "rethink_rate": weighted("rethink_rate"),
            "postjudge_positive_rate": weighted("postjudge_positive_rate"),
            "postjudge_avg_confidence": weighted("postjudge_avg_confidence"),
            "run_seconds_total": sum(float(row.get("run_seconds", 0.0)) for row in summary_rows),
        }

    def _write_summary_artifacts(
        self,
        summary_rows: list[dict],
        all_results: list[dict],
        run_config: dict,
        classifier_info: dict,
        model_load_seconds: float,
        completed_jobs: list[dict],
        skipped_jobs: list[dict],
        failed_jobs: list[dict],
    ) -> tuple[object, dict]:
        import pandas as pd

        matrix_summary = pd.DataFrame(summary_rows)
        summary_csv_path = self.run_dir / "matrix_summary.csv"
        summary_json_path = self.run_dir / "matrix_summary.json"
        partial_csv_path = self.run_dir / "matrix_summary_partial.csv"
        partial_json_path = self.run_dir / "matrix_summary_partial.json"
        by_variant_domain_csv_path = self.run_dir / "summary_by_variant_domain.csv"
        by_variant_domain_json_path = self.run_dir / "summary_by_variant_domain.json"
        all_predictions_path = self.run_dir / "all_predictions.csv"
        failed_jobs_path = self.run_dir / "failed_jobs.json"
        summary_payload_path = self.run_dir / "summary.json"

        matrix_summary.to_csv(partial_csv_path, index=False)
        partial_json_path.write_text(json.dumps(summary_rows, indent=2, default=str), encoding="utf-8")
        matrix_summary.to_csv(summary_csv_path, index=False)
        summary_json_path.write_text(json.dumps(summary_rows, indent=2, default=str), encoding="utf-8")
        matrix_summary.to_csv(by_variant_domain_csv_path, index=False)
        by_variant_domain_json_path.write_text(json.dumps(summary_rows, indent=2, default=str), encoding="utf-8")
        pd.DataFrame(all_results).to_csv(all_predictions_path, index=False)
        failed_jobs_path.write_text(json.dumps(failed_jobs, indent=2, default=str), encoding="utf-8")

        summary_payload = {
            "run_config": run_config,
            "classifier_metrics": classifier_info["classifier_metadata"]["metrics"],
            "model_load_seconds": model_load_seconds,
            "jobs": summary_rows,
            "completed_jobs": completed_jobs,
            "skipped_jobs": skipped_jobs,
            "failed_jobs": failed_jobs,
            "overall": self._overall_metrics(summary_rows),
            "summary_csv": str(summary_csv_path),
            "summary_by_variant_domain_csv": str(by_variant_domain_csv_path),
            "all_predictions_csv": str(all_predictions_path),
        }
        summary_payload_path.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
        return matrix_summary, summary_payload

    def run_matrix(self, port_wmdp, runtime_script_path: Path, artifact_info: dict, classifier_info: dict) -> dict:
        base_args = self._build_base_args(artifact_info, classifier_info)
        run_config = self._build_run_config(runtime_script_path, artifact_info, classifier_info)
        (self.run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

        matrix_jobs = self.build_matrix_jobs()
        summary_rows: list[dict] = []
        all_results: list[dict] = []
        completed_jobs: list[dict] = []
        skipped_jobs: list[dict] = []
        failed_jobs: list[dict] = []

        existing_jobs: dict[int, tuple[dict, list[dict]]] = {}
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

        matrix_summary = None
        summary_payload = None
        for job_index, job in enumerate(matrix_jobs, start=1):
            output_dir = self._job_output_dir(base_args, job)
            records = job["records"]
            print(f"\n=== Job {job_index}/{len(matrix_jobs)}: {job['variant']}/{job['domain']}, rows={len(records)}, prompt_source={job['prompt_source']} ===")
            try:
                if job_index in existing_jobs:
                    summary_row, results = existing_jobs[job_index]
                    summary_rows.append(summary_row)
                    all_results.extend(results)
                    skipped_jobs.append({"job_index": job_index, **job, "records": len(records), "output_dir": str(output_dir)})
                    print(f"Skipping completed job: {output_dir}")
                else:
                    if models is None:
                        raise RuntimeError("Internal error: models were not loaded for an incomplete job.")
                    args = SimpleNamespace(**vars(base_args))
                    args.wmdp_set = job["wmdp_set"]
                    start_run = time.perf_counter()
                    results, rethink_count = self._run_end_to_end_for_records(records, models, args, port_wmdp)
                    run_seconds = time.perf_counter() - start_run
                    metrics = self._job_metrics(job, results, rethink_count, model_load_seconds, run_seconds)
                    self._write_job_artifacts(output_dir, results, metrics)
                    summary_row = {**metrics, "output_dir": str(output_dir), "resume_status": "computed"}
                    summary_rows.append(summary_row)
                    all_results.extend(results)
                    completed_jobs.append({"job_index": job_index, **job, "records": len(records), "output_dir": str(output_dir)})
                    print(json.dumps(summary_row, indent=2, default=str))

                matrix_summary, summary_payload = self._write_summary_artifacts(
                    summary_rows,
                    all_results,
                    run_config,
                    classifier_info,
                    model_load_seconds,
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
                    summary_rows,
                    all_results,
                    run_config,
                    classifier_info,
                    model_load_seconds,
                    completed_jobs,
                    skipped_jobs,
                    failed_jobs,
                )
                print(json.dumps(failure, indent=2, default=str))
                if self.config["fail_fast"]:
                    raise

        if matrix_summary is None or summary_payload is None:
            matrix_summary, summary_payload = self._write_summary_artifacts(
                summary_rows,
                all_results,
                run_config,
                classifier_info,
                model_load_seconds,
                completed_jobs,
                skipped_jobs,
                failed_jobs,
            )
        print(json.dumps(summary_payload, indent=2, default=str)[:6000])
        return {
            "matrix_jobs": matrix_jobs,
            "summary_rows": summary_rows,
            "all_results": all_results,
            "matrix_summary": matrix_summary,
            "summary_payload": summary_payload,
            "completed_jobs": completed_jobs,
            "skipped_jobs": skipped_jobs,
            "failed_jobs": failed_jobs,
        }

    def verify(self, classifier_info: dict, matrix_result: dict) -> dict:
        summary_rows = matrix_result["summary_rows"]
        all_results = matrix_result["all_results"]
        matrix_summary = matrix_result["matrix_summary"]
        expected_jobs = len(self.config["wmdp_variants"]) * len(self.config["wmdp_domains"])
        if matrix_result["failed_jobs"]:
            raise RuntimeError(f"Scale run has failed jobs: {matrix_result['failed_jobs']}")
        if len(summary_rows) != expected_jobs:
            raise RuntimeError(f"Expected {expected_jobs} jobs, got {len(summary_rows)}")

        required_root_files = [
            self.run_dir / "artifact_audit.json",
            self.run_dir / "run_config.json",
            self.run_dir / "summary.json",
            self.run_dir / "matrix_summary.csv",
            self.run_dir / "matrix_summary.json",
            self.run_dir / "summary_by_variant_domain.csv",
            self.run_dir / "summary_by_variant_domain.json",
            self.run_dir / "all_predictions.csv",
            self.run_dir / "failed_jobs.json",
            classifier_info["classifier_artifact_dir"] / "classifier.joblib",
            classifier_info["classifier_artifact_dir"] / "classifier_metadata.json",
        ]
        missing_files = [str(path) for path in required_root_files if not Path(path).exists()]

        for row in summary_rows:
            out = Path(row["output_dir"])
            for name in self.required_job_files:
                path = out / name
                if not path.exists():
                    missing_files.append(str(path))
            if row["variant"] in {"noise_prefix", "composite"} and row["prompt_source"] != "full_question":
                raise RuntimeError(f"{row['variant']}/{row['domain']} used wrong prompt source: {row['prompt_source']}")
            if row["variant"] == "original" and row["prompt_source"] != "question_plus_choices":
                raise RuntimeError(f"original/{row['domain']} used wrong prompt source: {row['prompt_source']}")
            if int(row["rows"]) <= 0:
                raise RuntimeError(f"No rows for {row['variant']}/{row['domain']}")
            if not (0.0 <= float(row["valid_predictions_rate"]) <= 1.0):
                raise RuntimeError(f"Invalid valid_predictions_rate for {row['variant']}/{row['domain']}: {row['valid_predictions_rate']}")
            if not (0.0 <= float(row["rethink_rate"]) <= 1.0):
                raise RuntimeError(f"Invalid rethink_rate for {row['variant']}/{row['domain']}: {row['rethink_rate']}")
        if missing_files:
            raise RuntimeError(f"Missing expected artifacts: {missing_files}")

        overall = self._overall_metrics(summary_rows)
        if overall["rows"] != len(all_results):
            raise RuntimeError(f"Summary rows ({overall['rows']}) do not match predictions ({len(all_results)})")
        all_rethink = all(float(row["rethink_rate"]) == 1.0 for row in summary_rows)
        classifier_test = classifier_info["classifier_metadata"]["metrics"]["test"]
        result = {
            "status": "completed",
            "scale_run": True,
            "official_paper_checkpoint": False,
            "jobs": len(summary_rows),
            "rows": len(all_results),
            "max_samples": self.config["max_samples"],
            "row_count_mode": "full_dataset" if self.config["max_samples"] <= 0 else f"first_{self.config['max_samples']}_per_job",
            "classifier_test_accuracy": classifier_test["accuracy"],
            "classifier_test_macro_f1": classifier_test.get("macro_f1"),
            "overall_accuracy": overall["accuracy"],
            "overall_valid_predictions_rate": overall["valid_predictions_rate"],
            "overall_rethink_rate": overall["rethink_rate"],
            "overall_postjudge_positive_rate": overall["postjudge_positive_rate"],
            "completed_jobs": len(matrix_result["completed_jobs"]),
            "skipped_jobs": len(matrix_result["skipped_jobs"]),
            "all_jobs_rethink_rate_one": bool(all_rethink),
            "answer_expansion_before_postjudge": True,
            "run_dir": str(self.run_dir),
        }
        if all_rethink:
            print("WARNING: all jobs have rethink_rate=1.0")
        print("PAPER PORT WMDP RECREATED SCALE RUN COMPLETED")
        print("Jobs:", result["jobs"])
        print("Rows:", result["rows"])
        print("Row count mode:", result["row_count_mode"])
        print("Classifier test accuracy:", result["classifier_test_accuracy"])
        columns = [
            "variant",
            "domain",
            "rows",
            "prompt_source",
            "valid_predictions_rate",
            "rethink_rate",
            "postjudge_positive_rate",
            "run_seconds",
            "resume_status",
        ]
        print(matrix_summary[columns].to_string(index=False))
        print("Artifacts:", self.run_dir)
        print("Important: this is recreated-mode scale validation, not an official paper metric run.")
        return result


def run(project_root: str | Path, is_kaggle: bool, commit_sha: str) -> dict:
    return RecreatedScaleRunner(project_root, is_kaggle, commit_sha).run()
