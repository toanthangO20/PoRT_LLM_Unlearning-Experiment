from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from port_postjudge_rethink_oracle_diagnostic import PostjudgeRethinkOracleDiagnosticRunner
from port_recreated_smoke import env_text


class PostjudgeRoutingSemanticsDiagnosticRunner(PostjudgeRethinkOracleDiagnosticRunner):
    required_job_files = [
        "postjudge_routing_semantics_predictions.csv",
        "job_summary.csv",
        "job_summary.json",
        "prompt_examples.json",
    ]

    route_specs = [
        {
            "id": "paper_keep_label0_conf",
            "label": "Paper-style route with confidence",
            "mode": "label_keep",
            "keep_label": 0,
            "require_confidence": True,
            "keep_condition": "keep initial when postjudge label == 0 and confidence >= threshold; rethink otherwise",
        },
        {
            "id": "paper_keep_label0_no_conf",
            "label": "Paper-style route without confidence",
            "mode": "label_keep",
            "keep_label": 0,
            "require_confidence": False,
            "keep_condition": "keep initial when postjudge label == 0; rethink otherwise",
        },
        {
            "id": "inverted_keep_label1_conf",
            "label": "Inverted correctness route with confidence",
            "mode": "label_keep",
            "keep_label": 1,
            "require_confidence": True,
            "keep_condition": "keep initial when postjudge label == 1 and confidence >= threshold; rethink otherwise",
        },
        {
            "id": "inverted_keep_label1_no_conf",
            "label": "Inverted correctness route without confidence",
            "mode": "label_keep",
            "keep_label": 1,
            "require_confidence": False,
            "keep_condition": "keep initial when postjudge label == 1; rethink otherwise",
        },
        {
            "id": "confidence_keep_high",
            "label": "Confidence-only keep high",
            "mode": "confidence_keep_high",
            "keep_label": None,
            "require_confidence": True,
            "keep_condition": "keep initial when postjudge confidence >= threshold regardless of label; rethink otherwise",
        },
        {
            "id": "confidence_rethink_high",
            "label": "Confidence-only rethink high",
            "mode": "confidence_rethink_high",
            "keep_label": None,
            "require_confidence": True,
            "keep_condition": "rethink when postjudge confidence >= threshold regardless of label; keep initial otherwise",
        },
    ]

    route_family_specs = [
        PostjudgeRethinkOracleDiagnosticRunner.family_specs[0],
        PostjudgeRethinkOracleDiagnosticRunner.family_specs[2],
    ]

    def _read_config(self) -> dict:
        config = super()._read_config()
        explicit_run_name = env_text("PORT_RUN_NAME")
        config["run_name"] = explicit_run_name or f"paper_port_wmdp_postjudge_routing_semantics_diagnostic_{config['model_name']}"
        config["max_samples"] = int(env_text("PORT_MAX_SAMPLES", "32"))
        config["scale_run_family"] = "postjudge_routing_semantics_diagnostic"
        return config

    def resolve_or_bootstrap_artifacts(self) -> dict:
        artifact_info = super().resolve_or_bootstrap_artifacts()
        audit = dict(artifact_info["audit"])
        audit.update(
            {
                "artifact_note": (
                    "recreated mode uses public-data recreated artifacts; notebook 28 tests post-judge routing "
                    "semantics against raw and structure-gated rethink oracle upper bounds."
                ),
                "route_policies": self.route_specs,
                "limitations": [
                    "This is a recreated PoRT diagnostic built from public data, not an official paper checkpoint reproduction.",
                    "The metric is generated answer accuracy, not notebook 11 top-logit paper baseline accuracy.",
                    "Oracle rows use ground-truth correctness to choose between initial and rethink answers, so they are upper bounds only.",
                ],
            }
        )
        (self.run_dir / "artifact_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
        artifact_info["audit"] = audit
        return artifact_info

    @staticmethod
    def _route_prefix(family: str, route: dict) -> str:
        family_prefix = "gated" if family == "structure_gated" else family
        return f"{family_prefix}_route_{route['id']}"

    @staticmethod
    def _route_method(family: str, route: dict) -> str:
        family_prefix = "structure_gated" if family == "structure_gated" else family
        return f"{family_prefix}_route_{route['id']}"

    @staticmethod
    def _route_rethink_triggered(label: int, confidence: float, route: dict, threshold: float) -> bool:
        mode = route["mode"]
        if mode == "label_keep":
            keep = int(label) == int(route["keep_label"])
            if route.get("require_confidence"):
                keep = keep and float(confidence) >= float(threshold)
            return not keep
        if mode == "confidence_keep_high":
            return float(confidence) < float(threshold)
        if mode == "confidence_rethink_high":
            return float(confidence) >= float(threshold)
        raise ValueError(f"Unknown route mode: {mode}")

    def _add_route_predictions_for_family(self, row: dict, spec: dict, port_wmdp, item: dict, args) -> None:
        family = spec["family"]
        label = int(row[spec["postjudge_label_field"]])
        confidence = float(row[spec["postjudge_confidence_field"]])
        initial_answer = row[f"{spec['initial_prefix']}_answer"]
        rethink_answer = row[f"{spec['rethink_prefix']}_answer"]
        initial_correct = bool(row[f"{spec['initial_prefix']}_is_correct"])
        oracle_prefix = spec["oracle_prefix"]

        for route in self.route_specs:
            prefix = self._route_prefix(family, route)
            triggered = self._route_rethink_triggered(label, confidence, route, args.classifier_conf_threshold)
            answer = rethink_answer if triggered else initial_answer
            row[f"{prefix}_route_policy"] = route["id"]
            row[f"{prefix}_route_label"] = route["label"]
            row[f"{prefix}_keep_condition"] = route["keep_condition"]
            row[f"{prefix}_rethink_triggered"] = bool(triggered)
            row[f"{prefix}_decision"] = "rethink" if triggered else "initial"
            row.update(self._choice_fields(port_wmdp, prefix, answer, item))
            row[f"{family}_{route['id']}_matches_oracle"] = (
                row[f"{prefix}_predicted_index"] == row[f"{oracle_prefix}_predicted_index"]
            )
            row[f"{family}_{route['id']}_missed_oracle_rethink"] = (
                bool(row[f"{oracle_prefix}_uses_rethink"]) and not triggered
            )
            row[f"{family}_{route['id']}_rethink_hurt"] = (
                triggered and initial_correct and not bool(row[f"{prefix}_is_correct"])
            )
            row[f"{family}_{route['id']}_rethink_helped"] = (
                triggered and not initial_correct and bool(row[f"{prefix}_is_correct"])
            )

    def _route_summaries(self, rows: list[dict], job: dict, spec: dict) -> list[dict]:
        family = spec["family"]
        summaries = []
        for route in self.route_specs:
            prefix = self._route_prefix(family, route)
            route_key = f"{family}_{route['id']}"
            summaries.append(
                self._method_summary(
                    rows,
                    job,
                    self._route_method(family, route),
                    prefix,
                    family,
                    self._safe_bool_avg(rows, f"{prefix}_rethink_triggered"),
                    spec["postjudge_label_field"],
                    spec["postjudge_confidence_field"],
                    extra={
                        "route_policy": route["id"],
                        "route_label": route["label"],
                        "route_keep_condition": route["keep_condition"],
                        "selective_matches_oracle_rate": self._safe_bool_avg(rows, f"{route_key}_matches_oracle"),
                        "selective_missed_oracle_rethink_rate": self._safe_bool_avg(rows, f"{route_key}_missed_oracle_rethink"),
                        "selective_rethink_hurt_rate": self._safe_bool_avg(rows, f"{route_key}_rethink_hurt"),
                        "route_rethink_helped_rate": self._safe_bool_avg(rows, f"{route_key}_rethink_helped"),
                    },
                )
            )
        return summaries

    def _run_job_diagnostics(self, job_index: int, job: dict, models: dict, base_args: SimpleNamespace, port_wmdp) -> tuple[list[dict], list[dict], dict]:
        args = SimpleNamespace(**vars(base_args))
        args.wmdp_set = job["wmdp_set"]
        records = job["records"]
        raw_prompts = [item["prompt"] for item in records]

        self._set_generation_seed(self.config["seed"], job_index, 1)
        raw_initial_answers = self._generate_answers(raw_prompts, models, args, port_wmdp)
        raw_labels, raw_confidences, raw_expanded = self._classify_initial_answers(
            port_wmdp, raw_initial_answers, records, models, args
        )
        self._set_generation_seed(self.config["seed"], job_index, 2)
        raw_rethink_answers, raw_rethink_prompts = self._rethink_answers(
            raw_prompts, raw_initial_answers, models, args, port_wmdp
        )

        self._set_generation_seed(self.config["seed"], job_index, 3)
        gated_prompts = []
        gate_rows = []
        for start in range(0, len(raw_prompts), args.batch_size):
            batch_gated_prompts, batch_gate_rows = self._compile_and_gate_prompts(
                raw_prompts[start : start + args.batch_size],
                models,
                args,
                port_wmdp,
            )
            gated_prompts.extend(batch_gated_prompts)
            gate_rows.extend(batch_gate_rows)
        compiled_prompts = [row["compiled_prompt"] for row in gate_rows]

        self._set_generation_seed(self.config["seed"], job_index, 4)
        gated_initial_answers = self._generate_answers(gated_prompts, models, args, port_wmdp)
        gated_labels, gated_confidences, gated_expanded = self._classify_initial_answers(
            port_wmdp, gated_initial_answers, records, models, args
        )
        self._set_generation_seed(self.config["seed"], job_index, 5)
        gated_rethink_answers, gated_rethink_prompts = self._rethink_answers(
            gated_prompts, gated_initial_answers, models, args, port_wmdp
        )

        diagnostic_rows = []
        for idx, item in enumerate(records):
            gate_row = gate_rows[idx]
            row = {
                **item,
                **gate_row,
                "compiled_prompt_direct": compiled_prompts[idx],
                "structure_gated_prompt": gated_prompts[idx],
                "classifier_conf_threshold": args.classifier_conf_threshold,
            }
            self._add_family_predictions(
                row,
                self.route_family_specs[0],
                port_wmdp,
                item,
                raw_initial_answers[idx],
                raw_rethink_answers[idx],
                raw_labels[idx],
                raw_confidences[idx],
                raw_expanded[idx],
                raw_rethink_prompts[idx],
                args,
            )
            self._add_family_predictions(
                row,
                self.route_family_specs[1],
                port_wmdp,
                item,
                gated_initial_answers[idx],
                gated_rethink_answers[idx],
                gated_labels[idx],
                gated_confidences[idx],
                gated_expanded[idx],
                gated_rethink_prompts[idx],
                args,
            )
            for spec in self.route_family_specs:
                self._add_route_predictions_for_family(row, spec, port_wmdp, item, args)
            diagnostic_rows.append(row)

        job_summary = [self._raw_direct_summary(diagnostic_rows, job)]
        for spec in self.route_family_specs:
            job_summary.extend(self._family_summaries(diagnostic_rows, job, spec))
            job_summary.extend(self._route_summaries(diagnostic_rows, job, spec))
        for summary in job_summary:
            summary.update(
                {
                    "structure_gate_pass_rate": self._safe_bool_avg(diagnostic_rows, "structure_gate_pass"),
                    "structure_gate_fallback_to_raw_rate": self._safe_bool_avg(diagnostic_rows, "structure_gate_fallback_to_raw"),
                    "t5_compiled_prompt_choice_coverage_avg": self._safe_avg(diagnostic_rows, "t5_compiled_prompt_choice_coverage"),
                    "structure_gate_prompt_choice_coverage_avg": self._safe_avg(diagnostic_rows, "structure_gate_prompt_choice_coverage"),
                }
            )

        prompt_examples = {
            "variant": job["variant"],
            "domain": job["domain"],
            "prompt_source": job["prompt_source"],
            "route_policies": self.route_specs,
            "examples": [
                {
                    "row_index": diagnostic_rows[idx]["row_index"],
                    "raw_prompt_preview": diagnostic_rows[idx]["prompt"][:1200],
                    "compiled_prompt_preview": diagnostic_rows[idx]["compiled_prompt_direct"][:1200],
                    "structure_gated_prompt_preview": diagnostic_rows[idx]["structure_gated_prompt"][:1200],
                    "structure_gate_pass": diagnostic_rows[idx]["structure_gate_pass"],
                    "structure_gate_reasons": diagnostic_rows[idx]["structure_gate_reasons"],
                    "raw_initial_answer": diagnostic_rows[idx]["raw_initial_answer"],
                    "raw_rethink_answer": diagnostic_rows[idx]["raw_rethink_answer"],
                    "gated_initial_answer": diagnostic_rows[idx]["gated_initial_answer"],
                    "gated_rethink_answer": diagnostic_rows[idx]["gated_rethink_answer"],
                    "raw_postjudge_label": diagnostic_rows[idx]["raw_postjudge_label"],
                    "raw_postjudge_confidence": diagnostic_rows[idx]["raw_postjudge_confidence"],
                    "gated_postjudge_label": diagnostic_rows[idx]["gated_postjudge_label"],
                    "gated_postjudge_confidence": diagnostic_rows[idx]["gated_postjudge_confidence"],
                }
                for idx in range(min(3, len(diagnostic_rows)))
            ],
        }
        return diagnostic_rows, job_summary, prompt_examples

    def _load_existing_diagnostic_job(self, output_dir: Path, job: dict, expected_rows: int):
        if not self.config["resume_existing"]:
            return None
        if not all((output_dir / name).exists() for name in self.required_job_files):
            return None
        try:
            import pandas as pd

            rows = pd.read_csv(output_dir / "postjudge_routing_semantics_predictions.csv").to_dict(orient="records")
            summary = pd.read_csv(output_dir / "job_summary.csv").to_dict(orient="records")
        except Exception as exc:
            print(f"Existing routing semantics output is not readable, rerunning {job['variant']}/{job['domain']}: {exc}")
            return None
        if len(rows) != expected_rows:
            print(
                f"Existing routing semantics output has {len(rows)} rows, expected {expected_rows}; "
                f"rerunning {job['variant']}/{job['domain']}"
            )
            return None
        if not summary:
            return None
        for row in summary:
            if row.get("variant") != job["variant"] or row.get("domain") != job["domain"]:
                return None
            row["resume_status"] = "skipped_existing"
        return rows, summary

    def _write_job_artifacts(self, output_dir: Path, rows: list[dict], summary: list[dict], prompt_examples: dict) -> None:
        import pandas as pd

        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_dir / "postjudge_routing_semantics_predictions.csv", index=False)
        pd.DataFrame(summary).to_csv(output_dir / "job_summary.csv", index=False)
        (output_dir / "job_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        (output_dir / "prompt_examples.json").write_text(json.dumps(prompt_examples, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    @staticmethod
    def _weighted_summary(summary_rows: list[dict]) -> list[dict]:
        grouped: dict[str, dict] = {}
        weighted_fields = [
            "rethink_rate",
            "postjudge_positive_rate",
            "postjudge_avg_confidence",
            "selective_matches_oracle_rate",
            "selective_missed_oracle_rethink_rate",
            "selective_rethink_hurt_rate",
            "route_rethink_helped_rate",
            "structure_gate_pass_rate",
            "structure_gate_fallback_to_raw_rate",
            "t5_compiled_prompt_choice_coverage_avg",
            "structure_gate_prompt_choice_coverage_avg",
        ]
        metadata_fields = ["route_policy", "route_label", "route_keep_condition"]
        for row in summary_rows:
            method = row["method"]
            bucket = grouped.setdefault(
                method,
                {
                    "method": method,
                    "family": row.get("family"),
                    "prediction_prefix": row.get("prediction_prefix"),
                    "rows": 0,
                    "correct_count": 0,
                    "valid_predictions_count": 0,
                    "weighted": {field: 0.0 for field in weighted_fields},
                    "weighted_denom": {field: 0 for field in weighted_fields},
                    "metadata": {field: row.get(field) for field in metadata_fields if row.get(field) is not None},
                },
            )
            row_count = int(row["rows"])
            bucket["rows"] += row_count
            bucket["correct_count"] += int(row["correct_count"])
            bucket["valid_predictions_count"] += int(row["valid_predictions_count"])
            for field in metadata_fields:
                if field not in bucket["metadata"] and row.get(field) is not None:
                    bucket["metadata"][field] = row.get(field)
            for field in weighted_fields:
                value = row.get(field)
                if value is not None:
                    bucket["weighted"][field] += float(value) * row_count
                    bucket["weighted_denom"][field] += row_count

        result = []
        for bucket in grouped.values():
            row_count = bucket["rows"]
            item = {
                "method": bucket["method"],
                "family": bucket["family"],
                "prediction_prefix": bucket["prediction_prefix"],
                **bucket["metadata"],
                "rows": row_count,
                "correct_count": bucket["correct_count"],
                "accuracy": bucket["correct_count"] / row_count if row_count else None,
                "valid_predictions_count": bucket["valid_predictions_count"],
                "valid_predictions_rate": bucket["valid_predictions_count"] / row_count if row_count else None,
            }
            for field in weighted_fields:
                denom = bucket["weighted_denom"][field]
                item[field] = bucket["weighted"][field] / denom if denom else None
            result.append(item)
        return sorted(result, key=lambda item: str(item["method"]))

    def _build_run_config(self, runtime_script_path: Path, artifact_info: dict, classifier_info: dict) -> dict:
        run_config = super()._build_run_config(runtime_script_path, artifact_info, classifier_info)
        route_methods = [
            self._route_method(spec["family"], route)
            for spec in self.route_family_specs
            for route in self.route_specs
        ]
        run_config.update(
            {
                "purpose": "paper_port_wmdp_postjudge_routing_semantics_diagnostic",
                "scale_run_family": self.config["scale_run_family"],
                "route_policies": self.route_specs,
                "diagnostic_methods": [
                    "raw_direct_generation",
                    "raw_postjudge_no_rethink",
                    "raw_selective_rethink",
                    "raw_rethink_all",
                    "raw_oracle_initial_vs_rethink",
                    "structure_gated_no_rethink",
                    "structure_gated_selective_rethink",
                    "structure_gated_rethink_all",
                    "structure_gated_oracle_initial_vs_rethink",
                    *route_methods,
                ],
                "oracle_note": "Oracle methods use labels to select the correct answer between initial and rethink if either is correct.",
                "limitations": [
                    "This diagnostic uses generated answers, not notebook 11 top-logit paper baseline metrics.",
                    "This is a recreated PoRT diagnostic built from public data, not an official paper checkpoint reproduction.",
                    "Oracle methods are upper bounds and must not be reported as deployable PoRT metrics.",
                ],
            }
        )
        return run_config

    def _job_output_dir(self, base_args: SimpleNamespace, job: dict) -> Path:
        return (
            Path(base_args.output_dir)
            / self.config["model_name"].replace("/", "_")
            / "postjudge_routing_semantics"
            / job["variant"]
            / job["wmdp_set"]
        )

    def _write_summary_artifacts(
        self,
        run_config: dict,
        classifier_info: dict,
        model_load_seconds: float,
        all_rows: list[dict],
        job_summary_rows: list[dict],
        completed_jobs: list[dict],
        skipped_jobs: list[dict],
        failed_jobs: list[dict],
    ) -> dict:
        import pandas as pd

        predictions_path = self.run_dir / "all_postjudge_routing_semantics_predictions.csv"
        by_job_path = self.run_dir / "postjudge_routing_semantics_summary_by_job.csv"
        overall_path = self.run_dir / "postjudge_routing_semantics_summary_overall.csv"
        failed_jobs_path = self.run_dir / "failed_jobs.json"

        pd.DataFrame(all_rows).to_csv(predictions_path, index=False)
        pd.DataFrame(job_summary_rows).to_csv(by_job_path, index=False)
        overall_summary = self._weighted_summary(job_summary_rows)
        pd.DataFrame(overall_summary).to_csv(overall_path, index=False)
        failed_jobs_path.write_text(json.dumps(failed_jobs, indent=2, default=str), encoding="utf-8")

        by_method = {row["method"]: row for row in overall_summary}
        raw_direct = by_method.get("raw_direct_generation")
        deltas_vs_raw = {}
        deltas_vs_oracle = {}
        for method, row in by_method.items():
            if raw_direct and row.get("accuracy") is not None:
                deltas_vs_raw[f"{method}_minus_raw_direct"] = row["accuracy"] - raw_direct["accuracy"]
            family = row.get("family")
            oracle_method = None
            if family == "raw":
                oracle_method = "raw_oracle_initial_vs_rethink"
            elif family == "structure_gated":
                oracle_method = "structure_gated_oracle_initial_vs_rethink"
            if oracle_method and oracle_method in by_method and row.get("accuracy") is not None:
                deltas_vs_oracle[f"{method}_minus_{oracle_method}"] = row["accuracy"] - by_method[oracle_method]["accuracy"]

        summary_payload = {
            "run_config": run_config,
            "classifier_metrics": classifier_info["classifier_metadata"]["metrics"],
            "model_load_seconds": model_load_seconds,
            "completed_jobs": completed_jobs,
            "skipped_jobs": skipped_jobs,
            "failed_jobs": failed_jobs,
            "rows": len(all_rows),
            "summary_by_job": job_summary_rows,
            "overall_summary": overall_summary,
            "key_deltas_vs_raw_direct": deltas_vs_raw,
            "key_deltas_vs_family_oracle": deltas_vs_oracle,
            "all_postjudge_routing_semantics_predictions_csv": str(predictions_path),
            "postjudge_routing_semantics_summary_by_job_csv": str(by_job_path),
            "postjudge_routing_semantics_summary_overall_csv": str(overall_path),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
        return summary_payload

    def verify(self, classifier_info: dict, matrix_result: dict) -> dict:
        expected_jobs = len(self.config["wmdp_variants"]) * len(self.config["wmdp_domains"])
        if matrix_result["failed_jobs"]:
            raise RuntimeError(f"Routing semantics diagnostic run has failed jobs: {matrix_result['failed_jobs']}")
        if len(matrix_result["completed_jobs"]) + len(matrix_result["skipped_jobs"]) != expected_jobs:
            raise RuntimeError(
                "Expected "
                f"{expected_jobs} jobs, got completed+skipped={len(matrix_result['completed_jobs']) + len(matrix_result['skipped_jobs'])}"
            )

        required_root_files = [
            self.run_dir / "artifact_audit.json",
            self.run_dir / "run_config.json",
            self.run_dir / "summary.json",
            self.run_dir / "all_postjudge_routing_semantics_predictions.csv",
            self.run_dir / "postjudge_routing_semantics_summary_by_job.csv",
            self.run_dir / "postjudge_routing_semantics_summary_overall.csv",
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
            raise RuntimeError(f"Missing expected routing semantics artifacts: {missing_files}")

        overall = matrix_result["summary_payload"]["overall_summary"]
        by_method = {row["method"]: row for row in overall}
        raw_direct = by_method.get("raw_direct_generation")
        raw_current = by_method.get("raw_route_paper_keep_label0_conf")
        raw_inverted = by_method.get("raw_route_inverted_keep_label1_conf")
        raw_inverted_no_conf = by_method.get("raw_route_inverted_keep_label1_no_conf")
        raw_oracle = by_method.get("raw_oracle_initial_vs_rethink")
        gated_current = by_method.get("structure_gated_route_paper_keep_label0_conf")
        gated_inverted = by_method.get("structure_gated_route_inverted_keep_label1_conf")
        gated_oracle = by_method.get("structure_gated_oracle_initial_vs_rethink")
        result = {
            "status": "completed",
            "postjudge_routing_semantics_diagnostic": True,
            "official_paper_checkpoint": False,
            "jobs": expected_jobs,
            "rows": len(matrix_result["all_rows"]),
            "max_samples": self.config["max_samples"],
            "row_count_mode": "full_dataset" if self.config["max_samples"] <= 0 else f"first_{self.config['max_samples']}_per_job",
            "classifier_test_accuracy": classifier_info["classifier_metadata"]["metrics"]["test"]["accuracy"],
            "classifier_test_macro_f1": classifier_info["classifier_metadata"]["metrics"]["test"].get("macro_f1"),
            "raw_direct_accuracy": raw_direct["accuracy"] if raw_direct else None,
            "raw_current_route_accuracy": raw_current["accuracy"] if raw_current else None,
            "raw_inverted_route_accuracy": raw_inverted["accuracy"] if raw_inverted else None,
            "raw_inverted_no_conf_route_accuracy": raw_inverted_no_conf["accuracy"] if raw_inverted_no_conf else None,
            "raw_oracle_accuracy": raw_oracle["accuracy"] if raw_oracle else None,
            "structure_gated_current_route_accuracy": gated_current["accuracy"] if gated_current else None,
            "structure_gated_inverted_route_accuracy": gated_inverted["accuracy"] if gated_inverted else None,
            "structure_gated_oracle_accuracy": gated_oracle["accuracy"] if gated_oracle else None,
            "completed_jobs": len(matrix_result["completed_jobs"]),
            "skipped_jobs": len(matrix_result["skipped_jobs"]),
            "answer_expansion_before_postjudge": True,
            "run_dir": str(self.run_dir),
        }
        print("PAPER PORT WMDP POSTJUDGE ROUTING SEMANTICS DIAGNOSTIC COMPLETED")
        print("Jobs:", result["jobs"])
        print("Rows:", result["rows"])
        print("Row count mode:", result["row_count_mode"])
        print("Classifier test accuracy:", result["classifier_test_accuracy"])
        print("Raw direct accuracy:", result["raw_direct_accuracy"])
        print("Raw current route accuracy:", result["raw_current_route_accuracy"])
        print("Raw inverted route accuracy:", result["raw_inverted_route_accuracy"])
        print("Raw inverted no-confidence route accuracy:", result["raw_inverted_no_conf_route_accuracy"])
        print("Raw oracle accuracy:", result["raw_oracle_accuracy"])
        print("Structure-gated current route accuracy:", result["structure_gated_current_route_accuracy"])
        print("Structure-gated inverted route accuracy:", result["structure_gated_inverted_route_accuracy"])
        print("Structure-gated oracle accuracy:", result["structure_gated_oracle_accuracy"])
        print("Artifacts:", self.run_dir)
        print("Important: oracle methods are upper bounds, not deployable paper metrics.")
        return result


def run(project_root: str | Path, is_kaggle: bool, commit_sha: str) -> dict:
    return PostjudgeRoutingSemanticsDiagnosticRunner(project_root, is_kaggle, commit_sha).run()
