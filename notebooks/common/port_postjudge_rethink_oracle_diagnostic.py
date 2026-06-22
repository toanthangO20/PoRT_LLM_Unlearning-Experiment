from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from port_recreated_smoke import env_text
from port_recreated_structure_gate_scale_run import RecreatedStructureGateScaleRunner


class PostjudgeRethinkOracleDiagnosticRunner(RecreatedStructureGateScaleRunner):
    required_job_files = [
        "postjudge_rethink_oracle_predictions.csv",
        "job_summary.csv",
        "job_summary.json",
        "prompt_examples.json",
    ]

    family_specs = [
        {
            "family": "raw",
            "label": "Raw prompt",
            "initial_prefix": "raw_initial",
            "rethink_prefix": "raw_rethink",
            "selective_prefix": "raw_selective",
            "oracle_prefix": "raw_oracle",
            "postjudge_label_field": "raw_postjudge_label",
            "postjudge_confidence_field": "raw_postjudge_confidence",
            "initial_method": "raw_postjudge_no_rethink",
            "selective_method": "raw_selective_rethink",
            "rethink_method": "raw_rethink_all",
            "oracle_method": "raw_oracle_initial_vs_rethink",
            "initial_rethink_rate": 0.0,
            "rethink_all_rate": 1.0,
        },
        {
            "family": "compiled",
            "label": "T5 compiled prompt",
            "initial_prefix": "compiled_initial",
            "rethink_prefix": "compiled_rethink",
            "selective_prefix": "compiled_selective",
            "oracle_prefix": "compiled_oracle",
            "postjudge_label_field": "compiled_postjudge_label",
            "postjudge_confidence_field": "compiled_postjudge_confidence",
            "initial_method": "compiled_no_rethink",
            "selective_method": "compiled_selective_rethink",
            "rethink_method": "compiled_rethink_all",
            "oracle_method": "compiled_oracle_initial_vs_rethink",
            "initial_rethink_rate": 0.0,
            "rethink_all_rate": 1.0,
        },
        {
            "family": "structure_gated",
            "label": "Structure-gated prompt",
            "initial_prefix": "gated_initial",
            "rethink_prefix": "gated_rethink",
            "selective_prefix": "gated_selective",
            "oracle_prefix": "gated_oracle",
            "postjudge_label_field": "gated_postjudge_label",
            "postjudge_confidence_field": "gated_postjudge_confidence",
            "initial_method": "structure_gated_no_rethink",
            "selective_method": "structure_gated_selective_rethink",
            "rethink_method": "structure_gated_rethink_all",
            "oracle_method": "structure_gated_oracle_initial_vs_rethink",
            "initial_rethink_rate": 0.0,
            "rethink_all_rate": 1.0,
        },
    ]

    def _read_config(self) -> dict:
        config = super()._read_config()
        explicit_run_name = env_text("PORT_RUN_NAME")
        config["run_name"] = explicit_run_name or f"paper_port_wmdp_postjudge_rethink_oracle_diagnostic_{config['model_name']}"
        config["max_samples"] = int(env_text("PORT_MAX_SAMPLES", "32"))
        config["scale_run_family"] = "postjudge_rethink_oracle_diagnostic"
        return config

    def resolve_or_bootstrap_artifacts(self) -> dict:
        artifact_info = super().resolve_or_bootstrap_artifacts()
        audit = dict(artifact_info["audit"])
        audit.update(
            {
                "artifact_note": (
                    "recreated mode uses public-data recreated artifacts; notebook 27 isolates post-judge routing, "
                    "rethink generation, and row-level oracle upper bounds."
                ),
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

    def _classify_initial_answers(self, port_wmdp, answers: list[str], records: list[dict], models: dict, args) -> tuple[list[int], list[float], list[str]]:
        expanded = [
            self._expand_answer_text(port_wmdp, answer, item["choices"])
            for answer, item in zip(answers, records)
        ]
        labels, confidences = self._classify_expanded_answers(expanded, models, args)
        return [int(label) for label in labels], [float(conf) for conf in confidences], expanded

    def _selective_answer(self, label: int, confidence: float, initial_answer: str, rethink_answer: str, args) -> tuple[str, bool]:
        triggered = not (int(label) == 0 and float(confidence) >= args.classifier_conf_threshold)
        return (rethink_answer if triggered else initial_answer), bool(triggered)

    @staticmethod
    def _oracle_answer(initial_answer: str, rethink_answer: str, initial_correct: bool, rethink_correct: bool) -> tuple[str, str, bool]:
        if initial_correct:
            return initial_answer, "initial", bool(initial_correct)
        if rethink_correct:
            return rethink_answer, "rethink", True
        return initial_answer, "neither", False

    def _add_family_predictions(
        self,
        row: dict,
        spec: dict,
        port_wmdp,
        item: dict,
        initial_answer: str,
        rethink_answer: str,
        label: int,
        confidence: float,
        expanded_initial_answer: str,
        rethink_prompt: str,
        args,
    ) -> None:
        initial_prefix = spec["initial_prefix"]
        rethink_prefix = spec["rethink_prefix"]
        selective_prefix = spec["selective_prefix"]
        oracle_prefix = spec["oracle_prefix"]

        row.update(self._choice_fields(port_wmdp, initial_prefix, initial_answer, item))
        row.update(self._choice_fields(port_wmdp, rethink_prefix, rethink_answer, item))

        selective_answer, triggered = self._selective_answer(label, confidence, initial_answer, rethink_answer, args)
        row[f"{selective_prefix}_rethink_triggered"] = bool(triggered)
        row.update(self._choice_fields(port_wmdp, selective_prefix, selective_answer, item))

        oracle_answer, oracle_source, oracle_correct = self._oracle_answer(
            initial_answer,
            rethink_answer,
            bool(row[f"{initial_prefix}_is_correct"]),
            bool(row[f"{rethink_prefix}_is_correct"]),
        )
        row[f"{oracle_prefix}_source"] = oracle_source
        row[f"{oracle_prefix}_uses_rethink"] = oracle_source == "rethink"
        row.update(self._choice_fields(port_wmdp, oracle_prefix, oracle_answer, item))
        row[f"{oracle_prefix}_is_correct"] = bool(oracle_correct)

        row[spec["postjudge_label_field"]] = int(label)
        row[spec["postjudge_confidence_field"]] = float(confidence)
        row[f"{spec['family']}_expanded_initial_answer"] = expanded_initial_answer
        row[f"{spec['family']}_rethink_prompt"] = rethink_prompt
        row[f"{spec['family']}_selective_matches_oracle"] = (
            row[f"{selective_prefix}_predicted_index"] == row[f"{oracle_prefix}_predicted_index"]
        )
        row[f"{spec['family']}_selective_missed_oracle_rethink"] = (
            row[f"{oracle_prefix}_uses_rethink"] and not row[f"{selective_prefix}_rethink_triggered"]
        )
        row[f"{spec['family']}_selective_rethink_hurt"] = (
            row[f"{selective_prefix}_rethink_triggered"]
            and bool(row[f"{initial_prefix}_is_correct"])
            and not bool(row[f"{selective_prefix}_is_correct"])
        )

    @staticmethod
    def _safe_avg(rows: list[dict], field: str) -> float | None:
        values = [row.get(field) for row in rows if row.get(field) is not None]
        if not values:
            return None
        return sum(float(value) for value in values) / len(values)

    @staticmethod
    def _safe_bool_avg(rows: list[dict], field: str) -> float | None:
        values = [row.get(field) for row in rows if row.get(field) is not None]
        if not values:
            return None
        return sum(1.0 if value else 0.0 for value in values) / len(values)

    def _method_summary(
        self,
        rows: list[dict],
        job: dict,
        method: str,
        prediction_prefix: str,
        family: str,
        rethink_rate: float | None,
        postjudge_label_field: str | None = None,
        postjudge_confidence_field: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        total = len(rows)
        correct = sum(1 for row in rows if bool(row.get(f"{prediction_prefix}_is_correct")))
        valid = sum(1 for row in rows if row.get(f"{prediction_prefix}_predicted_index") is not None)
        summary = {
            "variant": job["variant"],
            "domain": job["domain"],
            "wmdp_set": job["wmdp_set"],
            "prompt_source": job["prompt_source"],
            "method": method,
            "family": family,
            "prediction_prefix": prediction_prefix,
            "rows": total,
            "correct_count": correct,
            "accuracy": correct / total if total else None,
            "valid_predictions_count": valid,
            "valid_predictions_rate": valid / total if total else None,
            "rethink_rate": rethink_rate,
        }
        if postjudge_label_field:
            positive = sum(1 for row in rows if int(row.get(postjudge_label_field, -1)) == 1)
            summary["postjudge_positive_rate"] = positive / total if total else None
        if postjudge_confidence_field:
            summary["postjudge_avg_confidence"] = self._safe_avg(rows, postjudge_confidence_field)
        if extra:
            summary.update(extra)
        return summary

    def _family_summaries(self, rows: list[dict], job: dict, spec: dict) -> list[dict]:
        family = spec["family"]
        initial_prefix = spec["initial_prefix"]
        rethink_prefix = spec["rethink_prefix"]
        selective_prefix = spec["selective_prefix"]
        oracle_prefix = spec["oracle_prefix"]
        post_label = spec["postjudge_label_field"]
        post_conf = spec["postjudge_confidence_field"]

        selective_rethink_rate = self._safe_bool_avg(rows, f"{selective_prefix}_rethink_triggered")
        oracle_rethink_rate = self._safe_bool_avg(rows, f"{oracle_prefix}_uses_rethink")
        oracle_accuracy = self._method_summary(
            rows,
            job,
            spec["oracle_method"],
            oracle_prefix,
            family,
            oracle_rethink_rate,
            post_label,
            post_conf,
            extra={
                "selective_matches_oracle_rate": self._safe_bool_avg(rows, f"{family}_selective_matches_oracle"),
                "selective_missed_oracle_rethink_rate": self._safe_bool_avg(rows, f"{family}_selective_missed_oracle_rethink"),
                "selective_rethink_hurt_rate": self._safe_bool_avg(rows, f"{family}_selective_rethink_hurt"),
            },
        )
        return [
            self._method_summary(
                rows,
                job,
                spec["initial_method"],
                initial_prefix,
                family,
                spec["initial_rethink_rate"],
                post_label,
                post_conf,
            ),
            self._method_summary(
                rows,
                job,
                spec["selective_method"],
                selective_prefix,
                family,
                selective_rethink_rate,
                post_label,
                post_conf,
                extra={
                    "selective_matches_oracle_rate": self._safe_bool_avg(rows, f"{family}_selective_matches_oracle"),
                    "selective_missed_oracle_rethink_rate": self._safe_bool_avg(rows, f"{family}_selective_missed_oracle_rethink"),
                    "selective_rethink_hurt_rate": self._safe_bool_avg(rows, f"{family}_selective_rethink_hurt"),
                },
            ),
            self._method_summary(
                rows,
                job,
                spec["rethink_method"],
                rethink_prefix,
                family,
                spec["rethink_all_rate"],
                post_label,
                post_conf,
            ),
            oracle_accuracy,
        ]

    def _raw_direct_summary(self, rows: list[dict], job: dict) -> dict:
        return self._method_summary(
            rows,
            job,
            "raw_direct_generation",
            "raw_initial",
            "raw",
            0.0,
            None,
            None,
            extra={
                "postjudge_positive_rate": None,
                "postjudge_avg_confidence": None,
            },
        )

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
        compiled_initial_answers = self._generate_answers(compiled_prompts, models, args, port_wmdp)
        compiled_labels, compiled_confidences, compiled_expanded = self._classify_initial_answers(
            port_wmdp, compiled_initial_answers, records, models, args
        )
        self._set_generation_seed(self.config["seed"], job_index, 5)
        compiled_rethink_answers, compiled_rethink_prompts = self._rethink_answers(
            compiled_prompts, compiled_initial_answers, models, args, port_wmdp
        )

        self._set_generation_seed(self.config["seed"], job_index, 6)
        gated_initial_answers = self._generate_answers(gated_prompts, models, args, port_wmdp)
        gated_labels, gated_confidences, gated_expanded = self._classify_initial_answers(
            port_wmdp, gated_initial_answers, records, models, args
        )
        self._set_generation_seed(self.config["seed"], job_index, 7)
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
                self.family_specs[0],
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
                self.family_specs[1],
                port_wmdp,
                item,
                compiled_initial_answers[idx],
                compiled_rethink_answers[idx],
                compiled_labels[idx],
                compiled_confidences[idx],
                compiled_expanded[idx],
                compiled_rethink_prompts[idx],
                args,
            )
            self._add_family_predictions(
                row,
                self.family_specs[2],
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
            diagnostic_rows.append(row)

        job_summary = [self._raw_direct_summary(diagnostic_rows, job)]
        for spec in self.family_specs:
            job_summary.extend(self._family_summaries(diagnostic_rows, job, spec))
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
                    "compiled_initial_answer": diagnostic_rows[idx]["compiled_initial_answer"],
                    "compiled_rethink_answer": diagnostic_rows[idx]["compiled_rethink_answer"],
                    "gated_initial_answer": diagnostic_rows[idx]["gated_initial_answer"],
                    "gated_rethink_answer": diagnostic_rows[idx]["gated_rethink_answer"],
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

            rows = pd.read_csv(output_dir / "postjudge_rethink_oracle_predictions.csv").to_dict(orient="records")
            summary = pd.read_csv(output_dir / "job_summary.csv").to_dict(orient="records")
        except Exception as exc:
            print(f"Existing oracle diagnostic output is not readable, rerunning {job['variant']}/{job['domain']}: {exc}")
            return None
        if len(rows) != expected_rows:
            print(f"Existing oracle diagnostic output has {len(rows)} rows, expected {expected_rows}; rerunning {job['variant']}/{job['domain']}")
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
        pd.DataFrame(rows).to_csv(output_dir / "postjudge_rethink_oracle_predictions.csv", index=False)
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
            "structure_gate_pass_rate",
            "structure_gate_fallback_to_raw_rate",
            "t5_compiled_prompt_choice_coverage_avg",
            "structure_gate_prompt_choice_coverage_avg",
        ]
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
                },
            )
            row_count = int(row["rows"])
            bucket["rows"] += row_count
            bucket["correct_count"] += int(row["correct_count"])
            bucket["valid_predictions_count"] += int(row["valid_predictions_count"])
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
        run_config.update(
            {
                "purpose": "paper_port_wmdp_postjudge_rethink_oracle_diagnostic",
                "scale_run_family": self.config["scale_run_family"],
                "diagnostic_methods": [
                    "raw_direct_generation",
                    "raw_postjudge_no_rethink",
                    "raw_selective_rethink",
                    "raw_rethink_all",
                    "raw_oracle_initial_vs_rethink",
                    "compiled_no_rethink",
                    "compiled_selective_rethink",
                    "compiled_rethink_all",
                    "compiled_oracle_initial_vs_rethink",
                    "structure_gated_no_rethink",
                    "structure_gated_selective_rethink",
                    "structure_gated_rethink_all",
                    "structure_gated_oracle_initial_vs_rethink",
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
            / "postjudge_rethink_oracle"
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

        predictions_path = self.run_dir / "all_postjudge_rethink_oracle_predictions.csv"
        by_job_path = self.run_dir / "postjudge_rethink_oracle_summary_by_job.csv"
        overall_path = self.run_dir / "postjudge_rethink_oracle_summary_overall.csv"
        failed_jobs_path = self.run_dir / "failed_jobs.json"

        pd.DataFrame(all_rows).to_csv(predictions_path, index=False)
        pd.DataFrame(job_summary_rows).to_csv(by_job_path, index=False)
        overall_summary = self._weighted_summary(job_summary_rows)
        pd.DataFrame(overall_summary).to_csv(overall_path, index=False)
        failed_jobs_path.write_text(json.dumps(failed_jobs, indent=2, default=str), encoding="utf-8")

        by_method = {row["method"]: row for row in overall_summary}
        raw_direct = by_method.get("raw_direct_generation")
        raw_selective = by_method.get("raw_selective_rethink")
        raw_oracle = by_method.get("raw_oracle_initial_vs_rethink")
        gated_selective = by_method.get("structure_gated_selective_rethink")
        gated_oracle = by_method.get("structure_gated_oracle_initial_vs_rethink")
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
            "key_deltas": {
                "raw_selective_minus_raw_direct": (
                    raw_selective["accuracy"] - raw_direct["accuracy"] if raw_selective and raw_direct else None
                ),
                "raw_oracle_minus_raw_direct": (
                    raw_oracle["accuracy"] - raw_direct["accuracy"] if raw_oracle and raw_direct else None
                ),
                "structure_gated_selective_minus_raw_direct": (
                    gated_selective["accuracy"] - raw_direct["accuracy"] if gated_selective and raw_direct else None
                ),
                "structure_gated_oracle_minus_raw_direct": (
                    gated_oracle["accuracy"] - raw_direct["accuracy"] if gated_oracle and raw_direct else None
                ),
            },
            "all_postjudge_rethink_oracle_predictions_csv": str(predictions_path),
            "postjudge_rethink_oracle_summary_by_job_csv": str(by_job_path),
            "postjudge_rethink_oracle_summary_overall_csv": str(overall_path),
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
            print(
                f"\n=== Oracle diagnostic job {job_index}/{len(matrix_jobs)}: "
                f"{job['variant']}/{job['domain']}, rows={len(job['records'])}, prompt_source={job['prompt_source']} ==="
            )
            try:
                if job_index in existing_jobs:
                    diagnostic_rows, job_summary = existing_jobs[job_index]
                    skipped_jobs.append(
                        {
                            "job_index": job_index,
                            "variant": job["variant"],
                            "domain": job["domain"],
                            "rows": len(job["records"]),
                            "output_dir": str(output_dir),
                        }
                    )
                    print(f"Skipping completed oracle diagnostic job: {output_dir}")
                else:
                    if models is None:
                        raise RuntimeError("Internal error: models were not loaded for an incomplete oracle diagnostic job.")
                    start_job = time.perf_counter()
                    diagnostic_rows, job_summary, prompt_examples = self._run_job_diagnostics(
                        job_index, job, models, base_args, port_wmdp
                    )
                    job_seconds = time.perf_counter() - start_job
                    for row in job_summary:
                        row["run_seconds"] = job_seconds
                        row["output_dir"] = str(output_dir)
                        row["resume_status"] = "computed"
                    self._write_job_artifacts(output_dir, diagnostic_rows, job_summary, prompt_examples)
                    completed_jobs.append(
                        {
                            "job_index": job_index,
                            "variant": job["variant"],
                            "domain": job["domain"],
                            "rows": len(job["records"]),
                            "run_seconds": job_seconds,
                            "output_dir": str(output_dir),
                        }
                    )
                    print(json.dumps(job_summary, indent=2, default=str)[:5000])

                all_rows.extend(diagnostic_rows)
                job_summary_rows.extend(job_summary)
                summary_payload = self._write_summary_artifacts(
                    run_config,
                    classifier_info,
                    model_load_seconds,
                    all_rows,
                    job_summary_rows,
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
                completed_jobs,
                skipped_jobs,
                failed_jobs,
            )
        print(json.dumps(summary_payload, indent=2, default=str)[:7000])
        return {
            "matrix_jobs": matrix_jobs,
            "all_rows": all_rows,
            "job_summary_rows": job_summary_rows,
            "summary_payload": summary_payload,
            "completed_jobs": completed_jobs,
            "skipped_jobs": skipped_jobs,
            "failed_jobs": failed_jobs,
        }

    def verify(self, classifier_info: dict, matrix_result: dict) -> dict:
        expected_jobs = len(self.config["wmdp_variants"]) * len(self.config["wmdp_domains"])
        if matrix_result["failed_jobs"]:
            raise RuntimeError(f"Oracle diagnostic run has failed jobs: {matrix_result['failed_jobs']}")
        if len(matrix_result["completed_jobs"]) + len(matrix_result["skipped_jobs"]) != expected_jobs:
            raise RuntimeError(
                "Expected "
                f"{expected_jobs} jobs, got completed+skipped={len(matrix_result['completed_jobs']) + len(matrix_result['skipped_jobs'])}"
            )

        required_root_files = [
            self.run_dir / "artifact_audit.json",
            self.run_dir / "run_config.json",
            self.run_dir / "summary.json",
            self.run_dir / "all_postjudge_rethink_oracle_predictions.csv",
            self.run_dir / "postjudge_rethink_oracle_summary_by_job.csv",
            self.run_dir / "postjudge_rethink_oracle_summary_overall.csv",
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
            raise RuntimeError(f"Missing expected oracle diagnostic artifacts: {missing_files}")

        overall = matrix_result["summary_payload"]["overall_summary"]
        by_method = {row["method"]: row for row in overall}
        raw_direct = by_method.get("raw_direct_generation")
        raw_selective = by_method.get("raw_selective_rethink")
        raw_oracle = by_method.get("raw_oracle_initial_vs_rethink")
        gated_selective = by_method.get("structure_gated_selective_rethink")
        gated_oracle = by_method.get("structure_gated_oracle_initial_vs_rethink")
        result = {
            "status": "completed",
            "postjudge_rethink_oracle_diagnostic": True,
            "official_paper_checkpoint": False,
            "jobs": expected_jobs,
            "rows": len(matrix_result["all_rows"]),
            "max_samples": self.config["max_samples"],
            "row_count_mode": "full_dataset" if self.config["max_samples"] <= 0 else f"first_{self.config['max_samples']}_per_job",
            "classifier_test_accuracy": classifier_info["classifier_metadata"]["metrics"]["test"]["accuracy"],
            "classifier_test_macro_f1": classifier_info["classifier_metadata"]["metrics"]["test"].get("macro_f1"),
            "raw_direct_accuracy": raw_direct["accuracy"] if raw_direct else None,
            "raw_selective_accuracy": raw_selective["accuracy"] if raw_selective else None,
            "raw_oracle_accuracy": raw_oracle["accuracy"] if raw_oracle else None,
            "structure_gated_selective_accuracy": gated_selective["accuracy"] if gated_selective else None,
            "structure_gated_oracle_accuracy": gated_oracle["accuracy"] if gated_oracle else None,
            "completed_jobs": len(matrix_result["completed_jobs"]),
            "skipped_jobs": len(matrix_result["skipped_jobs"]),
            "answer_expansion_before_postjudge": True,
            "run_dir": str(self.run_dir),
        }
        print("PAPER PORT WMDP POSTJUDGE/RETHINK ORACLE DIAGNOSTIC COMPLETED")
        print("Jobs:", result["jobs"])
        print("Rows:", result["rows"])
        print("Row count mode:", result["row_count_mode"])
        print("Classifier test accuracy:", result["classifier_test_accuracy"])
        print("Raw direct accuracy:", result["raw_direct_accuracy"])
        print("Raw selective accuracy:", result["raw_selective_accuracy"])
        print("Raw oracle accuracy:", result["raw_oracle_accuracy"])
        print("Structure-gated selective accuracy:", result["structure_gated_selective_accuracy"])
        print("Structure-gated oracle accuracy:", result["structure_gated_oracle_accuracy"])
        print("Artifacts:", self.run_dir)
        print("Important: oracle methods are upper bounds, not deployable paper metrics.")
        return result


def run(project_root: str | Path, is_kaggle: bool, commit_sha: str) -> dict:
    return PostjudgeRethinkOracleDiagnosticRunner(project_root, is_kaggle, commit_sha).run()
