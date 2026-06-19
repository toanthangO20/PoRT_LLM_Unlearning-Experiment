from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

COMMON_DIR = Path(__file__).resolve().parent
import sys

if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from port_prefix_compiler_source_diagnostic import PrefixCompilerSourceDiagnosticRunner
from port_recreated_smoke import env_bool, env_text


class PrefixQualityGateDiagnosticRunner(PrefixCompilerSourceDiagnosticRunner):
    required_job_files = [
        "prefix_quality_gate_predictions.csv",
        "job_summary.csv",
        "job_summary.json",
        "prompt_examples.json",
    ]

    def _read_config(self) -> dict:
        config = super()._read_config()
        config["run_name"] = env_text(
            "PORT_RUN_NAME",
            f"paper_port_wmdp_prefix_quality_gate_diagnostic_{config['model_name']}",
        )
        config["prefix_include_base_t5"] = env_bool("PORT_PREFIX_INCLUDE_BASE_T5", False)
        config["prefix_include_recreated_t5"] = env_bool("PORT_PREFIX_INCLUDE_RECREATED_T5", True)
        config["quality_gate_include_compiled_direct"] = env_bool("PORT_QUALITY_GATE_INCLUDE_COMPILED_DIRECT", True)
        config["quality_gate_include_structure_gate"] = env_bool("PORT_QUALITY_GATE_INCLUDE_STRUCTURE_GATE", True)
        config["quality_gate_include_repair_gate"] = env_bool("PORT_QUALITY_GATE_INCLUDE_REPAIR_GATE", True)
        config["quality_gate_min_choice_coverage"] = float(env_text("PORT_QUALITY_GATE_MIN_CHOICE_COVERAGE", "1.0"))
        config["quality_gate_min_len_ratio"] = float(env_text("PORT_QUALITY_GATE_MIN_LEN_RATIO", "0.50"))
        config["quality_gate_max_len_ratio"] = float(env_text("PORT_QUALITY_GATE_MAX_LEN_RATIO", "2.00"))
        config["quality_gate_require_prompt_instruction"] = env_bool("PORT_QUALITY_GATE_REQUIRE_PROMPT_INSTRUCTION", True)
        config["quality_gate_require_answer_instruction"] = env_bool("PORT_QUALITY_GATE_REQUIRE_ANSWER_INSTRUCTION", False)
        config["quality_gate_repair_max_prefix_chars"] = int(env_text("PORT_QUALITY_GATE_REPAIR_MAX_PREFIX_CHARS", "600"))
        config["scale_run_family"] = "prefix_quality_gate_diagnostic"
        return config

    def _enabled_policies(self) -> list[dict]:
        policies = []
        if self.config["quality_gate_include_compiled_direct"]:
            policies.append(
                {
                    "policy": "compiled_direct",
                    "label": "Compiled direct",
                    "description": "Use the T5 compiled prompt without any quality gate.",
                }
            )
        if self.config["quality_gate_include_structure_gate"]:
            policies.append(
                {
                    "policy": "structure_gate",
                    "label": "Structure gate",
                    "description": "Use the compiled prompt only when it preserves MCQ structure; otherwise fallback to raw prompt.",
                }
            )
        if self.config["quality_gate_include_repair_gate"]:
            policies.append(
                {
                    "policy": "repair_gate",
                    "label": "Repair gate",
                    "description": "Use the compiled prompt when it passes the gate; otherwise prepend a bounded compiled draft to the original prompt.",
                }
            )
        if not policies:
            raise RuntimeError("At least one PORT_QUALITY_GATE_INCLUDE_* policy must be enabled.")
        return policies

    @staticmethod
    def _policy_source(base_source: dict, policy: str) -> dict:
        return {
            "source_id": f"{base_source['source_id']}_{policy}",
            "source_type": f"{base_source['source_type']}_{policy}",
            "model_path": base_source.get("model_path"),
            "base_source_id": base_source["source_id"],
            "base_source_type": base_source["source_type"],
            "policy": policy,
        }

    def resolve_or_bootstrap_artifacts(self) -> dict:
        artifact_info = super().resolve_or_bootstrap_artifacts()
        audit = dict(artifact_info["audit"])
        audit.update(
            {
                "artifact_note": "Notebook 24 does not bootstrap or train artifacts; it tests quality-gated prompt policies on configured prefix compiler sources.",
                "artifact_source": "no_bootstrap_prefix_quality_gate_diagnostic",
                "quality_gate_policies": self._enabled_policies(),
                "quality_gate": self._quality_gate_config_payload(),
                "limitations": [
                    "This diagnostic uses generated answers, not notebook 11 top-logit paper baseline metrics.",
                    "Quality-gated and repaired prompts are recreated diagnostics, not official PoRT paper behavior.",
                    "This is not an official paper checkpoint reproduction unless official artifacts are explicitly supplied.",
                ],
            }
        )
        (self.run_dir / "artifact_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
        print(json.dumps(audit, indent=2, default=str))
        artifact_info["audit"] = audit
        return artifact_info

    def _quality_gate_config_payload(self) -> dict:
        return {
            "min_choice_coverage": self.config["quality_gate_min_choice_coverage"],
            "min_len_ratio": self.config["quality_gate_min_len_ratio"],
            "max_len_ratio": self.config["quality_gate_max_len_ratio"],
            "require_prompt_instruction": self.config["quality_gate_require_prompt_instruction"],
            "require_answer_instruction": self.config["quality_gate_require_answer_instruction"],
            "repair_max_prefix_chars": self.config["quality_gate_repair_max_prefix_chars"],
        }

    def _build_run_config(self, runtime_script_path: Path, artifact_info: dict, classifier_info: dict) -> dict:
        run_config = super()._build_run_config(runtime_script_path, artifact_info, classifier_info)
        policies = self._enabled_policies()
        run_config.update(
            {
                "purpose": "paper_port_wmdp_prefix_quality_gate_diagnostic",
                "diagnostic_methods": ["raw_direct_generation"]
                + [
                    f"{source['source_id']}_{policy['policy']}_generation"
                    for source in artifact_info["prefix_sources"]
                    for policy in policies
                ],
                "quality_gate_policies": policies,
                "quality_gate": self._quality_gate_config_payload(),
                "limitations": [
                    "This diagnostic tests whether prompt quality gates can stop recreated prefix compiler regressions.",
                    "The metric is generated answer accuracy, not notebook 11 top-logit paper baseline accuracy.",
                    "Quality-gated and repaired prompts are recreated diagnostics, not official PoRT paper behavior.",
                ],
            }
        )
        return run_config

    def train_weak_classifier(self, weak_dataset: dict[str, Path]) -> dict:
        artifact_dir = self.run_dir / "artifacts" / "not_used_classifier"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        classifier_path = artifact_dir / "classifier.joblib"
        classifier_path.write_text("not used by notebook 24 prefix quality gate diagnostic\n", encoding="utf-8")
        metadata = {
            "classifier_family": "not-used-prefix-quality-gate-diagnostic",
            "not_official_checkpoint": True,
            "metrics": {},
            "limitations": [
                "Notebook 24 does not run classifier gating.",
                "A placeholder classifier artifact is written only to keep common artifact checks simple.",
            ],
        }
        (artifact_dir / "classifier_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"classifier_head_ckpt": str(classifier_path), "note": "not used"}, indent=2, default=str))
        return {
            "classifier_base_model": "not-used-prefix-quality-gate-diagnostic",
            "classifier_head_ckpt": str(classifier_path),
            "classifier_artifact_dir": artifact_dir,
            "classifier_metadata": metadata,
        }

    def _load_existing_source_job(self, output_dir: Path, job: dict, source_id: str, expected_rows: int):
        if not self.config["resume_existing"]:
            return None
        if not all((output_dir / name).exists() for name in self.required_job_files):
            return None
        try:
            import pandas as pd

            rows = pd.read_csv(output_dir / "prefix_quality_gate_predictions.csv").to_dict(orient="records")
            summary = pd.read_csv(output_dir / "job_summary.csv").to_dict(orient="records")
        except Exception as exc:
            print(f"Existing quality gate output is not readable, rerunning {source_id} {job['variant']}/{job['domain']}: {exc}")
            return None
        if len(rows) != expected_rows:
            print(
                f"Existing quality gate output has {len(rows)} rows, expected {expected_rows}; "
                f"rerunning {source_id} {job['variant']}/{job['domain']}"
            )
            return None
        if not summary:
            return None
        for row in summary:
            if row.get("source_id") != source_id or row.get("variant") != job["variant"] or row.get("domain") != job["domain"]:
                return None
            row["resume_status"] = "skipped_existing"
        return rows, summary

    def _write_job_artifacts(self, output_dir: Path, rows: list[dict], summary: list[dict], prompt_examples: dict) -> None:
        import pandas as pd

        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_dir / "prefix_quality_gate_predictions.csv", index=False)
        pd.DataFrame(summary).to_csv(output_dir / "job_summary.csv", index=False)
        (output_dir / "job_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        (output_dir / "prompt_examples.json").write_text(json.dumps(prompt_examples, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def _gate_decision(self, prompt: str, compiled_prompt: str, used_fallback: bool) -> dict:
        compiled_quality = self._prefix_quality(prompt, compiled_prompt, used_fallback)
        prompt_quality = self._prefix_quality(prompt, prompt, False)
        reasons = []
        ratio = compiled_quality["compiled_prompt_char_len_ratio"]
        if compiled_quality["compiled_prompt_empty"]:
            reasons.append("empty_compiled_prompt")
        if compiled_quality["compiled_prompt_choice_coverage"] < self.config["quality_gate_min_choice_coverage"]:
            reasons.append("insufficient_choice_coverage")
        if ratio is None or ratio < self.config["quality_gate_min_len_ratio"] or ratio > self.config["quality_gate_max_len_ratio"]:
            reasons.append("compiled_length_ratio_out_of_bounds")
        if self.config["quality_gate_require_answer_instruction"] and not compiled_quality["compiled_prompt_has_answer_instruction"]:
            reasons.append("missing_answer_instruction")
        if (
            self.config["quality_gate_require_prompt_instruction"]
            and prompt_quality["compiled_prompt_has_answer_instruction"]
            and not compiled_quality["compiled_prompt_has_answer_instruction"]
        ):
            reasons.append("lost_original_answer_instruction")
        return {
            "gate_pass": not reasons,
            "gate_reasons": reasons,
            "compiled_quality": compiled_quality,
        }

    def _repair_prompt(self, prompt: str, compiled_prompt: str) -> str:
        compiled = str(compiled_prompt).strip()
        if not compiled:
            return str(prompt)
        max_chars = max(0, int(self.config["quality_gate_repair_max_prefix_chars"]))
        if max_chars and len(compiled) > max_chars:
            compiled = compiled[:max_chars].rstrip()
        return (
            "Prefix compiler draft:\n"
            f"{compiled}\n\n"
            "Original multiple-choice question:\n"
            f"{str(prompt).strip()}\n\n"
            "Answer with the single best letter A, B, C, or D."
        )

    def _final_prompt_for_policy(self, policy: str, prompt: str, compiled_prompt: str, used_fallback: bool) -> tuple[str, dict]:
        decision = self._gate_decision(prompt, compiled_prompt, used_fallback)
        if policy == "compiled_direct":
            return str(compiled_prompt), {
                **decision,
                "policy_action": "use_compiled_direct",
                "policy_fallback_to_raw": bool(used_fallback),
                "policy_repair_applied": False,
            }
        if policy == "structure_gate":
            if decision["gate_pass"]:
                return str(compiled_prompt), {
                    **decision,
                    "policy_action": "use_compiled",
                    "policy_fallback_to_raw": False,
                    "policy_repair_applied": False,
                }
            return str(prompt), {
                **decision,
                "policy_action": "fallback_raw",
                "policy_fallback_to_raw": True,
                "policy_repair_applied": False,
            }
        if policy == "repair_gate":
            if decision["gate_pass"]:
                return str(compiled_prompt), {
                    **decision,
                    "policy_action": "use_compiled",
                    "policy_fallback_to_raw": False,
                    "policy_repair_applied": False,
                }
            return self._repair_prompt(prompt, compiled_prompt), {
                **decision,
                "policy_action": "repair_with_raw_prompt",
                "policy_fallback_to_raw": False,
                "policy_repair_applied": True,
            }
        raise ValueError(f"Unknown quality gate policy: {policy}")

    def _run_policy_job(
        self,
        source_index: int,
        policy_index: int,
        policy: dict,
        source: dict,
        job_index: int,
        job: dict,
        raw_rows: list[dict],
        compiled_prompts: list[str],
        fallback_flags: list[bool],
        models: dict,
        base_args: SimpleNamespace,
        port_wmdp,
    ) -> tuple[list[dict], list[dict], dict]:
        args = SimpleNamespace(**vars(base_args))
        args.wmdp_set = job["wmdp_set"]
        records = job["records"]
        policy_name = policy["policy"]
        policy_source = self._policy_source(source, policy_name)

        final_prompts = []
        decisions = []
        for idx, item in enumerate(records):
            final_prompt, decision = self._final_prompt_for_policy(
                policy_name,
                item["prompt"],
                compiled_prompts[idx],
                fallback_flags[idx],
            )
            final_prompts.append(final_prompt)
            decisions.append(decision)

        self._set_generation_seed(self.config["seed"], job_index, 30 + source_index * 100 + policy_index * 10)
        answers = self._generate_answers(final_prompts, models, args, port_wmdp)
        self._clear_cuda_cache()

        rows = []
        for idx, item in enumerate(records):
            raw_index = self._as_optional_int(raw_rows[idx].get("raw_direct_predicted_index"))
            decision = decisions[idx]
            compiled_quality = decision["compiled_quality"]
            final_quality = self._prefix_quality(item["prompt"], final_prompts[idx], decision["policy_fallback_to_raw"])
            row = {
                **item,
                "source_id": policy_source["source_id"],
                "source_type": policy_source["source_type"],
                "model_path": source["model_path"],
                "base_source_id": source["source_id"],
                "base_source_type": source["source_type"],
                "policy": policy_name,
                "policy_label": policy["label"],
                "policy_action": decision["policy_action"],
                "policy_gate_pass": decision["gate_pass"],
                "policy_gate_reasons": ";".join(decision["gate_reasons"]),
                "policy_fallback_to_raw": decision["policy_fallback_to_raw"],
                "policy_repair_applied": decision["policy_repair_applied"],
                "t5_compiled_prompt": compiled_prompts[idx],
                "compiled_prompt": final_prompts[idx],
                "raw_direct_answer": raw_rows[idx].get("raw_direct_answer"),
                "raw_direct_predicted_index": raw_index,
                "t5_compiled_prompt_used_fallback": compiled_quality["compiled_prompt_used_fallback"],
                "t5_compiled_prompt_has_answer_instruction": compiled_quality["compiled_prompt_has_answer_instruction"],
                "t5_compiled_prompt_choice_coverage": compiled_quality["compiled_prompt_choice_coverage"],
                "t5_compiled_prompt_char_len_ratio": compiled_quality["compiled_prompt_char_len_ratio"],
                **final_quality,
            }
            row.update(self._choice_fields(port_wmdp, "prediction", answers[idx], item))
            row["answer"] = row["prediction_answer"]
            row["choice_letter"] = row["prediction_choice_letter"]
            row["predicted_index"] = row["prediction_predicted_index"]
            row["is_correct"] = row["prediction_is_correct"]
            row["same_as_raw_index"] = row["prediction_predicted_index"] == raw_index if raw_index is not None else False
            rows.append(row)

        summary = [self._source_summary(rows, policy_source, job)]
        prompt_examples = self._prompt_examples(job, rows)
        return rows, summary, prompt_examples

    def _source_summary(self, rows: list[dict], source: dict, job: dict) -> dict:
        summary = super()._source_summary(rows, source, job)

        def avg(name: str) -> float | None:
            values = [row.get(name) for row in rows if row.get(name) is not None]
            return sum(float(value) for value in values) / len(values) if values else None

        summary.update(
            {
                "base_source_id": source.get("base_source_id"),
                "policy": source.get("policy"),
                "policy_gate_pass_rate": avg("policy_gate_pass"),
                "policy_fallback_to_raw_rate": avg("policy_fallback_to_raw"),
                "policy_repair_applied_rate": avg("policy_repair_applied"),
                "t5_compiled_prompt_choice_coverage_avg": avg("t5_compiled_prompt_choice_coverage"),
                "t5_compiled_prompt_has_answer_instruction_rate": avg("t5_compiled_prompt_has_answer_instruction"),
                "t5_compiled_prompt_char_len_ratio_avg": avg("t5_compiled_prompt_char_len_ratio"),
            }
        )
        return summary

    @staticmethod
    def _weighted_summary(summary_rows: list[dict]) -> list[dict]:
        grouped: dict[str, dict] = {}
        weighted_fields = [
            "same_as_raw_index_rate",
            "compiled_prompt_used_fallback_rate",
            "compiled_prompt_same_as_original_rate",
            "compiled_prompt_has_answer_instruction_rate",
            "compiled_prompt_choice_coverage_avg",
            "prompt_char_len_avg",
            "compiled_prompt_char_len_avg",
            "compiled_prompt_char_len_ratio_avg",
            "policy_gate_pass_rate",
            "policy_fallback_to_raw_rate",
            "policy_repair_applied_rate",
            "t5_compiled_prompt_choice_coverage_avg",
            "t5_compiled_prompt_has_answer_instruction_rate",
            "t5_compiled_prompt_char_len_ratio_avg",
        ]
        for row in summary_rows:
            source_id = row["source_id"]
            bucket = grouped.setdefault(
                source_id,
                {
                    "source_id": source_id,
                    "source_type": row["source_type"],
                    "model_path": row.get("model_path"),
                    "method": row["method"],
                    "base_source_id": row.get("base_source_id"),
                    "policy": row.get("policy"),
                    "rows": 0,
                    "correct_count": 0,
                    "valid_predictions_count": 0,
                    **{f"{field}_numer": 0.0 for field in weighted_fields},
                    **{f"{field}_denom": 0 for field in weighted_fields},
                },
            )
            row_count = int(row["rows"])
            bucket["rows"] += row_count
            bucket["correct_count"] += int(row["correct_count"])
            bucket["valid_predictions_count"] += int(row["valid_predictions_count"])
            for field in weighted_fields:
                value = row.get(field)
                if value is not None:
                    bucket[f"{field}_numer"] += float(value) * row_count
                    bucket[f"{field}_denom"] += row_count

        result = []
        for bucket in grouped.values():
            rows = bucket["rows"]
            item = {
                "source_id": bucket["source_id"],
                "source_type": bucket["source_type"],
                "model_path": bucket.get("model_path"),
                "method": bucket["method"],
                "base_source_id": bucket.get("base_source_id"),
                "policy": bucket.get("policy"),
                "rows": rows,
                "correct_count": bucket["correct_count"],
                "accuracy": bucket["correct_count"] / rows if rows else None,
                "valid_predictions_count": bucket["valid_predictions_count"],
                "valid_predictions_rate": bucket["valid_predictions_count"] / rows if rows else None,
            }
            for field in weighted_fields:
                denom = bucket[f"{field}_denom"]
                item[field] = bucket[f"{field}_numer"] / denom if denom else None
            result.append(item)

        raw = next((item for item in result if item["source_id"] == "raw_direct"), None)
        if raw is not None and raw.get("accuracy") is not None:
            for item in result:
                item["accuracy_minus_raw"] = item["accuracy"] - raw["accuracy"] if item.get("accuracy") is not None else None
        return sorted(result, key=lambda item: item["source_id"])

    def _write_summary_artifacts(
        self,
        run_config: dict,
        classifier_info: dict,
        model_load_seconds: float,
        total_dataset_rows: int,
        prediction_rows: int,
        summary_rows: list[dict],
        completed_jobs: list[dict],
        skipped_jobs: list[dict],
        failed_jobs: list[dict],
    ) -> dict:
        import pandas as pd

        predictions_path = self.run_dir / "all_prefix_quality_gate_predictions.csv"
        by_job_path = self.run_dir / "prefix_quality_gate_summary_by_job.csv"
        overall_path = self.run_dir / "prefix_quality_gate_summary_overall.csv"
        failed_jobs_path = self.run_dir / "failed_jobs.json"

        pd.DataFrame(summary_rows).to_csv(by_job_path, index=False)
        overall_summary = self._weighted_summary(summary_rows)
        pd.DataFrame(overall_summary).to_csv(overall_path, index=False)
        failed_jobs_path.write_text(json.dumps(failed_jobs, indent=2, default=str), encoding="utf-8")

        summary_payload = {
            "run_config": run_config,
            "classifier_metrics": classifier_info["classifier_metadata"]["metrics"],
            "model_load_seconds": model_load_seconds,
            "completed_jobs": completed_jobs,
            "skipped_jobs": skipped_jobs,
            "failed_jobs": failed_jobs,
            "dataset_rows": total_dataset_rows,
            "prediction_rows": prediction_rows,
            "summary_by_job": summary_rows,
            "overall_summary": overall_summary,
            "all_prefix_quality_gate_predictions_csv": str(predictions_path),
            "prefix_quality_gate_summary_by_job_csv": str(by_job_path),
            "prefix_quality_gate_summary_overall_csv": str(overall_path),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
        return summary_payload

    def run_matrix(self, port_wmdp, runtime_script_path: Path, artifact_info: dict, classifier_info: dict) -> dict:
        base_args = self._build_base_args(artifact_info, classifier_info)
        run_config = self._build_run_config(runtime_script_path, artifact_info, classifier_info)
        (self.run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

        matrix_jobs = self.build_matrix_jobs()
        policies = self._enabled_policies()
        total_dataset_rows = sum(len(job["records"]) for job in matrix_jobs)
        prediction_rows = 0
        summary_rows: list[dict] = []
        completed_jobs: list[dict] = []
        skipped_jobs: list[dict] = []
        failed_jobs: list[dict] = []
        root_predictions_path = self.run_dir / "all_prefix_quality_gate_predictions.csv"
        if root_predictions_path.exists():
            root_predictions_path.unlink()

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

        raw_source = {"source_id": "raw_direct", "source_type": "raw", "model_path": None}
        source_load_seconds: dict[str, float] = {"raw_direct": 0.0}
        summary_payload = None

        def record_rows(source_id: str, job_index: int, job: dict, rows: list[dict], summary: list[dict], prompt_examples: dict | None, output_dir: Path, job_seconds: float | None):
            nonlocal prediction_rows, summary_payload
            if prompt_examples is not None:
                for row in summary:
                    row["run_seconds"] = job_seconds
                    row["t5_load_seconds"] = source_load_seconds.get(source_id, 0.0)
                    row["output_dir"] = str(output_dir)
                    row["resume_status"] = "computed"
                self._write_job_artifacts(output_dir, rows, summary, prompt_examples)
                completed_jobs.append(
                    {
                        "source_id": source_id,
                        "job_index": job_index,
                        "variant": job["variant"],
                        "domain": job["domain"],
                        "rows": len(job["records"]),
                        "run_seconds": job_seconds,
                        "output_dir": str(output_dir),
                    }
                )
                print(json.dumps(summary, indent=2, default=str))

            self._append_rows_to_csv(root_predictions_path, rows)
            prediction_rows += len(rows)
            summary_rows.extend(summary)
            summary_payload = self._write_summary_artifacts(
                run_config,
                classifier_info,
                model_load_seconds,
                total_dataset_rows,
                prediction_rows,
                summary_rows,
                completed_jobs,
                skipped_jobs,
                failed_jobs,
            )

        print("\n=== Prefix quality gate source 1: raw_direct ===")
        for job_index, job in enumerate(matrix_jobs, start=1):
            output_dir = self._job_output_dir(base_args, job, "raw_direct")
            print(f"\n=== Quality gate job {job_index}/{len(matrix_jobs)}: raw_direct {job['variant']}/{job['domain']}, rows={len(job['records'])} ===")
            try:
                existing = self._load_existing_source_job(output_dir, job, "raw_direct", len(job["records"]))
                if existing is not None:
                    rows, summary = existing
                    skipped_jobs.append(
                        {
                            "source_id": "raw_direct",
                            "job_index": job_index,
                            "variant": job["variant"],
                            "domain": job["domain"],
                            "rows": len(job["records"]),
                            "output_dir": str(output_dir),
                        }
                    )
                    print(f"Skipping completed raw quality gate job: {output_dir}")
                    record_rows("raw_direct", job_index, job, rows, summary, None, output_dir, None)
                else:
                    start_job = time.perf_counter()
                    rows, summary, prompt_examples = self._run_raw_job(job_index, job, models, base_args, port_wmdp)
                    record_rows("raw_direct", job_index, job, rows, summary, prompt_examples, output_dir, time.perf_counter() - start_job)
                del rows
                self._clear_cuda_cache()
            except Exception as exc:
                failure = {
                    "source_id": "raw_direct",
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
                    total_dataset_rows,
                    prediction_rows,
                    summary_rows,
                    completed_jobs,
                    skipped_jobs,
                    failed_jobs,
                )
                print(json.dumps(failure, indent=2, default=str))
                if self.config["fail_fast"]:
                    raise

        for source_index, source in enumerate(artifact_info["prefix_sources"], start=1):
            source_id = source["source_id"]
            print(f"\n=== Prefix quality gate compiler source {source_index}/{len(artifact_info['prefix_sources'])}: {source_id} ===")
            t5_tokenizer, t5_model, load_seconds = self._load_t5_source(source, base_args)
            models["t5_tokenizer"] = t5_tokenizer
            models["t5_model"] = t5_model
            source_load_seconds.update({self._policy_source(source, policy["policy"])["source_id"]: load_seconds for policy in policies})
            print(json.dumps({"source_id": source_id, "model_path": source["model_path"], "t5_load_seconds": load_seconds}, indent=2))

            try:
                for job_index, job in enumerate(matrix_jobs, start=1):
                    raw_output_dir = self._job_output_dir(base_args, job, "raw_direct")
                    raw_path = raw_output_dir / "prefix_quality_gate_predictions.csv"
                    if not raw_path.exists():
                        raise RuntimeError(f"Raw direct output must exist before quality gate policies run: {raw_path}")
                    raw_rows = self._read_raw_rows(raw_path)
                    prompts = [item["prompt"] for item in job["records"]]

                    self._set_generation_seed(self.config["seed"], job_index, 10 + source_index * 100)
                    compiled_prompts, fallback_flags = self._compile_prompts_with_loaded_source(prompts, models, base_args, port_wmdp)
                    self._clear_cuda_cache()

                    for policy_index, policy in enumerate(policies, start=1):
                        policy_source = self._policy_source(source, policy["policy"])
                        policy_source_id = policy_source["source_id"]
                        output_dir = self._job_output_dir(base_args, job, policy_source_id)
                        print(
                            f"\n=== Quality gate job {job_index}/{len(matrix_jobs)}: "
                            f"{policy_source_id} {job['variant']}/{job['domain']}, rows={len(job['records'])} ==="
                        )
                        try:
                            existing = self._load_existing_source_job(output_dir, job, policy_source_id, len(job["records"]))
                            if existing is not None:
                                rows, summary = existing
                                skipped_jobs.append(
                                    {
                                        "source_id": policy_source_id,
                                        "job_index": job_index,
                                        "variant": job["variant"],
                                        "domain": job["domain"],
                                        "rows": len(job["records"]),
                                        "output_dir": str(output_dir),
                                    }
                                )
                                print(f"Skipping completed quality gate job: {output_dir}")
                                record_rows(policy_source_id, job_index, job, rows, summary, None, output_dir, None)
                            else:
                                start_job = time.perf_counter()
                                rows, summary, prompt_examples = self._run_policy_job(
                                    source_index,
                                    policy_index,
                                    policy,
                                    source,
                                    job_index,
                                    job,
                                    raw_rows,
                                    compiled_prompts,
                                    fallback_flags,
                                    models,
                                    base_args,
                                    port_wmdp,
                                )
                                record_rows(policy_source_id, job_index, job, rows, summary, prompt_examples, output_dir, time.perf_counter() - start_job)
                            del rows
                            self._clear_cuda_cache()
                        except Exception as exc:
                            failure = {
                                "source_id": policy_source_id,
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
                                total_dataset_rows,
                                prediction_rows,
                                summary_rows,
                                completed_jobs,
                                skipped_jobs,
                                failed_jobs,
                            )
                            print(json.dumps(failure, indent=2, default=str))
                            if self.config["fail_fast"]:
                                raise
                    del raw_rows, compiled_prompts, fallback_flags
                    self._clear_cuda_cache()
            finally:
                models["t5_model"] = None
                models["t5_tokenizer"] = None
                try:
                    del t5_model, t5_tokenizer
                except UnboundLocalError:
                    pass
                self._clear_cuda_cache()

        if summary_payload is None:
            summary_payload = self._write_summary_artifacts(
                run_config,
                classifier_info,
                model_load_seconds,
                total_dataset_rows,
                prediction_rows,
                summary_rows,
                completed_jobs,
                skipped_jobs,
                failed_jobs,
            )
        print(json.dumps(summary_payload, indent=2, default=str)[:6000])
        return {
            "matrix_jobs": matrix_jobs,
            "prefix_sources": artifact_info["prefix_sources"],
            "quality_gate_policies": policies,
            "total_dataset_rows": total_dataset_rows,
            "prediction_rows": prediction_rows,
            "summary_rows": summary_rows,
            "summary_payload": summary_payload,
            "completed_jobs": completed_jobs,
            "skipped_jobs": skipped_jobs,
            "failed_jobs": failed_jobs,
        }

    def verify(self, classifier_info: dict, matrix_result: dict) -> dict:
        expected_jobs = len(self.config["wmdp_variants"]) * len(self.config["wmdp_domains"])
        expected_sources = 1 + len(matrix_result["prefix_sources"]) * len(matrix_result["quality_gate_policies"])
        if matrix_result["failed_jobs"]:
            raise RuntimeError(f"Prefix quality gate diagnostic has failed jobs: {matrix_result['failed_jobs']}")
        if len(matrix_result["completed_jobs"]) + len(matrix_result["skipped_jobs"]) != expected_jobs * expected_sources:
            raise RuntimeError(
                f"Expected {expected_jobs * expected_sources} source jobs, got "
                f"completed+skipped={len(matrix_result['completed_jobs']) + len(matrix_result['skipped_jobs'])}"
            )

        required_root_files = [
            self.run_dir / "artifact_audit.json",
            self.run_dir / "run_config.json",
            self.run_dir / "summary.json",
            self.run_dir / "all_prefix_quality_gate_predictions.csv",
            self.run_dir / "prefix_quality_gate_summary_by_job.csv",
            self.run_dir / "prefix_quality_gate_summary_overall.csv",
            self.run_dir / "failed_jobs.json",
            classifier_info["classifier_artifact_dir"] / "classifier.joblib",
            classifier_info["classifier_artifact_dir"] / "classifier_metadata.json",
        ]
        missing_files = [str(path) for path in required_root_files if not Path(path).exists()]
        base_args = self._build_base_args({"t5_model_path": ""}, classifier_info)
        source_ids = ["raw_direct"] + [
            self._policy_source(source, policy["policy"])["source_id"]
            for source in matrix_result["prefix_sources"]
            for policy in matrix_result["quality_gate_policies"]
        ]
        for source_id in source_ids:
            for job in matrix_result["matrix_jobs"]:
                output_dir = self._job_output_dir(base_args, job, source_id)
                for name in self.required_job_files:
                    if not (output_dir / name).exists():
                        missing_files.append(str(output_dir / name))
        if missing_files:
            raise RuntimeError(f"Missing expected prefix quality gate diagnostic artifacts: {missing_files}")

        overall = matrix_result["summary_payload"]["overall_summary"]
        result = {
            "status": "completed",
            "prefix_quality_gate_diagnostic": True,
            "official_paper_checkpoint": False,
            "jobs": expected_jobs,
            "sources": source_ids,
            "quality_gate_policies": matrix_result["quality_gate_policies"],
            "dataset_rows": matrix_result["total_dataset_rows"],
            "prediction_rows": matrix_result["prediction_rows"],
            "max_samples": self.config["max_samples"],
            "overall_summary": overall,
            "completed_jobs": len(matrix_result["completed_jobs"]),
            "skipped_jobs": len(matrix_result["skipped_jobs"]),
            "run_dir": str(self.run_dir),
        }
        print("PAPER PORT WMDP PREFIX QUALITY GATE DIAGNOSTIC COMPLETED")
        print("Jobs:", result["jobs"])
        print("Sources:", result["sources"])
        print("Dataset rows:", result["dataset_rows"])
        print("Prediction rows:", result["prediction_rows"])
        print("Overall summary:", json.dumps(overall, indent=2, default=str))
        print("Artifacts:", self.run_dir)
        print("Important: this is a quality-gate recreated diagnostic, not an official paper metric run.")
        return result


def run(project_root: str | Path, is_kaggle: bool, commit_sha: str) -> dict:
    return PrefixQualityGateDiagnosticRunner(project_root, is_kaggle, commit_sha).run()
