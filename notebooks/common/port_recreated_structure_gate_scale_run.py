from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from port_prefix_compiler_source_diagnostic import DEFAULT_RECREATED_ARTIFACT_MANIFEST_URL
from port_recreated_scale_run import RecreatedScaleRunner
from port_recreated_smoke import env_bool, env_text


class RecreatedStructureGateScaleRunner(RecreatedScaleRunner):
    @staticmethod
    def _comparison_references() -> dict:
        return {
            "notebook_20_recreated_scale": {
                "overall_accuracy": 0.2222222222222222,
                "overall_valid_predictions_rate": 0.9930555555555556,
                "overall_rethink_rate": 0.6770833333333334,
                "rows": 288,
            },
            "notebook_21_recreated_ablation": {
                "raw_direct_accuracy": 0.2916666666666667,
                "compiled_initial_accuracy": 0.2361111111111111,
                "rethink_all_accuracy": 0.21875,
                "best_threshold_final_accuracy": 0.21875,
                "rows": 288,
            },
            "notebook_25_structure_gate_counterfactual": {
                "raw_direct_accuracy": 0.2916666666666667,
                "structure_gate_accuracy": 0.2986111111111111,
                "structure_gate_reused_raw_prediction_rate": 0.8020833333333334,
                "rows": 288,
            },
        }

    def _read_config(self) -> dict:
        config = super()._read_config()
        config["run_name"] = env_text(
            "PORT_RUN_NAME",
            f"paper_port_wmdp_recreated_structure_gate_scale_run_{config['model_name']}",
        )
        config["scale_run_family"] = "recreated_port_structure_gate_scale"
        config["auto_download_recreated_artifact"] = env_bool("PORT_AUTO_DOWNLOAD_RECREATED_ARTIFACT", True)
        config["recreated_artifact_manifest_url"] = env_text(
            "PORT_RECREATED_ARTIFACT_MANIFEST_URL",
            DEFAULT_RECREATED_ARTIFACT_MANIFEST_URL,
        )
        config["bootstrap_recreated_if_missing"] = env_bool("PORT_BOOTSTRAP_RECREATED_IF_MISSING", False)
        config["quality_gate_min_choice_coverage"] = float(env_text("PORT_QUALITY_GATE_MIN_CHOICE_COVERAGE", "1.0"))
        config["quality_gate_min_len_ratio"] = float(env_text("PORT_QUALITY_GATE_MIN_LEN_RATIO", "0.50"))
        config["quality_gate_max_len_ratio"] = float(env_text("PORT_QUALITY_GATE_MAX_LEN_RATIO", "2.00"))
        config["quality_gate_require_prompt_instruction"] = env_bool("PORT_QUALITY_GATE_REQUIRE_PROMPT_INSTRUCTION", True)
        config["quality_gate_require_answer_instruction"] = env_bool("PORT_QUALITY_GATE_REQUIRE_ANSWER_INSTRUCTION", False)
        return config

    def _discover_candidate_zip(self) -> Path | None:
        candidates: list[Path] = []
        configured = self.config["recreated_artifact_zip_path"]
        if configured:
            candidates.append(Path(configured))
        candidates.append(Path("/kaggle/working/paper_port_recreated_artifacts_bootstrap.zip"))
        local_zip = self.project_root.parent / f"{self.project_root.name}_artifacts" / "paper_port_recreated_artifacts_bootstrap.zip"
        candidates.append(local_zip)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _download_default_recreated_artifact_zip(self) -> Path:
        manifest_url = self.config["recreated_artifact_manifest_url"]
        if not manifest_url:
            raise RuntimeError("PORT_RECREATED_ARTIFACT_MANIFEST_URL is empty.")

        destination = (
            Path("/kaggle/working/paper_port_recreated_artifacts_bootstrap.zip")
            if self.is_kaggle
            else self.run_dir / "downloads" / "paper_port_recreated_artifacts_bootstrap.zip"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)

        manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
        print(f"Downloading recreated artifact manifest: {manifest_url}")
        urllib.request.urlretrieve(manifest_url, manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        expected_size = int(manifest["artifact_size"])
        expected_sha = str(manifest["artifact_sha256"])
        if destination.exists():
            actual_size = destination.stat().st_size
            actual_sha = self._sha256_file(destination) if actual_size == expected_size else None
            if actual_size == expected_size and actual_sha == expected_sha:
                print(f"Using existing recreated artifact zip: {destination}")
                return destination
            print(f"Existing artifact zip failed validation, rebuilding: {destination}")
            destination.unlink()

        raw_base_url = manifest["raw_base_url"].rstrip("/")
        parts_dir = destination.with_suffix(destination.suffix + ".parts")
        parts_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths = []
        for index, chunk in enumerate(manifest["chunks"], start=1):
            chunk_name = chunk["name"]
            chunk_url = f"{raw_base_url}/{chunk_name}"
            chunk_path = parts_dir / chunk_name
            expected_chunk_size = int(chunk["size"])
            expected_chunk_sha = str(chunk["sha256"])

            needs_download = True
            if chunk_path.exists() and chunk_path.stat().st_size == expected_chunk_size:
                needs_download = self._sha256_file(chunk_path) != expected_chunk_sha
            if needs_download:
                print(f"Downloading artifact chunk {index}/{len(manifest['chunks'])}: {chunk_url}")
                urllib.request.urlretrieve(chunk_url, chunk_path)

            actual_chunk_size = chunk_path.stat().st_size
            actual_chunk_sha = self._sha256_file(chunk_path)
            if actual_chunk_size != expected_chunk_size or actual_chunk_sha != expected_chunk_sha:
                raise RuntimeError(
                    f"Artifact chunk validation failed for {chunk_name}: "
                    f"size={actual_chunk_size}/{expected_chunk_size} sha={actual_chunk_sha}/{expected_chunk_sha}"
                )
            chunk_paths.append(chunk_path)

        print(f"Assembling recreated artifact zip: {destination}")
        with destination.open("wb") as output:
            for chunk_path in chunk_paths:
                with chunk_path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        output.write(block)

        actual_size = destination.stat().st_size
        actual_sha = self._sha256_file(destination)
        if actual_size != expected_size or actual_sha != expected_sha:
            raise RuntimeError(
                f"Artifact zip validation failed: size={actual_size}/{expected_size} sha={actual_sha}/{expected_sha}"
            )
        print(
            json.dumps(
                {
                    "recreated_artifact_zip_path": str(destination),
                    "artifact_size": actual_size,
                    "artifact_sha256": actual_sha,
                    "manifest_url": manifest_url,
                },
                indent=2,
            )
        )
        return destination

    def resolve_or_bootstrap_artifacts(self) -> dict:
        auto_downloaded_zip = None
        if (
            not self.config["recreated_artifact_dir_env"]
            and not self.config["recreated_artifact_zip_url"]
            and self.config["auto_download_recreated_artifact"]
        ):
            auto_downloaded_zip = self._download_default_recreated_artifact_zip()
            self.config["recreated_artifact_zip_path"] = str(auto_downloaded_zip)

        artifact_info = super().resolve_or_bootstrap_artifacts()
        audit = dict(artifact_info["audit"])
        if auto_downloaded_zip is not None:
            audit["artifact_source"] = f"auto_download_manifest:{self.config['recreated_artifact_manifest_url']}"
            audit["auto_downloaded_zip"] = str(auto_downloaded_zip)
        audit.update(
            {
                "artifact_note": (
                    "recreated mode uses public-data recreated artifacts; notebook 26 integrates structure_gate "
                    "into the real recreated PoRT classifier/rethink scale path."
                ),
                "auto_download_recreated_artifact": self.config["auto_download_recreated_artifact"],
                "recreated_artifact_manifest_url": self.config["recreated_artifact_manifest_url"],
                "bootstrap_recreated_if_missing": self.config["bootstrap_recreated_if_missing"],
                "quality_gate": self._quality_gate_config_payload(),
                "limitations": [
                    "This is a recreated PoRT run built from public data, not an official paper checkpoint reproduction.",
                    "The post-judge classifier uses weak proxy labels from WMDP answer correctness.",
                    "The structure gate falls back to raw prompts when the recreated T5 prefix loses MCQ structure.",
                ],
            }
        )
        (self.run_dir / "artifact_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
        print(json.dumps(audit, indent=2, default=str))
        artifact_info["audit"] = audit
        return artifact_info

    def _quality_gate_config_payload(self) -> dict:
        return {
            "policy": "structure_gate",
            "min_choice_coverage": self.config["quality_gate_min_choice_coverage"],
            "min_len_ratio": self.config["quality_gate_min_len_ratio"],
            "max_len_ratio": self.config["quality_gate_max_len_ratio"],
            "require_prompt_instruction": self.config["quality_gate_require_prompt_instruction"],
            "require_answer_instruction": self.config["quality_gate_require_answer_instruction"],
            "reuse_raw_prediction": False,
        }

    @staticmethod
    def _prefix_quality(prompt: str, compiled_prompt: str, used_fallback: bool) -> dict:
        stripped_prompt = str(prompt).strip()
        stripped_compiled = str(compiled_prompt).strip()
        lower = stripped_compiled.lower()
        label_hits = 0
        for label in ["A", "B", "C", "D"]:
            if re.search(rf"(^|\n)\s*{label}[\.\)]\s+", stripped_compiled):
                label_hits += 1
        return {
            "compiled_prompt_used_fallback": bool(used_fallback),
            "compiled_prompt_empty": not bool(stripped_compiled),
            "compiled_prompt_same_as_original": stripped_compiled == stripped_prompt,
            "compiled_prompt_has_answer_instruction": "answer" in lower and "letter" in lower,
            "compiled_prompt_choice_label_count": label_hits,
            "compiled_prompt_choice_coverage": label_hits / 4.0,
            "prompt_char_len": len(stripped_prompt),
            "compiled_prompt_char_len": len(stripped_compiled),
            "compiled_prompt_char_len_ratio": (len(stripped_compiled) / len(stripped_prompt)) if stripped_prompt else None,
        }

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

    def _compile_and_gate_prompts(self, batch_prompts: list[str], models: dict, args, port_wmdp) -> tuple[list[str], list[dict]]:
        compiled_prompts = port_wmdp.run_prefix_compilation_step_batch(batch_prompts, models, args.example_library, args)
        retry_indices = [idx for idx, prompt in enumerate(compiled_prompts) if not str(prompt).strip()]
        if retry_indices:
            retry_questions = [batch_prompts[idx] for idx in retry_indices]
            retry_prompts = port_wmdp.run_prefix_compilation_step_batch(retry_questions, models, args.example_library, args)
            for idx, new_prompt in zip(retry_indices, retry_prompts):
                compiled_prompts[idx] = new_prompt

        gated_prompts = []
        gate_rows = []
        for raw_prompt, compiled_prompt in zip(batch_prompts, compiled_prompts):
            used_compile_fallback = False
            compiled_prompt = str(compiled_prompt)
            if not compiled_prompt.strip():
                compiled_prompt = raw_prompt
                used_compile_fallback = True

            decision = self._gate_decision(raw_prompt, compiled_prompt, used_compile_fallback)
            gate_pass = bool(decision["gate_pass"])
            final_prompt = compiled_prompt if gate_pass else raw_prompt
            final_quality = self._prefix_quality(raw_prompt, final_prompt, not gate_pass)
            compiled_quality = decision["compiled_quality"]
            gated_prompts.append(final_prompt)
            gate_rows.append(
                {
                    "compiled_prompt": compiled_prompt,
                    "structure_gate_prompt": final_prompt,
                    "structure_gate_pass": gate_pass,
                    "structure_gate_reasons": decision["gate_reasons"],
                    "structure_gate_fallback_to_raw": not gate_pass,
                    "t5_compiled_prompt_used_fallback": compiled_quality["compiled_prompt_used_fallback"],
                    "t5_compiled_prompt_same_as_original": compiled_quality["compiled_prompt_same_as_original"],
                    "t5_compiled_prompt_has_answer_instruction": compiled_quality["compiled_prompt_has_answer_instruction"],
                    "t5_compiled_prompt_choice_label_count": compiled_quality["compiled_prompt_choice_label_count"],
                    "t5_compiled_prompt_choice_coverage": compiled_quality["compiled_prompt_choice_coverage"],
                    "t5_compiled_prompt_char_len": compiled_quality["compiled_prompt_char_len"],
                    "t5_compiled_prompt_char_len_ratio": compiled_quality["compiled_prompt_char_len_ratio"],
                    "structure_gate_prompt_has_answer_instruction": final_quality["compiled_prompt_has_answer_instruction"],
                    "structure_gate_prompt_choice_label_count": final_quality["compiled_prompt_choice_label_count"],
                    "structure_gate_prompt_choice_coverage": final_quality["compiled_prompt_choice_coverage"],
                    "structure_gate_prompt_char_len": final_quality["compiled_prompt_char_len"],
                    "structure_gate_prompt_char_len_ratio": final_quality["compiled_prompt_char_len_ratio"],
                }
            )
        return gated_prompts, gate_rows

    def _run_end_to_end_for_records(self, records: list[dict], models: dict, args, port_wmdp) -> tuple[list[dict], int]:
        results = []
        rethink_total = 0
        prompts = [item["prompt"] for item in records]
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            batch_prompts = prompts[start : start + args.batch_size]
            processed_prompts, gate_rows = self._compile_and_gate_prompts(batch_prompts, models, args, port_wmdp)

            initial_answers = port_wmdp.get_llm_response_batch(processed_prompts, models, args)
            expanded_answers = [
                self._expand_answer_text(port_wmdp, answer, item["choices"])
                for answer, item in zip(initial_answers, batch_records)
            ]
            pred_labels, confidences = self._classify_expanded_answers(expanded_answers, models, args)

            final_answers = list(initial_answers)
            final_generation_prompts = list(processed_prompts)
            need_rethink_indices = [
                idx
                for idx, (label, conf) in enumerate(zip(pred_labels, confidences))
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
                        **gate_rows[rel_idx],
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

    def _build_run_config(self, runtime_script_path: Path, artifact_info: dict, classifier_info: dict) -> dict:
        run_config = super()._build_run_config(runtime_script_path, artifact_info, classifier_info)
        run_config.update(
            {
                "purpose": "paper_port_wmdp_recreated_structure_gate_scale_run",
                "scale_run_family": self.config["scale_run_family"],
                "quality_gate": self._quality_gate_config_payload(),
                "auto_download_recreated_artifact": self.config["auto_download_recreated_artifact"],
                "recreated_artifact_manifest_url": self.config["recreated_artifact_manifest_url"],
                "comparison_references": self._comparison_references(),
                "limitations": [
                    "This is a recreated PoRT run built from public data, not an official paper checkpoint reproduction.",
                    "The post-judge classifier uses weak proxy labels from WMDP answer correctness.",
                    "Unlike notebook 25, structure_gate fallback rows are regenerated through the real classifier/rethink path instead of reusing raw-direct predictions.",
                ],
            }
        )
        return run_config

    def _job_output_dir(self, base_args, job: dict) -> Path:
        return (
            Path(base_args.output_dir)
            / self.config["model_name"].replace("/", "_")
            / "structure_gate"
            / job["variant"]
            / job["wmdp_set"]
        )

    @staticmethod
    def _avg_bool(rows: list[dict], name: str) -> float | None:
        values = [row.get(name) for row in rows if row.get(name) is not None]
        return sum(1.0 if value else 0.0 for value in values) / len(values) if values else None

    @staticmethod
    def _avg_float(rows: list[dict], name: str) -> float | None:
        values = [row.get(name) for row in rows if row.get(name) is not None]
        return sum(float(value) for value in values) / len(values) if values else None

    @staticmethod
    def _job_metrics(job: dict, results: list[dict], rethink_count: int, model_load_seconds: float, run_seconds: float) -> dict:
        metrics = RecreatedScaleRunner._job_metrics(job, results, rethink_count, model_load_seconds, run_seconds)
        metrics.update(
            {
                "structure_gate_pass_rate": RecreatedStructureGateScaleRunner._avg_bool(results, "structure_gate_pass"),
                "structure_gate_fallback_to_raw_rate": RecreatedStructureGateScaleRunner._avg_bool(results, "structure_gate_fallback_to_raw"),
                "t5_compiled_prompt_choice_coverage_avg": RecreatedStructureGateScaleRunner._avg_float(results, "t5_compiled_prompt_choice_coverage"),
                "t5_compiled_prompt_has_answer_instruction_rate": RecreatedStructureGateScaleRunner._avg_bool(
                    results, "t5_compiled_prompt_has_answer_instruction"
                ),
                "t5_compiled_prompt_char_len_ratio_avg": RecreatedStructureGateScaleRunner._avg_float(
                    results, "t5_compiled_prompt_char_len_ratio"
                ),
                "structure_gate_prompt_choice_coverage_avg": RecreatedStructureGateScaleRunner._avg_float(
                    results, "structure_gate_prompt_choice_coverage"
                ),
                "structure_gate_prompt_has_answer_instruction_rate": RecreatedStructureGateScaleRunner._avg_bool(
                    results, "structure_gate_prompt_has_answer_instruction"
                ),
                "structure_gate_prompt_char_len_ratio_avg": RecreatedStructureGateScaleRunner._avg_float(
                    results, "structure_gate_prompt_char_len_ratio"
                ),
            }
        )
        return metrics

    @staticmethod
    def _overall_metrics(summary_rows: list[dict]) -> dict:
        overall = RecreatedScaleRunner._overall_metrics(summary_rows)
        total_rows = sum(int(row.get("rows", 0)) for row in summary_rows)

        def weighted(name: str) -> float | None:
            if total_rows == 0 or not all(name in row for row in summary_rows):
                return None
            return sum(float(row[name]) * int(row["rows"]) for row in summary_rows) / total_rows

        for name in [
            "structure_gate_pass_rate",
            "structure_gate_fallback_to_raw_rate",
            "t5_compiled_prompt_choice_coverage_avg",
            "t5_compiled_prompt_has_answer_instruction_rate",
            "t5_compiled_prompt_char_len_ratio_avg",
            "structure_gate_prompt_choice_coverage_avg",
            "structure_gate_prompt_has_answer_instruction_rate",
            "structure_gate_prompt_char_len_ratio_avg",
        ]:
            overall[name] = weighted(name)
        return overall

    def verify(self, classifier_info: dict, matrix_result: dict) -> dict:
        result = super().verify(classifier_info, matrix_result)
        overall = self._overall_metrics(matrix_result["summary_rows"])
        result.update(
            {
                "structure_gate_scale_run": True,
                "structure_gate_pass_rate": overall["structure_gate_pass_rate"],
                "structure_gate_fallback_to_raw_rate": overall["structure_gate_fallback_to_raw_rate"],
                "t5_compiled_prompt_choice_coverage_avg": overall["t5_compiled_prompt_choice_coverage_avg"],
                "structure_gate_prompt_choice_coverage_avg": overall["structure_gate_prompt_choice_coverage_avg"],
                "comparison_references": self._comparison_references(),
            }
        )
        print("Structure gate pass rate:", result["structure_gate_pass_rate"])
        print("Structure gate fallback-to-raw rate:", result["structure_gate_fallback_to_raw_rate"])
        print("T5 compiled choice coverage:", result["t5_compiled_prompt_choice_coverage_avg"])
        print("Gated prompt choice coverage:", result["structure_gate_prompt_choice_coverage_avg"])
        return result


def run(project_root: str | Path, is_kaggle: bool, commit_sha: str) -> dict:
    return RecreatedStructureGateScaleRunner(project_root, is_kaggle, commit_sha).run()
