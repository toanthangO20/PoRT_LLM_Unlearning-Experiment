from __future__ import annotations

import json
import hashlib
import sys
import urllib.request
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from port_recreated_scale_run import RecreatedScaleRunner
from port_prefix_compiler_source_diagnostic import DEFAULT_RECREATED_ARTIFACT_MANIFEST_URL
from port_recreated_smoke import env_bool, env_text


class RecreatedRawInvertedRouteScaleRunner(RecreatedScaleRunner):
    route_policy = {
        "id": "raw_route_inverted_keep_label1_no_conf",
        "label": "Raw inverted correctness route without confidence",
        "prompt_family": "raw",
        "keep_label": 1,
        "uses_confidence": False,
        "keep_condition": "keep initial when postjudge label == 1; rethink otherwise",
    }

    def _read_config(self) -> dict:
        config = super()._read_config()
        config["run_name"] = env_text(
            "PORT_RUN_NAME",
            f"paper_port_wmdp_recreated_raw_inverted_route_scale_run_{config['model_name']}",
        )
        config["scale_run_family"] = "recreated_port_raw_inverted_route_scale"
        config["auto_download_recreated_artifact"] = env_bool("PORT_AUTO_DOWNLOAD_RECREATED_ARTIFACT", True)
        config["recreated_artifact_manifest_url"] = env_text(
            "PORT_RECREATED_ARTIFACT_MANIFEST_URL",
            DEFAULT_RECREATED_ARTIFACT_MANIFEST_URL,
        )
        config["bootstrap_recreated_if_missing"] = env_bool("PORT_BOOTSTRAP_RECREATED_IF_MISSING", False)
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
            and not self.config["recreated_artifact_zip_path"]
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
                    "recreated mode uses public-data recreated artifacts; this run applies the deployable "
                    "raw inverted label-1 no-confidence route selected by notebook 28."
                ),
                "auto_download_recreated_artifact": self.config["auto_download_recreated_artifact"],
                "recreated_artifact_manifest_url": self.config["recreated_artifact_manifest_url"],
                "bootstrap_recreated_if_missing": self.config["bootstrap_recreated_if_missing"],
                "route_policy": self.route_policy,
                "limitations": [
                    "This is a recreated PoRT run built from public data, not an official paper checkpoint reproduction.",
                    "The post-judge classifier uses weak proxy labels from WMDP answer correctness.",
                    "This route uses raw prompts and does not use post-judge confidence for routing.",
                ],
            }
        )
        (self.run_dir / "artifact_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
        print(json.dumps(audit, indent=2, default=str))
        artifact_info["audit"] = audit
        return artifact_info

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
            "notebook_28_routing_semantics": {
                "raw_current_route_accuracy": 0.1840277777777778,
                "raw_inverted_conf_route_accuracy": 0.3125,
                "raw_inverted_no_conf_route_accuracy": 0.4236111111111111,
                "raw_oracle_accuracy": 0.4270833333333333,
                "structure_gated_inverted_route_accuracy": 0.2986111111111111,
                "rows": 288,
            },
        }

    def _build_run_config(self, runtime_script_path: Path, artifact_info: dict, classifier_info: dict) -> dict:
        run_config = super()._build_run_config(runtime_script_path, artifact_info, classifier_info)
        run_config.update(
            {
                "purpose": "paper_port_wmdp_recreated_raw_inverted_route_scale_run",
                "scale_run_family": self.config["scale_run_family"],
                "route_policy": self.route_policy,
                "deployable_routing_policy": True,
                "prompt_processing_family": "raw_prompt_no_prefix_compile",
                "auto_download_recreated_artifact": self.config["auto_download_recreated_artifact"],
                "recreated_artifact_manifest_url": self.config["recreated_artifact_manifest_url"],
                "comparison_references": self._comparison_references(),
                "limitations": [
                    "This is a recreated PoRT run built from public data, not an official paper checkpoint reproduction.",
                    "The post-judge classifier uses weak proxy labels from WMDP answer correctness.",
                    "This deployable route uses raw prompts and keeps the initial answer when the classifier label is 1; confidence is recorded but not used for routing.",
                    "Notebook 28 selected this route because it nearly matched the raw initial-vs-rethink oracle on the 32 rows/job diagnostic.",
                ],
            }
        )
        return run_config

    def _job_output_dir(self, base_args, job: dict) -> Path:
        return (
            Path(base_args.output_dir)
            / self.config["model_name"].replace("/", "_")
            / "raw_inverted_route"
            / job["variant"]
            / job["wmdp_set"]
        )

    def _run_end_to_end_for_records(self, records: list[dict], models: dict, args, port_wmdp) -> tuple[list[dict], int]:
        results = []
        rethink_total = 0
        prompts = [item["prompt"] for item in records]
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            processed_prompts = prompts[start : start + args.batch_size]

            initial_answers = port_wmdp.get_llm_response_batch(processed_prompts, models, args)
            expanded_answers = [
                self._expand_answer_text(port_wmdp, answer, item["choices"])
                for answer, item in zip(initial_answers, batch_records)
            ]
            pred_labels, confidences = self._classify_expanded_answers(expanded_answers, models, args)

            final_answers = list(initial_answers)
            final_generation_prompts = list(processed_prompts)
            rethink_answers_by_idx: dict[int, str] = {}
            rethink_prompts_by_idx: dict[int, str] = {}
            need_rethink_indices = [
                idx
                for idx, label in enumerate(pred_labels)
                if int(label) != int(self.route_policy["keep_label"])
            ]
            rethink_total += len(need_rethink_indices)
            if need_rethink_indices:
                rethink_prompts = [processed_prompts[idx] for idx in need_rethink_indices]
                rethink_initials = [initial_answers[idx] for idx in need_rethink_indices]
                rethink_answers, rethink_prompts_used = port_wmdp.run_rethink_step_batch(rethink_prompts, rethink_initials, models, args)
                for rel_idx, answer, rethink_prompt in zip(need_rethink_indices, rethink_answers, rethink_prompts_used):
                    final_answers[rel_idx] = answer
                    final_generation_prompts[rel_idx] = rethink_prompt
                    rethink_answers_by_idx[rel_idx] = answer
                    rethink_prompts_by_idx[rel_idx] = rethink_prompt

            for rel_idx, item in enumerate(batch_records):
                initial_letter = port_wmdp.extract_choice_from_answer(initial_answers[rel_idx], item["choices"])
                initial_index = ord(initial_letter) - ord("A") if initial_letter in self.choice_labels else None
                predicted_letter = port_wmdp.extract_choice_from_answer(final_answers[rel_idx], item["choices"])
                predicted_index = ord(predicted_letter) - ord("A") if predicted_letter in self.choice_labels else None
                is_correct = predicted_index == item["correct_answer_index"] if predicted_index is not None else False
                rethink_triggered = rel_idx in need_rethink_indices
                results.append(
                    {
                        **item,
                        "prompt_processing_family": "raw_prompt_no_prefix_compile",
                        "route_policy": self.route_policy["id"],
                        "route_keep_label": int(self.route_policy["keep_label"]),
                        "route_uses_confidence": bool(self.route_policy["uses_confidence"]),
                        "route_keep_condition": self.route_policy["keep_condition"],
                        "route_decision": "rethink" if rethink_triggered else "initial",
                        "initial_answer": initial_answers[rel_idx],
                        "initial_choice_letter": initial_letter,
                        "initial_predicted_index": initial_index,
                        "initial_is_correct": (
                            initial_index == item["correct_answer_index"] if initial_index is not None else False
                        ),
                        "expanded_initial_answer": expanded_answers[rel_idx],
                        "postjudge_label": int(pred_labels[rel_idx]),
                        "postjudge_confidence": float(confidences[rel_idx]),
                        "rethink_answer": rethink_answers_by_idx.get(rel_idx),
                        "rethink_prompt": rethink_prompts_by_idx.get(rel_idx),
                        "generated_answer": final_answers[rel_idx],
                        "generated_choice_letter": predicted_letter,
                        "predicted_index": predicted_index,
                        "is_correct": bool(is_correct),
                        "rethink_triggered": rethink_triggered,
                        "generation_prompt": final_generation_prompts[rel_idx],
                    }
                )
        return results, rethink_total

    @staticmethod
    def _job_metrics(job: dict, results: list[dict], rethink_count: int, model_load_seconds: float, run_seconds: float) -> dict:
        metrics = RecreatedScaleRunner._job_metrics(job, results, rethink_count, model_load_seconds, run_seconds)
        metrics.update(
            {
                "route_policy": RecreatedRawInvertedRouteScaleRunner.route_policy["id"],
                "route_keep_label": RecreatedRawInvertedRouteScaleRunner.route_policy["keep_label"],
                "route_uses_confidence": RecreatedRawInvertedRouteScaleRunner.route_policy["uses_confidence"],
                "prompt_processing_family": "raw_prompt_no_prefix_compile",
            }
        )
        return metrics

    def verify(self, classifier_info: dict, matrix_result: dict) -> dict:
        result = super().verify(classifier_info, matrix_result)
        summary_rows = matrix_result["summary_rows"]
        overall = self._overall_metrics(summary_rows)
        result.update(
            {
                "raw_inverted_route_scale_run": True,
                "route_policy": self.route_policy["id"],
                "route_keep_label": self.route_policy["keep_label"],
                "route_uses_confidence": self.route_policy["uses_confidence"],
                "prompt_processing_family": "raw_prompt_no_prefix_compile",
                "comparison_references": self._comparison_references(),
                "overall_accuracy_minus_notebook_20": (
                    overall["accuracy"] - self._comparison_references()["notebook_20_recreated_scale"]["overall_accuracy"]
                    if overall["accuracy"] is not None
                    else None
                ),
                "overall_accuracy_minus_raw_direct_notebook_21": (
                    overall["accuracy"] - self._comparison_references()["notebook_21_recreated_ablation"]["raw_direct_accuracy"]
                    if overall["accuracy"] is not None
                    else None
                ),
                "overall_accuracy_minus_notebook_28_raw_inverted_no_conf": (
                    overall["accuracy"] - self._comparison_references()["notebook_28_routing_semantics"]["raw_inverted_no_conf_route_accuracy"]
                    if overall["accuracy"] is not None
                    else None
                ),
            }
        )
        print("PAPER PORT WMDP RECREATED RAW INVERTED ROUTE SCALE RUN COMPLETED")
        print("Route policy:", result["route_policy"])
        print("Overall accuracy:", result["overall_accuracy"])
        print("Delta vs notebook 20:", result["overall_accuracy_minus_notebook_20"])
        print("Delta vs notebook 21 raw direct:", result["overall_accuracy_minus_raw_direct_notebook_21"])
        print("Delta vs notebook 28 same-route diagnostic:", result["overall_accuracy_minus_notebook_28_raw_inverted_no_conf"])
        return result


def run(project_root: str | Path, is_kaggle: bool, commit_sha: str) -> dict:
    return RecreatedRawInvertedRouteScaleRunner(project_root, is_kaggle, commit_sha).run()
