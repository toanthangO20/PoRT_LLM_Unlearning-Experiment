from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from port_generation_baseline_identity_ablation import GenerationBaselineIdentityAblationRunner
from port_recreated_smoke import env_bool, env_text


DEFAULT_RECREATED_ARTIFACT_MANIFEST_URL = (
    "https://raw.githubusercontent.com/toanthangO20/PoRT_LLM_Unlearning-Experiment/"
    "artifact-recreated-bootstrap-v1/manifest.json"
)


class PrefixCompilerSourceDiagnosticRunner(GenerationBaselineIdentityAblationRunner):
    required_job_files = [
        "prefix_compiler_predictions.csv",
        "job_summary.csv",
        "job_summary.json",
        "prompt_examples.json",
    ]

    def _read_config(self) -> dict:
        config = super()._read_config()
        config["run_name"] = env_text(
            "PORT_RUN_NAME",
            f"paper_port_wmdp_prefix_compiler_source_diagnostic_{config['model_name']}",
        )
        config["max_samples"] = int(env_text("PORT_MAX_SAMPLES", "32"))
        config["prefix_include_base_t5"] = env_bool("PORT_PREFIX_INCLUDE_BASE_T5", True)
        config["prefix_base_t5_model_path"] = env_text(
            "PORT_PREFIX_BASE_T5_MODEL_PATH",
            env_text("PORT_GENERATION_T5_MODEL_PATH", env_text("PORT_T5_MODEL_PATH", config["t5_base_model"])),
        )
        config["prefix_include_recreated_t5"] = env_bool("PORT_PREFIX_INCLUDE_RECREATED_T5", True)
        config["prefix_extra_t5_models"] = self._parse_extra_t5_models(env_text("PORT_PREFIX_EXTRA_T5_MODELS"))
        config["auto_download_recreated_artifact"] = env_bool("PORT_AUTO_DOWNLOAD_RECREATED_ARTIFACT", True)
        config["recreated_artifact_manifest_url"] = env_text(
            "PORT_RECREATED_ARTIFACT_MANIFEST_URL",
            DEFAULT_RECREATED_ARTIFACT_MANIFEST_URL,
        )
        config["scale_run_family"] = "prefix_compiler_source_diagnostic"
        return config

    @staticmethod
    def _parse_extra_t5_models(value: str | None) -> list[dict]:
        if not value:
            return []
        parts = [part.strip() for part in re.split(r"[;,]", value) if part.strip()]
        models = []
        for idx, part in enumerate(parts, start=1):
            if "=" in part:
                label, model_path = part.split("=", 1)
                label = label.strip()
                model_path = model_path.strip()
            else:
                label = f"extra_{idx}"
                model_path = part
            if not model_path:
                continue
            models.append({"source_id": PrefixCompilerSourceDiagnosticRunner._source_id(label), "label": label, "model_path": model_path})
        return models

    @staticmethod
    def _source_id(value: str) -> str:
        value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
        return value[:80] or "source"

    def _discover_recreated_t5_source(self) -> tuple[dict | None, list[dict]]:
        skipped = []
        configured_dir = self.config["recreated_artifact_dir_env"]
        if configured_dir:
            if not self._is_valid_recreated_artifact_dir(configured_dir):
                raise RuntimeError(f"Configured PORT_RECREATED_ARTIFACT_DIR is not a valid recreated artifact dir: {configured_dir}")
            artifact_dir = Path(configured_dir).resolve()
            return self._source_from_recreated_artifact_dir(artifact_dir, "env_dir"), skipped

        if self.config["recreated_artifact_zip_url"]:
            downloaded_zip = self._download_to(
                self.config["recreated_artifact_zip_url"],
                self.run_dir / "downloads" / "paper_port_recreated_artifacts_bootstrap.zip",
            )
            artifact_dir = self._extract_zip(downloaded_zip, self.run_dir / "recreated_artifact_zip")
            return self._source_from_recreated_artifact_dir(artifact_dir, "zip_url"), skipped

        candidate_zip = self._discover_candidate_zip()
        if candidate_zip:
            artifact_dir = self._extract_zip(candidate_zip, self.run_dir / "recreated_artifact_zip")
            return self._source_from_recreated_artifact_dir(artifact_dir, f"zip_path:{candidate_zip}"), skipped

        if self.config["auto_download_recreated_artifact"]:
            downloaded_zip = self._download_default_recreated_artifact_zip()
            artifact_dir = self._extract_zip(downloaded_zip, self.run_dir / "recreated_artifact_zip")
            return self._source_from_recreated_artifact_dir(
                artifact_dir,
                f"auto_download_manifest:{self.config['recreated_artifact_manifest_url']}",
            ), skipped

        skipped.append(
            {
                "source_id": "recreated_artifact",
                "reason": (
                    "No PORT_RECREATED_ARTIFACT_DIR, PORT_RECREATED_ARTIFACT_ZIP_URL, or recreated artifact zip was found, "
                    "and PORT_AUTO_DOWNLOAD_RECREATED_ARTIFACT=false."
                ),
            }
        )
        return None, skipped

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

        destination = Path("/kaggle/working/paper_port_recreated_artifacts_bootstrap.zip") if self.is_kaggle else self.run_dir / "downloads" / "paper_port_recreated_artifacts_bootstrap.zip"
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

    @staticmethod
    def _source_from_recreated_artifact_dir(artifact_dir: Path, artifact_source: str) -> dict:
        return {
            "source_id": "recreated_artifact",
            "label": "recreated_artifact",
            "model_path": str(artifact_dir / "artifacts" / "recreated_t5_ast_prefix_compiler"),
            "source_type": "recreated",
            "artifact_dir": str(artifact_dir),
            "artifact_source": artifact_source,
        }

    def resolve_or_bootstrap_artifacts(self) -> dict:
        placeholder_dir = self.run_dir / "prefix_compiler_no_bootstrap_artifacts"
        placeholder_dir.mkdir(parents=True, exist_ok=True)

        prefix_sources = []
        skipped_sources = []
        if self.config["prefix_include_base_t5"]:
            prefix_sources.append(
                {
                    "source_id": "base_t5",
                    "label": "base_t5",
                    "model_path": self.config["prefix_base_t5_model_path"],
                    "source_type": "base",
                    "artifact_source": "hf_or_local_model_path",
                }
            )

        if self.config["prefix_include_recreated_t5"]:
            recreated_source, skipped = self._discover_recreated_t5_source()
            skipped_sources.extend(skipped)
            if recreated_source is not None:
                prefix_sources.append(recreated_source)

        seen = {source["source_id"] for source in prefix_sources}
        for extra in self.config["prefix_extra_t5_models"]:
            source_id = extra["source_id"]
            while source_id in seen:
                source_id = f"{source_id}_extra"
            seen.add(source_id)
            prefix_sources.append(
                {
                    "source_id": source_id,
                    "label": extra["label"],
                    "model_path": extra["model_path"],
                    "source_type": "extra",
                    "artifact_source": "PORT_PREFIX_EXTRA_T5_MODELS",
                }
            )

        if not prefix_sources:
            raise RuntimeError("No prefix compiler source configured. Enable PORT_PREFIX_INCLUDE_BASE_T5 or set PORT_PREFIX_EXTRA_T5_MODELS.")

        audit = {
            "artifact_mode": self.config["artifact_mode"],
            "artifact_note": "Notebook 23 does not bootstrap or train artifacts; it only compares configured prefix compiler sources.",
            "artifact_source": "no_bootstrap_prefix_compiler_diagnostic",
            "official_paper_checkpoint": False,
            "prefix_sources": prefix_sources,
            "skipped_prefix_sources": skipped_sources,
            "pipeline_script_path": str(self.pipeline_script_path),
            "post_classifier_dir": str(self.post_classifier_dir),
            "eco_root": str(self.eco_root),
            "eco_config_path": str(self.eco_config_path),
            "example_library_path": str(self.example_library_path),
            "limitations": [
                "This diagnostic uses generated answers, not notebook 11 top-logit paper baseline metrics.",
                "A missing recreated artifact is recorded as skipped instead of triggering bootstrap/training.",
                "This is not an official paper checkpoint reproduction unless official artifacts are explicitly supplied.",
            ],
        }
        (self.run_dir / "artifact_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
        print(json.dumps(audit, indent=2, default=str))
        return {
            "artifact_dir": placeholder_dir,
            "t5_model_path": prefix_sources[0]["model_path"],
            "weak_dataset": {},
            "audit": audit,
            "prefix_sources": prefix_sources,
            "skipped_prefix_sources": skipped_sources,
        }

    def install_recreated_setup(self, port_wmdp, classifier_head_ckpt: str) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        def setup_all_models_prefix_diagnostic(args):
            main_device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
            dtype = getattr(torch, args.torch_dtype)

            llama_tokenizer = AutoTokenizer.from_pretrained(args.model_hub_name, trust_remote_code=True)
            if llama_tokenizer.pad_token is None:
                llama_tokenizer.pad_token = llama_tokenizer.eos_token
            llama_tokenizer.padding_side = "left"

            llama_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
            if getattr(llama_config, "pad_token_id", None) is None:
                llama_config.pad_token_id = llama_tokenizer.pad_token_id

            llama_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                config=llama_config,
                torch_dtype=dtype if main_device.type == "cuda" else torch.float32,
                attn_implementation="sdpa",
                trust_remote_code=True,
            ).to(main_device)
            llama_model.config.pad_token_id = llama_tokenizer.pad_token_id
            llama_model.eval()

            return {
                "t5_model": None,
                "t5_tokenizer": None,
                "prefix_llama_model": llama_model,
                "main_llama_model": llama_model,
                "llama_tokenizer": llama_tokenizer,
                "classifier_model": None,
                "classifier_tokenizer": None,
            }

        port_wmdp.setup_all_models = setup_all_models_prefix_diagnostic
        print("Installed prefix compiler source diagnostic setup (target model only; T5 sources load sequentially).")

    def _build_run_config(self, runtime_script_path: Path, artifact_info: dict, classifier_info: dict) -> dict:
        run_config = super()._build_run_config(runtime_script_path, artifact_info, classifier_info)
        run_config.update(
            {
                "purpose": "paper_port_wmdp_prefix_compiler_source_diagnostic",
                "diagnostic_methods": ["raw_direct_generation"]
                + [f"compiled_{source['source_id']}_generation" for source in artifact_info["prefix_sources"]],
                "prefix_sources": artifact_info["prefix_sources"],
                "skipped_prefix_sources": artifact_info["skipped_prefix_sources"],
                "limitations": [
                    "This diagnostic isolates prefix compiler source quality and does not run post-judge or rethink.",
                    "The metric is generated answer accuracy, not notebook 11 top-logit paper baseline accuracy.",
                    "This is a recreated/public-artifact diagnostic unless official artifacts are explicitly supplied.",
                ],
            }
        )
        return run_config

    def _job_output_dir(self, base_args: SimpleNamespace, job: dict, source_id: str) -> Path:
        return Path(base_args.output_dir) / self.config["model_name"].replace("/", "_") / source_id / job["variant"] / job["wmdp_set"]

    def _load_existing_source_job(self, output_dir: Path, job: dict, source_id: str, expected_rows: int):
        if not self.config["resume_existing"]:
            return None
        if not all((output_dir / name).exists() for name in self.required_job_files):
            return None
        try:
            import pandas as pd

            rows = pd.read_csv(output_dir / "prefix_compiler_predictions.csv").to_dict(orient="records")
            summary = pd.read_csv(output_dir / "job_summary.csv").to_dict(orient="records")
        except Exception as exc:
            print(f"Existing prefix diagnostic output is not readable, rerunning {source_id} {job['variant']}/{job['domain']}: {exc}")
            return None
        if len(rows) != expected_rows:
            print(
                f"Existing prefix diagnostic output has {len(rows)} rows, expected {expected_rows}; "
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
        pd.DataFrame(rows).to_csv(output_dir / "prefix_compiler_predictions.csv", index=False)
        pd.DataFrame(summary).to_csv(output_dir / "job_summary.csv", index=False)
        (output_dir / "job_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        (output_dir / "prompt_examples.json").write_text(json.dumps(prompt_examples, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    @staticmethod
    def _append_rows_to_csv(path: Path, rows: list[dict]) -> None:
        import pandas as pd

        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)

    @staticmethod
    def _clear_cuda_cache() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_t5_source(self, source: dict, args: SimpleNamespace):
        import torch
        from transformers import T5ForConditionalGeneration, T5TokenizerFast

        main_device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        started = time.perf_counter()
        tokenizer = T5TokenizerFast.from_pretrained(source["model_path"])
        model = T5ForConditionalGeneration.from_pretrained(source["model_path"]).to(main_device)
        model.eval()
        return tokenizer, model, time.perf_counter() - started

    def _compile_prompts_with_loaded_source(self, prompts: list[str], models: dict, args, port_wmdp) -> tuple[list[str], list[bool]]:
        compiled = []
        fallback_flags = []
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            batch_compiled = port_wmdp.run_prefix_compilation_step_batch(batch_prompts, models, args.example_library, args)
            retry_indices = [idx for idx, prompt in enumerate(batch_compiled) if not str(prompt).strip()]
            if retry_indices:
                retry_questions = [batch_prompts[idx] for idx in retry_indices]
                retry_prompts = port_wmdp.run_prefix_compilation_step_batch(retry_questions, models, args.example_library, args)
                for idx, retry_prompt in zip(retry_indices, retry_prompts):
                    batch_compiled[idx] = retry_prompt
            for idx, prompt in enumerate(batch_compiled):
                used_fallback = False
                prompt = str(prompt)
                if not prompt.strip():
                    prompt = batch_prompts[idx]
                    used_fallback = True
                compiled.append(prompt)
                fallback_flags.append(used_fallback)
        return compiled, fallback_flags

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

    def _source_summary(self, rows: list[dict], source: dict, job: dict) -> dict:
        total = len(rows)
        correct = sum(1 for row in rows if bool(row.get("prediction_is_correct", row.get("is_correct", False))))
        valid = sum(1 for row in rows if row.get("prediction_predicted_index", row.get("predicted_index")) is not None)

        def avg(name: str) -> float | None:
            values = [row.get(name) for row in rows if row.get(name) is not None]
            return sum(float(value) for value in values) / len(values) if values else None

        return {
            "variant": job["variant"],
            "domain": job["domain"],
            "wmdp_set": job["wmdp_set"],
            "prompt_source": job["prompt_source"],
            "source_id": source["source_id"],
            "source_type": source["source_type"],
            "model_path": source.get("model_path"),
            "method": "raw_direct_generation" if source["source_id"] == "raw_direct" else f"compiled_{source['source_id']}_generation",
            "rows": total,
            "correct_count": correct,
            "accuracy": correct / total if total else None,
            "valid_predictions_count": valid,
            "valid_predictions_rate": valid / total if total else None,
            "same_as_raw_index_rate": avg("same_as_raw_index"),
            "compiled_prompt_used_fallback_rate": avg("compiled_prompt_used_fallback"),
            "compiled_prompt_same_as_original_rate": avg("compiled_prompt_same_as_original"),
            "compiled_prompt_has_answer_instruction_rate": avg("compiled_prompt_has_answer_instruction"),
            "compiled_prompt_choice_coverage_avg": avg("compiled_prompt_choice_coverage"),
            "prompt_char_len_avg": avg("prompt_char_len"),
            "compiled_prompt_char_len_avg": avg("compiled_prompt_char_len"),
            "compiled_prompt_char_len_ratio_avg": avg("compiled_prompt_char_len_ratio"),
        }

    def _run_raw_job(self, job_index: int, job: dict, models: dict, base_args: SimpleNamespace, port_wmdp) -> tuple[list[dict], list[dict], dict]:
        args = SimpleNamespace(**vars(base_args))
        args.wmdp_set = job["wmdp_set"]
        records = job["records"]
        prompts = [item["prompt"] for item in records]

        self._set_generation_seed(self.config["seed"], job_index, 1)
        answers = self._generate_answers(prompts, models, args, port_wmdp)
        self._clear_cuda_cache()

        source = {"source_id": "raw_direct", "source_type": "raw", "model_path": None}
        rows = []
        for idx, item in enumerate(records):
            quality = self._prefix_quality(item["prompt"], item["prompt"], False)
            row = {
                **item,
                "source_id": source["source_id"],
                "source_type": source["source_type"],
                "model_path": None,
                "compiled_prompt": item["prompt"],
                "raw_direct_answer": answers[idx],
                "raw_direct_predicted_index": None,
                "same_as_raw_index": True,
                **quality,
            }
            row.update(self._choice_fields(port_wmdp, "prediction", answers[idx], item))
            row["answer"] = row["prediction_answer"]
            row["choice_letter"] = row["prediction_choice_letter"]
            row["predicted_index"] = row["prediction_predicted_index"]
            row["is_correct"] = row["prediction_is_correct"]
            row["raw_direct_predicted_index"] = row["prediction_predicted_index"]
            row["same_as_raw_index"] = True
            rows.append(row)
        summary = [self._source_summary(rows, source, job)]
        prompt_examples = self._prompt_examples(job, rows)
        return rows, summary, prompt_examples

    @staticmethod
    def _as_optional_int(value):
        if value is None:
            return None
        try:
            if str(value).lower() == "nan":
                return None
            return int(float(value))
        except Exception:
            return None

    def _run_prefix_source_job(
        self,
        source_index: int,
        job_index: int,
        source: dict,
        job: dict,
        raw_rows: list[dict],
        models: dict,
        base_args: SimpleNamespace,
        port_wmdp,
    ) -> tuple[list[dict], list[dict], dict]:
        args = SimpleNamespace(**vars(base_args))
        args.wmdp_set = job["wmdp_set"]
        records = job["records"]
        prompts = [item["prompt"] for item in records]

        self._set_generation_seed(self.config["seed"], job_index, 10 + source_index * 100)
        compiled_prompts, fallback_flags = self._compile_prompts_with_loaded_source(prompts, models, args, port_wmdp)
        self._clear_cuda_cache()

        self._set_generation_seed(self.config["seed"], job_index, 20 + source_index * 100)
        answers = self._generate_answers(compiled_prompts, models, args, port_wmdp)
        self._clear_cuda_cache()

        rows = []
        for idx, item in enumerate(records):
            raw_index = self._as_optional_int(raw_rows[idx].get("raw_direct_predicted_index"))
            quality = self._prefix_quality(item["prompt"], compiled_prompts[idx], fallback_flags[idx])
            row = {
                **item,
                "source_id": source["source_id"],
                "source_type": source["source_type"],
                "model_path": source["model_path"],
                "compiled_prompt": compiled_prompts[idx],
                "raw_direct_answer": raw_rows[idx].get("raw_direct_answer"),
                "raw_direct_predicted_index": raw_index,
                **quality,
            }
            row.update(self._choice_fields(port_wmdp, "prediction", answers[idx], item))
            row["answer"] = row["prediction_answer"]
            row["choice_letter"] = row["prediction_choice_letter"]
            row["predicted_index"] = row["prediction_predicted_index"]
            row["is_correct"] = row["prediction_is_correct"]
            row["same_as_raw_index"] = row["prediction_predicted_index"] == raw_index if raw_index is not None else False
            rows.append(row)
        summary = [self._source_summary(rows, source, job)]
        prompt_examples = self._prompt_examples(job, rows)
        return rows, summary, prompt_examples

    @staticmethod
    def _prompt_examples(job: dict, rows: list[dict]) -> dict:
        return {
            "variant": job["variant"],
            "domain": job["domain"],
            "prompt_source": job["prompt_source"],
            "source_id": rows[0]["source_id"] if rows else None,
            "examples": [
                {
                    "row_index": rows[idx]["row_index"],
                    "prompt_preview": rows[idx]["prompt"][:1200],
                    "compiled_prompt_preview": rows[idx]["compiled_prompt"][:1200],
                    "raw_direct_answer": rows[idx].get("raw_direct_answer"),
                    "answer": rows[idx].get("prediction_answer"),
                    "predicted_index": rows[idx].get("prediction_predicted_index"),
                    "correct_answer_index": rows[idx].get("correct_answer_index"),
                }
                for idx in range(min(3, len(rows)))
            ],
        }

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

        predictions_path = self.run_dir / "all_prefix_compiler_predictions.csv"
        by_job_path = self.run_dir / "prefix_compiler_summary_by_job.csv"
        overall_path = self.run_dir / "prefix_compiler_summary_overall.csv"
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
            "all_prefix_compiler_predictions_csv": str(predictions_path),
            "prefix_compiler_summary_by_job_csv": str(by_job_path),
            "prefix_compiler_summary_overall_csv": str(overall_path),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
        return summary_payload

    def _read_raw_rows(self, path: Path) -> list[dict]:
        import pandas as pd

        rows = pd.read_csv(path).to_dict(orient="records")
        return rows

    def run_matrix(self, port_wmdp, runtime_script_path: Path, artifact_info: dict, classifier_info: dict) -> dict:
        base_args = self._build_base_args(artifact_info, classifier_info)
        run_config = self._build_run_config(runtime_script_path, artifact_info, classifier_info)
        (self.run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

        matrix_jobs = self.build_matrix_jobs()
        total_dataset_rows = sum(len(job["records"]) for job in matrix_jobs)
        prediction_rows = 0
        summary_rows: list[dict] = []
        completed_jobs: list[dict] = []
        skipped_jobs: list[dict] = []
        failed_jobs: list[dict] = []
        root_predictions_path = self.run_dir / "all_prefix_compiler_predictions.csv"
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
        all_sources = [raw_source] + artifact_info["prefix_sources"]
        source_load_seconds: dict[str, float] = {"raw_direct": 0.0}
        summary_payload = None

        for source_index, source in enumerate(all_sources, start=1):
            source_id = source["source_id"]
            print(f"\n=== Prefix compiler source {source_index}/{len(all_sources)}: {source_id} ===")
            if source_id != "raw_direct":
                t5_tokenizer, t5_model, load_seconds = self._load_t5_source(source, base_args)
                models["t5_tokenizer"] = t5_tokenizer
                models["t5_model"] = t5_model
                source_load_seconds[source_id] = load_seconds
                print(json.dumps({"source_id": source_id, "model_path": source["model_path"], "t5_load_seconds": load_seconds}, indent=2))

            try:
                for job_index, job in enumerate(matrix_jobs, start=1):
                    output_dir = self._job_output_dir(base_args, job, source_id)
                    print(f"\n=== Prefix job {job_index}/{len(matrix_jobs)}: {source_id} {job['variant']}/{job['domain']}, rows={len(job['records'])} ===")
                    try:
                        existing = self._load_existing_source_job(output_dir, job, source_id, len(job["records"]))
                        if existing is not None:
                            rows, summary = existing
                            skipped_jobs.append(
                                {
                                    "source_id": source_id,
                                    "job_index": job_index,
                                    "variant": job["variant"],
                                    "domain": job["domain"],
                                    "rows": len(job["records"]),
                                    "output_dir": str(output_dir),
                                }
                            )
                            print(f"Skipping completed prefix diagnostic job: {output_dir}")
                        else:
                            start_job = time.perf_counter()
                            if source_id == "raw_direct":
                                rows, summary, prompt_examples = self._run_raw_job(job_index, job, models, base_args, port_wmdp)
                            else:
                                raw_output_dir = self._job_output_dir(base_args, job, "raw_direct")
                                raw_path = raw_output_dir / "prefix_compiler_predictions.csv"
                                if not raw_path.exists():
                                    raise RuntimeError(f"Raw direct output must exist before compiled source runs: {raw_path}")
                                raw_rows = self._read_raw_rows(raw_path)
                                rows, summary, prompt_examples = self._run_prefix_source_job(
                                    source_index,
                                    job_index,
                                    source,
                                    job,
                                    raw_rows,
                                    models,
                                    base_args,
                                    port_wmdp,
                                )
                            job_seconds = time.perf_counter() - start_job
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
                        del rows
                        self._clear_cuda_cache()
                    except Exception as exc:
                        failure = {
                            "source_id": source_id,
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
            finally:
                if source_id != "raw_direct":
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
        expected_sources = 1 + len(matrix_result["prefix_sources"])
        if matrix_result["failed_jobs"]:
            raise RuntimeError(f"Prefix compiler diagnostic has failed jobs: {matrix_result['failed_jobs']}")
        if len(matrix_result["completed_jobs"]) + len(matrix_result["skipped_jobs"]) != expected_jobs * expected_sources:
            raise RuntimeError(
                f"Expected {expected_jobs * expected_sources} source jobs, got "
                f"completed+skipped={len(matrix_result['completed_jobs']) + len(matrix_result['skipped_jobs'])}"
            )

        required_root_files = [
            self.run_dir / "artifact_audit.json",
            self.run_dir / "run_config.json",
            self.run_dir / "summary.json",
            self.run_dir / "all_prefix_compiler_predictions.csv",
            self.run_dir / "prefix_compiler_summary_by_job.csv",
            self.run_dir / "prefix_compiler_summary_overall.csv",
            self.run_dir / "failed_jobs.json",
            classifier_info["classifier_artifact_dir"] / "classifier.joblib",
            classifier_info["classifier_artifact_dir"] / "classifier_metadata.json",
        ]
        missing_files = [str(path) for path in required_root_files if not Path(path).exists()]
        base_args = self._build_base_args({"t5_model_path": ""}, classifier_info)
        source_ids = ["raw_direct"] + [source["source_id"] for source in matrix_result["prefix_sources"]]
        for source_id in source_ids:
            for job in matrix_result["matrix_jobs"]:
                output_dir = self._job_output_dir(base_args, job, source_id)
                for name in self.required_job_files:
                    if not (output_dir / name).exists():
                        missing_files.append(str(output_dir / name))
        if missing_files:
            raise RuntimeError(f"Missing expected prefix compiler diagnostic artifacts: {missing_files}")

        overall = matrix_result["summary_payload"]["overall_summary"]
        result = {
            "status": "completed",
            "prefix_compiler_source_diagnostic": True,
            "official_paper_checkpoint": False,
            "jobs": expected_jobs,
            "sources": source_ids,
            "dataset_rows": matrix_result["total_dataset_rows"],
            "prediction_rows": matrix_result["prediction_rows"],
            "max_samples": self.config["max_samples"],
            "overall_summary": overall,
            "completed_jobs": len(matrix_result["completed_jobs"]),
            "skipped_jobs": len(matrix_result["skipped_jobs"]),
            "run_dir": str(self.run_dir),
        }
        print("PAPER PORT WMDP PREFIX COMPILER SOURCE DIAGNOSTIC COMPLETED")
        print("Jobs:", result["jobs"])
        print("Sources:", result["sources"])
        print("Dataset rows:", result["dataset_rows"])
        print("Prediction rows:", result["prediction_rows"])
        print("Overall summary:", json.dumps(overall, indent=2, default=str))
        print("Artifacts:", self.run_dir)
        print("Important: this is a prefix compiler diagnostic, not an official paper metric run.")
        return result


def run(project_root: str | Path, is_kaggle: bool, commit_sha: str) -> dict:
    return PrefixCompilerSourceDiagnosticRunner(project_root, is_kaggle, commit_sha).run()
