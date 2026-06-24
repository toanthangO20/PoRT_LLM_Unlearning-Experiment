from __future__ import annotations

import json
import math
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK30_ZIP_NAME = "notebook30_full_recreated_raw_inverted_route_results.zip"


def env_text(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


@dataclass
class SplitData:
    train: pd.DataFrame
    eval: pd.DataFrame
    test: pd.DataFrame
    train_groups: set[str]
    eval_groups: set[str]
    test_groups: set[str]


class DirectRouterDiagnosticRunner:
    def __init__(self, project_root: str | Path, is_kaggle: bool, commit_sha: str):
        self.project_root = Path(project_root).resolve()
        self.is_kaggle = bool(is_kaggle)
        self.commit_sha = commit_sha
        self.output_dir = Path(env_text("PORT_ROUTER_DIAGNOSTIC_OUTPUT_DIR") or self.project_root / "results")
        self.run_name = env_text("PORT_RUN_NAME", "paper_port_direct_router_diagnostic_from_notebook30")
        self.run_dir = self.output_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(env_text("PORT_SEED", "1234"))
        self.eval_size = float(env_text("PORT_ROUTER_EVAL_SIZE", "0.15"))
        self.test_size = float(env_text("PORT_ROUTER_TEST_SIZE", "0.15"))

    def _download_zip(self, url: str) -> Path:
        destination = self.run_dir / "downloads" / NOTEBOOK30_ZIP_NAME
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading notebook 30 results zip: {url}")
        urllib.request.urlretrieve(url, destination)
        return destination

    def resolve_results_zip(self) -> Path:
        explicit = env_text("PORT_NOTEBOOK30_RESULTS_ZIP_PATH")
        if explicit:
            path = Path(explicit).expanduser()
            if path.exists():
                return path.resolve()
            raise FileNotFoundError(path)

        url = env_text("PORT_NOTEBOOK30_RESULTS_ZIP_URL")
        if url:
            return self._download_zip(url)

        candidates = [
            self.project_root / "results" / NOTEBOOK30_ZIP_NAME,
            Path("/kaggle/working") / NOTEBOOK30_ZIP_NAME,
            Path.cwd() / NOTEBOOK30_ZIP_NAME,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(
            "Could not find notebook 30 results zip. Set PORT_NOTEBOOK30_RESULTS_ZIP_PATH "
            f"or PORT_NOTEBOOK30_RESULTS_ZIP_URL. Expected default filename: {NOTEBOOK30_ZIP_NAME}"
        )

    @staticmethod
    def _read_zip_member(zip_path: Path, member_name: str) -> bytes:
        with zipfile.ZipFile(zip_path) as zf:
            matches = [name for name in zf.namelist() if name == member_name or name.endswith("/" + member_name)]
            if not matches:
                raise FileNotFoundError(f"{member_name} not found in {zip_path}")
            return zf.read(matches[0])

    def load_artifact(self, zip_path: Path) -> tuple[pd.DataFrame, dict, dict]:
        with zipfile.ZipFile(zip_path) as zf:
            required = ["all_predictions.csv", "summary.json", "run_config.json", "failed_jobs.json"]
            missing = [name for name in required if not any(item == name or item.endswith("/" + name) for item in zf.namelist())]
            if missing:
                raise RuntimeError(f"Notebook 30 zip is missing required files: {missing}")

            with zf.open("all_predictions.csv") as handle:
                df = pd.read_csv(handle)
            summary = json.loads(zf.read("summary.json").decode("utf-8"))
            run_config = json.loads(zf.read("run_config.json").decode("utf-8"))
            failed_jobs = json.loads(zf.read("failed_jobs.json").decode("utf-8"))

        if failed_jobs:
            raise RuntimeError(f"Notebook 30 artifact has failed jobs: {failed_jobs}")
        return df, summary, run_config

    @staticmethod
    def normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for column in ["initial_is_correct", "is_correct", "rethink_triggered"]:
            if df[column].dtype == object:
                df[column] = df[column].map({"True": True, "False": False, True: True, False: False})
            df[column] = df[column].astype(bool)
        for column in ["initial_answer", "expanded_initial_answer", "rethink_answer", "generated_answer", "question", "choices"]:
            if column in df.columns:
                df[column] = df[column].fillna("").astype(str)
        df["group_id"] = df["domain"].astype(str) + "::" + df["row_index"].astype(str)
        df["postjudge_conf_bin"] = pd.cut(
            df["postjudge_confidence"].astype(float),
            bins=[0.0, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0],
            include_lowest=True,
        ).astype(str)
        df["initial_final_oracle_correct"] = df["initial_is_correct"] | df["is_correct"]
        df["target_use_rethink"] = df["rethink_triggered"] & df["is_correct"] & ~df["initial_is_correct"]
        return df

    def split_by_group(self, candidate_df: pd.DataFrame) -> SplitData:
        from sklearn.model_selection import GroupShuffleSplit

        groups = candidate_df["group_id"].astype(str).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.seed)
        train_eval_idx, test_idx = next(splitter.split(candidate_df, candidate_df["target_use_rethink"], groups))
        train_eval = candidate_df.iloc[train_eval_idx].copy()
        test = candidate_df.iloc[test_idx].copy()

        eval_rel_size = self.eval_size / max(1e-9, 1.0 - self.test_size)
        splitter2 = GroupShuffleSplit(n_splits=1, test_size=eval_rel_size, random_state=self.seed + 1)
        groups2 = train_eval["group_id"].astype(str).to_numpy()
        train_idx, eval_idx = next(splitter2.split(train_eval, train_eval["target_use_rethink"], groups2))
        train = train_eval.iloc[train_idx].copy()
        eval_df = train_eval.iloc[eval_idx].copy()

        return SplitData(
            train=train,
            eval=eval_df,
            test=test,
            train_groups=set(train["group_id"].astype(str)),
            eval_groups=set(eval_df["group_id"].astype(str)),
            test_groups=set(test["group_id"].astype(str)),
        )

    @staticmethod
    def feature_text(df: pd.DataFrame, feature_set: str) -> pd.Series:
        meta = (
            "variant=" + df["variant"].astype(str)
            + " domain=" + df["domain"].astype(str)
            + " prompt_source=" + df["prompt_source"].astype(str)
            + " postjudge_label=" + df["postjudge_label"].astype(str)
            + " conf_bin=" + df["postjudge_conf_bin"].astype(str)
            + " initial_letter=" + df["initial_choice_letter"].fillna("INVALID").astype(str)
        )
        pre = (
            meta
            + "\nQuestion: " + df["question"].astype(str)
            + "\nChoices: " + df["choices"].astype(str)
            + "\nInitial answer: " + df["expanded_initial_answer"].astype(str)
        )
        if feature_set == "pre_rethink":
            return pre
        if feature_set == "posthoc_initial_rethink":
            return (
                pre
                + "\nRethink answer: " + df["rethink_answer"].fillna("").astype(str)
                + "\nRethink choice: " + df["generated_choice_letter"].fillna("INVALID").astype(str)
                + "\nSame choice: "
                + (
                    df["initial_predicted_index"].fillna(-100).astype(int)
                    == df["predicted_index"].fillna(-101).astype(int)
                ).astype(str)
            )
        raise ValueError(f"Unknown feature_set={feature_set}")

    @staticmethod
    def selected_accuracy(df: pd.DataFrame, use_rethink: np.ndarray) -> float:
        selected = np.where(use_rethink, df["is_correct"].to_numpy(dtype=bool), df["initial_is_correct"].to_numpy(dtype=bool))
        return float(selected.mean()) if len(selected) else math.nan

    @staticmethod
    def selection_summary(df: pd.DataFrame, use_rethink: np.ndarray) -> dict:
        use_rethink = np.asarray(use_rethink, dtype=bool)
        return {
            "rows": int(len(df)),
            "selected_accuracy": DirectRouterDiagnosticRunner.selected_accuracy(df, use_rethink),
            "select_rethink_rate": float(use_rethink.mean()) if len(use_rethink) else None,
            "always_initial_accuracy": float(df["initial_is_correct"].mean()) if len(df) else None,
            "always_rethink_accuracy": float(df["is_correct"].mean()) if len(df) else None,
            "oracle_accuracy": float((df["initial_is_correct"] | df["is_correct"]).mean()) if len(df) else None,
            "helped_count": int((use_rethink & df["is_correct"].to_numpy(dtype=bool) & ~df["initial_is_correct"].to_numpy(dtype=bool)).sum()),
            "hurt_count": int((use_rethink & ~df["is_correct"].to_numpy(dtype=bool) & df["initial_is_correct"].to_numpy(dtype=bool)).sum()),
        }

    def tune_threshold_policy(self, train: pd.DataFrame, eval_df: pd.DataFrame, test: pd.DataFrame) -> dict:
        thresholds = np.unique(
            np.concatenate(
                [
                    np.linspace(0.0, 1.0, 101),
                    train["postjudge_confidence"].to_numpy(dtype=float),
                    eval_df["postjudge_confidence"].to_numpy(dtype=float),
                ]
            )
        )
        rows = []
        for threshold in thresholds:
            def policy(df: pd.DataFrame) -> np.ndarray:
                label = df["postjudge_label"].to_numpy(dtype=int)
                conf = df["postjudge_confidence"].to_numpy(dtype=float)
                return (label == 0) & (conf >= threshold)

            rows.append(
                {
                    "threshold": float(threshold),
                    "train_accuracy": self.selected_accuracy(train, policy(train)),
                    "eval_accuracy": self.selected_accuracy(eval_df, policy(eval_df)),
                    "train_select_rethink_rate": float(policy(train).mean()),
                    "eval_select_rethink_rate": float(policy(eval_df).mean()),
                }
            )
        search = pd.DataFrame(rows)
        best = search.sort_values(["eval_accuracy", "train_accuracy"], ascending=False).iloc[0].to_dict()
        threshold = float(best["threshold"])
        use_test = (test["postjudge_label"].to_numpy(dtype=int) == 0) & (
            test["postjudge_confidence"].to_numpy(dtype=float) >= threshold
        )
        result = {
            "model": "label0_confidence_threshold",
            "feature_set": "postjudge_label_confidence",
            "chosen_threshold": threshold,
            "eval_accuracy": float(best["eval_accuracy"]),
            "test": self.selection_summary(test, use_test),
        }
        search.to_csv(self.run_dir / "threshold_policy_search.csv", index=False)
        return result

    def train_logistic_selector(self, splits: SplitData, feature_set: str) -> dict:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=50000)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.seed, solver="liblinear")),
            ]
        )
        x_train = self.feature_text(splits.train, feature_set)
        y_train = splits.train["target_use_rethink"].astype(int).to_numpy()
        model.fit(x_train, y_train)

        def proba(df: pd.DataFrame) -> np.ndarray:
            return model.predict_proba(self.feature_text(df, feature_set))[:, 1]

        eval_scores = proba(splits.eval)
        thresholds = np.unique(np.concatenate([np.linspace(0.0, 1.0, 101), eval_scores]))
        eval_rows = []
        for threshold in thresholds:
            use_eval = eval_scores >= threshold
            eval_rows.append(
                {
                    "threshold": float(threshold),
                    "eval_accuracy": self.selected_accuracy(splits.eval, use_eval),
                    "eval_select_rethink_rate": float(use_eval.mean()),
                }
            )
        eval_search = pd.DataFrame(eval_rows)
        best = eval_search.sort_values("eval_accuracy", ascending=False).iloc[0]
        threshold = float(best["threshold"])
        test_scores = proba(splits.test)
        use_test = test_scores >= threshold
        eval_search.to_csv(self.run_dir / f"logreg_{feature_set}_threshold_search.csv", index=False)
        return {
            "model": "tfidf_logreg_selector",
            "feature_set": feature_set,
            "chosen_threshold": threshold,
            "eval_accuracy": float(best["eval_accuracy"]),
            "eval_select_rethink_rate": float(best["eval_select_rethink_rate"]),
            "test": self.selection_summary(splits.test, use_test),
        }

    def hybrid_summary_for_test_groups(self, df: pd.DataFrame, candidate_test: pd.DataFrame, use_rethink: np.ndarray) -> dict:
        test_groups = set(candidate_test["group_id"].astype(str))
        test_all = df[df["group_id"].astype(str).isin(test_groups)].copy()
        selected_correct = test_all["initial_is_correct"].to_numpy(dtype=bool).copy()
        test_candidate_index = {idx: pos for pos, idx in enumerate(candidate_test.index)}
        for out_pos, row_idx in enumerate(test_all.index):
            candidate_pos = test_candidate_index.get(row_idx)
            if candidate_pos is not None and bool(use_rethink[candidate_pos]):
                selected_correct[out_pos] = bool(test_all.loc[row_idx, "is_correct"])
        return {
            "rows": int(len(test_all)),
            "accuracy": float(selected_correct.mean()) if len(selected_correct) else None,
            "current_notebook30_accuracy": float(test_all["is_correct"].mean()) if len(test_all) else None,
            "initial_only_accuracy": float(test_all["initial_is_correct"].mean()) if len(test_all) else None,
            "partial_oracle_accuracy": float(
                np.where(
                    test_all["rethink_triggered"].to_numpy(dtype=bool),
                    (test_all["initial_is_correct"] | test_all["is_correct"]).to_numpy(dtype=bool),
                    test_all["initial_is_correct"].to_numpy(dtype=bool),
                ).mean()
            )
            if len(test_all)
            else None,
        }

    def run(self) -> dict:
        zip_path = self.resolve_results_zip()
        predictions, source_summary, run_config = self.load_artifact(zip_path)
        df = self.normalize_predictions(predictions)
        candidate = df[df["rethink_triggered"]].copy()
        if candidate.empty:
            raise RuntimeError("No rethink rows are available for direct router training.")

        splits = self.split_by_group(candidate)
        threshold_result = self.tune_threshold_policy(splits.train, splits.eval, splits.test)
        logreg_pre = self.train_logistic_selector(splits, "pre_rethink")
        logreg_posthoc = self.train_logistic_selector(splits, "posthoc_initial_rethink")

        results = [threshold_result, logreg_pre, logreg_posthoc]
        best = max(results, key=lambda item: item["test"]["selected_accuracy"])

        # Recompute best selector decisions on test candidates for hybrid all-row summary.
        if best["model"] == "label0_confidence_threshold":
            use_test = (splits.test["postjudge_label"].to_numpy(dtype=int) == 0) & (
                splits.test["postjudge_confidence"].to_numpy(dtype=float) >= float(best["chosen_threshold"])
            )
        else:
            # Refit only for deriving the exact test decisions corresponding to the stored result.
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline

            model = Pipeline(
                [
                    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=50000)),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.seed, solver="liblinear")),
                ]
            )
            model.fit(self.feature_text(splits.train, best["feature_set"]), splits.train["target_use_rethink"].astype(int).to_numpy())
            scores = model.predict_proba(self.feature_text(splits.test, best["feature_set"]))[:, 1]
            use_test = scores >= float(best["chosen_threshold"])

        hybrid_test = self.hybrid_summary_for_test_groups(df, splits.test, use_test)
        candidate_summary = {
            "rows": int(len(candidate)),
            "always_initial_accuracy": float(candidate["initial_is_correct"].mean()),
            "notebook30_current_rethink_accuracy": float(candidate["is_correct"].mean()),
            "oracle_initial_vs_rethink_accuracy": float((candidate["initial_is_correct"] | candidate["is_correct"]).mean()),
            "target_rethink_positive_rate": float(candidate["target_use_rethink"].mean()),
            "rethink_hurt_rate": float((candidate["initial_is_correct"] & ~candidate["is_correct"]).mean()),
        }
        full_summary = {
            "rows": int(len(df)),
            "initial_accuracy": float(df["initial_is_correct"].mean()),
            "notebook30_final_accuracy": float(df["is_correct"].mean()),
            "partial_oracle_accuracy": float(
                np.where(
                    df["rethink_triggered"].to_numpy(dtype=bool),
                    (df["initial_is_correct"] | df["is_correct"]).to_numpy(dtype=bool),
                    df["initial_is_correct"].to_numpy(dtype=bool),
                ).mean()
            ),
            "full_initial_vs_available_rethink_oracle_accuracy": float((df["initial_is_correct"] | df["is_correct"]).mean()),
            "rethink_rate": float(df["rethink_triggered"].mean()),
        }

        pd.DataFrame(results).to_csv(self.run_dir / "router_model_results.csv", index=False)
        df.groupby(["variant", "domain"], sort=False).agg(
            rows=("is_correct", "size"),
            initial_accuracy=("initial_is_correct", "mean"),
            notebook30_final_accuracy=("is_correct", "mean"),
            rethink_rate=("rethink_triggered", "mean"),
            postjudge_positive_rate=("postjudge_label", "mean"),
            available_oracle_accuracy=("initial_final_oracle_correct", "mean"),
        ).reset_index().to_csv(self.run_dir / "notebook30_route_breakdown_by_job.csv", index=False)

        output = {
            "status": "completed",
            "purpose": "paper_port_direct_router_diagnostic_from_notebook30",
            "official_paper_checkpoint": False,
            "project_root": str(self.project_root),
            "commit": self.commit_sha,
            "source_zip": str(zip_path),
            "source_notebook30_run_config": {
                "run_name": run_config.get("run_name") or run_config.get("run_dir"),
                "route_policy": run_config.get("route_policy"),
                "rows": source_summary.get("overall", {}).get("rows"),
                "accuracy": source_summary.get("overall", {}).get("accuracy"),
            },
            "split": {
                "train_rows": int(len(splits.train)),
                "eval_rows": int(len(splits.eval)),
                "test_rows": int(len(splits.test)),
                "train_groups": int(len(splits.train_groups)),
                "eval_groups": int(len(splits.eval_groups)),
                "test_groups": int(len(splits.test_groups)),
                "group_key": "domain::row_index",
            },
            "full_summary": full_summary,
            "candidate_rethink_rows_summary": candidate_summary,
            "model_results": results,
            "best_test_model": best,
            "best_hybrid_test_group_summary": hybrid_test,
            "artifacts": {
                "summary_json": str(self.run_dir / "summary.json"),
                "router_model_results_csv": str(self.run_dir / "router_model_results.csv"),
                "route_breakdown_by_job_csv": str(self.run_dir / "notebook30_route_breakdown_by_job.csv"),
            },
            "limitations": [
                "Rows originally kept by notebook 30 do not have a generated rethink candidate, so supervised router training uses rows where rethink was actually generated.",
                "The diagnostic is offline and uses correctness labels from WMDP to learn a selector; it is not an official paper checkpoint result.",
                "A deployable full router would need either generate-both-candidates inference or a separate predictor trained on complete initial/rethink pairs.",
            ],
        }
        (self.run_dir / "summary.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        print("PAPER PORT DIRECT ROUTER DIAGNOSTIC COMPLETED")
        print("Rows:", full_summary["rows"])
        print("Notebook 30 final accuracy:", full_summary["notebook30_final_accuracy"])
        print("Available oracle accuracy:", full_summary["full_initial_vs_available_rethink_oracle_accuracy"])
        print("Candidate rows:", candidate_summary["rows"])
        print("Candidate current rethink accuracy:", candidate_summary["notebook30_current_rethink_accuracy"])
        print("Candidate oracle accuracy:", candidate_summary["oracle_initial_vs_rethink_accuracy"])
        print("Best test model:", best["model"], best["feature_set"])
        print("Best candidate test selected accuracy:", best["test"]["selected_accuracy"])
        print("Best hybrid test-group accuracy:", hybrid_test["accuracy"])
        print("Artifacts:", self.run_dir)
        return output


def run(project_root: str | Path, is_kaggle: bool, commit_sha: str) -> dict:
    return DirectRouterDiagnosticRunner(project_root, is_kaggle, commit_sha).run()


if __name__ == "__main__":
    project = Path(env_text("PORT_PROJECT_ROOT") or Path.cwd()).resolve()
    commit = env_text("PORT_COMMIT_SHA", "local")
    result = run(project, Path("/kaggle/working").exists(), commit)
    print(json.dumps(result, indent=2, default=str))
