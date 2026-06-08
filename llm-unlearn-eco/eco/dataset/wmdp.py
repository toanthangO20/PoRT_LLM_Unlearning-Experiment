import ast
import re
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

from eco.dataset.base import BaseDataset
from eco.paths import WMDP_DATASET_DIR


class WMDP(BaseDataset):
    dataset_type = "multiple_choice"
    path = str(WMDP_DATASET_DIR)
    name = "wmdp"
    subject = None
    subsets = ["test"]
    test_set = "test"
    choice_labels = ["A", "B", "C", "D"]
    eval_method = "choice_by_top_logit"
    metric = "accuracy"
    keys = ["prompt", "choices", "correct_answer"]
    eval_prompt_key = "prompt"
    eval_answer_key = "choices"

    def __init__(self, parquet_path=None, dataset_path=None, sample_size=None):
        super().__init__()
        self.parquet_path = Path(parquet_path) if parquet_path else None
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.sample_size = sample_size

    def _default_parquet_path(self):
        if self.subject is None:
            raise ValueError("subject must be set or parquet_path must be provided")
        return Path(self.path) / f"wmdp-{self.subject}" / "test-00000-of-00001.parquet"

    def download(self):
        if self.dataset_path is not None:
            dataset = load_from_disk(str(self.dataset_path))
            if isinstance(dataset, DatasetDict):
                self.dataset = dataset
            else:
                self.dataset = DatasetDict({"test": dataset})
            return

        parquet_path = self.parquet_path or self._default_parquet_path()
        if not parquet_path.exists():
            raise FileNotFoundError(f"WMDP parquet file not found: {parquet_path}")

        dataset = Dataset.from_parquet(str(parquet_path))
        if self.sample_size is not None:
            dataset = dataset.select(range(min(self.sample_size, len(dataset))))
        self.dataset = DatasetDict({"test": dataset})

    @classmethod
    def normalize_choices(cls, choices):
        if choices is None:
            raise ValueError("WMDP choices cannot be None")

        if hasattr(choices, "tolist"):
            choices = choices.tolist()

        if isinstance(choices, tuple):
            choices = list(choices)

        if isinstance(choices, list):
            if len(choices) == 1 and isinstance(choices[0], (list, tuple)):
                choices = list(choices[0])
            return [str(choice).strip() for choice in choices]

        if isinstance(choices, str):
            text = choices.strip()
            try:
                parsed = ast.literal_eval(text)
                if hasattr(parsed, "tolist"):
                    parsed = parsed.tolist()
                if isinstance(parsed, (list, tuple)):
                    return [str(choice).strip() for choice in parsed]
            except (SyntaxError, ValueError):
                pass

            quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", text, flags=re.DOTALL)
            flattened = [single or double for single, double in quoted]
            if flattened:
                return [choice.strip() for choice in flattened]

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if len(lines) > 1:
                return lines

        raise ValueError(f"Unsupported WMDP choices format: {type(choices)}")

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        if split_name not in self.dataset:
            raise KeyError(f"WMDP split '{split_name}' not found; available: {list(self.dataset.keys())}")

        dataset = self.dataset[split_name]
        dataset = dataset.map(
            lambda x: {
                "choice_text": self.normalize_choices(x["choices"]),
                "correct_answer": int(x["answer"]),
            }
        )
        dataset = dataset.map(
            lambda x: {
                "prompt": prompt_prefix
                + self.format_prompt(x["question"], x["choice_text"], self.choice_labels, self.subject),
                "choices": self.choice_labels,
            }
        )
        keep_columns = set(self.keys)
        remove_columns = [column for column in dataset.column_names if column not in keep_columns]
        if remove_columns:
            dataset = dataset.remove_columns(remove_columns)
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    @staticmethod
    def format_prompt(prompt, choice_text, choice_label, subject=None):
        topic = f" about {subject}" if subject else ""
        topic_line = f"The following are multiple choice questions (with answers){topic}.\n\n"
        question_line = f"Question:\n{prompt}\n"
        choice_lines = "\n".join(
            f"{label}. {text}" for label, text in zip(choice_label, choice_text)
        )
        answer_line = "\n\nAnswer:"
        return topic_line + question_line + choice_lines + answer_line


class WMDPBio(WMDP):
    name = "wmdp-bio"
    subject = "bio"


class WMDPChem(WMDP):
    name = "wmdp-chem"
    subject = "chem"


class WMDPCyber(WMDP):
    name = "wmdp-cyber"
    subject = "cyber"
