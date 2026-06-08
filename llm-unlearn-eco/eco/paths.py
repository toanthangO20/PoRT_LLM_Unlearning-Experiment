import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("PORT_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
ECO_ROOT = PROJECT_ROOT / "llm-unlearn-eco"
CONFIG_DIR = ECO_ROOT / "config"
MODEL_CONFIG_DIR = CONFIG_DIR / "model_config"
TASK_CONFIG_DIR = CONFIG_DIR / "task_config"
TOFU_DATASET_DIR = PROJECT_ROOT / "dataset" / "TOFU" / "original"
WMDP_DATASET_DIR = PROJECT_ROOT / "dataset" / "WMDP" / "original"
RESULTS_DIR = PROJECT_ROOT / "results"


def env_path(name, default):
    return Path(os.environ.get(name, default)).expanduser().resolve()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
