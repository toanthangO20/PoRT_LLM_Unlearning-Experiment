from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .io_utils import read_json


@dataclass
class SafePortConfig:
    path: Path
    raw: Dict[str, Any]
    package_root: Path
    project_root: Path
    output_dir: Path

    def section(self, name: str) -> Dict[str, Any]:
        return dict(self.raw.get(name, {}))

    def resolve(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        candidates = [
            (self.project_root / path).resolve(),
            (self.package_root / path).resolve(),
            (self.path.parent / path).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def output_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.output_dir / path).resolve()


def load_config(config_path: str) -> SafePortConfig:
    path = Path(config_path).resolve()
    raw = read_json(path)
    package_root = path.parent.parent.resolve()
    project_root_value = raw.get("project_root", ".")
    project_root = Path(project_root_value)
    if not project_root.is_absolute():
        project_root = (path.parent / project_root).resolve()
    output_dir = Path(raw.get("output_dir", "runs/safe_port"))
    if not output_dir.is_absolute():
        output_dir = (package_root / output_dir).resolve()
    return SafePortConfig(path=path, raw=raw, package_root=package_root, project_root=project_root, output_dir=output_dir)
