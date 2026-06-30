import argparse
import json
from pathlib import Path

from .belief import mine_beliefs
from .config import load_config
from .data import build_data
from .evaluation import evaluate
from .train_adapter import train_adapter


def _print_path(label: str, path: Path) -> None:
    print(json.dumps({label: str(path)}, indent=2, ensure_ascii=False))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="SAFE-PoRT local pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ["build-data", "mine-beliefs", "train-adapter", "evaluate"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, help="Path to JSON config")
        if name in {"mine-beliefs", "evaluate"}:
            p.add_argument("--dry-run", action="store_true", help="Skip model loading/generation")

    run_all = sub.add_parser("run-all")
    run_all.add_argument("--config", required=True, help="Path to JSON config")
    run_all.add_argument("--skip-beliefs", action="store_true")
    run_all.add_argument("--skip-train", action="store_true")
    run_all.add_argument("--skip-eval", action="store_true")
    run_all.add_argument("--dry-run-beliefs", action="store_true")
    run_all.add_argument("--dry-run-eval", action="store_true")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "build-data":
        _print_path("data_path", build_data(cfg))
    elif args.command == "mine-beliefs":
        _print_path("belief_path", mine_beliefs(cfg, dry_run=args.dry_run))
    elif args.command == "train-adapter":
        _print_path("adapter_path", train_adapter(cfg))
    elif args.command == "evaluate":
        _print_path("eval_dir", evaluate(cfg, dry_run=args.dry_run))
    elif args.command == "run-all":
        _print_path("data_path", build_data(cfg))
        if not args.skip_beliefs:
            _print_path("belief_path", mine_beliefs(cfg, dry_run=args.dry_run_beliefs))
        if not args.skip_train:
            _print_path("adapter_path", train_adapter(cfg))
        if not args.skip_eval:
            _print_path("eval_dir", evaluate(cfg, dry_run=args.dry_run_eval))


if __name__ == "__main__":
    main()

