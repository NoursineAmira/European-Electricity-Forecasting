import argparse

import pandas as pd

from . import experiment, pipeline
from .config import load_config
from .models import MODEL_NAMES

COMMANDS = ["fetch", "build", "validate", "all", "tune", "backtest", "evaluate", "intervals", "significance"]


def main() -> None:
    parser = argparse.ArgumentParser(prog="energy-forecast")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--countries", nargs="+", help="country codes (default: all in config)")
    parser.add_argument("--targets", nargs="+", choices=pipeline.TARGETS, default=list(pipeline.TARGETS))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    args = parser.parse_args()

    cfg = load_config()
    countries = args.countries or list(cfg.countries)
    unknown = set(countries) - set(cfg.countries)
    if unknown:
        parser.error(f"unknown countries: {', '.join(sorted(unknown))}")

    with pd.option_context("display.width", 200, "display.max_columns", None):
        if args.command == "tune":
            print(experiment.tune(cfg, countries, args.targets).to_string(index=False))
        elif args.command == "backtest":
            experiment.run_backtests(cfg, countries, args.targets, args.models)
        elif args.command in ("evaluate", "intervals", "significance"):
            table = getattr(experiment, args.command)(cfg, countries, args.targets)
            print(table.round(3).to_string(index=False))
        else:
            steps = ["fetch", "build", "validate"] if args.command == "all" else [args.command]
            for step in steps:
                getattr(pipeline, step)(cfg, countries)


if __name__ == "__main__":
    main()
