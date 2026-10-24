import argparse

import pandas as pd

from . import experiment, pipeline
from .config import load_config
from .models import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(prog="energy-forecast")
    parser.add_argument("command", choices=["fetch", "build", "validate", "all", "backtest", "evaluate"])
    parser.add_argument("--countries", nargs="+", help="country codes (default: all in config)")
    parser.add_argument("--targets", nargs="+", choices=pipeline.TARGETS, default=list(pipeline.TARGETS))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()

    cfg = load_config()
    countries = args.countries or list(cfg.countries)
    unknown = set(countries) - set(cfg.countries)
    if unknown:
        parser.error(f"unknown countries: {', '.join(sorted(unknown))}")

    if args.command == "backtest":
        experiment.run_backtests(cfg, countries, args.targets, args.models)
    elif args.command == "evaluate":
        metrics = experiment.evaluate(cfg, countries, args.targets)
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(metrics.round(1).to_string(index=False))
    else:
        steps = ["fetch", "build", "validate"] if args.command == "all" else [args.command]
        for step in steps:
            getattr(pipeline, step)(cfg, countries)


if __name__ == "__main__":
    main()
