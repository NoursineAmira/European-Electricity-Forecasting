import argparse

from . import pipeline
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="energy-forecast")
    parser.add_argument("command", choices=["fetch", "build", "validate", "all"])
    parser.add_argument("--countries", nargs="+", help="country codes (default: all in config)")
    args = parser.parse_args()

    cfg = load_config()
    countries = args.countries or list(cfg.countries)
    unknown = set(countries) - set(cfg.countries)
    if unknown:
        parser.error(f"unknown countries: {', '.join(sorted(unknown))}")

    steps = ["fetch", "build", "validate"] if args.command == "all" else [args.command]
    for step in steps:
        getattr(pipeline, step)(cfg, countries)


if __name__ == "__main__":
    main()
