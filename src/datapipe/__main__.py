"""``python -m datapipe`` runs the full pipeline."""

import argparse

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="datapipe", description="Run the orders pipeline end to end.")
    parser.add_argument("--out", default="reports", help="output directory (default: reports)")
    parser.add_argument("--orders", type=int, default=500, help="number of synthetic orders")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name, path in run_pipeline(args.out, args.orders, args.seed).items():
        print(f"{name:<20} {path}")


if __name__ == "__main__":
    main()
