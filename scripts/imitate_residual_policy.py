"""One fixed512-update offline actor imitation; does not count as RL or acceptance."""
import argparse
import json
from pathlib import Path

from orca_feetech.policy_imitation import clone_from_demonstrations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    proof = clone_from_demonstrations(args.source, args.output, args.reports, device=args.device)
    print(json.dumps({k:v for k,v in proof.items() if k != "datasets"}, indent=2))


if __name__ == "__main__":
    main()
