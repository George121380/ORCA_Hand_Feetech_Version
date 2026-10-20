"""Enable a training-only Gaussian-mean penalty without changing initial inference."""
import argparse
import json
from pathlib import Path

from orca_feetech.policy_regularization import regularize_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coefficient", type=float, default=.01)
    args = parser.parse_args()
    print(json.dumps(regularize_checkpoint(args.source, args.output, args.coefficient), indent=2))


if __name__ == "__main__":
    main()
