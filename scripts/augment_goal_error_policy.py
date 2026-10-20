"""Add deterministic goal-error policy features with verified initial output preservation."""
import argparse
import json
from pathlib import Path

from orca_feetech.policy_migration import augment_goal_error_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(augment_goal_error_checkpoint(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
