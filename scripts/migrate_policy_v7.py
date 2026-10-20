"""Expand a validated v6 checkpoint without changing its initial policy/value outputs."""
import argparse
import json
from pathlib import Path

from orca_feetech.policy_migration import migrate_v6_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate_v6_checkpoint(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
