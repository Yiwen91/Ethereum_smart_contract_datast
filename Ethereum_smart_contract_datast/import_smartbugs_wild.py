#!/usr/bin/env python3
"""
Import the SmartBugs Wild dataset into this project as the primary dataset.

By default, this script pulls only the contracts directory into:
  smartbugs_wild/contracts

Optional metadata files can also be copied with --include-metadata.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_URL = "https://github.com/smartbugs/smartbugs-wild.git"
METADATA_FILES = [
    "contracts.csv.tar.gz",
    "balances.json",
    "duplicates.json",
    "nb_lines.csv",
    "README.md",
    "LICENSE",
]


def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Import SmartBugs Wild into smartbugs_wild/")
    parser.add_argument(
        "--dest",
        default="smartbugs_wild",
        help="Destination directory inside the project (default: smartbugs_wild)",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Also copy top-level metadata files such as contracts.csv.tar.gz",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    dest_root = project_root / args.dest
    contracts_dest = dest_root / "contracts"
    dest_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="smartbugs_wild_") as tmp:
        clone_dir = Path(tmp) / "smartbugs-wild"
        # A shallow clone is more reliable here than mixing sparse-checkout paths
        # for directories and top-level files on Windows.
        run(["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)])

        copy_tree(clone_dir / "contracts", contracts_dest)

        if args.include_metadata:
            for name in METADATA_FILES:
                src = clone_dir / name
                if src.exists():
                    dst = dest_root / name
                    if src.is_dir():
                        copy_tree(src, dst)
                    else:
                        shutil.copy2(src, dst)

    print(f"Imported SmartBugs Wild contracts into {contracts_dest}")
    if args.include_metadata:
        print(f"Copied metadata files into {dest_root}")


if __name__ == "__main__":
    main()
