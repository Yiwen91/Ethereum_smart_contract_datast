#!/usr/bin/env python3
"""
Rebuild Colab upload zips from local experiment_splits and hf_models.

Outputs (in repo root):
  esc_primary_slither.zip   — Slither-labeled ESC splits
  smartbugs_secondary_slither.zip — Slither-labeled SmartBugs splits
  esc_primary.zip           — alias copy of esc_primary_slither.zip (legacy name)
  codebert_base.zip         — offline CodeBERT weights for Colab
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def _zip_dir(
    source_dir: Path,
    zip_path: Path,
    arc_prefix: str,
    *,
    compresslevel: int = 6,
) -> None:
    """Zip all files under source_dir using arc_prefix/filename inside the archive."""
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_suffix(zip_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    files = sorted(p for p in source_dir.rglob("*") if p.is_file())
    with zipfile.ZipFile(
        tmp,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compresslevel,
    ) as zf:
        for path in files:
            rel = path.relative_to(source_dir).as_posix()
            arcname = f"{arc_prefix}/{rel}"
            zf.write(path, arcname=arcname)

    if zip_path.exists():
        zip_path.unlink()
    tmp.replace(zip_path)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {zip_path.name}: {len(files)} files, {size_mb:.1f} MB")


def _zip_hf_model(model_dir: Path, zip_path: Path) -> None:
    """Zip as hf_models/codebert-base/... (matches --codebert-model-name hf_models/codebert-base)."""
    model_dir = model_dir.resolve()
    hf_root = model_dir.parent.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_suffix(zip_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    files = sorted(p for p in model_dir.rglob("*") if p.is_file())
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            rel = path.relative_to(hf_root.parent).as_posix()
            zf.write(path, arcname=rel)

    if zip_path.exists():
        zip_path.unlink()
    tmp.replace(zip_path)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {zip_path.name}: {len(files)} files, {size_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack Colab upload zip files.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root (default: script directory).",
    )
    parser.add_argument(
        "--skip-codebert",
        action="store_true",
        help="Skip rebuilding codebert_base.zip.",
    )
    parser.add_argument(
        "--skip-splits",
        action="store_true",
        help="Skip rebuilding split zips.",
    )
    args = parser.parse_args()
    root: Path = args.root.resolve()
    splits = root / "experiment_splits"

    if not args.skip_splits:
        esc = splits / "esc_primary_slither"
        sb = splits / "smartbugs_secondary_slither"
        if not esc.is_dir():
            raise SystemExit(f"Missing {esc}")
        if not sb.is_dir():
            raise SystemExit(f"Missing {sb}")

        esc_zip = root / "esc_primary_slither.zip"
        sb_zip = root / "smartbugs_secondary_slither.zip"
        _zip_dir(esc, esc_zip, "esc_primary_slither")
        _zip_dir(sb, sb_zip, "smartbugs_secondary_slither")

        legacy = root / "esc_primary.zip"
        shutil.copy2(esc_zip, legacy)
        print(f"Copied {esc_zip.name} -> {legacy.name} (legacy Colab name)")

    if not args.skip_codebert:
        model = root / "hf_models" / "codebert-base"
        if not model.is_dir():
            raise SystemExit(
                f"Missing {model}. Download with:\n"
                "  huggingface-cli download microsoft/codebert-base "
                f"--local-dir {model}"
            )
        _zip_hf_model(model, root / "codebert_base.zip")

    print("Done.")


if __name__ == "__main__":
    main()
