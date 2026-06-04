# Colab setup — paste this entire file into ONE cell after mounting Drive.
# Clones code from GitHub if Drive only has zips/data (no .py files).

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ========== EDIT ==========
DRIVE = Path("/content/drive/MyDrive/thesis")
GITHUB_REPO = "https://github.com/Yiwen91/Ethereum_smart_contract_datast.git"
CONTENT_REPO = Path("/content/Ethereum_smart_contract_datast")  # clone target (fast local disk)
# ==========================


def _is_repo(p: Path) -> bool:
    return (p / "train_experiment.py").is_file() and (p / "experiment_utils.py").is_file()


def _search_train_experiment(root: Path, max_depth: int = 6) -> Path | None:
    """Find train_experiment.py under root (limited depth)."""
    root = root.resolve()
    if not root.is_dir():
        return None
    if _is_repo(root):
        return root
    if max_depth <= 0:
        return None
    try:
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                found = _search_train_experiment(child, max_depth - 1)
                if found is not None:
                    return found
    except PermissionError:
        pass
    return None


def _find_zip(root: Path, name: str) -> Path | None:
    if (root / name).is_file():
        return root / name
    for hit in root.rglob(name):
        if hit.is_file():
            return hit
    return None


def _count_sol_files(root: Path, limit: int = 5000) -> int:
    if not root.is_dir():
        return 0
    n = 0
    for _ in root.rglob("*.sol"):
        n += 1
        if n >= limit:
            break
    return n


def discover_contracts_dir(drive: Path, marker: str) -> Path | None:
    """
    Find contract_dataset_ethereum or smartbugs_wild/contracts anywhere under drive.
    Picks the directory with the most .sol files.
    """
    candidates: list[Path] = [
        drive / marker,
        drive / "Ethereum_smart_contract_datast" / marker,
    ]
    if marker == "contract_dataset_ethereum":
        for hit in drive.rglob("contract_dataset_ethereum"):
            if hit.is_dir() and hit.name == "contract_dataset_ethereum":
                candidates.append(hit)
    elif marker.endswith("contracts"):
        for hit in drive.rglob("contracts"):
            if hit.is_dir() and "smartbugs" in str(hit).lower():
                candidates.append(hit)
        for hit in drive.rglob("smartbugs_wild"):
            c = hit / "contracts"
            if c.is_dir():
                candidates.append(c)

    best: Path | None = None
    best_n = 0
    seen: set[str] = set()
    for c in candidates:
        key = str(c.resolve())
        if key in seen:
            continue
        seen.add(key)
        n = _count_sol_files(c)
        if n > best_n:
            best_n = n
            best = c.resolve()
    if best is not None:
        print(f"[data] {marker} -> {best}  (.sol count sample up to {best_n})")
    return best


def _resolved_count(split) -> int:
    return sum(
        1
        for r in split.records
        if r.get("contract_file") and Path(r["contract_file"]).is_file()
    )


def ensure_repo() -> Path:
    """Return repo root with all Python scripts; clone to /content if needed."""
    for candidate in [
        CONTENT_REPO,
        CONTENT_REPO / "Ethereum_smart_contract_datast",
        DRIVE / "Ethereum_smart_contract_datast",
        DRIVE / "Ethereum_smart_contract_datast" / "Ethereum_smart_contract_datast",
        DRIVE,
    ]:
        if candidate.is_dir():
            found = _search_train_experiment(candidate, max_depth=5)
            if found is not None:
                print(f"[repo] Using existing code at {found}")
                return found

    print(f"[repo] Cloning {GITHUB_REPO} -> {CONTENT_REPO}")
    if CONTENT_REPO.exists():
        subprocess.run(["rm", "-rf", str(CONTENT_REPO)], check=False)
    CONTENT_REPO.parent.mkdir(parents=True, exist_ok=True)
    code = subprocess.run(
        ["git", "clone", "--depth", "1", GITHUB_REPO, str(CONTENT_REPO)],
        capture_output=True,
        text=True,
    )
    if code.returncode != 0:
        raise RuntimeError(
            "git clone failed:\n"
            f"{code.stderr}\n"
            "Upload the project folder manually to Drive, or check the GitHub URL."
        )

    found = _search_train_experiment(CONTENT_REPO, max_depth=5)
    if found is None:
        raise FileNotFoundError(
            f"Clone completed but train_experiment.py not found under {CONTENT_REPO}"
        )
    print(f"[repo] Cloned OK -> {found}")
    return found


REPO = ensure_repo()
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

print("REPO =", REPO)
print("CWD  =", Path.cwd())

# Unpack splits: look for zips in REPO first, then anywhere under DRIVE
(REPO / "experiment_splits").mkdir(parents=True, exist_ok=True)
for zname in ["esc_primary_slither.zip", "smartbugs_secondary_slither.zip", "esc_primary.zip"]:
    zp = _find_zip(REPO, zname) or _find_zip(DRIVE, zname)
    if zp is None:
        if zname.startswith("esc_primary"):
            print(f"[warn] missing zip: {zname}")
        continue
    dest_name = "esc_primary_slither" if zname == "esc_primary.zip" else zname.replace(".zip", "")
    # esc_primary.zip contains esc_primary_slither/ inside
    subprocess.run(["unzip", "-q", "-o", str(zp), "-d", "experiment_splits"], cwd=REPO, check=False)
    print(f"[zip] {zp} -> experiment_splits/")

cb_zip = _find_zip(REPO, "codebert_base.zip") or _find_zip(DRIVE, "codebert_base.zip")
if cb_zip and not (REPO / "hf_models/codebert-base/config.json").is_file():
    subprocess.run(["unzip", "-q", "-o", str(cb_zip), "-d", str(REPO)], check=False)
    print(f"[zip] {cb_zip}")

# Dependencies (no apt solc)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "pip"], check=False)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "slither-analyzer",
        "scikit-learn",
        "ijson",
        "transformers",
        "shap",
        "matplotlib",
        "accelerate",
        "solc-select",
    ],
    check=False,
)
for v in ["0.4.26", "0.5.17", "0.6.12", "0.7.6", "0.8.20", "0.8.26"]:
    subprocess.run(["solc-select", "install", v], check=False)
subprocess.run(["solc-select", "use", "0.8.20"], check=False)
solc_bin = Path.home() / ".solc-select" / "bin"
if solc_bin.is_dir():
    os.environ["PATH"] = f"{solc_bin}:{os.environ.get('PATH', '')}"
subprocess.run(["solc", "--version"], check=False)

# Contracts on Drive (search thesis/ and thesis/Ethereum_smart_contract_datast/)
ESC_CONTRACTS = discover_contracts_dir(DRIVE, "contract_dataset_ethereum")
SB_CONTRACTS = discover_contracts_dir(DRIVE, "smartbugs_wild/contracts")
if SB_CONTRACTS is None:
    SB_CONTRACTS = discover_contracts_dir(DRIVE, "contracts")
if ESC_CONTRACTS is None:
    raise FileNotFoundError(
        f"No contract_dataset_ethereum found under {DRIVE}.\n"
        f"Upload/unzip it to e.g. {DRIVE / 'Ethereum_smart_contract_datast' / 'contract_dataset_ethereum'}"
    )
if SB_CONTRACTS is None:
    raise FileNotFoundError(
        f"No smartbugs_wild/contracts found under {DRIVE}.\n"
        f"Expected: {DRIVE / 'Ethereum_smart_contract_datast' / 'smartbugs_wild' / 'contracts'}"
    )

from experiment_utils import load_named_split  # noqa: E402

for split, cdir in [
    ("experiment_splits/esc_primary_slither", ESC_CONTRACTS),
    ("experiment_splits/smartbugs_secondary_slither", SB_CONTRACTS),
]:
    val_json = REPO / split / "val.json"
    if not val_json.is_file():
        raise FileNotFoundError(
            f"Missing {val_json}\n"
            f"Put esc_primary_slither.zip on Drive under {DRIVE} and re-run this cell."
        )
    s = load_named_split(
        "val",
        val_json,
        max_samples=100,
        project_root=REPO,
        contracts_dir=cdir,
    )
    ok = _resolved_count(s)
    print(f"{split}: resolved {ok}/100  contracts_dir={cdir}")
    if ok < 50:
        raise RuntimeError(
            f"Only {ok}/100 .sol files found.\n"
            f"Upload contract_dataset_ethereum to:\n  {DRIVE / 'contract_dataset_ethereum'}"
        )

# Expose for next cells
globals().update(REPO=REPO, DRIVE=DRIVE, ESC_CONTRACTS=ESC_CONTRACTS, SB_CONTRACTS=SB_CONTRACTS)
print("Setup OK. REPO =", REPO)
