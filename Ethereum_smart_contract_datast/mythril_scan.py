#!/usr/bin/env python3
"""
Mythril-based vulnerability scanning for smart contract datasets.

Supplements the primary dataset by scanning contracts with Mythril (static analysis).
Produces standardized labels compatible with the main pipeline.

Research targets:
- Reentrancy (SWC-107)
- Integer overflow/underflow (SWC-101)
- Transaction-ordering dependence / front-running (SWC-114)
- Uninitialized storage pointers (SWC-109)
- Unchecked external calls (SWC-104)

Usage:
  py mythril_scan.py <contracts_dir> [--output-dir OUTPUT] [--copy-staging]
  py mythril_scan.py --from-json standardized_dataset.json  # scan files from existing dataset
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
MYTHRIL_SWC_MAP = SCRIPT_DIR / "mythril_swc_mapping.json"
MYTHRIL_DOCKER_CACHE = SCRIPT_DIR / ".mythril_docker_cache"


def load_mythril_swc_mapping() -> Dict:
    """Load Mythril issue title -> SWC mapping."""
    if MYTHRIL_SWC_MAP.exists():
        with open(MYTHRIL_SWC_MAP, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("mythril_to_swc", {})
    return {}


def get_solc_version_from_pragma(sol_file: str) -> Optional[str]:
    """Extract Solidity version from pragma (e.g. ^0.4.24 -> 0.4.26). Returns None if not found."""
    try:
        with open(sol_file, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(1024)
        m = re.search(r"pragma\s+solidity\s+[\^>=<]?\s*([0-9]+)\.([0-9]+)\.([0-9]+)", head, re.I)
        if m:
            major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if major == 0 and minor == 4:
                return "0.4.26"  # Safe for most 0.4.x
            if major == 0 and minor == 5:
                return "0.5.17"
            return f"{major}.{minor}.{patch}"
    except Exception:
        pass
    return None


def is_docker_available() -> bool:
    """Return True only if Docker CLI exists and the daemon is reachable."""
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and bool((result.stdout or "").strip())
    except Exception:
        return False


def find_mythril():
    """Return ('myth', None) or ('docker', 'mythril/myth') when the backend is usable."""
    for cmd in ("myth", "mythril"):
        if shutil.which(cmd):
            return (cmd, None)
    if is_docker_available():
        return ("docker", "mythril/myth")
    return (None, None)


def _parse_mythril_json(stdout: str, stderr: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Parse Mythril JSON from stdout even when the command exits non-zero."""
    output = (stdout or "").strip()
    if output:
        try:
            return json.loads(output), None
        except json.JSONDecodeError:
            pass
    error = (stderr or "").strip() or output or "Unknown error"
    return None, error


def _ensure_docker_cache_dirs() -> Tuple[str, str]:
    """Create persistent host cache directories for dockerized solc installs."""
    svm_dir = MYTHRIL_DOCKER_CACHE / "svm"
    solcx_dir = MYTHRIL_DOCKER_CACHE / "solcx"
    svm_dir.mkdir(parents=True, exist_ok=True)
    solcx_dir.mkdir(parents=True, exist_ok=True)
    return (
        str(svm_dir.resolve()).replace("\\", "/"),
        str(solcx_dir.resolve()).replace("\\", "/"),
    )


def _build_docker_mythril_cmd(sol_file: str, docker_image: str, solv: Optional[str]) -> List[str]:
    """Build a Docker command that can install and use the requested solc version."""
    sol_path = Path(sol_file).resolve()
    host_dir = str(sol_path.parent).replace("\\", "/")
    filename = sol_path.name
    svm_dir, solcx_dir = _ensure_docker_cache_dirs()

    if solv:
        inner = (
            f"svm install {solv} >/dev/null 2>&1; "
            "sync-svm-solc-versions-with-solcx >/dev/null 2>&1; "
            f"myth analyze /tmp/{filename} -o json --solv {solv}"
        )
        return [
            "docker", "run", "--rm",
            "-v", f"{host_dir}:/tmp",
            "-v", f"{svm_dir}:/home/mythril/.svm",
            "-v", f"{solcx_dir}:/home/mythril/.solcx",
            docker_image,
            "bash", "-lc", inner,
        ]

    return [
        "docker", "run", "--rm",
        "-v", f"{host_dir}:/tmp",
        "-v", f"{svm_dir}:/home/mythril/.svm",
        "-v", f"{solcx_dir}:/home/mythril/.solcx",
        docker_image,
        "analyze", f"/tmp/{filename}", "-o", "json",
    ]


def _run_mythril_cmd(sol_file: str, myth_cmd: str, docker_image: Optional[str], solv: Optional[str], timeout: int) -> Tuple[Optional[Dict], Optional[str]]:
    """Run Mythril. Returns (parsed_json, error_message)."""
    try:
        if myth_cmd == "docker" and docker_image:
            cmd = _build_docker_mythril_cmd(sol_file, docker_image, solv)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        else:
            cmd = [myth_cmd, "analyze", str(sol_file), "-o", "json"]
            if solv:
                cmd.extend(["--solv", solv])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path(sol_file).parent),
            )
        report, parse_error = _parse_mythril_json(result.stdout, result.stderr)
        if report is not None:
            return report, None
        return None, parse_error
    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except json.JSONDecodeError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


def run_mythril_analyze(sol_file: str, timeout: int = 600) -> Optional[Dict]:
    """
    Run Mythril analyze on a .sol file. Returns parsed JSON or None.
    Uses native myth if available, otherwise Docker (mythril/myth).
    For 0.4.x/0.5.x contracts, passes --solv so Mythril compiles correctly.
    """
    myth_cmd, docker_image = find_mythril()
    if not myth_cmd:
        return None
    solv = get_solc_version_from_pragma(sol_file)
    report, err = _run_mythril_cmd(sol_file, myth_cmd, docker_image, solv, timeout)
    if report is not None and report.get("success", True):
        return report
    if report and "SolidityVersionMismatch" in str(report.get("error", "")) and not solv:
        solv = "0.4.26"
        report, _ = _run_mythril_cmd(sol_file, myth_cmd, docker_image, solv, timeout)
    if report is not None:
        return report
    return {"success": False, "error": err or "Unknown error", "issues": []}


def parse_mythril_issues(report: Dict, swc_map: Dict) -> List[Dict]:
    """
    Parse Mythril issues into standardized format: {swc_id, title, description, ...}.
    """
    issues = report.get("issues", []) or report.get("analysis", {}).get("issues", [])
    out = []
    for issue in issues:
        title = issue.get("title", "") or issue.get("name", "").strip()
        swc_id = issue.get("swc-id") or issue.get("swc_id", "")
        desc = issue.get("description", "")
        severity = issue.get("severity", "")
        loc = issue.get("location", "")
        if not swc_id and title:
            for key, val in swc_map.items():
                if key.lower() in title.lower():
                    swc_id = val.get("swc_id", "")
                    break
        out.append({
            "title": title,
            "swc_id": swc_id,
            "description": desc,
            "severity": severity,
            "location": loc,
        })
    return out


def scan_contract(sol_file: str, swc_map: Dict, timeout: int = 600) -> Dict:
    """
    Scan one contract with Mythril. Returns:
    { success, file, issues, swc_ids, vulnerabilities }
    """
    report = run_mythril_analyze(sol_file, timeout=timeout)
    if not report or not report.get("success", True):
        return {
            "success": False,
            "file": sol_file,
            "issues": [],
            "swc_ids": [],
            "vulnerabilities": [],
            "error": (report or {}).get("error", "Unknown error"),
        }
    issues = parse_mythril_issues(report, swc_map)
    swc_ids = list(dict.fromkeys(i["swc_id"] for i in issues if i["swc_id"]))
    vulns = [i["title"] for i in issues if i["title"]]
    return {
        "success": True,
        "file": sol_file,
        "issues": issues,
        "swc_ids": swc_ids,
        "vulnerabilities": vulns,
    }


def copy_contracts(
    src_dir: str,
    dest_dir: str,
    pattern: str = "*.sol",
    recursive: bool = True,
) -> List[Path]:
    """Copy .sol files to staging directory for Mythril scan."""
    src = Path(src_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if recursive:
        files = list(src.rglob(pattern))
    else:
        files = list(src.glob(pattern))
    copied = []
    for f in files:
        if f.is_file():
            rel = f.relative_to(src)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            copied.append(target)
    return copied


def scan_directory(
    contracts_dir: str,
    output_dir: str = "mythril_scan_output",
    copy_staging: bool = False,
    max_files: Optional[int] = None,
    timeout: int = 600,
) -> Dict:
    """
    Scan all .sol files in directory. Optionally copy to staging first.
    Returns aggregated results.
    """
    contracts_dir = Path(contracts_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    swc_map = load_mythril_swc_mapping()

    if copy_staging:
        staging = output_path / "staging"
        staging.mkdir(exist_ok=True)
        paths = copy_contracts(str(contracts_dir), str(staging))
        scan_root = staging
    else:
        paths = list(contracts_dir.rglob("*.sol"))
        scan_root = contracts_dir

    results = []
    total = len(paths)
    if max_files:
        paths = paths[: max_files]
    for i, p in enumerate(paths, 1):
        if i % 50 == 0:
            print(f"  Scanning {i}/{min(total, max_files or total)}...")
        r = scan_contract(str(p), swc_map, timeout=timeout)
        r["file"] = str(p)
        results.append(r)

    # Save results
    out_file = output_path / "mythril_scan_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"contracts": results, "swc_mapping": swc_map}, f, indent=2)

    # Summary
    with_vulns = [r for r in results if r.get("vulnerabilities")]
    summary = {
        "total_scanned": len(results),
        "failed_scans": sum(1 for r in results if not r.get("success")),
        "with_vulnerabilities": len(with_vulns),
        "total_issues": sum(len(r.get("issues", [])) for r in results),
        "swc_counts": {},
    }
    for r in results:
        for sid in r.get("swc_ids", []):
            summary["swc_counts"][sid] = summary["swc_counts"].get(sid, 0) + 1

    summary_file = output_path / "mythril_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"results": results, "summary": summary, "output_dir": str(output_path)}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Scan contracts with Mythril for vulnerability labeling (research supplement)."
    )
    parser.add_argument("input_dir", nargs="?", help="Directory containing .sol files")
    parser.add_argument(
        "--output-dir", "-o",
        default="mythril_scan_output",
        help="Output directory for scan results (default: mythril_scan_output)",
    )
    parser.add_argument(
        "--copy-staging",
        action="store_true",
        help="Copy contracts to staging directory before scanning",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files to scan (for testing)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-contract timeout in seconds (default: 600)",
    )
    args = parser.parse_args()

    myth_cmd, _ = find_mythril()
    if not myth_cmd:
        if shutil.which("docker"):
            print("Docker is installed, but the Docker engine is not running.")
            print("Start Docker Desktop, wait until it shows 'Engine running', then rerun the scan.")
            print("Optional check: docker version")
        else:
            print("Mythril not found. On Windows, pip install often fails (pyethash/ckzg).")
            print("Use Docker instead:")
            print("  1. Install Docker Desktop: https://www.docker.com/products/docker-desktop")
            print("  2. Start Docker Desktop and wait for the engine to run")
            print("  3. Run: docker pull mythril/myth")
            print("  4. This script will auto-detect Docker and use it.")
            print("Alternatively (Linux/Mac): py -m pip install mythril")
        sys.exit(1)

    if not args.input_dir:
        print("Usage: py mythril_scan.py <contracts_dir> [--output-dir OUTPUT] [--copy-staging]")
        sys.exit(1)

    if not Path(args.input_dir).is_dir():
        print(f"Not a directory: {args.input_dir}")
        sys.exit(1)

    if myth_cmd == "docker":
        print("Using Mythril via Docker (mythril/myth).")
    print(f"Scanning contracts in {args.input_dir} with Mythril...")
    data = scan_directory(
        args.input_dir,
        output_dir=args.output_dir,
        copy_staging=args.copy_staging,
        max_files=args.max_files,
        timeout=args.timeout,
    )
    s = data["summary"]
    print(f"\nDone. Scanned {s['total_scanned']} contracts, {s['with_vulnerabilities']} with vulnerabilities.")
    print(f"Results saved to {data['output_dir']}/mythril_scan_results.json")
    print(f"Summary: {data['output_dir']}/mythril_summary.json")
    if s.get("swc_counts"):
        print("SWC counts:", s["swc_counts"])


if __name__ == "__main__":
    main()
