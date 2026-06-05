"""Shared helpers for the CUDA-to-SYCL pilot benchmark tooling.

All tools are designed to degrade gracefully: if a toolchain (nvcc, c2s/dpct,
icpx) or a device (NVIDIA GPU, SYCL device) is missing, the corresponding
pipeline step is recorded as a ``skipped_*`` status instead of aborting the
whole run.

Status vocabulary used across ``metadata.json`` ``status`` fields and reports
--------------------------------------------------------------------------
unknown                     not attempted yet
pass                        step succeeded
fail                        step attempted and failed
skipped_no_cuda_toolkit     nvcc not installed
skipped_no_cuda_gpu         no usable NVIDIA GPU at runtime
skipped_no_syclomatic       c2s / dpct not installed
skipped_no_sycl_compiler    icpx / clang++ (SYCL) not installed
skipped_no_sycl_device      no usable SYCL device at runtime
skipped_not_built           a prerequisite build step did not produce a binary
skipped_not_migrated        SYCLomatic output is missing
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone

# --- Repository layout ------------------------------------------------------

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
CASES_DIR = os.path.join(REPO_ROOT, "cases")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
CATEGORIES = ["easy", "medium", "hpc", "ai", "library_api"]


# --- Case discovery ---------------------------------------------------------

def iter_cases(category_filter=None, case_filter=None):
    """Yield absolute paths to every case directory holding a metadata.json.

    Cases are returned sorted by (category, case folder) for stable reports.
    """
    results = []
    for category in CATEGORIES:
        cat_dir = os.path.join(CASES_DIR, category)
        if not os.path.isdir(cat_dir):
            continue
        if category_filter and category != category_filter:
            continue
        for name in sorted(os.listdir(cat_dir)):
            case_dir = os.path.join(cat_dir, name)
            if not os.path.isdir(case_dir):
                continue
            if not os.path.isfile(os.path.join(case_dir, "metadata.json")):
                continue
            if case_filter and name != case_filter:
                continue
            results.append(case_dir)
    return results


# --- metadata.json I/O (order preserving) -----------------------------------

def load_metadata(case_dir):
    path = os.path.join(case_dir, "metadata.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh, object_pairs_hook=OrderedDict)


def save_metadata(case_dir, meta):
    path = os.path.join(case_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")


def set_status(meta, key, value):
    """Set ``meta['status'][key]`` defensively (creates the dict if needed)."""
    meta.setdefault("status", OrderedDict())[key] = value


# --- Toolchain / device detection -------------------------------------------

def which(name):
    return shutil.which(name)


def find_cuda_compiler():
    return which("nvcc")


def find_syclomatic():
    """Return the SYCLomatic / DPCT executable name if available."""
    for name in ("c2s", "dpct"):
        if which(name):
            return name
    return None


def find_sycl_compiler():
    for name in ("icpx", "icx", "clang++"):
        if which(name):
            return name
    return None


def has_nvidia_gpu():
    """Best-effort check for a usable NVIDIA GPU."""
    smi = which("nvidia-smi")
    if not smi:
        return False
    try:
        out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=15)
        return out.returncode == 0 and "GPU" in out.stdout
    except Exception:
        return False


def has_sycl_device():
    """Best-effort check for a usable SYCL device via sycl-ls."""
    ls = which("sycl-ls")
    if not ls:
        return False
    try:
        out = subprocess.run([ls], capture_output=True, text=True, timeout=15)
        return out.returncode == 0 and out.stdout.strip() != ""
    except Exception:
        return False


# --- Command execution with logging -----------------------------------------

def run_logged(command, cwd, log_path, timeout=600, env=None):
    """Run ``command`` (str via shell) in ``cwd``, tee combined output to a log.

    Returns (returncode, combined_output). returncode is 124 on timeout and
    127 if the command could not be started.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    header = (
        f"# command: {command}\n"
        f"# cwd:     {cwd}\n"
        f"# time:    {datetime.now(timezone.utc).isoformat()}\n"
        f"{'-' * 70}\n"
    )
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        body = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        body = (exc.stdout or "") + (exc.stderr or "") + f"\n[TIMEOUT after {timeout}s]\n"
        rc = 124
    except FileNotFoundError as exc:
        body = f"[command not found] {exc}\n"
        rc = 127
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(body)
        fh.write(f"\n{'-' * 70}\n# exit code: {rc}\n")
    return rc, body


def log_path(case_dir, name):
    return os.path.join(case_dir, "logs", name)


# --- Misc helpers ------------------------------------------------------------

def case_id_of(case_dir):
    return os.path.basename(case_dir)


def category_of(case_dir):
    return os.path.basename(os.path.dirname(case_dir))


def eprint(*args):
    print(*args, file=sys.stderr)
