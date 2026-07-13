from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import psutil


@dataclass(frozen=True)
class RuntimeInfo:
    is_kaggle: bool
    device: str
    gpu_name: str
    cuda_available: bool
    project_root: Path
    data_root: Path
    output_root: Path
    log_root: Path


def is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:
        return "none"


def detect_runtime(project_root: Optional[str | Path] = None) -> RuntimeInfo:
    root = Path(project_root or Path.cwd()).resolve()
    kaggle = is_kaggle()
    cuda = False
    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    device = "cuda" if cuda else "cpu"
    gpu_name = detect_gpu_name() if cuda else "none"
    data_root = Path(os.environ.get("DATA_ROOT", root / "data"))
    output_root = Path(os.environ.get("OUTPUT_ROOT", root / "results"))
    log_root = Path(os.environ.get("LOG_ROOT", root / "logs"))
    return RuntimeInfo(
        is_kaggle=kaggle,
        device=device,
        gpu_name=gpu_name,
        cuda_available=cuda,
        project_root=root,
        data_root=data_root,
        output_root=output_root,
        log_root=log_root,
    )


def memory_snapshot() -> Dict[str, float]:
    vm = psutil.virtual_memory()
    return {
        "cpu_percent": float(psutil.cpu_percent(interval=None)),
        "ram_percent": float(vm.percent),
        "ram_used_gb": float(vm.used / (1024**3)),
        "ram_total_gb": float(vm.total / (1024**3)),
    }
