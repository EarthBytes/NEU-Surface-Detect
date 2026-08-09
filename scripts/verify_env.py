"""Verify that the development environment is correctly configured."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
REQUIRED_PACKAGES = [
    ("torch", "PyTorch"),
    ("torchvision", "torchvision"),
    ("fastapi", "FastAPI"),
    ("mlflow", "MLflow"),
    ("pandas", "pandas"),
    ("numpy", "NumPy"),
    ("PIL", "Pillow"),
    ("yaml", "PyYAML"),
]


def check_python_version() -> bool:
    version = sys.version_info[:2]
    ok = version >= MIN_PYTHON
    label = f"{version[0]}.{version[1]}"
    required = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}+"
    status = "OK" if ok else "FAIL"
    print(f"[{status}] Python {label} (required: {required})")
    return ok


def check_package(module_name: str, display_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {display_name} ({version})")
        return True
    except ImportError:
        print(f"[FAIL] {display_name} — not installed")
        return False


def check_torch() -> bool:
    try:
        import torch

        print(f"[OK] PyTorch ({torch.__version__})")
        print(f"     CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError:
        print("[FAIL] PyTorch — not installed")
        return False


def check_project_layout() -> bool:
    root = Path(__file__).resolve().parent.parent
    expected = [
        "data_ingestion",
        "training",
        "models",
        "inference",
        "monitoring",
        ".github/workflows",
        "requirements.txt",
        "requirements-dev.txt",
    ]
    missing = [name for name in expected if not (root / name).exists()]
    if missing:
        print(f"[FAIL] Missing project paths: {', '.join(missing)}")
        return False
    print("[OK] Project structure")
    return True


def main() -> int:
    print("Verifying NEU-Surface-Detect development environment\n")

    checks = [
        check_python_version(),
        check_project_layout(),
        check_torch(),
    ]
    for module_name, display_name in REQUIRED_PACKAGES[1:]:
        if module_name in ("torch", "torchvision"):
            continue
        checks.append(check_package(module_name, display_name))

    try:
        import torchvision

        print(f"[OK] torchvision ({torchvision.__version__})")
    except ImportError:
        print("[FAIL] torchvision — not installed")
        checks.append(False)

    print()
    if all(checks):
        print("All checks passed.")
        return 0

    print("Some checks failed. Run: bash scripts/setup_env.sh")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
