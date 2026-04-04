"""
Load initial_inputs.npy and initial_outputs.npy from every function_* folder
under this directory.

Requires: pip install numpy
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np


class FunctionArrays(TypedDict):
    inputs: np.ndarray
    outputs: np.ndarray
    dir: Path


def _initial_data_root() -> Path:
    return Path(__file__).resolve().parent


def discover_function_dirs(root: Path | None = None) -> list[Path]:
    """Return function_* directories under root, sorted by numeric suffix."""
    base = root if root is not None else _initial_data_root()

    def sort_key(p: Path) -> tuple[int, str]:
        name = p.name
        if name.startswith("function_"):
            suffix = name.removeprefix("function_")
            if suffix.isdigit():
                return (int(suffix), name)
        return (10**9, name)

    dirs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("function_")]
    return sorted(dirs, key=sort_key)


def load_one_function(function_dir: Path) -> FunctionArrays:
    """Load inputs and outputs from a single function_* directory."""
    inputs_path = function_dir / "initial_inputs.npy"
    outputs_path = function_dir / "initial_outputs.npy"
    if not inputs_path.is_file():
        raise FileNotFoundError(f"Missing {inputs_path}")
    if not outputs_path.is_file():
        raise FileNotFoundError(f"Missing {outputs_path}")
    return {
        "inputs": np.load(inputs_path),
        "outputs": np.load(outputs_path),
        "dir": function_dir,
    }


def load_all_functions(root: Path | None = None) -> dict[str, FunctionArrays]:
    """
    Load numpy arrays for each function_* folder.

    Keys are directory names (e.g. 'function_1').
    """
    base = root if root is not None else _initial_data_root()
    out: dict[str, FunctionArrays] = {}
    for d in discover_function_dirs(base):
        out[d.name] = load_one_function(d)
    return out


def main() -> None:
    data = load_all_functions()
    for name, bundle in data.items():
        x = bundle["inputs"]
        y = bundle["outputs"]
        print(f"{name}: inputs {x.shape} {x.dtype}, outputs {y.shape} {y.dtype}")


if __name__ == "__main__":
    main()
