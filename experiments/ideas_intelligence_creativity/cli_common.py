"""Helpers shared by run_experiment_1.py and run_experiment_3.py.

Output rules (SPEC.md section 5): the output directory defaults to ``EXP_DIR/results``; the publish
directory must resolve inside ``REPO_ROOT`` and is checked before anything is computed; JSON is
written strictly (no NaN or Infinity) with sorted keys, so a rerun is byte-identical.
"""

import json
import multiprocessing
import shutil
import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]      # the directory that contains experiments/


def check_publish_dir(arg):
    """Resolve ``--publish-plots-dir``; refuse (exit 2) unless it resolves inside REPO_ROOT.

    Symlinks are followed and ``..`` is normalised before the check. Returns None when ``arg`` is None.
    """
    if arg is None:
        return None
    target = Path(arg).expanduser().resolve()
    if not target.is_relative_to(REPO_ROOT.resolve()):
        print(f"refused: --publish-plots-dir {arg} resolves to {target}, which is outside the repository "
              f"root {REPO_ROOT}; nothing was written", file=sys.stderr)
        raise SystemExit(2)
    return target


def prepare_output_dir(arg) -> Path:
    """The output directory (default ``EXP_DIR/results``), created if missing."""
    out = Path(arg).expanduser() if arg is not None else EXP_DIR / "results"
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _to_builtin(obj):
    """Numpy scalars and arrays as Python types, recursively; tuples as lists."""
    if isinstance(obj, dict):
        return {_to_builtin(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_json_strict(obj, path) -> None:
    """Write ``obj`` as strict JSON (sorted keys, indent 1, trailing newline). Raises on NaN or Infinity.

    The text is built before the file is opened, so a refused object leaves no partial file.
    """
    text = json.dumps(_to_builtin(obj), allow_nan=False, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def publish_plots(plots_dir, publish_dir) -> list:
    """Copy every ``*.html`` in ``plots_dir`` byte-identically into ``publish_dir`` (created if missing).

    Nothing else is copied. Returns the destination paths.
    """
    publish_dir = Path(publish_dir)
    publish_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(Path(plots_dir).glob("*.html")):
        dst = publish_dir / src.name
        shutil.copyfile(src, dst)
        copied.append(dst)
    return copied


def parallel_map(fn, items, workers: int) -> list:
    """``[fn(x) for x in items]``, over a process pool when ``workers > 1``. Result order follows ``items``."""
    if workers < 1:
        raise ValueError(f"workers must be at least 1, not {workers}")
    items = list(items)
    if workers == 1:
        return [fn(x) for x in items]
    with multiprocessing.Pool(workers) as pool:
        return pool.map(fn, items)
