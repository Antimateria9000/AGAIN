from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / "var" / "cache"
TMP_ROOT = ROOT / "var" / "tmp"
PYTHON_CACHE = CACHE_ROOT / "python"
MATPLOTLIB_CACHE = CACHE_ROOT / "matplotlib"
RUNTIME_TMP = TMP_ROOT / "runtime"

for directory in (CACHE_ROOT, TMP_ROOT, PYTHON_CACHE, MATPLOTLIB_CACHE, RUNTIME_TMP):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))
os.environ.setdefault("TMP", str(RUNTIME_TMP))
os.environ.setdefault("TEMP", str(RUNTIME_TMP))
os.environ.setdefault("TMPDIR", str(RUNTIME_TMP))

if getattr(sys, "pycache_prefix", None) is None:
    sys.pycache_prefix = str(PYTHON_CACHE)
