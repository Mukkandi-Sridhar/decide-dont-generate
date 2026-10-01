"""Sync every lab (labs/chNN.py, jupytext 'percent' format) to a notebook and execute it in mock mode.

usage: python tools/run_labs.py [ch01 ch21 ...] [--no-exec]
Exit code is non-zero if any notebook fails.
"""
import os
import sys
import time
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def run(path: Path, execute: bool = True) -> bool:
    nb = jupytext.read(path)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = path.with_suffix(".ipynb")
    if execute:
        env_before = dict(os.environ)
        os.environ.pop("JEVKIT_LIVE", None)
        os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
        t = time.time()
        try:
            NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {path.name}: {type(e).__name__}: {str(e)[-800:]}")
            return False
        finally:
            os.environ.clear()
            os.environ.update(env_before)
        print(f"ok   {path.name} ({time.time() - t:.0f}s)")
        # store notebooks without outputs so readers run them fresh
        for c in nb.cells:
            if c.cell_type == "code":
                c.outputs, c.execution_count = [], None
    nbformat.write(nb, out)
    return True


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    execute = "--no-exec" not in sys.argv
    labs = sorted((ROOT / "labs").glob("ch*.py"))
    if args:
        labs = [p for p in labs if p.stem in args]
    ok = all([run(p, execute) for p in labs])
    sys.exit(0 if ok else 1)
