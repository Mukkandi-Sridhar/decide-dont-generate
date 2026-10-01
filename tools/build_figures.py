"""Draw the book's figures: python tools/build_figures.py all   (or name sources: ch01 ch21 ...)

Each figures/src/<id>.py draws the figures for one chapter and writes them to figures/<id>/ as PDF and SVG, and the
numbers the text quotes to results/<id>.json. The source ids come from the book's first draft, so they don't match
the final chapter numbers: ch04.py draws Chapter 3's figures, ch21.py Chapter 14's, and so on."""

import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevkit import figs  # noqa: E402


def build(ch: str, only: set[str] | None = None) -> int:
    src = ROOT / "figures" / "src" / f"{ch}.py"
    if not src.exists():
        print(f"[skip] {ch}: no figure source")
        return 0
    spec = importlib.util.spec_from_file_location(f"figsrc_{ch}", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "record"):            # write results/<id>.json, the numbers the text quotes
        t = time.time()
        mod.record()
        print(f"  {ch}/record  ({time.time() - t:.1f}s)")
    n = 0
    for name, fn in figs.registered(ch):
        if name == "summary":                # one-page summaries are not part of the book any more
            continue
        if only and name not in only:
            continue
        t = time.time()
        f = fn()
        figs.save(f, ch, name)
        n += 1
        print(f"  {ch}/{name}  ({time.time() - t:.1f}s)")
    return n


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = set(args[i + 1:])
        args = args[:i]
    chs = sorted(p.stem for p in (ROOT / "figures" / "src").glob("*.py")) if args == ["all"] else args
    total = sum(build(c, only) for c in chs)
    print(f"built {total} figures")
