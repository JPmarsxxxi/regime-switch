"""Result caching keyed on the source of the modules a study depends on, so slow tests and report scripts share
one run per code version."""

import hashlib
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "reports" / "data"


def code_hash(deps):
    h = hashlib.sha256()
    for d in sorted(deps):
        h.update((ROOT / d).read_bytes())
    return h.hexdigest()[:12]


def cached(name, fn, deps):
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}-{code_hash(deps)}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    out = fn()
    path.write_bytes(pickle.dumps(out))
    return out


def verdict(ok):
    return "PASS" if ok else "FAIL"
