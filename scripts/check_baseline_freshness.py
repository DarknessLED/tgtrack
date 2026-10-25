"""Audit whether a results directory is older than the sources it depends on.

Implements METHOD.md rule 2 ("fresh baseline rule"): an artifact is only quotable if no file it
depends on is newer than it.  Exit code 1 if any stale artifact is found, so it can gate a report.

    python scripts/check_baseline_freshness.py                # audit every results/* dir
    python scripts/check_baseline_freshness.py results/e029_dancetrack0007
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_ONLY = {"check_baseline_freshness.py"}
DEPS = [ROOT / "src" / "tgtrack", ROOT / "scripts", ROOT / "configs"]


def _newest_dep() -> tuple[float, str]:
    best, where = 0.0, ""
    for d in DEPS:
        if not d.exists():
            continue
        for f in d.rglob("*.py"):
            if "__pycache__" in f.parts:
                continue
            if f.name in AUDIT_ONLY:
                continue
            m = f.stat().st_mtime
            if m > best:
                best, where = m, str(f.relative_to(ROOT))
    return best, where


def _artifact_mtime(d: Path) -> tuple[float, str]:
    """Newest summary.json (preferred) or newest file in the dir."""
    summ = sorted(d.glob("summary.json"))
    if summ:
        return max(p.stat().st_mtime for p in summ), "summary.json"
    files = [p for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    if not files:
        return 0.0, "(empty)"
    p = max(files, key=lambda q: q.stat().st_mtime)
    return p.stat().st_mtime, str(p.relative_to(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="*", help="results directories (default: all of results/*)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    a = ap.parse_args(argv)

    dep_t, dep_where = _newest_dep()
    targets = [Path(x) for x in a.dirs] if a.dirs else sorted(
        p for p in (ROOT / "results").iterdir() if p.is_dir())
    rows = []
    for d in targets:
        d = d if d.is_absolute() else ROOT / d
        if not d.is_dir():
            continue
        art_t, art_where = _artifact_mtime(d)
        stale = art_t < dep_t
        rows.append({
            "dir": str(d.relative_to(ROOT)).replace("\\\\", "/"),
            "artifact": art_where,
            "artifact_age_s": round(dep_t - art_t, 1) if stale else -round(art_t - dep_t, 1),
            "stale": stale,
        })
    if a.json:
        print(json.dumps({"newest_dep": dep_where, "dirs": rows}, indent=2))
    else:
        print(f"newest dependency: {dep_where}")
        bad = [r for r in rows if r["stale"]]
        for r in sorted(rows, key=lambda q: -q["artifact_age_s"]):
            tag = "STALE " if r["stale"] else "ok    "
            print(f"  {tag} {r['dir']:<44} {r['artifact']:<14} {r['artifact_age_s']:>10.1f}s")
        print(f"\\n{len(bad)} stale of {len(rows)} directories")
    return 1 if any(r["stale"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
