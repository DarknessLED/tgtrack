#!/usr/bin/env python
"""Build the portable handoff package for `C:\\Projects\\track`.

Why: the work must be inheritable by another agent in another workspace, and pushed to
GitHub so it can be read from a second machine. A raw `git add -A` would push 70 MB of
model weights and 72 MB of demo videos; a docs-only export would lose the numbers that
make the report checkable. So the package is defined by an explicit allow-list:

    docs/      every .md            (reports, ledger, surveys, plans, reviews)
    src/       every .py            (the library)
    scripts/   every .py            (reproducible entry points)
    tests/     every .py            (the guarantees)
    configs/   every file
    results/   markdown, csv, and SMALL json summaries -- never weights, videos,
               per-frame dumps, logs or raw run trees

Everything excluded is listed in `MANIFEST.md` together with how to regenerate it
(download URLs / commands), because a handoff that hides its own gaps is worthless.

Usage:
    python scripts/build_handoff.py                 # dist/tgtrack_handoff/
    python scripts/build_handoff.py --zip           # ... plus a .zip next to it
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "dist"

# (source dir, include-subdirs?) -- copied wholesale with a suffix filter
TEXT_DIRS = {
    "docs": ("*.md",),
    "configs": ("*",),
}
CODE_DIRS = {
    "src": ("*.py",),
    "scripts": ("*.py",),
    "tests": ("*.py",),
}
ROOT_FILES = ("README.md", "conftest.py", "pytest.ini", "requirements.txt")

# results/ is selective: text artefacts that hold the measured numbers
RESULT_SUFFIXES = (".md", ".csv")
RESULT_JSON_ALLOW = ("summary.json", "report.json", "specs.json")
BIG_BYTES = 1_500_000

# never shipped, with the reason printed in the manifest
EXCLUDED_PATTERNS = [
    "data/**", "_x/**", "_p2/**", ".git/**", "**/__pycache__/**",
    "results/weights/**", "results/**/*.mp4", "results/**/*.pt", "results/**/*.onnx",
    "results/**/*.zip", "results/**/raw.json", "results/**/per_run.json",
    "results/**/logs/**", "results/gate_sweep/runs/**", "dist/**",
]


def _skip(path: Path) -> bool:
    rel = path.as_posix()
    for pat in EXCLUDED_PATTERNS:
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch("/" + rel, "*/" + pat):
            return True
        # directory-level patterns such as results/weights/**
        if pat.endswith("/**") and rel.startswith(pat[:-3]):
            return True
    return False


def collect() -> list[Path]:
    files: list[Path] = []
    for d, suffixes in TEXT_DIRS.items():
        for f in sorted((REPO / d).rglob("*")):
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            if any(fnmatch.fnmatch(f.name, s) for s in suffixes):
                files.append(f)
    for d, suffixes in CODE_DIRS.items():
        for f in sorted((REPO / d).rglob("*")):
            if f.is_file() and any(fnmatch.fnmatch(f.name, s) for s in suffixes):
                files.append(f)
    for name in ROOT_FILES:
        if (REPO / name).is_file():
            files.append(REPO / name)

    # results: text + small summaries only
    for f in sorted((REPO / "results").rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(REPO).as_posix()
        if _skip(f):
            continue
        keep = (
            f.suffix.lower() in RESULT_SUFFIXES
            or (f.suffix.lower() == ".json" and f.name in RESULT_JSON_ALLOW)
        )
        if keep and f.stat().st_size <= BIG_BYTES:
            files.append(f)
    return [f for f in files if not _skip(f)]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def build_manifest(stage: Path, copied: list[Path]) -> str:
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    by_dir: dict[str, list[Path]] = {}
    for f in copied:
        by_dir.setdefault(f.relative_to(REPO).parts[0], []).append(f)
    total = sum(f.stat().st_size for f in copied)
    L: list[str] = []
    a = L.append
    a("# MANIFEST — what is in this package")
    a("")
    a(f"Generated {now} by `scripts/build_handoff.py`. {len(copied)} files, "
      f"{total / 1e6:.2f} MB uncompressed.")
    a("")
    a("This file is generated; do not edit by hand.")
    a("")
    a("## Included")
    a("")
    a("| area | files | size | why it is here |")
    a("|---|---|---|---|")
    why = {
        "docs": "reports, the experiment ledger, surveys, plans, external reviews",
        "src": "the library itself (`tgtrack`)",
        "scripts": "every reproducible entry point; each experiment cites one",
        "tests": "the guarantees: 470+ tests encoding every measured behaviour",
        "configs": "tracker configs (ultralytics-compatible) used by the runs",
        "results": "the measured numbers as markdown/csv/small json summaries",
    }
    for d in sorted(by_dir):
        n = len(by_dir[d])
        s = sum(f.stat().st_size for f in by_dir[d])
        a(f"| `{d}/` | {n} | {s / 1e6:.2f} MB | {why.get(d, '—')} |")
    a(f"| root files | {sum(1 for f in copied if len(f.relative_to(REPO).parts) == 1)} | — | "
      "README, pytest config, requirements |")
    a("")
    a("## Deliberately EXCLUDED (and how to regenerate)")
    a("")
    a("| excluded | size on the origin machine | how to get it |")
    a("|---|---|---|")
    a("| `data/raw/`, `data/sequences/` | 6.51 GB used (17.6 GB of downloads) | "
      "dataset URLs + conversion commands are in `README.md` §Reproducing and "
      "`docs/BRIEF.md` §Datasets |")
    a("| `results/weights/*.pt`, `*.onnx` | 70 MB | `ultralytics` auto-download "
      "(`yolo26m.pt`, `yolo26n-reid.onnx`) or `hf-mirror.com`; exact URLs in `docs/BRIEF.md` |")
    a("| `results/**/*.mp4` demo videos | 72 MB | re-render with `scripts/run_video.py` / "
      "`scripts/demo_click.py` |")
    a("| `results/**/per_run.json`, `raw.json` | tens of MB | per-frame dumps; regenerate "
      "with the command recorded in the matching `docs/EXPERIMENTS.md` entry |")
    a("| `results/**/logs/`, `results/gate_sweep/runs/` | — | regenerated by re-running the "
      "sweep (`scripts/gate_sweep.py`) |")
    a("")
    a("## Integrity")
    a("")
    a("First 40 files by path, with sha256 prefix (12 hex), so a copy can be verified:")
    a("")
    a("| path | bytes | sha256[:12] |")
    a("|---|---|---|")
    for f in sorted(copied, key=lambda p: p.relative_to(REPO).as_posix())[:40]:
        rel = f.relative_to(REPO).as_posix()
        a(f"| `{rel}` | {f.stat().st_size} | `{sha(f)}` |")
    a("")
    a("(Full list is `find . -type f | sort` in the package root.)")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", action="store_true", help="also write a .zip next to the folder")
    ap.add_argument("--out", default="dist/tgtrack_handoff", help="staging directory")
    args = ap.parse_args(argv)

    stage = REPO / args.out
    # The staging directory is ALSO the published git repository, so a wholesale rmtree would
    # destroy .git and fail on read-only git objects (WinError 5). Purge everything except .git,
    # tolerate locked files, and prune stale entries after the copy.
    if stage.exists():
        for child in stage.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
    stage.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for f in collect():
        rel = f.relative_to(REPO)
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        copied.append(f)

    # keep the handoff note if it exists, and always (re)write the manifest
    handoff_src = REPO / "docs" / "HANDOFF.md"
    if handoff_src.is_file():
        shutil.copy2(handoff_src, stage / "docs" / "HANDOFF.md")
        copied.append(handoff_src)
    (stage / "MANIFEST.md").write_text(build_manifest(stage, copied), encoding="utf-8")

    # prune anything that is no longer on the allow-list (keeps the package exactly reproducible)
    wanted = {f.relative_to(REPO) for f in copied} | {Path("MANIFEST.md")}
    for f in sorted(stage.rglob("*"), reverse=True):
        if not f.is_file():
            continue
        rel = f.relative_to(stage)
        if rel.parts and rel.parts[0] == ".git":
            continue
        if rel not in wanted:
            try:
                f.unlink()
            except OSError:
                pass

    total = sum(f.stat().st_size for f in copied)
    print(f"[handoff] {len(copied)} files, {total / 1e6:.2f} MB -> {stage}")

    if args.zip:
        zpath = DIST / ("tgtrack_handoff_%s.zip" % datetime.now().strftime("%Y%m%d_%H%M"))
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for f in sorted(stage.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(stage.parent))
        print(f"[handoff] zip {zpath.stat().st_size / 1e6:.2f} MB -> {zpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
