#!/usr/bin/env python3
"""Vendor upstream paper repositories into external/ with provenance.

Reads the reproduction shortlist, clones every paper that has a code link,
records the original URL + resolved HEAD commit into external/MANIFEST.json,
then strips each clone's .git so it becomes part of this repo.

The SHA capture matters: once .git is gone we can no longer tell which commit
we vendored, so the manifest is the only reproducibility anchor.

Usage:
    python scripts/vendor_repos.py            # clone everything missing
    python scripts/vendor_repos.py --force    # re-clone even if dir exists
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHORTLIST = ROOT / "data" / "interp_final_selection.json"
VENDOR_DIR = ROOT / "external"
# Manifest lives outside the (gitignored) external/ so provenance stays tracked.
MANIFEST = ROOT / "data" / "vendored_repos.json"


def clean_url(url: str) -> str:
    """Strip trailing punctuation/whitespace and normalize to a clone URL."""
    url = url.strip().rstrip(".,);")
    if not url.endswith(".git"):
        url = url + ".git"
    return url


def slug_for(url: str) -> str:
    """owner-repo slug, e.g. ApolloResearch/deception-detection -> deception-detection."""
    m = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?$", url)
    if not m:
        return re.sub(r"\W+", "-", url).strip("-")
    return m.group(2)


def load_targets() -> list[dict]:
    data = json.loads(SHORTLIST.read_text())
    targets = []
    for p in data["papers"]:
        code = p.get("code")
        if not code:
            continue
        url = clean_url(code)
        targets.append(
            {
                "title": p["title"],
                "rank": p.get("rank"),
                "focus": p.get("focus"),
                "url": url,
                "slug": slug_for(url),
            }
        )
    return targets


def run(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        cmd, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-clone existing dirs")
    args = ap.parse_args()

    VENDOR_DIR.mkdir(exist_ok=True)
    targets = load_targets()
    print(f"{len(targets)} repos to vendor -> {VENDOR_DIR.relative_to(ROOT)}/\n")

    records = []
    for t in targets:
        dest = VENDOR_DIR / t["slug"]
        rec = {**t, "status": None, "commit": None}

        if dest.exists():
            if args.force:
                shutil.rmtree(dest)
            else:
                print(f"  skip  {t['slug']:<40} (exists)")
                rec["status"] = "exists"
                records.append(rec)
                continue

        try:
            # shallow clone: we discard history anyway, this is fast
            run(["git", "clone", "--depth", "1", t["url"], str(dest)])
            rec["commit"] = run(["git", "rev-parse", "HEAD"], cwd=dest)
            shutil.rmtree(dest / ".git")
            rec["status"] = "vendored"
            print(f"  ok    {t['slug']:<40} {rec['commit'][:10]}")
        except subprocess.CalledProcessError as e:
            rec["status"] = "failed"
            rec["error"] = (e.stderr or "").strip().splitlines()[-1:] or ["unknown"]
            print(f"  FAIL  {t['slug']:<40} {rec['error']}")
            if dest.exists():
                shutil.rmtree(dest)
        records.append(rec)

    manifest = {
        "generated": date.today().isoformat(),
        "source": str(SHORTLIST.relative_to(ROOT)),
        "note": "Upstream repos vendored with .git stripped; commit = HEAD at clone time.",
        "repos": records,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    ok = sum(r["status"] in ("vendored", "exists") for r in records)
    fail = sum(r["status"] == "failed" for r in records)
    print(f"\n{ok} ready, {fail} failed -> {MANIFEST.relative_to(ROOT)}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
