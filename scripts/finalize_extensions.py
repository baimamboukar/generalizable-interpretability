#!/usr/bin/env python3
"""Validate the web-agent extension links and merge them into the final dataset.

Steps:
  1. Validate every extension `link` actually resolves. arXiv links are checked
     against the arXiv API (existence + title match, to catch hallucinated ids);
     all other links are checked over HTTP.
  2. Rebuild each shortlisted paper (full selection record) with a validated
     `extensions` field.
  3. Write interp_final.json (+ a console validation report).

Inputs : interp_selection.json (the 20 selected, with metadata)
         interp_with_extensions.json (web-agent result: papers -> extensions)
Output : interp_final.json
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
SEL = DATA / "interp_selection.json"
EXT = DATA / "interp_with_extensions.json"
OUT = DATA / "interp_final.json"

UA = "interp-link-validator (mohamedgaye.mhd@gmail.com)"
ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})")
STOP = set("a an the of for and or to in on with via using through from into is are we our "
           "this that study paper model models language large".split())


def toks(t):
    return {w for w in re.sub(r"[^a-z0-9 ]", " ", (t or "").lower()).split()
            if w and w not in STOP and len(w) > 2}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def http_check(url):
    """Return (ok, status, note). ok=None means uncertain (blocked/ambiguous)."""
    req = urllib.request.Request(url, method="GET",
                                 headers={"User-Agent": UA, "Range": "bytes=0-2047"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return True, r.status, "reachable"
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 429):
            return None, e.code, "exists but blocked to bots"
        if e.code in (404, 410):
            return False, e.code, "not found"
        return None, e.code, "uncertain"
    except Exception as e:
        return None, None, f"unreachable: {type(e).__name__}"


def arxiv_lookup(ids):
    """Batch arxiv API: id -> title (or None if missing)."""
    out = {}
    for i in range(0, len(ids), 25):
        chunk = ids[i:i + 25]
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)})
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}),
                                         timeout=40).read()
            root = ET.fromstring(raw)
            ns = "{http://www.w3.org/2005/Atom}"
            for e in root.findall(ns + "entry"):
                idt = e.findtext(ns + "id") or ""
                m = ARXIV_ID.search(idt)
                title = (e.findtext(ns + "title") or "").strip().replace("\n", " ")
                # arxiv returns an error entry without a real id for bad ids
                if m and "arxiv.org/abs" in idt:
                    out[m.group(1)] = title
        except Exception as ex:
            print(f"  arxiv API error: {ex}", file=sys.stderr)
        time.sleep(3)
    return out


def validate_all(papers):
    # gather links
    arxiv_ids, other_links = set(), set()
    for p in papers:
        for e in p.get("extensions", []):
            link = (e.get("link") or "").strip()
            if not link:
                continue
            m = ARXIV_ID.search(link) if ("arxiv" in link.lower() or "48550" in link) else None
            if m:
                arxiv_ids.add(m.group(1))
            else:
                other_links.add(link)

    arxiv_titles = arxiv_lookup(sorted(arxiv_ids)) if arxiv_ids else {}
    http_status = {}
    for link in sorted(other_links):
        ok, status, note = http_check(link)
        http_status[link] = {"ok": ok, "status": status, "note": note}
        print(f"  http {str(status):>4} {note:<28} {link[:70]}", file=sys.stderr)
        time.sleep(0.3)

    # annotate
    for p in papers:
        for e in p.get("extensions", []):
            link = (e.get("link") or "").strip()
            m = ARXIV_ID.search(link) if ("arxiv" in link.lower() or "48550" in link) else None
            if not link:
                e["link_check"] = {"ok": False, "note": "no link provided"}
            elif m:
                aid = m.group(1)
                if aid in arxiv_titles:
                    tj = round(jaccard(toks(e.get("title")), toks(arxiv_titles[aid])), 2)
                    e["link_check"] = {
                        "ok": True, "source": "arxiv", "arxiv_id": aid,
                        "arxiv_title": arxiv_titles[aid], "title_match": tj,
                        "note": "ok" if tj >= 0.4 else "WARNING: title mismatch — verify link",
                    }
                else:
                    e["link_check"] = {"ok": False, "source": "arxiv", "arxiv_id": aid,
                                       "note": "arXiv id not found"}
            else:
                st = http_status.get(link, {})
                e["link_check"] = {"ok": st.get("ok"), "source": "http",
                                   "status": st.get("status"), "note": st.get("note")}
    return arxiv_titles, http_status


def main():
    sel = json.loads(SEL.read_text(encoding="utf-8"))
    selrecs = {r["rank"]: r for r in (sel["selection"] if isinstance(sel, dict) else sel)}
    ext = json.loads(EXT.read_text(encoding="utf-8"))
    papers = ext["papers"]

    validate_all(papers)

    final = []
    n_checked = n_ok = n_uncertain = n_bad = n_removed = 0
    for wp in sorted(papers, key=lambda p: p["rank"]):
        base = dict(selrecs.get(wp["rank"], {"title": wp["title"], "rank": wp["rank"]}))
        kept = []
        for e in wp.get("extensions", []):
            n_checked += 1
            chk = e.get("link_check", {})
            note = chk.get("note", "") or ""
            if "mismatch" in note:           # hallucinated: id resolves to a different paper
                n_removed += 1
                continue                      # drop it
            if chk.get("ok") is True:
                n_ok += 1
            elif chk.get("ok") is None:
                n_uncertain += 1
            else:
                n_bad += 1
            kept.append(e)
        # recompute counts from the kept extensions so they stay consistent
        counts = {"reproduce": 0, "extend": 0, "improve": 0}
        for e in kept:
            for t in e.get("type", []):
                if t in counts:
                    counts[t] += 1
        counts["total"] = len(kept)
        base.pop("abstract", None)           # keep final compact
        base["extensions"] = kept
        base["counts"] = counts
        base["n_extensions"] = len(kept)
        base["search_notes"] = wp.get("search_notes", "")
        final.append(base)

    payload = {
        "metadata": {
            "title": "Reproduction shortlist with validated extension studies",
            "built_by": "scripts/finalize_extensions.py",
            "sources": {
                "selection": "data/interp_selection.json (scripts/select_interp_targets.py)",
                "extensions": "data/interp_with_extensions.json (web-agent pass)",
            },
            "selection_metadata": sel.get("metadata", {}) if isinstance(sel, dict) else {},
            "extension_metadata": ext.get("metadata", {}),
            "link_validation": {
                "method": ("arXiv links checked against the arXiv API (existence + title-match to "
                           "flag wrong/hallucinated ids); other links checked over HTTP. Extensions "
                           "whose link resolves to a different paper (title mismatch) are removed."),
                "extensions_checked": n_checked,
                "links_ok": n_ok,
                "links_uncertain": n_uncertain,
                "links_broken": n_bad,
                "removed_hallucinated": n_removed,
                "extensions_kept": n_checked - n_removed,
            },
        },
        "papers": final,
        "summary": {
            "papers": len(final),
            "papers_with_extensions": sum(1 for p in final if p["extensions"]),
            "extensions_total": n_checked - n_removed,
            "extensions_valid_links": n_ok,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nchecked: {n_checked} | ok: {n_ok} | uncertain: {n_uncertain} | broken: {n_bad} "
          f"| removed (hallucinated): {n_removed} | kept: {n_checked - n_removed}", file=sys.stderr)
    print(f"wrote {OUT.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
