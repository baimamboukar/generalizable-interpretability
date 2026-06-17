#!/usr/bin/env python3
"""Find extension studies (reproduce / extend / improve) for each shortlisted paper.

For every paper in interp_selection.json -> selection, this resolves the paper on
OpenAlex (free citation graph, no API key), pulls the works that CITE it, and tags
each citing work as a likely reproduction, extension, or improvement based on its
title/abstract. Writes interp_extensions.json and EXTENSIONS.md.

Reliability notes (read these):
  - The citation LIST is real OpenAlex data. Counts of total citations are reliable.
  - OpenAlex does not expose citation *intent*, so the reproduce/extend/improve tag
    is a keyword heuristic over the citing paper's own title+abstract — treat the
    tagged items as CANDIDATES to verify, not a final classification.
  - Title-based matching can mismatch; `match_confidence` (title-word Jaccard) and
    the matched OpenAlex title are recorded so every match can be checked.
  - Very recent papers (2025/2026) may have few or no citations yet.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOCS = ROOT / "docs"
DATA.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)
SRC = DATA / "interp_selection.json"
OUT_JSON = DATA / "interp_extensions.json"
OUT_MD = DOCS / "EXTENSIONS.md"

MAILTO = "mohamedgaye.mhd@gmail.com"      # OpenAlex "polite pool"
TOP_N = 20                                 # papers to process (the capped shortlist)
MAX_CITERS = 500                           # safety cap on citing works per paper
MATCH_MIN = 0.45                           # min title Jaccard to accept a match
PER_PAGE = 200

TAGGERS = {
    "reproduce": re.compile(r"\b(reproduc|replicat|re-?implement)\w*", re.I),
    "extend": re.compile(r"\b(extend|extension|generali[sz]|build(s|ing)?\s+(on|upon)|"
                         r"based on|follow[- ]?up|beyond)\b", re.I),
    "improve": re.compile(r"\b(improv|enhanc|outperform|stronger|refin(e|ing|ed)|boost|"
                          r"better than|more (robust|accurate|faithful))\b", re.I),
}
STOP = set("a an the of for and or to in on with via using through from into is are "
           "we our this that study paper model models language large".split())


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": f"interp-extensions ({MAILTO})"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 + 2 * attempt)
                continue
            return None                       # 400/404 etc. — skip gracefully
        except Exception:
            time.sleep(1 + attempt)
    return None


def title_tokens(t):
    return {w for w in re.sub(r"[^a-z0-9 ]", " ", (t or "").lower()).split()
            if w and w not in STOP and len(w) > 2}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def reconstruct_abstract(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))[:1200]


def resolve(title, year):
    clean = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", title)).strip()
    q = urllib.parse.quote(clean[:280])
    url = (f"https://api.openalex.org/works?search={q}&per_page=5"
           f"&select=id,title,publication_year,cited_by_count,doi&mailto={MAILTO}")
    resp = get(url)
    data = (resp or {}).get("results", [])
    want = title_tokens(title)
    best, best_j = None, 0.0
    for w in data:
        j = jaccard(want, title_tokens(w.get("title", "")))
        if j > best_j:
            best, best_j = w, j
    return best, round(best_j, 2)


def fetch_citers(work_id):
    wid = work_id.rsplit("/", 1)[-1]
    out, cursor = [], "*"
    while cursor and len(out) < MAX_CITERS:
        url = (f"https://api.openalex.org/works?filter=cites:{wid}&per_page={PER_PAGE}"
               f"&cursor={cursor}&mailto={MAILTO}"
               "&select=id,title,publication_year,doi,abstract_inverted_index,"
               "authorships,primary_location")
        page = get(url)
        if not page:
            break
        out.extend(page.get("results", []))
        cursor = page.get("meta", {}).get("next_cursor")
        time.sleep(0.2)
    return out[:MAX_CITERS]


def classify(work):
    text = (work.get("title") or "") + " " + reconstruct_abstract(
        work.get("abstract_inverted_index"))
    return [tag for tag, rx in TAGGERS.items() if rx.search(text)]


def short_authors(work):
    names = [a.get("author", {}).get("display_name", "")
             for a in (work.get("authorships") or [])[:3]]
    names = [n for n in names if n]
    s = ", ".join(names)
    if len(work.get("authorships") or []) > 3:
        s += " et al."
    return s


def venue_of(work):
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("display_name", "")


def main():
    sel = json.loads(SRC.read_text(encoding="utf-8"))
    papers = (sel["selection"] if isinstance(sel, dict) else sel)[:TOP_N]
    results = []
    for i, p in enumerate(papers, 1):
        title = p["title"]
        print(f"[{i}/{len(papers)}] {title[:60]}", file=sys.stderr)
        match, conf = resolve(title, p.get("year"))
        entry = {
            "rank": p.get("rank", i), "title": title, "focus": p.get("focus"),
            "venue": p.get("venue"), "year": p.get("year"), "composite": p.get("composite"),
            "openalex_id": match.get("id") if match else None,
            "openalex_title_matched": match.get("title") if match else None,
            "match_confidence": conf,
            "doi": (match.get("doi") if match else None),
        }
        if not match or conf < MATCH_MIN:
            entry.update({"resolved": False, "total_citations": None,
                          "counts": {}, "extensions": [],
                          "note": "no confident OpenAlex match — verify title manually"})
            results.append(entry)
            time.sleep(0.3)
            continue

        citers = fetch_citers(match["id"])
        exts = []
        counts = {"reproduce": 0, "extend": 0, "improve": 0, "extension_like": 0}
        for w in citers:
            tags = classify(w)
            if not tags:
                continue
            for t in tags:
                counts[t] += 1
            counts["extension_like"] += 1
            exts.append({
                "title": w.get("title"), "year": w.get("publication_year"),
                "type": tags, "doi": w.get("doi"), "venue": venue_of(w),
                "authors": short_authors(w),
                "openalex_id": w.get("id"),
            })
        exts.sort(key=lambda e: (-(len(e["type"])), -(e["year"] or 0)))
        entry.update({
            "resolved": True,
            "total_citations": match.get("cited_by_count"),
            "citers_examined": len(citers),
            "citers_capped": len(citers) >= MAX_CITERS,
            "counts": counts,
            "extensions": exts,
        })
        results.append(entry)
        print(f"    cites={match.get('cited_by_count')} extension-like={counts['extension_like']}",
              file=sys.stderr)
        time.sleep(0.3)

    total_ext = sum(r["counts"].get("extension_like", 0) for r in results)
    resolved = sum(r.get("resolved", False) for r in results)
    payload = {
        "metadata": {
            "title": "Extension studies for the reproduction shortlist",
            "generated_by": "scripts/find_extensions.py",
            "source": "data/interp_selection.json -> selection (top %d)" % TOP_N,
            "citation_source": "OpenAlex API (https://api.openalex.org), polite pool",
            "method": ("Resolve each shortlisted paper on OpenAlex by title, fetch citing works, "
                       "and tag each as reproduce/extend/improve by keyword over its title+abstract."),
            "reliability": ("Citation counts are real OpenAlex data; the reproduce/extend/improve "
                            "tags are heuristic CANDIDATES (OpenAlex has no citation-intent field) — "
                            "verify before quoting. match_confidence is title-word Jaccard; entries "
                            "below %.2f are flagged unresolved." % MATCH_MIN),
            "caps": {"max_citers_per_paper": MAX_CITERS, "match_min_confidence": MATCH_MIN},
            "papers_resolved": resolved, "papers_total": len(results),
            "total_extension_like": total_ext,
        },
        "papers": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # readable summary
    L = ["# Extension Studies — Reproduction Shortlist\n",
         f"For each of the top {TOP_N} shortlisted papers, citing works on OpenAlex tagged as likely "
         "**reproduce / extend / improve**. Tags are heuristic candidates — verify before quoting "
         "(see `interp_extensions.json` metadata).\n",
         f"\n**{resolved}/{len(results)} papers resolved** on OpenAlex; "
         f"**{total_ext} extension-like citing studies** found in total.\n",
         "\n| # | Paper | Focus | Total cites | Extension-like | Reprod. | Extend | Improve | Match |\n"
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"]
    for r in results:
        c = r.get("counts", {})
        tc = r["total_citations"] if r["total_citations"] is not None else "–"
        if not r.get("resolved"):
            L.append(f"| {r['rank']} | {r['title'][:50]} | {r['focus']} | – | – | – | – | – | "
                     f"⚠ {r['match_confidence']} |\n")
            continue
        L.append(f"| {r['rank']} | {r['title'][:50]} | {r['focus']} | {tc} | "
                 f"{c.get('extension_like',0)} | {c.get('reproduce',0)} | {c.get('extend',0)} | "
                 f"{c.get('improve',0)} | {r['match_confidence']} |\n")
    L.append("\n## Top extension candidates per paper\n")
    for r in results:
        if not r.get("extensions"):
            continue
        L.append(f"\n### {r['rank']}. {r['title']}\n")
        for e in r["extensions"][:6]:
            doi = f" · {e['doi']}" if e.get("doi") else ""
            L.append(f"- *[{'/'.join(e['type'])}]* {e['title']} ({e['year']}) — "
                     f"{e['authors']}{doi}\n")
    OUT_MD.write_text("".join(L), encoding="utf-8")
    print(f"\nresolved {resolved}/{len(results)}; total extension-like {total_ext}", file=sys.stderr)
    print(f"wrote {OUT_JSON.name}, {OUT_MD.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
