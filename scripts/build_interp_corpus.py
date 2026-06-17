#!/usr/bin/env python3
"""Build a unified interpretability corpus from the conference dumps in papers-base/.

Scans every paper in papers-base/*.json (~52k across 9 venues), keeps the ones
relevant to our reoriented direction — probing, steering / representation
engineering, SAEs / dictionary learning, and mechanistic interpretability more
broadly — and writes a single de-duplicated corpus that we filter down later.

Each kept paper is tagged with a primary `category`, the matched terms, and a
relevance `score` (title hits count more than abstract hits). Accepted papers go
to the main corpus; withdrawn / rejected / under-review matches are written to a
separate file so nothing is lost but the main pool stays clean.

Outputs:
    interp_corpus.json             accepted, matched, de-duplicated, ranked
    interp_corpus_unaccepted.json  matched but withdrawn / rejected
    INTERP_CORPUS.md               human-readable summary (counts + top papers)

This is a recall-oriented first pass: tune TERMS / KEEP_SCORE to taste, then
re-run. No network access — everything comes from the local dumps.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "papers-base"
DATA = ROOT / "data"
DOCS = ROOT / "docs"
DATA.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)
OUT_CORPUS = DATA / "interp_corpus.json"
OUT_UNACCEPTED = DATA / "interp_corpus_unaccepted.json"
OUT_MD = DOCS / "INTERP_CORPUS.md"

KEEP_SCORE = 3            # minimum weighted score to keep a paper
TITLE_BONUS = 1          # extra weight when a term is found in the title

# venue label + year parsed from the file stem (e.g. "iclr2025" -> ICLR, 2025)
VENUE_NAME = {"iclr": "ICLR", "icml": "ICML", "nips": "NeurIPS", "neurips": "NeurIPS",
              "aaai": "AAAI", "emnlp": "EMNLP", "naacl": "NAACL", "ijcai": "IJCAI"}

# --- topic taxonomy -------------------------------------------------------------
# (regex, category, weight, strong). A "strong" term is unambiguous enough that a
# single hit (weight 3) clears KEEP_SCORE on its own. Weaker terms need company.
TERMS = [
    # ---- probing ----
    (r"linear prob(e|es|ing)", "probing", 3, True),
    (r"activation prob(e|es|ing)", "probing", 3, True),
    (r"diagnostic classifier", "probing", 3, True),
    (r"probing classifier", "probing", 3, True),
    (r"representation prob(e|es|ing)", "probing", 3, True),
    (r"concept prob(e|es|ing)", "probing", 3, True),
    (r"sparse probing", "probing", 3, True),
    (r"\bprobing\b", "probing", 2, False),
    (r"\bprobe(s)?\b", "probing", 1, False),
    # ---- steering / representation engineering ----
    (r"steering vector", "steering", 3, True),
    (r"activation steering", "steering", 3, True),
    (r"activation addition", "steering", 3, True),
    (r"\bActAdd\b", "steering", 3, True),
    (r"representation engineering", "steering", 3, True),
    (r"\bRepE\b", "steering", 3, True),
    (r"persona vector", "steering", 3, True),
    (r"concept vector", "steering", 3, True),
    (r"inference[- ]time intervention", "steering", 3, True),
    (r"activation engineering", "steering", 3, True),
    (r"contrastive activation", "steering", 3, True),
    (r"feature steering", "steering", 3, True),
    (r"model steering", "steering", 3, True),
    (r"steer(ing)? (the|a|llm|language model|model|behavior|behaviour|generation)", "steering", 2, False),
    # ---- SAE / dictionary learning ----
    (r"sparse autoencoder", "sae", 3, True),
    (r"sparse auto-encoder", "sae", 3, True),
    (r"\bSAEs?\b", "sae", 3, True),
    (r"crosscoder", "sae", 3, True),
    (r"transcoder", "sae", 3, True),
    (r"dictionary learning", "sae", 3, True),
    (r"sparse dictionary", "sae", 3, True),
    # ---- circuits / mechanistic core ----
    (r"mechanistic interpretab", "mech", 3, True),
    (r"activation patching", "mech", 3, True),
    (r"causal tracing", "mech", 3, True),
    (r"path patching", "mech", 3, True),
    (r"interchange intervention", "mech", 3, True),
    (r"distributed alignment search", "mech", 3, True),
    (r"logit lens", "mech", 3, True),
    (r"patchscopes", "mech", 3, True),
    (r"induction head", "mech", 3, True),
    (r"superposition", "mech", 2, False),
    (r"polysemantic", "mech", 3, True),
    (r"monosemantic", "mech", 3, True),
    (r"linear representation hypothesis", "mech", 3, True),
    (r"causal mediation", "mech", 3, True),
    (r"concept erasure", "mech", 3, True),
    (r"\bcircuit(s)?\b", "mech", 1, False),
    (r"feature attribution", "mech", 1, False),
    # ---- general interpretability signal ----
    (r"interpretab", "interp", 2, False),
    (r"residual stream", "interp", 2, False),
    (r"latent direction", "interp", 2, False),
    (r"hidden state(s)?", "interp", 1, False),
    (r"internal representation", "interp", 1, False),
    (r"knowledge editing", "interp", 1, False),
    (r"model editing", "interp", 1, False),
]
COMPILED = [(re.compile(p, re.IGNORECASE), cat, w, strong) for p, cat, w, strong in TERMS]
CATEGORY_PRIORITY = ["probing", "steering", "sae", "mech", "interp"]

# Context gates: ambiguous terms only count when an interp/LLM anchor co-occurs in
# the document. Guards acronym collisions like "SAE" = Standard American English
# and "dictionary learning" used for image/signal sparse coding rather than
# feature interpretability.
GATES = {
    r"\bSAEs?\b": [r"sparse", r"auto-?encoder", r"monosemantic", r"dictionary",
                   r"feature direction", r"latent feature"],
    r"dictionary learning": [r"monosemantic", r"polysemantic", r"interpretab",
                             r"residual stream", r"sparse autoencoder", r"feature direction",
                             r"dictionary feature"],
}
GATE_RX = {p: [re.compile(a, re.IGNORECASE) for a in anchors] for p, anchors in GATES.items()}


def venue_year(stem: str):
    m = re.match(r"([a-z]+)(\d{4})", stem.lower())
    if not m:
        return stem.upper(), None
    return VENUE_NAME.get(m.group(1), m.group(1).upper()), int(m.group(2))


def is_accepted(status: str) -> bool:
    s = (status or "").lower()
    if not s:
        return True  # published-proceedings dumps with blank status
    return not ("reject" in s or "withdraw" in s)


def score_paper(title, abstract, keywords, primary_area):
    title_l = title or ""
    rest = " \n ".join(x for x in (abstract, keywords, primary_area) if x)
    full = title_l + " \n " + rest
    cat_score = defaultdict(float)
    matched = set()
    strong_hit = False
    kw_interp = "interpretab" in (keywords or "").lower() or "interpretab" in (primary_area or "").lower()
    for rx, cat, w, strong in COMPILED:
        in_title = bool(rx.search(title_l))
        in_rest = bool(rx.search(rest))
        if not (in_title or in_rest):
            continue
        if rx.pattern in GATE_RX and not any(a.search(full) for a in GATE_RX[rx.pattern]):
            continue  # ambiguous term without interp context — skip
        matched.add(rx.pattern)
        cat_score[cat] += w + (TITLE_BONUS if in_title else 0)
        if strong:
            strong_hit = True
    total = sum(cat_score.values())
    # require either a strong term or an explicit interpretability keyword, so a
    # lone weak hit ("probe", "circuit") does not pull in off-topic papers.
    if not (strong_hit or kw_interp):
        return None
    if total < KEEP_SCORE:
        return None
    primary = max(cat_score, key=lambda c: (cat_score[c], -CATEGORY_PRIORITY.index(c)))
    return primary, round(total, 1), dict(cat_score), sorted(matched)


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def clean_int(v):
    try:
        n = int(v)
        return n if n >= 0 else None
    except (TypeError, ValueError):
        return None


def main():
    files = sorted(SRC_DIR.glob("*.json"))
    by_title = {}            # norm_title -> record (deduped, best kept)
    unaccepted = []
    seen = 0
    for fp in files:
        venue, year = venue_year(fp.stem)
        data = json.loads(fp.read_text(encoding="utf-8"))
        seen += len(data)
        for p in data:
            title = (p.get("title") or "").strip()
            if not title:
                continue
            res = score_paper(title, p.get("abstract", ""), p.get("keywords", ""),
                              p.get("primary_area", ""))
            if res is None:
                continue
            primary, total, cat_score, matched = res
            authors = [a.strip() for a in re.split(r"[;]", p.get("author", "") or "") if a.strip()]
            rec = {
                "title": title,
                "authors": authors,
                "venue": venue, "year": year,
                "status": p.get("status", ""),
                "track": p.get("track", ""),
                "category": primary,
                "score": total,
                "category_scores": {k: round(v, 1) for k, v in cat_score.items()},
                "matched_terms": matched,
                "citations": clean_int(p.get("gs_citation")),
                "github": (p.get("github") or "").strip(),
                "url": (p.get("site") or p.get("openreview") or p.get("pdf") or "").strip(),
                "pdf": (p.get("pdf") or "").strip(),
                "keywords": (p.get("keywords") or "").strip(),
                "primary_area": (p.get("primary_area") or "").strip(),
                "abstract": (p.get("abstract") or "").strip(),
                "id": p.get("id", ""),
                "accepted": is_accepted(p.get("status", "")),
                "venues": [f"{venue} {year}"],
            }
            key = norm_title(title)
            if key in by_title:                      # same paper across venues
                prev = by_title[key]
                if f"{venue} {year}" not in prev["venues"]:
                    prev["venues"].append(f"{venue} {year}")
                # keep the higher-scoring / more-cited / accepted copy as primary
                better = (rec["accepted"], rec["score"], rec["citations"] or 0) > \
                         (prev["accepted"], prev["score"], prev["citations"] or 0)
                if better:
                    rec["venues"] = prev["venues"]
                    by_title[key] = rec
            else:
                by_title[key] = rec

    records = list(by_title.values())
    accepted = [r for r in records if r["accepted"]]
    rejected = [r for r in records if not r["accepted"]]

    def sort_key(r):
        return (-r["score"], -(r["citations"] or 0), r["title"].lower())
    accepted.sort(key=sort_key)
    rejected.sort(key=sort_key)

    OUT_CORPUS.write_text(json.dumps(accepted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_UNACCEPTED.write_text(json.dumps(rejected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- summary ----
    by_cat = Counter(r["category"] for r in accepted)
    by_venue = Counter(r["venue"] for r in accepted)
    with_code = sum(1 for r in accepted if r["github"])
    L = ["# Unified Interpretability Corpus\n",
         f"Harvested from {len(files)} conference dumps in `papers-base/` "
         f"({seen:,} papers scanned). Topic match on probing, steering, SAEs, and "
         "mechanistic interpretability (title + abstract + keywords).\n",
         f"\n**{len(accepted)} accepted** matched papers (de-duplicated), "
         f"plus {len(rejected)} withdrawn/rejected matches in "
         "`interp_corpus_unaccepted.json`. {0} have public code.\n".format(with_code),
         "\n## By category (accepted)\n\n| Category | Papers |\n| --- | --- |\n"]
    for c in CATEGORY_PRIORITY:
        L.append(f"| {c} | {by_cat.get(c, 0)} |\n")
    L.append("\n## By venue (accepted)\n\n| Venue | Papers |\n| --- | --- |\n")
    for v, n in by_venue.most_common():
        L.append(f"| {v} | {n} |\n")
    L.append("\n## Top 30 accepted by relevance score\n\n| Score | Cat | Cites | Title | Venue |\n"
             "| --- | --- | --- | --- | --- |\n")
    for r in accepted[:30]:
        L.append(f"| {r['score']:g} | {r['category']} | {r['citations'] if r['citations'] is not None else '–'} "
                 f"| {r['title'][:80]} | {r['venue']} {r['year']} |\n")
    L.append("\n---\nRegenerate: `python3 scripts/build_interp_corpus.py`. "
             "Tune `TERMS` / `KEEP_SCORE` to widen or narrow the net.\n")
    OUT_MD.write_text("".join(L), encoding="utf-8")

    print(f"scanned {seen:,} papers across {len(files)} venues")
    print(f"kept {len(accepted)} accepted + {len(rejected)} unaccepted (deduped from "
          f"{len(records)} unique matched titles)")
    print("by category (accepted):", dict(by_cat))
    print("by venue (accepted):", dict(by_venue))
    print(f"with public code: {with_code}")
    print(f"wrote {OUT_CORPUS.name}, {OUT_UNACCEPTED.name}, {OUT_MD.name}")


if __name__ == "__main__":
    main()
