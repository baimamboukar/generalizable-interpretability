#!/usr/bin/env python3
"""Filter the unified interp corpus down to a probes-and-steering shortlist.

Reads interp_corpus.json and selects up to MAX_SELECTED reproduction targets,
scored to match the project's line of attack (per mentor guidance):

  - mainly probes and steering vectors (probing weighted above steering, since
    the plan is to land probes first, then extend to steering incl. persona);
  - a clean, reusable codebase to build the cross-model pipeline on (public code
    is rewarded);
  - cross-model readiness: studies that already span several model FAMILIES and a
    GRADUAL SIZE range are favored, because the goal is to run the same probe
    across many models and sizes;
  - impact and recency as secondary signals.

Model families, parameter sizes, and code links are extracted from the abstract
(the conference dumps do not carry them as fields), so the shortlist can be
verified by hand. Writes:
    interp_selection.json   ranked shortlist, figure-ready
    INTERP_SELECTION.md     readable table + per-paper notes
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOCS = ROOT / "docs"
DATA.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)
SRC = DATA / "interp_corpus.json"
OUT_JSON = DATA / "interp_selection.json"
OUT_MD = DOCS / "INTERP_SELECTION.md"

MAX_SELECTED = 20          # cap (not a target — take the strong ones up to here)
PROBING_W, STEERING_W = 1.0, 0.85   # probing edges out steering ("probes first")

# Composite-score budget (points per component; sums to 100). Single source of
# truth: score() and the exported metadata both read from here.
WEIGHTS = {"focus": 30, "reproducibility": 22, "cross_model": 22,
           "impact": 8, "recency": 4, "topic": 14}
CODE_POINTS = {"repo": 22, "promised": 12, "unknown": 6}
RECENCY_POINTS = {2026: 4, 2025: 3.5, 2024: 2.5}

# --- model-family detection (normalized family <- regexes) ----------------------
FAMILY_PATTERNS = {
    "Llama": [r"\bll?ama-?\d?", r"\bvicuna\b", r"\balpaca\b"],
    "GPT": [r"\bgpt-?2\b", r"\bgpt-?j\b", r"\bgpt-?neo(x)?\b", r"\bgpt-?3\b", r"\bgpt-?4\b"],
    "Pythia": [r"\bpythia\b"],
    "Gemma": [r"\bgemma-?\d?\b"],
    "Qwen": [r"\bqwen\d?\b"],
    "Mistral": [r"\bmistral\b", r"\bmixtral\b"],
    "Falcon": [r"\bfalcon\b"],
    "OPT": [r"\bopt-\d"],
    "BLOOM": [r"\bbloom(z)?\b"],
    "Phi": [r"\bphi-\d"],
    "OLMo": [r"\bolmo\b"],
    "Yi": [r"\byi-\d"],
    "DeepSeek": [r"\bdeepseek\b"],
    "BERT": [r"\bbert\b", r"\broberta\b", r"\bdeberta\b", r"\balbert\b"],
    "T5": [r"\bt5\b", r"\bflan-?t5\b", r"\bul2\b"],
}
FAMILY_RX = {fam: [re.compile(p, re.IGNORECASE) for p in pats] for fam, pats in FAMILY_PATTERNS.items()}

# LLM/NLP context gate: keeps interpretability probing/steering of language models
# and drops the unrelated vision / self-supervised "linear probing" *evaluation*
# protocol (frozen backbone + linear classifier on ImageNet etc.).
LLM_CTX = re.compile(
    r"\b(language model|large language model|LLMs?|GPT[- ]?\d?|Llama|Gemma|Qwen|Mistral|"
    r"Mixtral|Pythia|Falcon|OLMo|Vicuna|BERT|RoBERTa|natural language|NLP|linguistic|"
    r"sentence|token(s|ization|izer)?|in-context|instruction[- ]?follow|prompt|dialog|chat|"
    r"truthful|refus(al|e|es)|persona|jailbreak|hallucinat|deception|sentiment|"
    r"question answering|knowledge editing|factual)\b", re.IGNORECASE)

# Drop self-supervised / vision-transfer papers (where "linear probing" is an eval
# protocol) unless they carry a genuine interpretability signal.
EXCLUDE_CTX = re.compile(
    r"\b(contrastive language-image|CLIP\b|self-supervised|imagenet|federated|"
    r"vision-language|representation learner|transfer learning|linear evaluation|"
    r"semantic segmentation|object detection|image classification|few-shot classification)\b",
    re.IGNORECASE)
STRONG_INTERP = re.compile(
    r"\b(interpretab|concept|truthful|decept|refus|persona|sycophan|hallucinat|"
    r"knowledge|factual|sentiment|steering vector|activation steering|residual stream|"
    r"hidden state|linear representation|jailbreak|belief|world model|safety|alignment)\b",
    re.IGNORECASE)

# Hard-drop non-text modalities (audio / speech / EEG / ECG / biosignals): the
# pipeline targets text language models.
MODALITY_EXCLUDE = re.compile(
    r"\b(EEG|ECG|EMG|electroencephalo\w*|electrocardio\w*|biosignal|\baudio\b|acoustic|"
    r"phoneme|speech recognition|spoken language|automatic speech|\bMIDI\b|"
    r"music generation|sound event)\b", re.IGNORECASE)

# Safety / alignment topics to incentivize (each distinct hit adds to the topic score).
SAFETY_TERMS = [re.compile(p, re.IGNORECASE) for p in [
    r"\bsafety\b", r"\balign(ment|ed)\b", r"\bharm(ful|less|s)?\b", r"\brefus", r"\bjailbreak",
    r"\bdecep(tion|tive)", r"\btruthful|honest", r"\bsycophan", r"\btoxic", r"\bhallucinat",
    r"\bRLHF\b|reward model|preference (optimization|tuning)", r"\bguardrail|red[- ]?team",
    r"\bmisuse|unsafe|backdoor", r"\bmoral|ethical", r"\bbias|fairness"]]

SIZE_B = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,2})?)\s?[Bb](?![a-zA-Z])")
SIZE_M = re.compile(r"(?<![\w.])(\d{2,4})\s?[Mm](?![a-zA-Z])")
GITHUB = re.compile(r"https?://github\.com/[\w.\-]+/[\w.\-]+", re.IGNORECASE)
CODE_PROMISE = re.compile(r"\b(code|implementation|we)\b[^.]{0,40}\b(available|released?|release|"
                          r"open[- ]?source(d)?|public(ly)?)\b", re.IGNORECASE)
PERSONA = re.compile(r"\bpersona\b", re.IGNORECASE)


def detect_families(text):
    return sorted({fam for fam, rxs in FAMILY_RX.items() if any(r.search(text) for r in rxs)})


def detect_sizes(text):
    sizes = []
    for m in SIZE_B.finditer(text):
        v = float(m.group(1))
        if 0.01 <= v <= 1000:
            sizes.append(round(v, 2))
    for m in SIZE_M.finditer(text):
        v = float(m.group(1)) / 1000.0
        if 0.005 <= v <= 1.0:
            sizes.append(round(v, 3))
    return sorted(set(sizes))


def detect_code(rec, text):
    if rec.get("github"):
        return rec["github"], "repo"
    m = GITHUB.search(text)
    if m:
        return m.group(0), "repo"
    if CODE_PROMISE.search(text):
        return None, "promised"
    return None, "unknown"


def score(rec, fams, sizes, code_kind, has_persona, n_safety):
    cs = rec.get("category_scores", {})
    base = PROBING_W * cs.get("probing", 0) + STEERING_W * cs.get("steering", 0)

    # focus on probes/steering (0..WEIGHTS["focus"])
    focus = WEIGHTS["focus"] * min(base / 9.0, 1.0)

    # reproducibility / reusable code (0..WEIGHTS["reproducibility"])
    repro = CODE_POINTS[code_kind]

    # cross-model readiness: families + gradual size span (0..WEIGHTS["cross_model"])
    fam_max = WEIGHTS["cross_model"] - 8
    fam_pts = fam_max * min(len(fams), 4) / 4
    if len(sizes) >= 2 and sizes[-1] / max(sizes[0], 0.01) >= 4:
        size_pts = 8                        # genuinely gradual span (e.g. 1B -> 70B)
    elif len(sizes) >= 2:
        size_pts = 5
    elif len(sizes) == 1:
        size_pts = 2
    else:
        size_pts = 0
    cross = fam_pts + size_pts

    # impact (0..WEIGHTS["impact"]) and recency (0..WEIGHTS["recency"])
    c = rec.get("citations") or 0
    impact = WEIGHTS["impact"] * min(math.log10(c + 1) / 2.0, 1.0)
    recency = RECENCY_POINTS.get(rec.get("year"), 1.5)

    # topic incentive: safety / alignment work + persona steering (0..14)
    safety_pts = 2 * min(n_safety, 5)       # up to 10 for safety/alignment relevance
    persona_pts = 4 if has_persona else 0
    topic = min(safety_pts + persona_pts, WEIGHTS["topic"])

    composite = round(min(focus + repro + cross + impact + recency + topic, 100), 1)
    return composite, {
        "focus": round(focus, 1), "reproducibility": repro, "cross_model": round(cross, 1),
        "impact": round(impact, 1), "recency": recency,
        "safety": safety_pts, "persona_bonus": persona_pts,
    }


def main():
    corpus = json.loads(SRC.read_text(encoding="utf-8"))
    cands = []
    funnel = {"corpus": len(corpus), "has_probe_or_steer_signal": 0,
              "in_llm_context": 0, "after_ssl_eval_filter": 0, "after_modality_filter": 0}
    for rec in corpus:
        cs = rec.get("category_scores", {})
        if cs.get("probing", 0) + cs.get("steering", 0) <= 0:
            continue                         # no probe/steering signal -> out of scope
        funnel["has_probe_or_steer_signal"] += 1
        text = f"{rec.get('title','')} \n {rec.get('abstract','')} \n {rec.get('keywords','')}"
        if not LLM_CTX.search(text):
            continue                         # vision/SSL "linear probing" eval, not LLM interp
        funnel["in_llm_context"] += 1
        if EXCLUDE_CTX.search(text) and not STRONG_INTERP.search(text):
            continue                         # SSL/vision-transfer eval without interp intent
        funnel["after_ssl_eval_filter"] += 1
        if MODALITY_EXCLUDE.search(text):
            continue                         # audio / speech / EEG / ECG — not text LMs
        funnel["after_modality_filter"] += 1
        fams = detect_families(text)
        sizes = detect_sizes(text)
        repo, code_kind = detect_code(rec, text)
        has_persona = bool(PERSONA.search(text))
        n_safety = sum(bool(rx.search(text)) for rx in SAFETY_TERMS)
        composite, parts = score(rec, fams, sizes, code_kind, has_persona, n_safety)
        focus_cat = "probing" if cs.get("probing", 0) >= cs.get("steering", 0) else "steering"
        cands.append({
            "title": rec["title"],
            "focus": focus_cat,
            "category": rec["category"],
            "venue": rec["venue"], "year": rec["year"],
            "composite": composite, "score_breakdown": parts,
            "families": fams, "n_families": len(fams),
            "sizes_b": sizes, "size_min_b": sizes[0] if sizes else None,
            "size_max_b": sizes[-1] if sizes else None,
            "code": repo, "code_status": code_kind,
            "persona": has_persona, "safety_hits": n_safety,
            "citations": rec.get("citations"),
            "relevance_score": rec.get("score"),
            "matched_terms": rec.get("matched_terms", []),
            "url": rec.get("url", ""), "pdf": rec.get("pdf", ""),
            "authors": rec.get("authors", []),
            "abstract": rec.get("abstract", ""),
        })

    cands.sort(key=lambda r: (-r["composite"], -(r["citations"] or 0)))
    shortlist = cands[:MAX_SELECTED]
    for i, r in enumerate(shortlist, 1):
        r["rank"] = i

    # ---- aggregate stats over the shortlist ----
    fam_counter = Counter(f for r in shortlist for f in r["families"])
    sized = [r for r in shortlist if r["size_min_b"] is not None]
    stats = {
        "n_selected": len(shortlist),
        "by_focus": dict(Counter(r["focus"] for r in shortlist)),
        "by_venue": dict(Counter(f"{r['venue']} {r['year']}" for r in shortlist).most_common()),
        "by_year": dict(sorted(Counter(r["year"] for r in shortlist).items())),
        "code_status": dict(Counter(r["code_status"] for r in shortlist)),
        "with_safety_signal": sum(r["safety_hits"] > 0 for r in shortlist),
        "with_persona": sum(r["persona"] for r in shortlist),
        "model_families_covered": dict(fam_counter.most_common()),
        "with_extracted_sizes": len(sized),
        "size_span_b": [min(r["size_min_b"] for r in sized),
                        max(r["size_max_b"] for r in sized)] if sized else None,
        "composite_range": [shortlist[-1]["composite"], shortlist[0]["composite"]] if shortlist else None,
    }

    metadata = {
        "title": "Probes & steering reproduction shortlist",
        "generated_by": "scripts/select_interp_targets.py",
        "source_corpus": "data/interp_corpus.json (built by scripts/build_interp_corpus.py)",
        "objective": (
            "Pick reproduction targets for a clean, scalable pipeline that runs probes — then "
            "steering vectors (incl. persona) — across many model families and sizes, reproducing "
            "each paper's original result before extending to new models. Probes are prioritized "
            "first per mentor guidance."),
        "max_selected": MAX_SELECTED,
        "selection_funnel": funnel,
        "scope_filters": [
            "Has a probing or steering signal in the source corpus (SAE-only / circuit-only excluded).",
            "Sits in an LLM/NLP context (drops vision/self-supervised 'linear probing' used as an "
            "evaluation protocol).",
            "Not a self-supervised / vision-transfer eval paper unless it carries a genuine "
            "interpretability signal.",
            "Text modality only — audio / speech / EEG / ECG / biosignal papers are dropped.",
        ],
        "relevance_score": {
            "field": "relevance_score (carried from the corpus harvester)",
            "how": ("Each paper is matched against a term taxonomy over title + abstract + keywords. "
                    "Strong unambiguous terms (e.g. 'steering vector', 'linear probe', 'mechanistic "
                    "interpretability') weight 3, medium terms 2, weak/context terms 1, with a +1 "
                    "bonus when a term appears in the title. A paper is kept only if it hits a strong "
                    "term or an explicit interpretability keyword and totals at least 3."),
        },
        "composite_score": {
            "formula": "focus + reproducibility + cross_model + impact + recency + topic, capped at 100",
            "weights_max_points": WEIGHTS,
            "components": {
                "focus": ("Probe/steering alignment from the corpus category scores; probing weighted "
                          f"{PROBING_W} vs steering {STEERING_W} ('probes first')."),
                "reproducibility": (f"Reusable code to build the pipeline on. {CODE_POINTS} points for "
                                    "repo link / code promised / none detected (from abstract or fields)."),
                "cross_model": ("Cross-model readiness: model families (up to "
                                f"{WEIGHTS['cross_model'] - 8} pts for >=4 families) plus a gradual "
                                "parameter-size span (8 pts when max/min >= 4x, e.g. 1B->70B)."),
                "impact": f"Citations, log-scaled: {WEIGHTS['impact']} * min(log10(cites+1)/2, 1).",
                "recency": f"Publication year: {RECENCY_POINTS} (older -> 1.5).",
                "topic": ("Safety/alignment incentive: 2 pts per distinct safety/alignment term "
                          f"(<=10) + 4 for persona steering, capped at {WEIGHTS['topic']}."),
            },
        },
        "field_notes": {
            "families/sizes": "Parsed from the abstract as a cross-model-readiness signal; verify per paper.",
            "code_status": "repo = link found, promised = code release mentioned, unknown = none "
                           "detected (absence is not proof there is no repo).",
            "citations": "Google Scholar count from the conference dump; null when unavailable.",
        },
        "stats": stats,
    }

    payload = {"metadata": metadata, "selection": shortlist}
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- readable summary ----
    n_probe = sum(r["focus"] == "probing" for r in shortlist)
    n_steer = sum(r["focus"] == "steering" for r in shortlist)
    n_code = sum(r["code_status"] == "repo" for r in shortlist)
    L = ["# Probes & Steering — Reproduction Shortlist\n",
         f"Filtered from {len(corpus)} interp-corpus papers to the {len(cands)} text-LM "
         f"probing/steering studies, ranked by alignment with the line of attack "
         "(probes first, then steering incl. persona; reusable code; cross-model readiness; "
         "safety/alignment topics). Audio/speech/EEG/ECG modalities are excluded.\n",
         f"\n**{len(shortlist)} targets** (cap {MAX_SELECTED}): {n_probe} probing, {n_steer} steering. "
         f"{n_code} have a code repo detected in the abstract.\n",
         "\n| # | Composite | Focus | Families | Sizes (B) | Code | Cites | Title | Venue |\n"
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"]
    for r in shortlist:
        fam = ", ".join(r["families"]) or "–"
        sz = (f'{r["size_min_b"]:g}–{r["size_max_b"]:g}' if r["size_min_b"] is not None
              and r["size_min_b"] != r["size_max_b"] else
              (f'{r["size_min_b"]:g}' if r["size_min_b"] is not None else "–"))
        code = "✓" if r["code_status"] == "repo" else ("~" if r["code_status"] == "promised" else "–")
        cites = r["citations"] if r["citations"] is not None else "–"
        L.append(f"| {r['rank']} | {r['composite']:g} | {r['focus']} | {fam} | {sz} | {code} | "
                 f"{cites} | {r['title'][:64]} | {r['venue']} {r['year']} |\n")
    L.append("\n## Notes\n")
    L.append("- **Code** column: ✓ repo link found, ~ code promised in abstract, – none detected "
             "(absence is not proof there is no repo — verify before building).\n")
    L.append("- **Families / Sizes** are parsed from the abstract as a cross-model-readiness signal; "
             "confirm against each paper.\n")
    L.append("- Scope is probes + steering only; SAE-only and circuit-only papers are excluded by "
             "design but remain in `data/interp_corpus.json`.\n")

    L.append("\n## How the shortlist is built\n")
    L.append("**Funnel** (corpus → selected):\n\n| Stage | Papers |\n| --- | --- |\n")
    funnel_labels = {
        "corpus": "Interp corpus",
        "has_probe_or_steer_signal": "Has probe/steering signal",
        "in_llm_context": "In LLM/NLP context",
        "after_ssl_eval_filter": "After dropping SSL/vision-eval",
        "after_modality_filter": "After dropping audio/EEG/ECG (in scope)",
    }
    for k, lab in funnel_labels.items():
        L.append(f"| {lab} | {funnel[k]} |\n")
    L.append(f"| **Selected (cap {MAX_SELECTED})** | **{len(shortlist)}** |\n")
    L.append("\n**Relevance score** (from the corpus harvester): term taxonomy over title + abstract "
             "+ keywords — strong unambiguous terms weight 3, medium 2, context 1, +1 if in the title; "
             "kept only with a strong term or an explicit interpretability keyword (min total 3).\n")
    L.append("\n**Composite score** = focus + reproducibility + cross-model + impact + recency + topic "
             "(capped at 100). Points budget:\n\n| Component | Max | What it rewards |\n| --- | --- | --- |\n")
    comp_desc = {
        "focus": f"probe/steering alignment (probing ×{PROBING_W} vs steering ×{STEERING_W})",
        "reproducibility": "public/reusable code to build the pipeline on",
        "cross_model": "model families + gradual parameter-size span",
        "impact": "citations (log-scaled)",
        "recency": "publication year",
        "topic": "safety/alignment terms (+persona steering)",
    }
    for comp, mx in WEIGHTS.items():
        L.append(f"| {comp} | {mx} | {comp_desc[comp]} |\n")
    L.append("\nFull machine-readable methodology and per-paper score breakdowns are in the "
             "`metadata` block of `data/interp_selection.json`.\n")

    L.append("\n---\nRegenerate: `python3 scripts/select_interp_targets.py` "
             f"(cap = `MAX_SELECTED` = {MAX_SELECTED}).\n")
    OUT_MD.write_text("".join(L), encoding="utf-8")

    print(f"in-scope (probe/steering): {len(cands)} ; shortlisted: {len(shortlist)} "
          f"({n_probe} probing, {n_steer} steering, {n_code} with repo)")
    print("top 8:")
    for r in shortlist[:8]:
        print(f"  {r['rank']:>2} {r['composite']:>5} {r['focus']:<8} fam={r['n_families']} "
              f"sz={r['size_min_b']}-{r['size_max_b']} code={r['code_status']:<8} {r['title'][:54]}")
    print(f"wrote {OUT_JSON.name}, {OUT_MD.name}")


if __name__ == "__main__":
    main()
