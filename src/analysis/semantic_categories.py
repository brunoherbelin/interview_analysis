#!/usr/bin/env python3
"""
Semantic-category analysis — topics/opinions "beyond words", grounded in the KG.

Instead of surface lemmas (lotus, little, then), each FRED concept is lifted to a
coarse semantic category along two axes:

  - ACT   (what the participant is doing): verb frames -> WordNet verb supersense
           perception ("describing what they see"), emotion ("emotions felt"),
           cognition ("reflecting"), motion, communication, ...
  - THEME (what they talk about): noun concepts -> WordNet noun supersense
           body, feeling, time, location/space, objects/environment, ...

The categories come from the FRED graph's disambiguated predicate frames and typed
concepts (not raw text), aggregated across participants with sentiment (utterance
signed score) and SenticNet Hourglass emotion. Pools Sync + Async.

Run: venv/bin/python -m src.analysis.semantic_categories
"""

import json
import csv
from collections import defaultdict, Counter
from functools import lru_cache
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import config, logger
from src.text.opinions import (
    ConceptAffect, _concept_lemma, _load_utterance_sentiment,
    CONCEPT_STOPLIST, FRED_DOMAIN, PROPBANK_MARK,
)

FRIENDLY = {
    "verb.perception": "perception (see/look)", "verb.emotion": "emotion (feel)",
    "verb.cognition": "cognition (think/know)", "verb.motion": "motion (move)",
    "verb.communication": "communication (say)", "verb.body": "body action",
    "verb.contact": "contact/touch", "verb.change": "change", "verb.stative": "state/being",
    "verb.creation": "creation", "verb.consumption": "consumption", "verb.social": "social",
    "verb.possession": "possession", "verb.competition": "competition",
    "noun.body": "body", "noun.feeling": "feeling", "noun.cognition": "cognition/percept",
    "noun.location": "location/space", "noun.artifact": "objects (artifact)",
    "noun.object": "natural object", "noun.plant": "plant/flora", "noun.time": "time",
    "noun.event": "event", "noun.state": "state", "noun.attribute": "attribute",
    "noun.communication": "communication", "noun.substance": "substance",
    "noun.phenomenon": "phenomenon", "noun.quantity": "quantity", "noun.person": "person",
    "noun.group": "group", "noun.act": "act", "noun.animal": "animal",
}


@lru_cache(maxsize=8192)
def _supersense(lemma: str, pos: str):
    from nltk.corpus import wordnet as wn
    ss = wn.synsets(lemma.split()[0].replace(" ", "_"), pos=pos)
    return ss[0].lexname() if ss else None


def _results_dir() -> Path:
    d = Path(config.get("paths.results_dir", "data/results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
def extract_categorized(ttl_path: Path, affect: ConceptAffect) -> list:
    """One record per (utterance, category): axis, category, score, moodtags."""
    from rdflib import Graph
    from nltk.corpus import wordnet as wn  # noqa: ensures corpus is importable

    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    meta = list(g.query("""
        PREFIX ia: <http://interview-analysis.local/ontology#>
        SELECT ?vm ?type ?base WHERE {
          ?i a ia:Interview ; ia:vm ?vm ; ia:interviewType ?type ; ia:baseFilename ?base .
        } LIMIT 1"""))
    if not meta:
        return []
    vm, itype, base = (str(x) for x in meta[0])
    speaker = ttl_path.stem.rsplit(".", 1)[-1]
    sentiment = _load_utterance_sentiment(vm, base, speaker)

    # utterance -> {axis_category -> set(lemmas)}
    per_utt = defaultdict(lambda: defaultdict(set))
    starts = {}
    for idx, start, cls in g.query("""
        PREFIX ia: <http://interview-analysis.local/ontology#>
        SELECT ?idx ?start ?cls WHERE {
          ?u a ia:Utterance ; ia:index ?idx ; ia:startTime ?start .
          ?ind ia:fromUtterance ?u ; a ?cls .
        }"""):
        i = int(idx); starts[i] = float(start)
        cs = str(cls)
        lemma = _concept_lemma(cs)
        if not lemma or lemma in CONCEPT_STOPLIST:
            continue
        if PROPBANK_MARK in cs:
            ss = _supersense(lemma, wn.VERB)
            if ss:
                per_utt[i][("act", ss)].add(lemma)
        elif cs.startswith(FRED_DOMAIN):
            ss = _supersense(lemma, wn.NOUN)
            if ss:
                per_utt[i][("theme", ss)].add(lemma)

    records = []
    for i, cats in per_utt.items():
        score = sentiment.get(round(starts[i], 2))
        for (axis, cat), lemmas in cats.items():
            moods = []
            for lm in lemmas:
                a = affect.lookup(lm)
                if a:
                    moods += a["moodtags"]
            records.append({"vm": vm, "interview": itype, "axis": axis,
                            "category": cat, "score": score, "moodtags": moods})
    return records


def aggregate(records: list, axis: str) -> list:
    by_cat = defaultdict(list)
    for r in records:
        if r["axis"] == axis:
            by_cat[r["category"]].append(r)

    out = []
    for cat, recs in by_cat.items():
        by_part = defaultdict(list)
        for r in recs:
            if r["score"] is not None:
                by_part[r["vm"]].append(r["score"])
        part_scores = {p: round(mean(v), 4) for p, v in by_part.items() if v}
        if not part_scores:
            continue
        vals = list(part_scores.values())
        consensus = round(mean(vals), 4)
        pos = sum(1 for v in vals if v > 0.05); neg = sum(1 for v in vals if v < -0.05)
        label = ("contested" if pos and neg else
                 "consensus_positive" if consensus > 0.05 else
                 "consensus_negative" if consensus < -0.05 else "neutral")

        def cond(c):
            xs = [r["score"] for r in recs if r["interview"] == c and r["score"] is not None]
            return round(mean(xs), 4) if xs else None

        out.append({
            "axis": axis, "category": cat, "friendly": FRIENDLY.get(cat, cat.split(".")[-1]),
            "n_participants": len(part_scores), "n_mentions": len(recs),
            "consensus_score": consensus, "spread": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "label": label, "participant_scores": part_scores,
            "sync_score": cond("Sync"), "async_score": cond("Async"),
            "top_moods": Counter(m for r in recs for m in r["moodtags"]).most_common(4),
        })
    out.sort(key=lambda s: (s["n_participants"], s["n_mentions"]), reverse=True)
    return out


# --------------------------------------------------------------------------- #
def plot_heatmap(cats: list, title: str, path: Path):
    parts = sorted({p for c in cats for p in c["participant_scores"]})
    M = np.full((len(cats), len(parts)), np.nan)
    for i, c in enumerate(cats):
        for j, p in enumerate(parts):
            if p in c["participant_scores"]:
                M[i, j] = c["participant_scores"][p]
    vmax = max(0.2, float(np.nanmax(np.abs(M))))
    cmap = plt.get_cmap("RdYlGn").copy(); cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(1.1 * len(parts) + 4, 0.44 * len(cats) + 1.5))
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(parts))); ax.set_xticklabels(parts, rotation=30, ha="right")
    ax.set_yticks(range(len(cats))); ax.set_yticklabels([c["friendly"] for c in cats])
    for i in range(len(cats)):
        for j in range(len(parts)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, fraction=0.03, pad=0.02, label="mean sentiment")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    logger.info(f"Saved {path}")


def plot_emotions_felt(records: list, path: Path):
    """Hourglass emotions co-occurring with 'emotion' acts — the 'emotions felt' view."""
    moods = Counter(m for r in records
                    if r["axis"] == "act" and r["category"] == "verb.emotion"
                    for m in r["moodtags"])
    if not moods:
        logger.warning("No emotion-act moodtags; skipping emotions-felt plot.")
        return
    items = moods.most_common(8)[::-1]
    labels, counts = zip(*items)
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(items) + 1.2))
    ax.barh(range(len(items)), counts, color="#c44e52")
    ax.set_yticks(range(len(items))); ax.set_yticklabels(labels)
    ax.set_xlabel("occurrences within 'emotion (feel)' acts")
    ax.set_title("Emotions felt — SenticNet Hourglass within emotion acts", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    logger.info(f"Saved {path}")


def save_csv(acts, themes, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["axis", "category", "friendly", "n_participants", "n_mentions",
                    "consensus_score", "spread", "label", "sync_score", "async_score", "top_moods"])
        for c in acts + themes:
            moods = ";".join(f"{m}:{n}" for m, n in c["top_moods"])
            w.writerow([c["axis"], c["category"], c["friendly"], c["n_participants"],
                        c["n_mentions"], c["consensus_score"], c["spread"], c["label"],
                        c["sync_score"], c["async_score"], moods])
    logger.info(f"Saved {path}")


def main() -> int:
    kg_root = Path(config.get("paths.processed_dir")) / "knowledge_graph"
    ttls = sorted(kg_root.glob("*/*.ttl"))
    if not ttls:
        logger.error("No knowledge graphs found; run --kg first.")
        return 1

    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    affect = ConceptAffect()
    records = []
    for ttl in ttls:
        records += extract_categorized(ttl, affect)
        logger.info(f"{ttl.name}: cumulative {len(records)} category records")

    acts = aggregate(records, "act")
    themes = aggregate(records, "theme")

    out = _results_dir()
    save_csv(acts, themes, out / "semantic_categories.csv")
    plot_heatmap(acts, "ACT categories × participant sentiment\n(what participants do: perceive / feel / think ...)",
                 out / "semantic_act_heatmap.png")
    plot_heatmap(themes, "THEME categories × participant sentiment\n(what they talk about)",
                 out / "semantic_theme_heatmap.png")
    plot_emotions_felt(records, out / "emotions_felt_hourglass.png")

    print("\n=== ACT categories (what participants are doing), by coverage ===")
    for c in acts[:10]:
        moods = ",".join(m for m, _ in c["top_moods"]) or "-"
        print(f"  {c['friendly']:26s} P={c['n_participants']}/4 n={c['n_mentions']:4d} "
              f"cons={c['consensus_score']:+.2f} [{c['label']:18s}] {moods}")
    print("\n=== THEME categories (what they talk about), by coverage ===")
    for c in themes[:10]:
        print(f"  {c['friendly']:26s} P={c['n_participants']}/4 n={c['n_mentions']:4d} "
              f"cons={c['consensus_score']:+.2f} [{c['label']}]")
    logger.info("Semantic-category analysis complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
