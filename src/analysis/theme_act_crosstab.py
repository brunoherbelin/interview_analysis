#!/usr/bin/env python3
"""
ACT x THEME cross-tab — sentiment by what participants do AND what about.

For every participant utterance we read its ACT categories (verb frames -> verb
supersense) and THEME categories (noun concepts -> noun supersense) from the FRED
graph. Each (act, theme) that co-occur in an utterance inherit that utterance's
sentiment; cells are averaged per participant, then across participants (so no
single talkative participant dominates). Pools Sync + Async.

Produces an ACT x THEME sentiment map (e.g. "perception of the body", "emotion
about the environment") plus a co-occurrence frequency map.

Run: venv/bin/python -m src.analysis.theme_act_crosstab
"""

import csv
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from src.config import config, logger
from src.text.opinions import (
    _concept_lemma, _load_utterance_sentiment,
    CONCEPT_STOPLIST, FRED_DOMAIN, PROPBANK_MARK,
)
from src.analysis.semantic_categories import _supersense, FRIENDLY

N_ACTS = 8
N_THEMES = 10
MIN_PARTICIPANTS = 2  # only show a cell backed by >= this many participants
MIN_PAIRS_TEST = 6    # only significance-test a cell backed by >= this many PAIRED participants
FDR_ALPHA = 0.05


def _results_dir() -> Path:
    d = Path(config.get("paths.results_dir", "data/results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _trial_order(base_filename: str):
    """First/Second trial from the filename, e.g. VM27-1-Sync -> 'First'."""
    parts = base_filename.split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        return {"1": "First", "2": "Second"}.get(parts[1])
    return None


def utterance_categories(ttl_path: Path) -> list:
    """Per utterance: {vm, interview, order, score, acts:set, themes:set}."""
    from rdflib import Graph
    from nltk.corpus import wordnet as wn

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

    per = defaultdict(lambda: {"acts": set(), "themes": set()})
    starts = {}
    for idx, start, cls in g.query("""
        PREFIX ia: <http://interview-analysis.local/ontology#>
        SELECT ?idx ?start ?cls WHERE {
          ?u a ia:Utterance ; ia:index ?idx ; ia:startTime ?start .
          ?ind ia:fromUtterance ?u ; a ?cls .
        }"""):
        i = int(idx); starts[i] = float(start); cs = str(cls)
        lemma = _concept_lemma(cs)
        if not lemma or lemma in CONCEPT_STOPLIST:
            continue
        if PROPBANK_MARK in cs:
            ss = _supersense(lemma, wn.VERB)
            if ss:
                per[i]["acts"].add(ss)
        elif cs.startswith(FRED_DOMAIN):
            ss = _supersense(lemma, wn.NOUN)
            if ss:
                per[i]["themes"].add(ss)

    order = _trial_order(base)
    return [{"vm": vm, "interview": itype, "order": order,
             "score": sentiment.get(round(starts[i], 2)),
             "acts": d["acts"], "themes": d["themes"]}
            for i, d in per.items()]


def build_crosstab(utts: list, acts=None, themes=None):
    """Return (acts, themes, sentiment_matrix, count_matrix, coverage_matrix).

    If acts/themes are given, use them verbatim (so panels share axes); otherwise
    pick the most prevalent ones.
    """
    if acts is None:
        acts = [a for a, _ in Counter(a for u in utts for a in u["acts"]).most_common(N_ACTS)]
    if themes is None:
        themes = [t for t, _ in Counter(t for u in utts for t in u["themes"]).most_common(N_THEMES)]
    ai = {a: i for i, a in enumerate(acts)}
    tj = {t: j for j, t in enumerate(themes)}

    # cell -> vm -> list of utterance scores
    cell = defaultdict(lambda: defaultdict(list))
    for u in utts:
        if u["score"] is None:
            continue
        for a in u["acts"] & set(acts):
            for t in u["themes"] & set(themes):
                cell[(ai[a], tj[t])][u["vm"]].append(u["score"])

    S = np.full((len(acts), len(themes)), np.nan)
    C = np.zeros((len(acts), len(themes)), dtype=int)
    P = np.zeros((len(acts), len(themes)), dtype=int)
    for (i, j), by_vm in cell.items():
        part_means = [mean(v) for v in by_vm.values() if v]
        C[i, j] = sum(len(v) for v in by_vm.values())
        P[i, j] = len(part_means)
        if len(part_means) >= MIN_PARTICIPANTS:
            S[i, j] = mean(part_means)
    return acts, themes, S, C, P


def plot_sentiment(acts, themes, S, P, path: Path):
    vmax = max(0.2, float(np.nanmax(np.abs(S)))) if np.isfinite(S).any() else 0.2
    cmap = plt.get_cmap("RdYlGn").copy(); cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(1.05 * len(themes) + 3.5, 0.62 * len(acts) + 2))
    im = ax.imshow(np.ma.masked_invalid(S), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(themes)))
    ax.set_xticklabels([FRIENDLY.get(t, t.split(".")[-1]) for t in themes], rotation=35, ha="right")
    ax.set_yticks(range(len(acts)))
    ax.set_yticklabels([FRIENDLY.get(a, a.split(".")[-1]) for a in acts])
    for i in range(len(acts)):
        for j in range(len(themes)):
            if not np.isnan(S[i, j]):
                ax.text(j, i, f"{S[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("THEME (what they talk about)")
    ax.set_ylabel("ACT (what they do)")
    ax.set_title("Sentiment by ACT × THEME  (mean across participants)\n"
                 f"grey = fewer than {MIN_PARTICIPANTS} participants", fontsize=10)
    fig.colorbar(im, fraction=0.03, pad=0.02, label="mean sentiment")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    logger.info(f"Saved {path}")


def plot_frequency(acts, themes, C, path: Path):
    fig, ax = plt.subplots(figsize=(1.05 * len(themes) + 3.5, 0.62 * len(acts) + 2))
    im = ax.imshow(C, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(themes)))
    ax.set_xticklabels([FRIENDLY.get(t, t.split(".")[-1]) for t in themes], rotation=35, ha="right")
    ax.set_yticks(range(len(acts)))
    ax.set_yticklabels([FRIENDLY.get(a, a.split(".")[-1]) for a in acts])
    for i in range(len(acts)):
        for j in range(len(themes)):
            if C[i, j]:
                ax.text(j, i, str(C[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if C[i, j] > C.max() * 0.6 else "black")
    ax.set_xlabel("THEME"); ax.set_ylabel("ACT")
    ax.set_title("Co-occurrence frequency (utterances) by ACT × THEME", fontsize=10)
    fig.colorbar(im, fraction=0.03, pad=0.02, label="utterances")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    logger.info(f"Saved {path}")


def cell_participant_means(utts, act, theme):
    """Per-participant mean utterance sentiment for one (act, theme) cell."""
    return list(cell_means_by_vm(utts, act, theme).values())


def cell_means_by_vm(utts, act, theme) -> dict:
    """vm -> mean utterance sentiment for one (act, theme) cell (for paired tests)."""
    by_vm = defaultdict(list)
    for u in utts:
        if u["score"] is not None and act in u["acts"] and theme in u["themes"]:
            by_vm[u["vm"]].append(u["score"])
    return {v: mean(s) for v, s in by_vm.items() if s}


def _bh_fdr(pvals: list) -> list:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out.tolist()


def plot_sentiment_pair(acts, themes, S_a, S_b, labels, path: Path):
    """Two ACT × THEME sentiment maps side by side (label_a | label_b), shared scale."""
    la, lb = labels
    finite = np.concatenate([S_a[np.isfinite(S_a)], S_b[np.isfinite(S_b)]])
    vmax = max(0.2, float(np.abs(finite).max())) if finite.size else 0.2
    cmap = plt.get_cmap("RdYlGn").copy(); cmap.set_bad("#eeeeee")
    ylab = [FRIENDLY.get(a, a.split(".")[-1]) for a in acts]
    xlab = [FRIENDLY.get(t, t.split(".")[-1]) for t in themes]

    fig, axes = plt.subplots(1, 2, figsize=(2.0 * len(themes) + 4, 0.6 * len(acts) + 2),
                             sharey=True)
    for ax, S, title in zip(axes, [S_a, S_b], [la, lb]):
        im = ax.imshow(np.ma.masked_invalid(S), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(themes))); ax.set_xticklabels(xlab, rotation=35, ha="right")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("THEME")
        for i in range(len(acts)):
            for j in range(len(themes)):
                if not np.isnan(S[i, j]):
                    ax.text(j, i, f"{S[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    axes[0].set_yticks(range(len(acts))); axes[0].set_yticklabels(ylab)
    axes[0].set_ylabel("ACT")
    fig.suptitle(f"Sentiment by ACT × THEME — {la} vs {lb}  "
                 f"(grey = <{MIN_PARTICIPANTS} participants)", fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="mean sentiment")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_frequency_pair(acts, themes, C_a, C_b, labels, path: Path):
    """Two ACT × THEME co-occurrence count maps side by side, shared scale."""
    la, lb = labels
    vmax = max(1, int(max(C_a.max(), C_b.max())))
    ylab = [FRIENDLY.get(a, a.split(".")[-1]) for a in acts]
    xlab = [FRIENDLY.get(t, t.split(".")[-1]) for t in themes]

    fig, axes = plt.subplots(1, 2, figsize=(2.0 * len(themes) + 4, 0.6 * len(acts) + 2),
                             sharey=True)
    for ax, C, title in zip(axes, [C_a, C_b], [la, lb]):
        im = ax.imshow(C, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(themes))); ax.set_xticklabels(xlab, rotation=35, ha="right")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("THEME")
        for i in range(len(acts)):
            for j in range(len(themes)):
                if C[i, j]:
                    ax.text(j, i, str(int(C[i, j])), ha="center", va="center", fontsize=7,
                            color="white" if C[i, j] > vmax * 0.6 else "black")
    axes[0].set_yticks(range(len(acts))); axes[0].set_yticklabels(ylab)
    axes[0].set_ylabel("ACT")
    fig.suptitle(f"Co-occurrence frequency by ACT × THEME — {la} vs {lb}", fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="utterances")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_pair_by_factor(utts, acts, themes, key, labels, factor, out):
    """Build the two-panel sentiment + frequency maps for one factor (condition/order)."""
    la, lb = labels
    a_utts = [u for u in utts if u.get(key) == la]
    b_utts = [u for u in utts if u.get(key) == lb]
    _, _, S_a, C_a, _ = build_crosstab(a_utts, acts, themes)
    _, _, S_b, C_b, _ = build_crosstab(b_utts, acts, themes)
    plot_sentiment_pair(acts, themes, S_a, S_b, labels,
                        out / f"theme_act_sentiment_by_{factor}.png")
    plot_frequency_pair(acts, themes, C_a, C_b, labels,
                        out / f"theme_act_frequency_by_{factor}.png")
    return a_utts, b_utts


def select_pairs(sync_utts, async_utts, acts, themes, named, n_total=9):
    """Named pairs (with support) + the largest Sync/Async gaps."""
    def supported(a, t):
        return (len(cell_participant_means(sync_utts, a, t)) >= MIN_PARTICIPANTS and
                len(cell_participant_means(async_utts, a, t)) >= MIN_PARTICIPANTS)

    selected = [p for p in named if p[0] in acts and p[1] in themes and supported(*p)]
    cand = []
    for a in acts:
        for t in themes:
            if (a, t) in selected or not supported(a, t):
                continue
            d = abs(mean(cell_participant_means(async_utts, a, t)) -
                    mean(cell_participant_means(sync_utts, a, t)))
            cand.append((d, (a, t)))
    for _, p in sorted(cand, reverse=True):
        if len(selected) >= n_total:
            break
        selected.append(p)
    return selected


def plot_selected_bars(pairs, sync_utts, async_utts, path: Path):
    """Grouped bars (white=Sync, black=Async) for selected ACT × THEME pairs."""
    labels, sy, sye, ay, aye = [], [], [], [], []
    for a, t in pairs:
        sm = cell_participant_means(sync_utts, a, t)
        am = cell_participant_means(async_utts, a, t)
        labels.append(f"{a.split('.')[-1]}\n×{t.split('.')[-1]}")
        sy.append(np.mean(sm)); sye.append(np.std(sm, ddof=1) / np.sqrt(len(sm)) if len(sm) > 1 else 0)
        ay.append(np.mean(am)); aye.append(np.std(am, ddof=1) / np.sqrt(len(am)) if len(am) > 1 else 0)

    x = np.arange(len(pairs)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.15 * len(pairs) + 2, 4.6))
    ax.bar(x - w / 2, sy, w, yerr=sye, capsize=3, color="white", edgecolor="black", label="Sync")
    ax.bar(x + w / 2, ay, w, yerr=aye, capsize=3, color="black", label="Async")
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean sentiment (± SEM across participants)")
    ax.set_title("Sync vs Async for selected ACT × THEME pairs", fontsize=11)
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    logger.info(f"Saved {path}")


def save_condition_csv(acts, themes, sync_utts, async_utts, path: Path) -> int:
    """Per-cell Sync vs Async sentiment, with a PAIRED significance test.

    Each participant present in a cell under BOTH conditions contributes a paired
    difference (Sync − Async); we report Cohen's dz, the Wilcoxon p, and a
    Benjamini-Hochberg FDR correction across all tested cells — so the table shows
    which colour differences are real and which are within noise. Returns the number
    of FDR-significant cells.
    """
    cells = []
    for a in acts:
        for t in themes:
            sv = cell_means_by_vm(sync_utts, a, t)
            av = cell_means_by_vm(async_utts, a, t)
            if not sv and not av:
                continue
            vms = [v for v in sv if v in av]
            rec = {"a": a, "t": t,
                   "sync": round(mean(list(sv.values())), 4) if sv else "",
                   "n_sync": len(sv), "async_": round(mean(list(av.values())), 4) if av else "",
                   "n_async": len(av), "n_pairs": len(vms),
                   "mean_diff": "", "dz": "", "wp": None, "p_fdr": ""}
            if len(vms) >= MIN_PAIRS_TEST:
                s = np.array([sv[v] for v in vms]); asy = np.array([av[v] for v in vms])
                d = s - asy
                sd = float(d.std(ddof=1))
                rec["mean_diff"] = round(float(d.mean()), 4)
                rec["dz"] = round(float(d.mean() / sd), 4) if sd > 0 else 0.0
                if np.any(d):
                    try:
                        rec["wp"] = float(wilcoxon(s, asy).pvalue)
                    except ValueError:
                        rec["wp"] = None
            cells.append(rec)

    testable = [c for c in cells if c["wp"] is not None]
    for c, q in zip(testable, _bh_fdr([c["wp"] for c in testable])):
        c["p_fdr"] = round(q, 4)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["act", "theme", "sync_sentiment", "sync_participants",
                    "async_sentiment", "async_participants", "n_pairs",
                    "mean_diff_sync_minus_async", "cohen_dz", "wilcoxon_p",
                    "p_fdr", "fdr_significant"])
        for c in cells:
            sig = c["p_fdr"] != "" and c["p_fdr"] < FDR_ALPHA
            w.writerow([FRIENDLY.get(c["a"], c["a"]), FRIENDLY.get(c["t"], c["t"]),
                        c["sync"], c["n_sync"], c["async_"], c["n_async"], c["n_pairs"],
                        c["mean_diff"], c["dz"],
                        round(c["wp"], 4) if c["wp"] is not None else "", c["p_fdr"], sig])

    n_sig = sum(1 for c in cells if c["p_fdr"] != "" and c["p_fdr"] < FDR_ALPHA)
    logger.info(f"Saved {path} ({len(testable)} cells tested, {n_sig} FDR-significant)")
    return n_sig


def save_csv(acts, themes, S, C, P, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["act", "theme", "mean_sentiment", "n_utterances", "n_participants"])
        for i, a in enumerate(acts):
            for j, t in enumerate(themes):
                if C[i, j]:
                    s = "" if np.isnan(S[i, j]) else round(float(S[i, j]), 4)
                    w.writerow([FRIENDLY.get(a, a), FRIENDLY.get(t, t), s, C[i, j], P[i, j]])
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

    utts = []
    for ttl in ttls:
        utts += utterance_categories(ttl)
    logger.info(f"Collected {len(utts)} utterances from {len(ttls)} interviews")

    # pooled axes (so all panels share the same acts/themes ordering)
    acts, themes, S, C, P = build_crosstab(utts)
    out = _results_dir()
    save_csv(acts, themes, S, C, P, out / "theme_act_crosstab.csv")
    plot_sentiment(acts, themes, S, P, out / "theme_act_sentiment.png")
    plot_frequency(acts, themes, C, out / "theme_act_frequency.png")

    # split by condition and by trial order, aligned to the pooled axes
    # (each produces a sentiment pair map + a frequency pair map)
    sync_utts, async_utts = plot_pair_by_factor(
        utts, acts, themes, "interview", ("Sync", "Async"), "condition", out)
    plot_pair_by_factor(
        utts, acts, themes, "order", ("First", "Second"), "order", out)
    n_sig = save_condition_csv(acts, themes, sync_utts, async_utts,
                               out / "theme_act_crosstab_by_condition.csv")

    named = [("verb.motion", "noun.state"), ("verb.perception", "noun.event"),
             ("verb.emotion", "noun.body")]
    pairs = select_pairs(sync_utts, async_utts, acts, themes, named)
    plot_selected_bars(pairs, sync_utts, async_utts, out / "theme_act_selected_bars.png")

    print(f"\n=== Selected ACT × THEME pairs: Sync vs Async "
          f"({n_sig} cell(s) FDR-significant, see by_condition.csv) ===")
    for a, t in pairs:
        sm = cell_participant_means(sync_utts, a, t)
        am = cell_participant_means(async_utts, a, t)
        print(f"  {a.split('.')[-1]:12s} × {t.split('.')[-1]:12s} "
              f"Sync={mean(sm):+.2f} Async={mean(am):+.2f}  Δ={mean(am)-mean(sm):+.2f}")
    logger.info("ACT × THEME cross-tab (incl. Sync/Async) complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
