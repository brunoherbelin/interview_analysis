#!/usr/bin/env python3
"""
Paired within-subject contrast — what participants report differently by condition.

The clustering (participant_clustering.py) showed that Sync/Async do NOT form
separable global "experience signatures". That doesn't mean there is no difference:
it can be a systematic *shift* on specific features rather than a split into groups.
Since the condition is a KNOWN label and the design is within-subject (each VM has
both a Sync and an Async interview), the right tool is a paired contrast, not
clustering.

For a broad, interpretable feature pool grounded in the KG — Hourglass emotion
prevalences, ACT-category shares, THEME-category shares, the most frequent
ACT × THEME cells, and mean SenticNet polarity — we compute, per feature, the
paired difference (Sync − Async) across participants, its effect size (Cohen's dz),
paired t-test and Wilcoxon p-values, and a Benjamini–Hochberg FDR correction across
all tested features. The significant (or, absent those, the largest) contrasts are
plotted, giving a direct answer to "what do people report more of in each condition".

The same script does the trial-ORDER contrast (First − Second) via --factor order, and
the condition×ORDER INTERACTION via --interaction. The latter is a 2×2 crossover:
condition and order are within-subject but confounded (each participant is Sync-first or
Async-first), so the interaction is the carryover/sequence effect, tested BETWEEN the two
starting groups (SyncFirst vs AsyncFirst) by comparing subject totals (First + Second),
which cancel both main effects (Grizzle 1965; Senn, *Cross-over Trials*).

Run: venv/bin/python -m src.analysis.condition_contrast            # Sync − Async
     venv/bin/python -m src.analysis.condition_contrast --factor order   # First − Second
     venv/bin/python -m src.analysis.condition_contrast --interaction    # condition×order
"""

import csv
import json
import math
import argparse
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, ttest_ind

from src.config import config, logger
from src.text.opinions import (
    ConceptAffect, _concept_lemma, CONCEPT_STOPLIST, FRED_DOMAIN, PROPBANK_MARK,
)
from src.analysis.semantic_categories import _supersense, FRIENDLY

FACTORS = {
    "condition": {"key": "interview", "levels": ("Sync", "Async")},
    "order": {"key": "order", "levels": ("First", "Second")},
}
N_CELLS = 12          # most frequent ACT × THEME co-occurrences to include
MIN_PAIRS = 8         # need at least this many paired participants to test a feature
MIN_GROUP = 5         # min participants per starting sequence for the interaction test
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


# --------------------------------------------------------------------------- #
# Feature extraction from the KG
# --------------------------------------------------------------------------- #
def load_interview(ttl_path: Path, affect: ConceptAffect):
    """Per interview: {vm, interview, order, utts:[{acts,themes,moods}], n, polarity}."""
    from rdflib import Graph
    from nltk.corpus import wordnet as wn

    g = Graph(); g.parse(str(ttl_path), format="turtle")
    meta = list(g.query("""
        PREFIX ia: <http://interview-analysis.local/ontology#>
        SELECT ?vm ?type ?base WHERE {
          ?i a ia:Interview ; ia:vm ?vm ; ia:interviewType ?type ; ia:baseFilename ?base .
        } LIMIT 1"""))
    if not meta:
        return None
    vm, itype, base = (str(x) for x in meta[0])

    per = defaultdict(lambda: {"acts": set(), "themes": set(), "moods": set()})
    polarities = []
    for idx, cls in g.query("""
        PREFIX ia: <http://interview-analysis.local/ontology#>
        SELECT ?idx ?cls WHERE {
          ?u a ia:Utterance ; ia:index ?idx .
          ?ind ia:fromUtterance ?u ; a ?cls .
        }"""):
        i = int(idx); cs = str(cls)
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
        else:
            continue
        a = affect.lookup(lemma)
        if a:
            per[i]["moods"].update(a["moodtags"])
            polarities.append(a["polarity"])

    if not per:
        return None
    return {"vm": vm, "interview": itype, "order": _trial_order(base),
            "utts": list(per.values()), "n": len(per),
            "polarity": mean(polarities) if polarities else math.nan}


def choose_columns(interviews: list, min_present: int):
    """Feature columns kept because they are prevalent enough to be testable."""
    present = Counter()
    cell_count = Counter()
    for iv in interviews:
        emos, acts, themes = set(), set(), set()
        for u in iv["utts"]:
            acts |= u["acts"]; themes |= u["themes"]; emos |= u["moods"]
            for a in u["acts"]:
                for t in u["themes"]:
                    cell_count[(a, t)] += 1
        for m in emos:
            present[f"emo:{m}"] += 1
        for a in acts:
            present[f"act:{a}"] += 1
        for t in themes:
            present[f"theme:{t}"] += 1

    cols = [k for k, c in present.items() if c >= min_present]
    cols += [f"cell:{a}|{t}" for (a, t), _ in cell_count.most_common(N_CELLS)]
    cols.append("polarity")
    return cols


def interview_vector(iv: dict, cols: list) -> dict:
    """Per-interview feature values: shares of utterances (+ mean polarity)."""
    n = iv["n"]
    emo_c, act_c, theme_c, cell_c = Counter(), Counter(), Counter(), Counter()
    for u in iv["utts"]:
        for m in u["moods"]:
            emo_c[m] += 1
        for a in u["acts"]:
            act_c[a] += 1
        for t in u["themes"]:
            theme_c[t] += 1
        for a in u["acts"]:
            for t in u["themes"]:
                cell_c[(a, t)] += 1

    feats = {}
    for c in cols:
        if c == "polarity":
            feats[c] = iv["polarity"]
        elif c.startswith("emo:"):
            feats[c] = emo_c[c[4:]] / n
        elif c.startswith("act:"):
            feats[c] = act_c[c[4:]] / n
        elif c.startswith("theme:"):
            feats[c] = theme_c[c[6:]] / n
        elif c.startswith("cell:"):
            a, t = c[5:].split("|")
            feats[c] = cell_c[(a, t)] / n
    return feats


def feature_label(name: str) -> str:
    if name == "polarity":
        return "sentiment (SenticNet)"
    kind, rest = name.split(":", 1)
    if kind == "emo":
        return f"emotion: {rest}"
    if kind in ("act", "theme"):
        return FRIENDLY.get(rest, rest.split(".")[-1])
    if kind == "cell":
        a, t = rest.split("|")
        return f"{a.split('.')[-1]} × {t.split('.')[-1]}"
    return name


# --------------------------------------------------------------------------- #
# Paired contrast + FDR
# --------------------------------------------------------------------------- #
def _bh_fdr(pvals: list) -> list:
    """Benjamini–Hochberg FDR-adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out.tolist()


def paired_contrast(rows: list, cols: list, levels: tuple) -> list:
    """For each feature, the paired (level_a − level_b) contrast across participants."""
    la, lb = levels
    by_vm = defaultdict(dict)
    for r in rows:
        by_vm[r["vm"]][r["level"]] = r["feats"]
    paired = [v for v, d in by_vm.items() if la in d and lb in d]

    results = []
    for f in cols:
        A, B = [], []
        for v in paired:
            a, b = by_vm[v][la].get(f), by_vm[v][lb].get(f)
            if a is None or b is None or math.isnan(a) or math.isnan(b):
                continue
            A.append(a); B.append(b)
        n = len(A)
        rec = {"feature": f, "label": feature_label(f), "n_pairs": n,
               "mean_diff": None, "dz": None, "t_p": None, "wilcoxon_p": None,
               "p_fdr": None, "higher_in": None, "testable": n >= MIN_PAIRS}
        if rec["testable"]:
            A, B = np.array(A), np.array(B)
            diff = A - B
            sd = float(diff.std(ddof=1))
            rec["mean_diff"] = float(diff.mean())
            rec["dz"] = float(diff.mean() / sd) if sd > 0 else 0.0
            rec["higher_in"] = la if rec["mean_diff"] > 0 else lb
            rec["t_p"] = float(ttest_rel(A, B).pvalue)
            try:
                rec["wilcoxon_p"] = float(wilcoxon(A, B).pvalue) if np.any(diff) else None
            except ValueError:
                rec["wilcoxon_p"] = None
        results.append(rec)

    tested = [r for r in results if r["testable"]]
    # FDR over the primary (Wilcoxon, else t) p-value of each tested feature
    primary = [r["wilcoxon_p"] if r["wilcoxon_p"] is not None else r["t_p"] for r in tested]
    for r, q in zip(tested, _bh_fdr(primary)):
        r["p_fdr"] = q

    results.sort(key=lambda r: (abs(r["dz"]) if r["dz"] is not None else -1), reverse=True)
    return results


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def save_csv(results: list, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "label", "n_pairs", "mean_diff", "cohen_dz",
                    "t_p", "wilcoxon_p", "p_fdr", "higher_in", "fdr_significant"])
        for r in results:
            sig = r["p_fdr"] is not None and r["p_fdr"] < FDR_ALPHA
            w.writerow([r["feature"], r["label"], r["n_pairs"],
                        _round(r["mean_diff"]), _round(r["dz"]), _round(r["t_p"]),
                        _round(r["wilcoxon_p"]), _round(r["p_fdr"]),
                        r["higher_in"] or "", sig])
    logger.info(f"Saved {path}")


def _round(x, nd=4):
    return round(x, nd) if isinstance(x, float) else ""


def plot_contrasts(results: list, factor: str, levels: tuple, path: Path):
    la, lb = levels
    tested = [r for r in results if r["testable"]]
    sig = [r for r in tested if r["p_fdr"] is not None and r["p_fdr"] < FDR_ALPHA]
    show = sig if len(sig) >= 3 else tested[:10]     # results already sorted by |dz|
    show = sorted(show, key=lambda r: r["dz"])
    if not show:
        logger.warning("No testable features to plot.")
        return

    dz = [r["dz"] for r in show]
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in dz]
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(show) + 1.8))
    ax.barh(range(len(show)), dz, color=colors)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels([r["label"] for r in show], fontsize=8)
    for i, r in enumerate(show):
        star = " *" if (r["p_fdr"] is not None and r["p_fdr"] < FDR_ALPHA) else ""
        ax.text(dz[i] + (0.02 if dz[i] >= 0 else -0.02), i, f"{dz[i]:+.2f}{star}",
                va="center", ha="left" if dz[i] >= 0 else "right", fontsize=7, color="0.3")
    ax.axvline(0, color="0.6", lw=1)
    ax.set_xlabel(f"Cohen's dz   (→ higher in {la}    ← higher in {lb})")
    head = (f"{len(sig)} FDR-significant" if sig
            else "none FDR-significant — showing largest effects")
    ax.set_title(f"{factor.capitalize()} paired contrast: {la} − {lb}\n"
                 f"({head}; * = FDR<{FDR_ALPHA})", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    logger.info(f"Saved {path}")


def plot_feature_scatter(rows, results, levels, factor, path,
                         x_feat="emo:joy", group_sub="body"):
    """Scatter the interviews in (x_feat, aggregated group) space, marker = level.

    The y-axis aggregates the FDR-significant features whose name contains
    `group_sub` (e.g. all significant body contrasts) into one composite: the mean
    of their per-interview z-scores (so features on different scales weigh equally).
    A visual companion to the bar chart — shows how the two levels are *distributed*.
    """
    la, lb = levels
    sig = {r["feature"] for r in results
           if r["testable"] and r["p_fdr"] is not None and r["p_fdr"] < FDR_ALPHA}
    y_feats = sorted(f for f in sig if group_sub in f)
    have_x = any(x_feat in r["feats"] for r in rows)
    if not have_x or not y_feats:
        logger.info(f"Scatter skipped (need '{x_feat}' and >=1 significant "
                    f"'{group_sub}' feature; have x={have_x}, y={len(y_feats)}).")
        return

    Y = np.array([[r["feats"][f] for f in y_feats] for r in rows])
    yscore = ((Y - Y.mean(0)) / (Y.std(0) + 1e-9)).mean(axis=1)
    x = np.array([r["feats"][x_feat] for r in rows])
    lvl = [r["level"] for r in rows]
    markers = {la: "o", lb: "^"}
    colors = {la: "#1f77b4", lb: "#d62728"}

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for lv in (la, lb):
        idx = [i for i, l in enumerate(lvl) if l == lv]
        ax.scatter(x[idx], yscore[idx], marker=markers[lv], color=colors[lv], s=70,
                   edgecolor="k", alpha=0.85, label=lv, zorder=3)
        ax.scatter(x[idx].mean(), yscore[idx].mean(), marker=markers[lv],
                   color=colors[lv], s=380, edgecolor="k", linewidth=2, zorder=4)
    for i, r in enumerate(rows):
        ax.annotate(r["vm"], (x[i], yscore[i]), fontsize=6, color="0.45",
                    xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="0.85", lw=1, zorder=0)
    ax.set_xlabel(f"{feature_label(x_feat)} (share of utterances)")
    ax.set_ylabel(f"body focus — mean z of {len(y_feats)} significant "
                  f"'{group_sub}' contrasts")
    ax.set_title(f"{factor.capitalize()}: {la} vs {lb} in (joy, body) space\n"
                 f"large markers = group means", fontsize=10)
    ax.legend(title=factor)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    logger.info(f"Saved {path}")


# --------------------------------------------------------------------------- #
# Condition × Order interaction (2×2 crossover / carryover)
# --------------------------------------------------------------------------- #
def interaction_contrast(interviews, cols) -> tuple:
    """Test the condition×order interaction over the feature pool.

    Condition and order are within-subject but confounded (each participant is
    Sync-first or Async-first), so this is a 2×2 crossover: the interaction is the
    carryover/sequence effect, tested BETWEEN the two starting groups. Per feature,
    the subject total (First + Second) cancels both main effects, so we compare
    totals between SyncFirst and AsyncFirst with Mann-Whitney U (+ Welch t), FDR.
    Returns (results, n_syncfirst, n_asyncfirst).
    """
    recs = [{"vm": iv["vm"], "condition": iv["interview"], "order": iv["order"],
             "feats": interview_vector(iv, cols)} for iv in interviews]
    by_vm = defaultdict(dict)
    for r in recs:
        if r["order"] in ("First", "Second"):
            by_vm[r["vm"]][r["order"]] = r
    subjects = [("SyncFirst" if d["First"]["condition"] == "Sync" else "AsyncFirst",
                 d["First"], d["Second"])
                for d in by_vm.values() if "First" in d and "Second" in d]
    n_sf = sum(1 for s in subjects if s[0] == "SyncFirst")
    n_af = len(subjects) - n_sf

    results, fdr_p = [], []
    for f in cols:
        totals = {"SyncFirst": [], "AsyncFirst": []}
        for seq, first, second in subjects:
            a, b = first["feats"].get(f), second["feats"].get(f)
            if a is None or b is None or math.isnan(a) or math.isnan(b):
                continue
            totals[seq].append(a + b)
        A, B = np.array(totals["SyncFirst"]), np.array(totals["AsyncFirst"])
        rec = {"feature": f, "label": feature_label(f), "n_syncfirst": int(A.size),
               "n_asyncfirst": int(B.size), "mean_total_syncfirst": None,
               "mean_total_asyncfirst": None, "U": None, "p": None, "welch_p": None,
               "rank_biserial": None, "p_fdr": None, "higher_in": None,
               "testable": A.size >= MIN_GROUP and B.size >= MIN_GROUP}
        if rec["testable"]:
            u = mannwhitneyu(A, B, alternative="two-sided")
            t = ttest_ind(A, B, equal_var=False)
            rec.update(mean_total_syncfirst=float(A.mean()), mean_total_asyncfirst=float(B.mean()),
                       U=float(u.statistic), p=float(u.pvalue), welch_p=float(t.pvalue),
                       rank_biserial=float(1 - 2 * u.statistic / (A.size * B.size)),
                       higher_in="SyncFirst" if A.mean() > B.mean() else "AsyncFirst")
            fdr_p.append(float(u.pvalue))
        results.append(rec)

    for r, q in zip([r for r in results if r["testable"]], _bh_fdr(fdr_p)):
        r["p_fdr"] = q
    results.sort(key=lambda r: abs(r["rank_biserial"]) if r["rank_biserial"] is not None else -1,
                 reverse=True)
    return results, n_sf, n_af


def plot_interaction(results, n_sf, n_af, path):
    tested = [r for r in results if r["testable"]]
    if not tested:
        logger.warning("No testable features for interaction; skipping plot.")
        return
    sig = [r for r in tested if r["p_fdr"] is not None and r["p_fdr"] < FDR_ALPHA]
    show = sorted(sig if len(sig) >= 3 else tested[:12], key=lambda r: r["rank_biserial"])
    rb = [r["rank_biserial"] for r in show]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in rb]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(show) + 1.9))
    ax.barh(range(len(show)), rb, color=colors)
    ax.set_yticks(range(len(show))); ax.set_yticklabels([r["label"] for r in show], fontsize=8)
    for i, r in enumerate(show):
        star = " *" if (r["p_fdr"] is not None and r["p_fdr"] < FDR_ALPHA) else ""
        ax.text(rb[i] + (0.02 if rb[i] >= 0 else -0.02), i, f"{rb[i]:+.2f}{star}",
                va="center", ha="left" if rb[i] >= 0 else "right", fontsize=7, color="0.3")
    ax.axvline(0, color="0.6", lw=1)
    nsig = len(sig)
    ax.set_xlabel(f"rank-biserial   (→ higher total in SyncFirst    ← in AsyncFirst)")
    ax.set_title(f"Condition × Order interaction (carryover): SyncFirst vs AsyncFirst totals\n"
                 f"n={n_sf} vs {n_af}; {nsig} FDR-significant; * = FDR<{FDR_ALPHA}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    logger.info(f"Saved {path}")


def run_interaction(interviews, cols, out) -> int:
    results, n_sf, n_af = interaction_contrast(interviews, cols)
    tested = [r for r in results if r["testable"]]
    if not tested:
        logger.error("No features had enough participants per sequence for the interaction test.")
        return 1
    sig = [r for r in tested if r["p_fdr"] is not None and r["p_fdr"] < FDR_ALPHA]

    with open(out / "interaction_contrast.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "label", "n_syncfirst", "n_asyncfirst",
                    "mean_total_syncfirst", "mean_total_asyncfirst", "mannwhitney_U",
                    "p", "welch_p", "rank_biserial", "p_fdr", "higher_total_in", "fdr_significant"])
        for r in results:
            s = r["p_fdr"] is not None and r["p_fdr"] < FDR_ALPHA
            w.writerow([r["feature"], r["label"], r["n_syncfirst"], r["n_asyncfirst"],
                        _round(r["mean_total_syncfirst"]), _round(r["mean_total_asyncfirst"]),
                        _round(r["U"]), _round(r["p"]), _round(r["welch_p"]),
                        _round(r["rank_biserial"]), _round(r["p_fdr"]), r["higher_in"] or "", s])
    logger.info(f"Saved {out/'interaction_contrast.csv'}")
    plot_interaction(results, n_sf, n_af, out / "interaction_contrast.png")
    with open(out / "interaction_contrast_summary.json", "w", encoding="utf-8") as f:
        json.dump({"analysis": "condition_x_order_interaction",
                   "n_syncfirst": n_sf, "n_asyncfirst": n_af,
                   "n_features_tested": len(tested), "n_fdr_significant": len(sig),
                   "top": [{k: r[k] for k in ("feature", "label", "mean_total_syncfirst",
                            "mean_total_asyncfirst", "U", "p", "rank_biserial", "p_fdr",
                            "higher_in")} for r in tested[:15]]},
                  f, indent=2, ensure_ascii=False)

    print(f"\n=== Condition × Order interaction (2×2 crossover, "
          f"{n_sf} SyncFirst vs {n_af} AsyncFirst), {len(tested)} features tested ===")
    print(f"  test = Mann-Whitney on subject totals (First+Second) between starting groups; "
          f"FDR-significant: {len(sig)}")
    print(f"  {'feature':28s} {'rb':>6s} {'totSF':>7s} {'totAF':>7s} {'p':>6s} {'p_fdr':>6s}  higher")
    for r in tested[:12]:
        star = "*" if (r["p_fdr"] is not None and r["p_fdr"] < FDR_ALPHA) else " "
        print(f" {star}{r['label'][:27]:28s} {r['rank_biserial']:+6.2f} "
              f"{r['mean_total_syncfirst']:+7.3f} {r['mean_total_asyncfirst']:+7.3f} "
              f"{r['p']:6.3f} {r['p_fdr']:6.3f}  {r['higher_in']}")
    logger.info("Condition × Order interaction analysis complete!")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired within-subject feature contrast")
    ap.add_argument("--factor", choices=list(FACTORS), default="condition",
                    help="pairing factor: condition (Sync−Async) or order (First−Second)")
    ap.add_argument("--interaction", action="store_true",
                    help="test the condition×order interaction (SyncFirst vs AsyncFirst) "
                         "instead of a single-factor paired contrast")
    args = ap.parse_args()
    spec = FACTORS[args.factor]
    key, levels = spec["key"], spec["levels"]

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
    interviews = [iv for iv in (load_interview(t, affect) for t in ttls) if iv]
    logger.info(f"Loaded {len(interviews)} interviews")

    cols = choose_columns(interviews, min_present=max(MIN_PAIRS, len(interviews) // 5))

    if args.interaction:
        return run_interaction(interviews, cols, _results_dir())

    rows = [{"vm": iv["vm"], "level": iv[key], "feats": interview_vector(iv, cols)}
            for iv in interviews if iv[key] in levels]

    results = paired_contrast(rows, cols, levels)
    n_pairs = max((r["n_pairs"] for r in results), default=0)
    if not n_pairs:
        logger.error(f"No paired participants for factor '{args.factor}'.")
        return 1

    out = _results_dir()
    save_csv(results, out / f"{args.factor}_contrast.csv")
    plot_contrasts(results, args.factor, levels, out / f"{args.factor}_contrast.png")
    plot_feature_scatter(rows, results, levels, args.factor,
                         out / f"{args.factor}_scatter_joy_body.png")

    tested = [r for r in results if r["testable"]]
    sig = [r for r in tested if r["p_fdr"] is not None and r["p_fdr"] < FDR_ALPHA]
    with open(out / f"{args.factor}_contrast_summary.json", "w", encoding="utf-8") as f:
        json.dump({"factor": args.factor, "levels": list(levels), "n_pairs": n_pairs,
                   "n_features_tested": len(tested), "n_fdr_significant": len(sig),
                   "top": [{k: r[k] for k in ("feature", "label", "mean_diff", "dz",
                            "wilcoxon_p", "p_fdr", "higher_in")} for r in tested[:15]]},
                  f, indent=2, ensure_ascii=False)

    la, lb = levels
    print(f"\n=== {args.factor.capitalize()} paired contrast ({la} − {lb}), "
          f"n={n_pairs} participants, {len(tested)} features tested ===")
    print(f"  FDR-significant (<{FDR_ALPHA}): {len(sig)}")
    print(f"  {'feature':30s} {'dz':>6s} {'Δ':>8s} {'wilcox_p':>9s} {'p_fdr':>7s}  higher_in")
    for r in tested[:12]:
        star = "*" if (r["p_fdr"] is not None and r["p_fdr"] < FDR_ALPHA) else " "
        wp = f"{r['wilcoxon_p']:.3f}" if r["wilcoxon_p"] is not None else "  -  "
        print(f" {star}{r['label'][:29]:30s} {r['dz']:+6.2f} {r['mean_diff']:+8.4f} "
              f"{wp:>9s} {r['p_fdr']:7.3f}  {r['higher_in']}")
    logger.info(f"{args.factor.capitalize()} contrast complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
