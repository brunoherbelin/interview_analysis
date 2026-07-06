#!/usr/bin/env python3
"""
Aspect 1 — Sync vs Async paired comparison of participant sentiment.

Each VM is one participant measured in BOTH conditions, so this is a within-subject
(paired) design (n = number of participants). Three interview-level scores are
compared, from three different methods:

  - roberta : mean transformer signed score  (p_pos - p_neg)      [sentiment stage]
  - vader   : mean VADER compound                                  [sentiment stage]
  - opinion : mean offline-SenticNet polarity over the FRED topics [concept level]

For each measure we run a paired t-test and a Wilcoxon signed-rank test, report the
mean paired difference (95% CI) and Cohen's dz, and check agreement among the three
methods (Pearson/Spearman). Outputs go to data/results/.

NOTE: with only a handful of participants this validates the *procedure*, not
significance (Wilcoxon cannot reach p<0.05 below n=6).

Run: venv/bin/python -m src.analysis.sync_async_sentiment
"""

import json
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.config import config, logger
from src.text.opinions import extract_interview, ConceptAffect

MEASURES = ["roberta", "vader", "opinion"]
MEASURE_LABELS = {"roberta": "RoBERTa (transformer)", "vader": "VADER (lexicon)",
                  "opinion": "Opinion (SenticNet concept)"}


def _results_dir() -> Path:
    d = Path(config.get("paths.results_dir", "data/results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _interview_score_from_kg(ttl_path: Path, affect: ConceptAffect):
    """Mean SenticNet concept polarity over the interview's FRED topics."""
    interview = extract_interview(ttl_path)
    if not interview:
        return None
    pols = []
    for u in interview["utterances"].values():
        for topic in u["topics"]:
            a = affect.lookup(topic)
            if a is not None:
                pols.append(a["polarity"])
    return float(np.mean(pols)) if pols else None


def collect_scores() -> list:
    """One row per interview: {vm, condition, roberta, vader, opinion}."""
    sent_root = Path(config.get("paths.processed_dir")) / "sentiment"
    kg_root = Path(config.get("paths.processed_dir")) / "knowledge_graph"
    affect = ConceptAffect()

    rows = []
    for sent_path in sorted(sent_root.glob("*/*.sentiment.json")):
        data = json.load(open(sent_path, encoding="utf-8"))
        segs = data.get("segments", [])
        if not segs:
            continue
        vm = data["vm"]
        condition = data["interview"]  # "Sync" / "Async"
        stem = sent_path.name[:-len(".sentiment.json")]  # <base>.<SPEAKER>
        ttl_path = kg_root / vm / f"{stem}.ttl"

        rows.append({
            "vm": vm,
            "condition": condition,
            "roberta": float(np.mean([s["signed_score"] for s in segs])),
            "vader": float(np.mean([s["vader_compound"] for s in segs])),
            "opinion": _interview_score_from_kg(ttl_path, affect) if ttl_path.exists() else None,
        })
    return rows


def _pairs(rows, measure):
    """Return aligned (sync, async) arrays paired by participant (vm)."""
    by_vm = {}
    for r in rows:
        if r[measure] is None:
            continue
        by_vm.setdefault(r["vm"], {})[r["condition"]] = r[measure]
    vms = sorted(vm for vm, c in by_vm.items() if "Sync" in c and "Async" in c)
    sync = np.array([by_vm[v]["Sync"] for v in vms])
    asy = np.array([by_vm[v]["Async"] for v in vms])
    return vms, sync, asy


def paired_tests(rows) -> dict:
    out = {}
    for m in MEASURES:
        vms, sync, asy = _pairs(rows, m)
        n = len(vms)
        diff = sync - asy
        md = float(diff.mean()) if n else None
        sd = float(diff.std(ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n else 0.0
        tcrit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0

        t_res = stats.ttest_rel(sync, asy) if n > 1 else None
        try:
            w_res = stats.wilcoxon(sync, asy) if n > 1 and np.any(diff) else None
        except ValueError:
            w_res = None

        out[m] = {
            "n_pairs": n,
            "participants": vms,
            "mean_sync": float(sync.mean()) if n else None,
            "mean_async": float(asy.mean()) if n else None,
            "sd_sync": float(sync.std(ddof=1)) if n > 1 else None,
            "sd_async": float(asy.std(ddof=1)) if n > 1 else None,
            "mean_diff_sync_minus_async": md,
            "diff_95ci": [md - tcrit * se, md + tcrit * se] if n > 1 else None,
            "cohen_dz": (md / sd) if sd > 0 else None,
            "paired_t": {"t": float(t_res.statistic), "p": float(t_res.pvalue)} if t_res else None,
            "wilcoxon": {"W": float(w_res.statistic), "p": float(w_res.pvalue)} if w_res else None,
        }
    return out


def method_agreement(rows) -> dict:
    """Pearson/Spearman among the three measures across all interviews."""
    cols = {m: np.array([r[m] for r in rows if all(r[x] is not None for x in MEASURES)])
            for m in MEASURES}
    agree = {}
    for i, a in enumerate(MEASURES):
        for b in MEASURES[i + 1:]:
            pr = stats.pearsonr(cols[a], cols[b])
            sr = stats.spearmanr(cols[a], cols[b])
            agree[f"{a}~{b}"] = {
                "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
                "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue),
            }
    return agree


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_paired(rows, tests, path: Path):
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(4 * len(MEASURES), 4.2))
    for ax, m in zip(axes, MEASURES):
        vms, sync, asy = _pairs(rows, m)
        for i, vm in enumerate(vms):
            ax.plot([0, 1], [sync[i], asy[i]], "-o", color="0.6", alpha=0.8, zorder=1)
            ax.annotate(vm, (1, asy[i]), fontsize=7, color="0.4", xytext=(4, 0),
                        textcoords="offset points", va="center")
        ax.plot([0, 1], [sync.mean(), asy.mean()], "-o", color="C3", lw=3,
                markersize=9, zorder=3, label="mean")
        ax.axhline(0, color="0.85", lw=1, zorder=0)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Sync", "Async"])
        ax.set_xlim(-0.3, 1.4)
        t = tests[m]
        sub = (f"Δ={t['mean_diff_sync_minus_async']:+.3f}, dz={t['cohen_dz']:+.2f}\n"
               f"t-test p={t['paired_t']['p']:.3f}"
               + (f", W p={t['wilcoxon']['p']:.3f}" if t['wilcoxon'] else ""))
        ax.set_title(f"{MEASURE_LABELS[m]}\n{sub}", fontsize=9)
        ax.set_ylabel("sentiment score")
    fig.suptitle(f"Sync vs Async — paired by participant (n={tests['roberta']['n_pairs']})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_correlation(rows, path: Path):
    cols = np.array([[r[m] for m in MEASURES]
                     for r in rows if all(r[x] is not None for x in MEASURES)])
    corr = np.corrcoef(cols.T)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(MEASURES))); ax.set_xticklabels(MEASURES, rotation=30, ha="right")
    ax.set_yticks(range(len(MEASURES))); ax.set_yticklabels(MEASURES)
    for i in range(len(MEASURES)):
        for j in range(len(MEASURES)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title(f"Method agreement (Pearson r, n={len(cols)} interviews)", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved {path}")


def save_csv(rows, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["vm", "condition"] + MEASURES)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info(f"Saved {path}")


def main() -> int:
    rows = collect_scores()
    if not rows:
        logger.error("No sentiment/KG outputs found; run --sentiment and --kg first.")
        return 1

    tests = paired_tests(rows)
    agreement = method_agreement(rows)

    out = _results_dir()
    save_csv(rows, out / "sync_async_sentiment_scores.csv")
    with open(out / "sync_async_sentiment_tests.json", "w", encoding="utf-8") as f:
        json.dump({"paired_tests": tests, "method_agreement": agreement,
                   "n_interviews": len(rows)}, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {out/'sync_async_sentiment_tests.json'}")
    plot_paired(rows, tests, out / "sync_async_paired.png")
    plot_correlation(rows, out / "sync_async_method_correlation.png")

    # console summary
    print("\n=== Sync vs Async (paired) ===")
    for m in MEASURES:
        t = tests[m]
        print(f"{MEASURE_LABELS[m]:28s} n={t['n_pairs']} "
              f"Sync={t['mean_sync']:+.3f} Async={t['mean_async']:+.3f} "
              f"Δ={t['mean_diff_sync_minus_async']:+.3f} dz={t['cohen_dz']:+.2f} "
              f"t_p={t['paired_t']['p']:.3f}"
              + (f" W_p={t['wilcoxon']['p']:.3f}" if t['wilcoxon'] else ""))
    print("\n=== Method agreement (Pearson r) ===")
    for k, v in agreement.items():
        print(f"  {k:18s} r={v['pearson_r']:+.2f} (p={v['pearson_p']:.3f})")
    logger.info("Sync/Async sentiment analysis complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
