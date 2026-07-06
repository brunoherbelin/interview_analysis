#!/usr/bin/env python3
"""
Human verification of the sentiment scores — distribution sanity check.

Plots per-utterance distributions of the three sentiment measures so you can eyeball
whether they look reasonable (roughly normal):
  - roberta : transformer signed score (p_pos - p_neg)      [sentiment stage]
  - vader   : VADER compound                                 [sentiment stage]
  - opinion : mean SenticNet concept polarity of the utterance's concepts [KG + SenticNet]

Each panel shows the histogram, a KDE, and a normal-reference curve (same mean/sd),
plus mean/sd/skew. No normality test is shown on purpose: at these sample sizes such
tests almost always reject and would mislead — the shape and skew are the honest read.

Run: venv/bin/python scripts/human_check_sentiments.py
Output: data/results/sentiment_histograms.png
"""

import sys
import json
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, skew

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import config, logger
from src.text.opinions import extract_interview, ConceptAffect


def collect():
    """Return per-utterance arrays for the three measures."""
    sent_root = Path(config.get("paths.processed_dir")) / "sentiment"
    kg_root = Path(config.get("paths.processed_dir")) / "knowledge_graph"

    roberta, vader = [], []
    for p in sorted(sent_root.glob("*/*.sentiment.json")):
        for s in json.load(open(p, encoding="utf-8")).get("segments", []):
            roberta.append(float(s["signed_score"]))
            vader.append(float(s["vader_compound"]))

    affect = ConceptAffect()
    opinion = []
    for ttl in sorted(kg_root.glob("*/*.ttl")):
        interview = extract_interview(ttl)
        if not interview:
            continue
        for u in interview["utterances"].values():
            pols = [affect.lookup(c)["polarity"] for c in u["topics"] if affect.lookup(c)]
            if pols:
                opinion.append(mean(pols))
    return {"roberta": roberta, "vader": vader, "opinion": opinion}


def _panel(ax, data, title):
    data = np.asarray(data, dtype=float)
    ax.hist(data, bins=30, density=True, color="0.75", edgecolor="white")
    mu, sd = float(data.mean()), float(data.std())
    xs = np.linspace(data.min(), data.max(), 200)
    if sd > 1e-6:
        try:
            ax.plot(xs, gaussian_kde(data)(xs), color="C0", lw=2, label="KDE")
        except Exception:
            pass
        ax.plot(xs, norm.pdf(xs, mu, sd), "--", color="C3", lw=1.5, label="normal ref")
    ax.set_title(f"{title}\nN={len(data)}, mean={mu:+.2f}, sd={sd:.2f}, skew={skew(data):+.2f}",
                 fontsize=9)
    ax.set_xlabel("score")
    ax.legend(fontsize=7)


def main() -> int:
    data = collect()
    if not data["roberta"]:
        logger.error("No sentiment scores found; run --sentiment and --kg first.")
        return 1

    labels = {"roberta": "RoBERTa (per utterance)", "vader": "VADER (per utterance)",
              "opinion": "Opinion / SenticNet (per utterance)"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key in zip(axes, ["roberta", "vader", "opinion"]):
        _panel(ax, data[key], labels[key])
    axes[0].set_ylabel("density")
    fig.suptitle("Distribution of sentiment scores", fontsize=11)
    fig.tight_layout()

    out = Path(config.get("paths.results_dir", "data/results"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sentiment_histograms.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved {path}")

    print("\n=== Sentiment score distributions ===")
    for key in ["roberta", "vader", "opinion"]:
        d = np.asarray(data[key])
        print(f"  {labels[key]:34s} N={len(d):4d} mean={d.mean():+.3f} "
              f"sd={d.std():.3f} skew={skew(d):+.2f}")
    logger.info("Sentiment histograms complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
