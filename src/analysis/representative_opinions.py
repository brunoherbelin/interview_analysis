#!/usr/bin/env python3
"""
Aspect 2 — most representative shared *opinions* across all participants.

Pools Sync + Async and reads the shared-opinion table (data/processed/opinions/
opinions.json), which is built from the knowledge-graph → SenticNet pipeline, to
represent, across the whole participant group:

  - the most representative opinion topics (by participant coverage, then mentions),
  - each topic's consensus sentiment and whether it is shared or contested,
  - a topic x participant sentiment heatmap (who feels what, and how),
  - the corpus-wide emotion profile (SenticNet Hourglass moodtags).

Note: this depends on the knowledge graphs, so it only covers participants whose
KGs were built. For a topic view that covers *every* participant, see the
BERTopic-based representative_topics.py.

Writes tables + figures to data/results/.

Run: venv/bin/python -m src.analysis.representative_opinions
"""

import json
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.config import config, logger

TOP_N = 18
LABEL_COLORS = {
    "consensus_positive": "#2ca02c",
    "consensus_negative": "#d62728",
    "contested": "#ff7f0e",
    "neutral": "#7f7f7f",
    "individual": "#c7c7c7",
}


def _results_dir() -> Path:
    d = Path(config.get("paths.results_dir", "data/results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_opinions() -> dict:
    path = Path(config.get("paths.processed_dir")) / "opinions" / "opinions.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run --opinions first.")
    return json.load(open(path, encoding="utf-8"))


# --------------------------------------------------------------------------- #
def save_csv(shared, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["topic", "n_participants", "n_mentions", "consensus_score",
                    "spread", "label", "sync_score", "async_score", "top_moods"])
        for s in shared:
            moods = ";".join(f"{m}:{c}" for m, c in s["top_moods"])
            w.writerow([s["topic"], s["n_participants"], s["n_mentions"],
                        s["consensus_score"], s["spread"], s["label"],
                        s["sync_score"], s["async_score"], moods])
    logger.info(f"Saved {path}")


def plot_heatmap(top, path: Path):
    """Top topics x participants, cell = that participant's mean sentiment."""
    participants = sorted({p for s in top for p in s["participant_scores"]})
    M = np.full((len(top), len(participants)), np.nan)
    for i, s in enumerate(top):
        for j, p in enumerate(participants):
            if p in s["participant_scores"]:
                M[i, j] = s["participant_scores"][p]

    vmax = max(0.2, float(np.nanmax(np.abs(M))))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#eeeeee")

    fig, ax = plt.subplots(figsize=(1.1 * len(participants) + 3, 0.42 * len(top) + 1.5))
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(participants))); ax.set_xticklabels(participants, rotation=30, ha="right")
    ax.set_yticks(range(len(top))); ax.set_yticklabels([s["topic"] for s in top])
    for i in range(len(top)):
        for j in range(len(participants)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                        color="black")
    ax.set_title("Shared opinions: topic × participant sentiment\n"
                 "(green = positive, red = negative, grey = not mentioned)", fontsize=10)
    fig.colorbar(im, fraction=0.03, pad=0.02, label="mean sentiment")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_coverage_bar(top, path: Path):
    """Top topics by participant coverage, coloured by consensus label."""
    topics = [s["topic"] for s in top][::-1]
    cover = [s["n_participants"] for s in top][::-1]
    cons = [s["consensus_score"] for s in top][::-1]
    colors = [LABEL_COLORS.get(s["label"], "#7f7f7f") for s in top][::-1]

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(top) + 1.5))
    ax.barh(range(len(topics)), cover, color=colors)
    ax.set_yticks(range(len(topics))); ax.set_yticklabels(topics)
    for i, (c, sc) in enumerate(zip(cover, cons)):
        ax.text(c + 0.05, i, f"{sc:+.2f}", va="center", fontsize=7, color="0.3")
    ax.set_xlabel("participants mentioning the topic (coverage)")
    ax.set_title("Most representative opinions across participants\n"
                 "(bar colour = consensus; number = mean sentiment)", fontsize=10)
    ax.legend(handles=[Patch(color=v, label=k) for k, v in LABEL_COLORS.items()],
              fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_emotion_profile(shared, path: Path):
    """Corpus-wide SenticNet Hourglass emotion frequencies."""
    moods = Counter()
    for s in shared:
        for mood, count in s["top_moods"]:
            moods[mood] += count
    if not moods:
        logger.warning("No emotion tags available; skipping emotion profile.")
        return
    items = moods.most_common(10)[::-1]
    labels, counts = zip(*items)
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(items) + 1.2))
    ax.barh(range(len(items)), counts, color="#4c72b0")
    ax.set_yticks(range(len(items))); ax.set_yticklabels(labels)
    ax.set_xlabel("occurrences across participants' topics")
    ax.set_title("Corpus emotion profile (SenticNet Hourglass)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved {path}")


def main() -> int:
    data = load_opinions()
    shared = data.get("shared", [])
    if not shared:
        logger.error("No shared opinions found; run --opinions first.")
        return 1

    # already sorted by (coverage, mentions); representative = the most shared
    top = shared[:TOP_N]
    out = _results_dir()

    save_csv(shared, out / "representative_opinions.csv")
    plot_heatmap(top, out / "opinion_participant_heatmap.png")
    plot_coverage_bar(top, out / "top_opinions_coverage.png")
    plot_emotion_profile(shared, out / "opinion_emotion_profile.png")

    # console summary
    print(f"\n=== Most representative opinions (of {data['n_topics']} topics total) ===")
    for s in top[:12]:
        moods = ",".join(m for m, _ in s["top_moods"]) or "-"
        print(f"  {s['topic']:14s} coverage={s['n_participants']}/4  mentions={s['n_mentions']:3d}  "
              f"consensus={s['consensus_score']:+.2f} [{s['label']:18s}] {moods}")
    logger.info("Representative-topics analysis complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
