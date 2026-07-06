#!/usr/bin/env python3
"""
Aspect 2 — most representative *topics* across all participants (BERTopic).

Reads the topic-modeling outputs (data/processed/topics/): the global topic model
(topics_model.json) and the per-interview topic assignments, to represent, across
the whole participant group:

  - the most representative topics (by participant coverage, then prevalence),
  - each topic's keyword label and corpus size,
  - a topic x participant *prevalence* heatmap (who talks about what, how much),
  - the Sync vs Async prevalence contrast.

Unlike representative_opinions.py (which reads the knowledge-graph opinion table
and therefore only covers participants whose KGs were built), this script depends
only on topic modeling, so it covers *every* participant.

Each VM is one participant; its Sync and Async interviews are pooled per VM. A
topic's prevalence for a participant is the share of that participant's utterances
assigned to the topic (outliers, topic -1, count toward the total but not topics).

Writes tables + figures to data/results/.

Run: venv/bin/python -m src.analysis.representative_topics
"""

import json
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import config, logger

TOP_N = 18
OUTLIER_TOPIC = -1


def _results_dir() -> Path:
    d = Path(config.get("paths.results_dir", "data/results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _topics_dir() -> Path:
    return Path(config.get("paths.processed_dir")) / "topics"


def load_topic_model() -> dict:
    path = _topics_dir() / "topics_model.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run --topics first.")
    return json.load(open(path, encoding="utf-8"))


def load_interviews() -> list:
    """Per-interview topic distributions: {vm, interview, counts{tid:int}, total}."""
    interviews = []
    for path in sorted(_topics_dir().glob("*/*.topics.json")):
        r = json.load(open(path, encoding="utf-8"))
        counts = {int(tid): int(c) for tid, c in r["topic_distribution"].items()}
        interviews.append({
            "vm": r["vm"],
            "interview": r["interview"],
            "counts": counts,
            "total": sum(counts.values()),   # all participant utterances, incl. outliers
        })
    return interviews


def _short_label(topic: dict) -> str:
    """Compact, readable topic label from its top keywords."""
    kws = topic.get("keywords", [])
    return f"{topic['topic_id']}: " + ", ".join(kws[:3]) if kws else str(topic["topic_id"])


# --------------------------------------------------------------------------- #
def build_topic_table(model: dict, interviews: list) -> list:
    """Per topic: participant coverage, prevalence, per-participant + Sync/Async share."""
    meta = {t["topic_id"]: t for t in model["topics"]}

    vm_counts = defaultdict(lambda: defaultdict(int))   # vm  -> tid -> count
    vm_total = defaultdict(int)                          # vm  -> total utterances
    cond_counts = defaultdict(lambda: defaultdict(int))  # cond -> tid -> count
    cond_total = defaultdict(int)                        # cond -> total utterances
    for iv in interviews:
        vm, cond = iv["vm"], iv["interview"]
        vm_total[vm] += iv["total"]
        cond_total[cond] += iv["total"]
        for tid, c in iv["counts"].items():
            vm_counts[vm][tid] += c
            cond_counts[cond][tid] += c

    def cond_share(cond, tid):
        tot = cond_total.get(cond, 0)
        return round(cond_counts[cond][tid] / tot, 4) if tot else None

    topics = []
    for tid, m in meta.items():
        if tid == OUTLIER_TOPIC:
            continue
        part_share = {vm: round(vm_counts[vm][tid] / vm_total[vm], 4)
                      for vm in vm_counts if vm_counts[vm].get(tid, 0) and vm_total[vm]}
        if not part_share:
            continue
        topics.append({
            "topic_id": tid,
            "label": _short_label(m),
            "keywords": m.get("keywords", []),
            "size": m.get("size", sum(vm_counts[vm][tid] for vm in vm_counts)),
            "n_participants": len(part_share),
            "n_mentions": sum(vm_counts[vm][tid] for vm in vm_counts),
            "participant_share": part_share,
            "sync_share": cond_share("Sync", tid),
            "async_share": cond_share("Async", tid),
        })

    # most shared (broad coverage), then most prevalent, first
    topics.sort(key=lambda t: (t["n_participants"], t["n_mentions"]), reverse=True)
    return topics


def save_csv(topics: list, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["topic_id", "label", "keywords", "size", "n_participants",
                    "n_mentions", "sync_share", "async_share"])
        for t in topics:
            w.writerow([t["topic_id"], t["label"], " ".join(t["keywords"]),
                        t["size"], t["n_participants"], t["n_mentions"],
                        t["sync_share"], t["async_share"]])
    logger.info(f"Saved {path}")


def plot_heatmap(top: list, path: Path):
    """Top topics x participants, cell = share of that participant's utterances."""
    participants = sorted({p for t in top for p in t["participant_share"]})
    M = np.full((len(top), len(participants)), np.nan)
    for i, t in enumerate(top):
        for j, p in enumerate(participants):
            if p in t["participant_share"]:
                M[i, j] = t["participant_share"][p]

    vmax = max(0.05, float(np.nanmax(M)))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#eeeeee")

    fig, ax = plt.subplots(figsize=(0.34 * len(participants) + 4, 0.42 * len(top) + 1.5))
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(participants)))
    ax.set_xticklabels(participants, rotation=90, ha="center", fontsize=7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([t["label"] for t in top], fontsize=8)
    ax.set_title("Representative topics: topic × participant prevalence\n"
                 "(darker = larger share of that participant's speech; grey = absent)",
                 fontsize=10)
    fig.colorbar(im, fraction=0.03, pad=0.02, label="share of participant's utterances")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_coverage_bar(top: list, path: Path):
    """Top topics by participant coverage; annotate with corpus size."""
    labels = [t["label"] for t in top][::-1]
    cover = [t["n_participants"] for t in top][::-1]
    size = [t["size"] for t in top][::-1]

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(top) + 1.5))
    ax.barh(range(len(labels)), cover, color="#4c72b0")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i, (c, sz) in enumerate(zip(cover, size)):
        ax.text(c + 0.1, i, f"n={sz}", va="center", fontsize=7, color="0.3")
    ax.set_xlabel("participants discussing the topic (coverage)")
    ax.set_title("Most representative topics across participants\n"
                 "(bar = participant coverage; n = utterances in topic corpus-wide)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved {path}")


def main() -> int:
    model = load_topic_model()
    interviews = load_interviews()
    if not interviews:
        logger.error("No per-interview topic files found; run --topics first.")
        return 1

    topics = build_topic_table(model, interviews)
    if not topics:
        logger.error("No non-outlier topics found in the topic model.")
        return 1

    top = topics[:TOP_N]
    out = _results_dir()
    n_participants = len({iv["vm"] for iv in interviews})

    save_csv(topics, out / "representative_topics.csv")
    plot_heatmap(top, out / "topic_participant_heatmap.png")
    plot_coverage_bar(top, out / "top_topics_coverage.png")

    # console summary
    print(f"\n=== Most representative topics "
          f"(of {model['n_topics']} topics, {n_participants} participants) ===")
    for t in top[:12]:
        sync = f"{t['sync_share']:.3f}" if t["sync_share"] is not None else "  -  "
        asyn = f"{t['async_share']:.3f}" if t["async_share"] is not None else "  -  "
        print(f"  [{t['topic_id']:3d}] coverage={t['n_participants']:2d}/{n_participants}  "
              f"mentions={t['n_mentions']:4d}  sync={sync} async={asyn}  "
              f"{', '.join(t['keywords'][:5])}")
    logger.info("Representative-topics analysis complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
