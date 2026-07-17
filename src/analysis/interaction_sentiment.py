#!/usr/bin/env python3
"""
Aspect 1c — Condition × Order interaction (2×2 crossover / carryover test).

Each participant does BOTH conditions, one per trial, in a randomised order — a 2×2
crossover in which condition (Sync/Async) and order (First/Second) are within-subject
but perfectly confounded (a participant is either Sync-first or Async-first). The
condition×order interaction is therefore the crossover *carryover / sequence* effect,
tested BETWEEN subjects by comparing the two starting groups (SyncFirst vs AsyncFirst).

Standard 2×2 crossover result (Grizzle 1965; Senn, *Cross-over Trials*): with y1, y2 the
participant's first- and second-trial scores, the subject **total** (y1 + y2) contains
each condition and each period exactly once, so the condition and order main effects are
constant across sequences — a difference in totals between SyncFirst and AsyncFirst
isolates the interaction. We test it with Mann-Whitney U (+ Welch t) per measure, apply
FDR across measures, and draw the 2×2 interaction plot (cell means ± SEM).

  - interaction (carryover)  = totals differ between sequences  (BETWEEN subjects)
  - condition main effect     = Aspect 1  (sync_async_sentiment.py, WITHIN subjects)
  - order main effect         = Aspect 1b (order_sentiment.py, WITHIN subjects)

Measures (interview-level, as in Aspects 1 & 1b): roberta, vader, opinion.

Run: venv/bin/python -m src.analysis.interaction_sentiment
"""

import json
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.config import config, logger
from src.text.opinions import extract_interview, ConceptAffect
from src.analysis.condition_contrast import _bh_fdr

MEASURES = ["roberta", "vader", "opinion"]
MEASURE_LABELS = {"roberta": "RoBERTa (transformer)", "vader": "VADER (lexicon)",
                  "opinion": "Opinion (SenticNet concept)"}
CONDITIONS = ["Sync", "Async"]
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
    """One row per interview: {vm, condition, order, roberta, vader, opinion}."""
    sent_root = Path(config.get("paths.processed_dir")) / "sentiment"
    kg_root = Path(config.get("paths.processed_dir")) / "knowledge_graph"
    affect = ConceptAffect()

    rows = []
    for sent_path in sorted(sent_root.glob("*/*.sentiment.json")):
        data = json.load(open(sent_path, encoding="utf-8"))
        segs = data.get("segments", [])
        if not segs:
            continue
        order = _trial_order(data.get("base_filename", ""))
        if order is None:
            logger.warning(f"{sent_path.name}: cannot read trial order; skipping")
            continue
        stem = sent_path.name[:-len(".sentiment.json")]
        ttl_path = kg_root / data["vm"] / f"{stem}.ttl"
        rows.append({
            "vm": data["vm"], "condition": data["interview"], "order": order,
            "roberta": float(np.mean([s["signed_score"] for s in segs])),
            "vader": float(np.mean([s["vader_compound"] for s in segs])),
            "opinion": _interview_score_from_kg(ttl_path, affect) if ttl_path.exists() else None,
        })
    return rows


def build_subjects(rows: list) -> list:
    """One record per participant with both trials + the starting sequence."""
    by_vm = defaultdict(dict)
    for r in rows:
        by_vm[r["vm"]][r["order"]] = r
    subjects = []
    for vm, d in by_vm.items():
        if "First" in d and "Second" in d:
            seq = "SyncFirst" if d["First"]["condition"] == "Sync" else "AsyncFirst"
            subjects.append({"vm": vm, "seq": seq, "First": d["First"], "Second": d["Second"]})
    return subjects


def interaction_tests(subjects: list) -> dict:
    """Per measure: between-sequence test of subject totals (= carryover/interaction)
    plus the 2×2 (condition × order) cell means for plotting."""
    out = {}
    fdr_keys, fdr_p = [], []
    for m in MEASURES:
        totals = {"SyncFirst": [], "AsyncFirst": []}
        cells = defaultdict(list)                       # (condition, order) -> values
        for s in subjects:
            f, se = s["First"][m], s["Second"][m]
            if f is None or se is None:
                continue
            totals[s["seq"]].append(f + se)             # subject total y1 + y2
            cells[(s["First"]["condition"], "First")].append(f)
            cells[(s["Second"]["condition"], "Second")].append(se)

        A = np.array(totals["SyncFirst"]); B = np.array(totals["AsyncFirst"])
        rec = {"n_syncfirst": int(A.size), "n_asyncfirst": int(B.size)}
        if A.size >= 2 and B.size >= 2:
            u = stats.mannwhitneyu(A, B, alternative="two-sided")
            t = stats.ttest_ind(A, B, equal_var=False)
            rec.update({
                "mean_total_syncfirst": float(A.mean()),
                "mean_total_asyncfirst": float(B.mean()),
                "mannwhitney_U": float(u.statistic), "p": float(u.pvalue),
                "welch_t": float(t.statistic), "welch_p": float(t.pvalue),
                "rank_biserial": float(1 - 2 * u.statistic / (A.size * B.size)),
            })
            fdr_keys.append(m); fdr_p.append(float(u.pvalue))
        rec["cells"] = {
            f"{c}_{o}": {"mean": float(np.mean(v)),
                         "sem": float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                         "n": len(v)}
            for (c, o), v in cells.items()}
        out[m] = rec

    for k, q in zip(fdr_keys, _bh_fdr(fdr_p)):
        out[k]["p_fdr"] = q
    return out


# --------------------------------------------------------------------------- #
def plot_interaction(tests: dict, path: Path):
    """2×2 interaction plot per measure: x = order, one line per condition."""
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(4 * len(MEASURES), 4.2))
    colors = {"Sync": "C0", "Async": "C3"}
    for ax, m in zip(axes, MEASURES):
        cells = tests[m]["cells"]
        for cond in CONDITIONS:
            ys = [cells.get(f"{cond}_{o}", {}).get("mean") for o in ("First", "Second")]
            es = [cells.get(f"{cond}_{o}", {}).get("sem", 0.0) for o in ("First", "Second")]
            ax.errorbar([0, 1], ys, yerr=es, marker="o", color=colors[cond], lw=2,
                        capsize=3, label=cond)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["First", "Second"]); ax.set_xlim(-0.3, 1.3)
        pf = tests[m].get("p_fdr")
        sub = f"interaction p={tests[m].get('p', float('nan')):.3f}"
        if pf is not None:
            sub += f", FDR={pf:.3f}"
        ax.set_title(f"{MEASURE_LABELS[m]}\n{sub}", fontsize=9)
        ax.set_ylabel("mean score"); ax.legend(fontsize=8, title="condition")
    fig.suptitle("Condition × Order interaction — 2×2 crossover cell means (± SEM)\n"
                 "non-parallel lines = interaction (carryover)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    logger.info(f"Saved {path}")


def save_csv(subjects: list, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        head = ["vm", "sequence"]
        for m in MEASURES:
            head += [f"{m}_first", f"{m}_second", f"{m}_total", f"{m}_first_minus_second"]
        w.writerow(head)
        for s in subjects:
            row = [s["vm"], s["seq"]]
            for m in MEASURES:
                f1, f2 = s["First"][m], s["Second"][m]
                if f1 is None or f2 is None:
                    row += ["", "", "", ""]
                else:
                    row += [round(f1, 4), round(f2, 4), round(f1 + f2, 4), round(f1 - f2, 4)]
            w.writerow(row)
    logger.info(f"Saved {path}")


def main() -> int:
    rows = collect_scores()
    subjects = build_subjects(rows)
    if len(subjects) < 4:
        logger.error("Too few complete participants for an interaction test.")
        return 1
    n_sf = sum(1 for s in subjects if s["seq"] == "SyncFirst")
    n_af = len(subjects) - n_sf
    logger.info(f"{len(subjects)} participants: {n_sf} SyncFirst, {n_af} AsyncFirst")

    tests = interaction_tests(subjects)
    out = _results_dir()
    save_csv(subjects, out / "interaction_scores.csv")
    with open(out / "interaction_tests.json", "w", encoding="utf-8") as f:
        json.dump({"n_participants": len(subjects), "n_syncfirst": n_sf, "n_asyncfirst": n_af,
                   "measures": tests}, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {out/'interaction_tests.json'}")
    plot_interaction(tests, out / "interaction_plot.png")

    print(f"\n=== Condition × Order interaction (2×2 crossover, "
          f"{n_sf} SyncFirst vs {n_af} AsyncFirst) ===")
    print("  test = Mann-Whitney on subject totals (SyncFirst − AsyncFirst); "
          "non-zero => the condition effect depends on order (carryover)")
    for m in MEASURES:
        t = tests[m]
        if "p" not in t:
            print(f"  {MEASURE_LABELS[m]:28s} (insufficient data)")
            continue
        pf = f"{t['p_fdr']:.3f}" if "p_fdr" in t else "  -  "
        print(f"  {MEASURE_LABELS[m]:28s} total_SF={t['mean_total_syncfirst']:+.3f} "
              f"total_AF={t['mean_total_asyncfirst']:+.3f}  U={t['mannwhitney_U']:.0f} "
              f"p={t['p']:.3f} p_fdr={pf}  rank-biserial={t['rank_biserial']:+.2f}")
    logger.info("Condition × Order interaction analysis complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
