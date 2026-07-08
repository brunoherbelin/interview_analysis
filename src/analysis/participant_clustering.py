#!/usr/bin/env python3
"""
Interpretable clustering of interviews by "experience signature".

Instead of clustering a large, redundant feature matrix (curse of dimensionality,
uninterpretable), each interview is reduced to a small, NAMED signature chosen from
prior analyses — features that are prevalent across participants (so they describe
the shared experience) and/or discriminate Sync vs Async:

  - joy / eagerness / pleasantness — prevalence of the corpus's top Hourglass emotions
  - body — thinking about / feeling in the body (cognition/emotion acts × body theme)
  - motion — describing movement (motion acts over state/object/artifact themes),
    a Sync/Async discriminator
  - polarity — mean SenticNet concept polarity (its uneven spread helps split clusters)

Two-step, validated clustering (the established fix for small-n high-dim data):
standardize -> choose k by silhouette -> k-means (baseline) + Gaussian Mixture
(soft) -> bootstrap stability -> PCA view. Every cluster is describable from its
feature profile, and related to Sync/Async.

Refs: Hourglass of Emotions (Cambria et al.); two-step reduce-then-cluster for
interpretable small-sample clustering (e.g. Dalmaijer et al. 2022, power for
cluster analysis). Applies meaningfully at n>=~30 participants; runs now as a
smoke test on the available interviews.

Run: venv/bin/python -m src.analysis.participant_clustering
"""

import json
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

from src.config import config, logger
from src.text.opinions import (
    ConceptAffect, _concept_lemma,
    CONCEPT_STOPLIST, FRED_DOMAIN, PROPBANK_MARK,
)
from src.analysis.semantic_categories import _supersense

# Criterion 1 — the most prevalent Hourglass emotions in the corpus (per-interview
# intensity varies). MOODTAGS is the single source of truth (the emotion feature
# columns are built from it); re-derive it if the SenticNet source/labels change.
MOODTAGS = ["joy", "eagerness", "pleasantness"]
FEATURES = MOODTAGS + ["body", "motion", "polarity"]
FEATURE_LABELS = {
    **{m: f"emotion: {m}" for m in MOODTAGS},
    "body": "body (cognition/emotion)", "motion": "motion × state/object",
    "polarity": "sentiment (SenticNet)",
}
# Criterion 2 — ACT × THEME cells: thinking/feeling about the body, and motion
# dynamics (state/object/artifact themes) which discriminate Sync/Async.
BODY_ACTS = {"verb.cognition", "verb.emotion"}
MOTION_THEMES = {"noun.state", "noun.object", "noun.artifact"}
K_MAX = 4
RANDOM_STATE = 42


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


def interview_features(ttl_path: Path, affect: ConceptAffect):
    """Compute the compact experience signature for one interview.

    All features are shares of the participant's utterances (∈ [0, 1]) except
    `polarity`, the mean SenticNet concept polarity over the interview.
    """
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

    per = defaultdict(lambda: {"acts": set(), "themes": set(), "lemmas": set()})
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
            per[i]["lemmas"].add(lemma)
        elif cs.startswith(FRED_DOMAIN):
            ss = _supersense(lemma, wn.NOUN)
            if ss:
                per[i]["themes"].add(ss)
            per[i]["lemmas"].add(lemma)

    if not per:
        return None
    n = len(per)

    # Hourglass emotion prevalence (per utterance) + mean SenticNet polarity (per concept)
    mood_counts = {m: 0 for m in MOODTAGS}
    polarities = []
    for d in per.values():
        moods = set()
        for lm in d["lemmas"]:
            a = affect.lookup(lm)
            if a:
                moods.update(a["moodtags"])
                polarities.append(a["polarity"])
        for m in MOODTAGS:
            if m in moods:
                mood_counts[m] += 1

    def act_theme_share(acts_ok, themes_ok):
        return sum(1 for d in per.values()
                   if (d["acts"] & acts_ok) and (d["themes"] & themes_ok)) / n

    row = {
        "vm": vm, "interview": itype, "base": base, "order": _trial_order(base),
        "body": act_theme_share(BODY_ACTS, {"noun.body"}),
        "motion": act_theme_share({"verb.motion"}, MOTION_THEMES),
        "polarity": mean(polarities) if polarities else 0.0,
    }
    for m in MOODTAGS:
        row[m] = mood_counts[m] / n
    return row


def select_k(X, k_range):
    """Silhouette (k-means) and BIC (GMM) across k; return (best_k, diagnostics)."""
    diag = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        sil = float(silhouette_score(X, km.labels_)) if k < len(X) else float("nan")
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=RANDOM_STATE).fit(X)
        diag[k] = {"silhouette": sil, "bic": float(gm.bic(X))}
    best_k = max(k_range, key=lambda k: diag[k]["silhouette"])
    return best_k, diag


def bootstrap_stability(X, k, B=200, frac=0.8):
    """Mean adjusted-Rand between full-data labels and subsample re-clusterings."""
    ref = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
    n = len(X); m = max(k + 1, int(frac * n))
    if m >= n:
        return float("nan")
    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    for _ in range(B):
        idx = rng.choice(n, m, replace=False)
        lab = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X[idx])
        scores.append(adjusted_rand_score(ref[idx], lab))
    return float(np.mean(scores))


def plot_profiles(rows, Xz, labels, k, path: Path, features):
    """Heatmap: standardized feature means per cluster (the interpretable summary)."""
    means = np.array([Xz[labels == c].mean(axis=0) for c in range(k)]).T  # features x clusters
    vmax = max(0.5, float(np.abs(means).max()))
    fig, ax = plt.subplots(figsize=(1.4 * k + 3, 0.5 * len(features) + 1.5))
    im = ax.imshow(means, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    sizes = [int((labels == c).sum()) for c in range(k)]
    ax.set_xticks(range(k)); ax.set_xticklabels([f"C{c}\n(n={sizes[c]})" for c in range(k)])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([FEATURE_LABELS[f] for f in features])
    for i in range(len(features)):
        for c in range(k):
            ax.text(c, i, f"{means[i, c]:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Cluster profiles (standardized feature means)", fontsize=10)
    fig.colorbar(im, fraction=0.04, pad=0.03, label="z-score")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    logger.info(f"Saved {path}")


def plot_pca(rows, Xz, labels, k, path: Path, group_key="interview",
             markers=None, marker_title="Sync / Async"):
    """PCA scatter: colour = cluster, marker = a chosen grouping (condition or order)."""
    markers = markers or {"Sync": "o", "Async": "^"}
    pcs = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xz)
    P = pcs.transform(Xz)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(rows):
        ax.scatter(P[i, 0], P[i, 1], color=cmap(labels[i] % 10),
                   marker=markers.get(r.get(group_key), "s"), s=80, edgecolor="k", zorder=3)
        ax.annotate(f"{r['vm']}", (P[i, 0], P[i, 1]), fontsize=7, xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]*100:.0f}%)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(c % 10), label=f"C{c}")
               for c in range(k)]
    handles += [plt.Line2D([], [], marker=m, ls="", color="0.4", label=g)
                for g, m in markers.items()]
    ax.legend(handles=handles, fontsize=8, ncol=2)
    ax.set_title(f"Interviews in signature space (PCA)\nmarker = {marker_title}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
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
    rows = [f for f in (interview_features(t, affect) for t in ttls) if f]
    n = len(rows)
    logger.info(f"Built signatures for {n} interviews")
    if n < 4:
        logger.error("Too few interviews to cluster.")
        return 1

    # Variance guard: a near-constant feature only adds noise after standardizing.
    stds = np.array([r[f] for r in rows for f in FEATURES]).reshape(len(rows), -1).std(axis=0)
    features = [f for f, s in zip(FEATURES, stds) if s > 1e-9]
    dropped = [f for f, s in zip(FEATURES, stds) if s <= 1e-9]
    if dropped:
        logger.warning(f"Dropping near-constant feature(s) from clustering: {', '.join(dropped)}")

    X = np.array([[r[f] for f in features] for r in rows])
    Xz = StandardScaler().fit_transform(X)

    k_range = list(range(2, min(K_MAX, n - 1) + 1))
    best_k, diag = select_k(Xz, k_range)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(Xz)
    labels = km.labels_
    gm = GaussianMixture(n_components=best_k, covariance_type="full",
                         random_state=RANDOM_STATE).fit(Xz)
    gm_labels = gm.predict(Xz)
    stability = bootstrap_stability(Xz, best_k)

    # cluster x condition
    cond_tab = defaultdict(lambda: defaultdict(int))
    for r, c in zip(rows, labels):
        cond_tab[int(c)][r["interview"]] += 1

    out = _results_dir()
    with open(out / "cluster_features.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["vm", "interview", "base"] + FEATURES + ["cluster", "gmm_cluster"])
        for r, c, gc in zip(rows, labels, gm_labels):
            w.writerow([r["vm"], r["interview"], r["base"]] +
                       [round(r[f], 4) for f in FEATURES] + [int(c), int(gc)])
    with open(out / "cluster_validation.json", "w", encoding="utf-8") as f:
        json.dump({"n_interviews": n, "features": features, "chosen_k": best_k,
                   "selection": diag, "kmeans_bootstrap_stability_ARI": stability,
                   "kmeans_vs_gmm_ARI": float(adjusted_rand_score(labels, gm_labels)),
                   "cluster_by_condition": {str(k): dict(v) for k, v in cond_tab.items()}},
                  f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {out/'cluster_features.csv'} and cluster_validation.json")

    plot_profiles(rows, Xz, labels, best_k, out / "cluster_profiles.png", features)
    plot_pca(rows, Xz, labels, best_k, out / "cluster_pca.png",
             group_key="interview", markers={"Sync": "o", "Async": "^"},
             marker_title="Sync / Async")
    plot_pca(rows, Xz, labels, best_k, out / "cluster_pca_order.png",
             group_key="order", markers={"First": "o", "Second": "^"},
             marker_title="First / Second trial")

    print(f"\n=== Clustering: n={n} interviews, {len(features)} features, chosen k={best_k} ===")
    print("  silhouette by k:", {k: round(diag[k]['silhouette'], 3) for k in k_range})
    print(f"  bootstrap stability (ARI): {stability:.3f}   kmeans~gmm ARI: "
          f"{adjusted_rand_score(labels, gm_labels):.3f}")
    print("  cluster × condition:")
    for c in sorted(cond_tab):
        print(f"    C{c}: {dict(cond_tab[c])}")
    logger.info("Participant clustering complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
