#!/usr/bin/env python3
"""
Semantic search over the participant-feedback corpus (SentenceTransformers).

Two tools over the same embedded corpus (participant utterances only):

  Tier 1 — search:  query the corpus in natural language and get the most
            semantically similar participant statements, with metadata (participant,
            Sync/Async, First/Second, sentiment, timestamp, the question they were
            answering). Optional cross-encoder re-ranking of the top candidates.

  Tier 2 — probe:   turn a *probe* (a short statement of an experience, e.g. "I lost
            the sense of my body") into a per-utterance relevance score, aggregate it
            per participant × condition, and run the same within-subject paired
            contrast we use elsewhere (Wilcoxon + Benjamini-Hochberg FDR) to ask
            whether Sync/Async — or First/Second trial — elicits more of that theme.

This is the "Distributed Dictionary Representations" idea (Garten et al. 2018) /
semantic projection (Grand et al. 2022): a concept-as-vector used as a graded
measure. Asymmetric query→passage retrieval uses a multi-qa model (see config.
semantic.model). Corpus is ~8k utterances, so search is exact (no ANN needed);
embeddings are cached.

CAVEAT: semantic probes can "find" support for almost any wording — Tier 2 is only
trustworthy with pre-declared probes, the paired FDR test, and reading the example
matches (written to *_examples.json) to check each probe measures what you intend.

Run:
  venv/bin/python -m src.analysis.semantic_search search --query "felt weightless and floating"
  venv/bin/python -m src.analysis.semantic_search search --query "..." --rerank --top-k 15
  venv/bin/python -m src.analysis.semantic_search probe            # default probe set, both factors
  venv/bin/python -m src.analysis.semantic_search probe --probes-file my_probes.json
"""

import csv
import json
import hashlib
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from src.config import config, logger
from src.text.sentiment import (
    group_speaker_transcriptions, identify_participant, _interview_type,
)
from src.text.opinions import _load_utterance_sentiment
from src.text.topics import is_low_content
from src.analysis.condition_contrast import _bh_fdr

FACTORS = {  # name -> (record key, (level_a, level_b))
    "condition": ("interview", ("Sync", "Async")),
    "order": ("order", ("First", "Second")),
}

# Exploratory default probes for this VR/embodiment relaxation study. First-person
# statements of an experience, each with an expected `valence` used by the "felt"
# mode: positive/negative gate similarity by sentiment that agrees with the probe
# (so "did NOT feel anxious" no longer counts as anxiety); neutral gates by emotional
# intensity |sentiment|. Override with --probes-file (JSON: {name: query} or
# {name: {"query": ..., "valence": "positive|negative|neutral"}}).
DEFAULT_PROBES = {
    "presence_immersion": {"query": "I felt like I was really there, fully immersed in the environment", "valence": "neutral"},
    "bodily_dissolution": {"query": "I lost the sense of my body, felt disembodied or dissolved", "valence": "neutral"},
    "relaxation_calm": {"query": "I felt calm, relaxed and at peace", "valence": "positive"},
    "anxiety_unease": {"query": "I felt anxious, uneasy or uncomfortable", "valence": "negative"},
    "time_distortion": {"query": "my sense of time changed, time felt distorted", "valence": "neutral"},
    "connection": {"query": "I felt connected to something larger than myself", "valence": "positive"},
    "vividness_realness": {"query": "the experience felt vivid and real", "valence": "neutral"},
    "floating_lightness": {"query": "I felt light, floating or weightless", "valence": "positive"},
    "body_awareness": {"query": "I became strongly aware of sensations in my body", "valence": "neutral"},
    "emotional_intensity": {"query": "I felt a strong, intense emotion", "valence": "neutral"},
}


def normalize_probes(probes: dict):
    """Accept {name: query} or {name: {query, valence}}; return (queries, valences)."""
    queries, valences = {}, {}
    for name, val in probes.items():
        if isinstance(val, dict):
            queries[name] = val["query"]
            valences[name] = val.get("valence", "neutral")
        else:
            queries[name] = val
            valences[name] = "neutral"
    return queries, valences


def _results_dir() -> Path:
    d = Path(config.get("paths.results_dir", "data/results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_dir() -> Path:
    d = Path(config.get("paths.processed_dir")) / "semantic"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _trial_order(base_filename: str):
    """First/Second trial from the filename, e.g. VM27-1-Sync -> 'First'."""
    parts = base_filename.split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        return {"1": "First", "2": "Second"}.get(parts[1])
    return None


# --------------------------------------------------------------------------- #
# Corpus + embeddings
# --------------------------------------------------------------------------- #
def collect_utterances() -> list:
    """Every (content-bearing) participant utterance with its metadata + sentiment."""
    trans_dir = Path(config.get("paths.processed_dir")) / "transcriptions"
    recs = []
    for vm_dir in sorted(p for p in trans_dir.iterdir() if p.is_dir()):
        vm = vm_dir.name
        for audio_filename, speakers in group_speaker_transcriptions(vm_dir).items():
            participant, _ = identify_participant(speakers)
            if participant is None:
                continue
            base = audio_filename[:-4] if audio_filename.lower().endswith(".wav") else audio_filename
            itype = _interview_type(audio_filename)
            order = _trial_order(base)
            sent = _load_utterance_sentiment(vm, base, participant)
            for seg in speakers[participant].get("segments", []):
                text = seg.get("text", "").strip()
                if not text or is_low_content(text):
                    continue
                recs.append({
                    "vm": vm, "interview": itype, "order": order, "base": base,
                    "start": float(seg["start"]), "end": float(seg.get("end", seg["start"])),
                    "text": text, "previous_other": seg.get("previous_other", ""),
                    "sentiment": sent.get(round(float(seg["start"]), 2)),
                })
    return recs


def _fingerprint(texts, model_name) -> str:
    h = hashlib.sha1(model_name.encode("utf-8"))
    for t in texts:
        h.update(t.encode("utf-8")); h.update(b"\0")
    return h.hexdigest()


def resolve_model_name(cli_model=None) -> str:
    return cli_model or config.get("semantic.model", "multi-qa-MiniLM-L6-cos-v1")


def uses_cosine(model_name: str) -> bool:
    """Cosine models normalize embeddings; *-dot-* models use raw dot product."""
    return "dot" not in model_name.lower()


def load_model(name: str):
    from sentence_transformers import SentenceTransformer
    device = config.get("processing.device", "cpu")
    logger.info(f"Loading semantic model {name} on {device} "
                f"({'cosine/normalized' if uses_cosine(name) else 'dot-product'})")
    return SentenceTransformer(name, device=device if device == "cuda" else "cpu")


def embed_corpus(model, recs, name, normalize) -> np.ndarray:
    """Encode (and cache) the corpus document embeddings (normalized iff cosine model)."""
    texts = [r["text"] for r in recs]
    fp = _fingerprint(texts, name)
    tag = name.split("/")[-1]
    emb_path = _cache_dir() / f"corpus_{tag}.npy"
    meta_path = _cache_dir() / f"corpus_{tag}.json"
    if emb_path.exists() and meta_path.exists():
        meta = json.load(open(meta_path))
        if meta.get("fingerprint") == fp:
            logger.info(f"Loaded cached embeddings ({len(texts)} utterances, {tag})")
            return np.load(emb_path)

    logger.info(f"Encoding {len(texts)} participant utterances with {tag} ...")
    emb = model.encode_document(texts, batch_size=64, convert_to_numpy=True,
                                normalize_embeddings=normalize, show_progress_bar=True)
    np.save(emb_path, emb)
    json.dump({"fingerprint": fp, "model": name, "count": len(texts)}, open(meta_path, "w"))
    return emb


def encode_queries(model, queries, normalize) -> np.ndarray:
    return model.encode_query(list(queries), convert_to_numpy=True,
                              normalize_embeddings=normalize)


# --------------------------------------------------------------------------- #
# Tier 1 — search
# --------------------------------------------------------------------------- #
def search(query, recs, emb, model, normalize, top_k=10, rerank=False) -> list:
    q = encode_queries(model, [query], normalize)[0]
    scores = emb @ q                                    # cosine (normalized) or dot
    n_pool = max(top_k, 50) if rerank else top_k
    top = np.argsort(-scores)[:n_pool]
    hits = [(float(scores[i]), int(i)) for i in top]

    if rerank:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(config.get("semantic.rerank_model",
                                     "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        ce_scores = ce.predict([(query, recs[i]["text"]) for _, i in hits])
        hits = sorted(zip(ce_scores.tolist(), [i for _, i in hits]), reverse=True)

    return [(float(s), recs[i]) for s, i in hits[:top_k]]


def print_hits(query, hits):
    print(f"\n=== Top {len(hits)} for: {query!r} ===")
    for score, r in hits:
        sent = f"{r['sentiment']:+.2f}" if r["sentiment"] is not None else "  -  "
        print(f"[{score:.3f}] {r['vm']} {r['interview']}/{r['order']} "
              f"@{r['start']:.0f}s  sent={sent}")
        if r["previous_other"]:
            print(f"    Q: {r['previous_other'][:90]}")
        print(f"    A: {r['text']}")


# --------------------------------------------------------------------------- #
# Tier 2 — probe / contrast
# --------------------------------------------------------------------------- #
def felt_matrix(P, sentiment, valences, names) -> np.ndarray:
    """Gate the topic-similarity matrix P by sentiment agreeing with each probe's
    valence: positive->max(0,s), negative->max(0,-s), neutral->|s| (intensity)."""
    M = np.empty_like(P)
    for j, name in enumerate(names):
        v = valences[name]
        if v == "positive":
            gate = np.maximum(0.0, sentiment)
        elif v == "negative":
            gate = np.maximum(0.0, -sentiment)
        else:
            gate = np.abs(sentiment)
        M[:, j] = P[:, j] * gate
    return M


def grouped_topk(M, recs, factor_key, names, top_k) -> dict:
    """(vm, level) -> {probe: mean of the top-k utterance scores} (length-robust)."""
    groups = defaultdict(list)
    for i, r in enumerate(recs):
        lvl = r.get(factor_key)
        if lvl is not None:
            groups[(r["vm"], lvl)].append(i)
    out = {}
    for key, idxs in groups.items():
        sub = M[idxs]
        k = min(top_k, sub.shape[0])
        topk = np.sort(sub, axis=0)[-k:]                 # k largest per probe
        out[key] = {names[j]: float(topk[:, j].mean()) for j in range(len(names))}
    return out


def paired_contrast(scores, probes, levels, min_pairs) -> list:
    la, lb = levels
    by_vm = defaultdict(dict)
    for (vm, lvl), d in scores.items():
        by_vm[vm][lvl] = d
    paired = [vm for vm, dd in by_vm.items() if la in dd and lb in dd]

    results = []
    for p in probes:
        A, B = [], []
        for vm in paired:
            a, b = by_vm[vm][la].get(p), by_vm[vm][lb].get(p)
            if a is not None and b is not None:
                A.append(a); B.append(b)
        n = len(A)
        rec = {"probe": p, "n_pairs": n, "mean_diff": None, "dz": None,
               "wilcoxon_p": None, "p_fdr": None, "higher_in": None, "testable": n >= min_pairs}
        if rec["testable"]:
            A, B = np.array(A), np.array(B)
            d = A - B
            sd = float(d.std(ddof=1))
            rec["mean_diff"] = float(d.mean())
            rec["dz"] = float(d.mean() / sd) if sd > 0 else 0.0
            rec["higher_in"] = la if d.mean() > 0 else lb
            try:
                rec["wilcoxon_p"] = float(wilcoxon(A, B).pvalue) if np.any(d) else None
            except ValueError:
                rec["wilcoxon_p"] = None
        results.append(rec)

    testable = [r for r in results if r["testable"]]
    for r, q in zip(testable, _bh_fdr([r["wilcoxon_p"] if r["wilcoxon_p"] is not None else 1.0
                                       for r in testable])):
        r["p_fdr"] = q
    results.sort(key=lambda r: abs(r["dz"]) if r["dz"] is not None else -1, reverse=True)
    return results


def plot_probe_contrast(results, factor, levels, path, mode="topical"):
    la, lb = levels
    tested = [r for r in results if r["testable"]]
    if not tested:
        logger.warning(f"No testable probes for {factor}; skipping plot.")
        return
    tested = sorted(tested, key=lambda r: r["dz"])
    dz = [r["dz"] for r in tested]
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in dz]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(tested) + 1.8))
    ax.barh(range(len(tested)), dz, color=colors)
    ax.set_yticks(range(len(tested)))
    ax.set_yticklabels([r["probe"] for r in tested], fontsize=8)
    for i, r in enumerate(tested):
        star = " *" if (r["p_fdr"] is not None and r["p_fdr"] < 0.05) else ""
        ax.text(dz[i] + (0.02 if dz[i] >= 0 else -0.02), i, f"{dz[i]:+.2f}{star}",
                va="center", ha="left" if dz[i] >= 0 else "right", fontsize=7, color="0.3")
    ax.axvline(0, color="0.6", lw=1)
    nsig = sum(1 for r in tested if r["p_fdr"] is not None and r["p_fdr"] < 0.05)
    ax.set_xlabel(f"Cohen's dz   (→ higher in {la}    ← higher in {lb})")
    ax.set_title(f"Semantic probes ({mode}) — {factor}: {la} − {lb}\n"
                 f"({nsig} FDR-significant; * = FDR<0.05)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    logger.info(f"Saved {path}")


def run_probe(recs, emb, model, probes_raw, factors, modes, normalize, model_tag):
    out = _results_dir()
    min_pairs = int(config.get("semantic.min_pairs", 8))
    top_k = int(config.get("semantic.salience_top_k", 3))

    queries, valences = normalize_probes(probes_raw)
    names = list(queries)
    Q = encode_queries(model, queries.values(), normalize)  # (n_probes, dim)
    P = emb @ Q.T                                          # topic similarity (n_utts, n_probes)
    sentiment = np.array([r["sentiment"] if r["sentiment"] is not None else 0.0 for r in recs])

    for mode in modes:                                    # "topical" | "felt"
        base = "semantic_probe" if mode == "topical" else "semantic_felt"
        prefix = f"{base}_{model_tag}"
        M = P if mode == "topical" else felt_matrix(P, sentiment, valences, names)
        scores_by_factor = {f: grouped_topk(M, recs, FACTORS[f][0], names, top_k)
                            for f in factors}

        # wide per-(participant, level) scores
        with open(out / f"{prefix}_scores.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["vm", "factor", "level"] + names)
            for factor in factors:
                for (vm, lvl), d in sorted(scores_by_factor[factor].items()):
                    w.writerow([vm, factor, lvl] + [round(d[p], 4) for p in names])
        logger.info(f"Saved {out/f'{prefix}_scores.csv'}")

        # per-factor paired contrast (+plot)
        for factor in factors:
            levels = FACTORS[factor][1]
            results = paired_contrast(scores_by_factor[factor], names, levels, min_pairs)
            with open(out / f"{prefix}_{factor}_contrast.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["probe", "n_pairs", "mean_diff", "cohen_dz", "wilcoxon_p",
                            "p_fdr", "higher_in", "fdr_significant"])
                for r in results:
                    sig = r["p_fdr"] is not None and r["p_fdr"] < 0.05
                    w.writerow([r["probe"], r["n_pairs"], _r(r["mean_diff"]), _r(r["dz"]),
                                _r(r["wilcoxon_p"]), _r(r["p_fdr"]), r["higher_in"] or "", sig])
            plot_probe_contrast(results, factor, levels,
                                out / f"{prefix}_{factor}_contrast.png", mode)

            la, lb = levels
            tested = [r for r in results if r["testable"]]
            nsig = sum(1 for r in tested if r["p_fdr"] is not None and r["p_fdr"] < 0.05)
            print(f"\n=== Semantic probes [{mode}] — {factor} ({la} − {lb}), "
                  f"{len(tested)} tested, {nsig} FDR-significant ===")
            for r in tested:
                star = "*" if (r["p_fdr"] is not None and r["p_fdr"] < 0.05) else " "
                wp = f"{r['wilcoxon_p']:.3f}" if r["wilcoxon_p"] is not None else "  -  "
                print(f" {star}{r['probe']:20s} dz={r['dz']:+.2f} Δ={r['mean_diff']:+.4f} "
                      f"wilcox_p={wp} p_fdr={r['p_fdr']:.3f}  {r['higher_in']}")

        # validation: top matches per probe under this mode
        examples = {}
        for j, p in enumerate(names):
            top = np.argsort(-M[:, j])[:5]
            examples[p] = [{"score": round(float(M[i, j]), 4), "vm": recs[i]["vm"],
                            "interview": recs[i]["interview"], "order": recs[i]["order"],
                            "sentiment": recs[i]["sentiment"], "text": recs[i]["text"]}
                           for i in top]
        json.dump({"mode": mode, "probes": probes_raw, "examples": examples},
                  open(out / f"{prefix}_examples.json", "w"), indent=2, ensure_ascii=False)
        logger.info(f"Saved {out/f'{prefix}_examples.json'}")


def _r(x, nd=4):
    return round(x, nd) if isinstance(x, float) else ""


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Semantic search over participant feedback")
    sub = ap.add_subparsers(dest="cmd", required=True)
    model_help = "SentenceTransformer model (default: config semantic.model)"
    s = sub.add_parser("search", help="query the corpus for similar participant statements")
    s.add_argument("--query", required=True)
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--rerank", action="store_true", help="cross-encoder re-rank the top candidates")
    s.add_argument("--model", default=None, help=model_help)
    p = sub.add_parser("probe", help="paired condition/order contrast over semantic probes")
    p.add_argument("--probes-file", help="JSON {name: query} or {name: {query, valence}}")
    p.add_argument("--factor", choices=["condition", "order", "both"], default="both")
    p.add_argument("--mode", choices=["topical", "felt", "both"], default="both",
                   help="topical = similarity only; felt = similarity gated by agreeing sentiment")
    p.add_argument("--model", default=None, help=model_help)
    args = ap.parse_args()

    recs = collect_utterances()
    if not recs:
        logger.error("No participant utterances found; run --transcribe (and --sentiment) first.")
        return 1
    logger.info(f"Corpus: {len(recs)} participant utterances")
    model_name = resolve_model_name(args.model)
    normalize = uses_cosine(model_name)
    model = load_model(model_name)
    emb = embed_corpus(model, recs, model_name, normalize)

    if args.cmd == "search":
        print_hits(args.query, search(args.query, recs, emb, model, normalize,
                                      args.top_k, args.rerank))
        return 0

    probes = DEFAULT_PROBES
    if args.probes_file:
        probes = json.load(open(args.probes_file, encoding="utf-8"))
    factors = ["condition", "order"] if args.factor == "both" else [args.factor]
    modes = ["topical", "felt"] if args.mode == "both" else [args.mode]
    run_probe(recs, emb, model, probes, factors, modes, normalize, model_name.split("/")[-1])
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
