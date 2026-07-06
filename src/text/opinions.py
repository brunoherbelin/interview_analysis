#!/usr/bin/env python3
"""
Frame-based shared-opinion analysis (Sentilo-inspired), fully local.

For each participant we already have a FRED knowledge graph (the opinion *holder*
is always the participant). This stage:

  1. extracts the concepts each utterance is *about* from the FRED graph (topics),
  2. scores each (participant, topic) opinion by combining the contextual
     utterance sentiment (transformer signed score) with concept-level affect from
     the offline SenticNet knowledge base (polarity + Hourglass emotion),
  3. canonicalizes topics across participants (WordNet lemma + embedding merge),
  4. measures which opinions are *shared*: coverage, consensus vs contested, and
     the Sync/Async contrast.

References: Reforgiato Recupero et al. (2015) "Sentilo: Frame-Based Sentiment
Analysis" (Cognitive Computation); Gangemi et al. (2014) "Frame-Based Detection of
Opinion Holders and Topics" (IEEE CIM); SenticNet (Cambria et al.).
"""

import re
import json
from collections import defaultdict, Counter
from functools import lru_cache
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Any, List, Optional, Tuple

from src.config import config, logger
from src.text.knowledge_graph import FRED_DOMAIN, IA

PROPBANK_MARK = "propbank"

# Grammatical / non-topical FRED & PropBank concepts to drop.
CONCEPT_STOPLIST = {
    "person", "neuter", "male", "female", "thing", "entity", "more", "less",
    "exact", "amr-unknown", "multi-sentence", "have-degree", "have-quant",
    "be", "have", "do", "so", "and", "or", "possible", "it", "this", "that",
    "expressive", "truth-value", "yeah", "yes", "no", "kind", "you", "your",
    "okay", "ordinal-entity", "ordinal entity", "one", "way", "lot", "bit",
    "sort", "part", "someone", "something", "anything", "everything", "nothing",
}


@lru_cache(maxsize=4096)
def _is_topic_noun(lemma: str) -> bool:
    """Keep a concept only if it has a WordNet *noun* sense (a discussable topic)."""
    from nltk.corpus import wordnet as wn
    return bool(wn.synsets(lemma.replace(" ", "_"), pos=wn.NOUN))

# Merge topics whose embeddings are at least this similar.
_SIM_THRESHOLD = 0.72


# --------------------------------------------------------------------------- #
# SenticNet (offline) concept affect
# --------------------------------------------------------------------------- #
class ConceptAffect:
    """Offline SenticNet lookup: concept -> polarity + Hourglass emotion."""

    def __init__(self):
        from senticnet.senticnet import SenticNet
        self.sn = SenticNet()

    def lookup(self, concept: str) -> Optional[Dict[str, Any]]:
        try:
            d = self.sn.concept(concept)
        except KeyError:
            return None
        return {
            "polarity": float(d["polarity_value"]),
            "moodtags": [m.lstrip("#") for m in d.get("moodtags", [])],
            # the 4 Hourglass affective dimensions (introspection/temper/attitude/sensitivity)
            "sentics": {k: float(v) for k, v in d.get("sentics", {}).items()},
        }


# --------------------------------------------------------------------------- #
# Extraction from a FRED graph
# --------------------------------------------------------------------------- #
def _concept_lemma(class_uri: str) -> Optional[str]:
    """Recover a concept lemma from a FRED/PropBank type URI."""
    local = class_uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    if class_uri.startswith(FRED_DOMAIN):
        local = re.sub(r"^u\d+_", "", local)            # strip per-utterance prefix
    local = re.sub(r"-\d+$", "", local)                 # strip PropBank sense id
    lemma = local.replace("_", " ").strip().lower()
    if not lemma or len(lemma) < 3 or lemma in CONCEPT_STOPLIST:
        return None
    if any(ch.isdigit() for ch in lemma):
        return None
    return lemma


def extract_interview(ttl_path: Path) -> Optional[Dict[str, Any]]:
    """Parse a participant KG into interview metadata + per-utterance topics."""
    from rdflib import Graph
    g = Graph()
    g.parse(str(ttl_path), format="turtle")

    meta_q = g.query("""
        PREFIX ia: <http://interview-analysis.local/ontology#>
        SELECT ?vm ?type ?part ?base WHERE {
          ?i a ia:Interview ; ia:vm ?vm ; ia:interviewType ?type ;
             ia:participant ?part ; ia:baseFilename ?base .
        } LIMIT 1""")
    meta = list(meta_q)
    if not meta:
        logger.warning(f"{ttl_path.name}: no ia:Interview metadata; skipping")
        return None
    vm, itype, participant, base = (str(x) for x in meta[0])

    # utterance -> {start, text, topics}
    rows = g.query("""
        PREFIX ia: <http://interview-analysis.local/ontology#>
        SELECT ?idx ?start ?text ?cls WHERE {
          ?u a ia:Utterance ; ia:index ?idx ; ia:startTime ?start ; ia:text ?text .
          ?ind ia:fromUtterance ?u ; a ?cls .
        }""")
    utterances: Dict[int, Dict[str, Any]] = {}
    for idx, start, text, cls in rows:
        u = utterances.setdefault(int(idx), {
            "index": int(idx), "start": float(start), "text": str(text), "topics": set()
        })
        cls_str = str(cls)
        # Topics are noun concepts: in FRED these are `fred:` classes, whereas
        # verbs/events become PropBank rolesets (stance, not topic) — so exclude those.
        if cls_str.startswith(FRED_DOMAIN):
            lemma = _concept_lemma(cls_str)
            if lemma and _is_topic_noun(lemma):
                u["topics"].add(lemma)

    return {"vm": vm, "interview": itype, "participant": participant,
            "base_filename": base, "utterances": utterances}


def _load_utterance_sentiment(vm: str, base: str, participant: str) -> Dict[float, float]:
    """Map utterance start-time -> transformer signed score from the sentiment stage."""
    path = (Path(config.get("paths.processed_dir")) / "sentiment" / vm
            / f"{base}.{participant}.sentiment.json")
    if not path.exists():
        return {}
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}
    return {round(float(s["start"]), 2): float(s.get("signed_score", 0.0))
            for s in data.get("segments", [])}


# --------------------------------------------------------------------------- #
# Topic canonicalization across participants
# --------------------------------------------------------------------------- #
def canonicalize(lemmas: List[str]) -> Dict[str, str]:
    """Map each lemma to a canonical topic (WordNet singularization + embedding merge)."""
    from nltk.stem import WordNetLemmatizer
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    lem = WordNetLemmatizer()
    freq = Counter(lemmas)
    # normalize (singularize head noun)
    norm = {w: lem.lemmatize(w) for w in freq}
    norm_freq = Counter()
    for w, c in freq.items():
        norm_freq[norm[w]] += c
    uniques = [w for w, _ in norm_freq.most_common()]  # frequent first

    # embedding-based greedy merge
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        device = config.get("processing.device", "cpu")
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device if device == "cuda" else "cpu")
        embs = model.encode(uniques, normalize_embeddings=True)
        reps: List[int] = []
        canon_of: Dict[str, str] = {}
        for i, w in enumerate(uniques):
            best_j, best_sim = -1, 0.0
            for j in reps:
                sim = float(np.dot(embs[i], embs[j]))
                if sim > best_sim:
                    best_sim, best_j = sim, j
            if best_j >= 0 and best_sim >= _SIM_THRESHOLD:
                canon_of[w] = uniques[best_j]
            else:
                reps.append(i)
                canon_of[w] = w
    except Exception as e:
        logger.warning(f"Embedding canonicalization unavailable ({e}); using lemma identity")
        canon_of = {w: w for w in uniques}

    # compose: raw lemma -> normalized -> canonical
    return {w: canon_of[norm[w]] for w in freq}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def analyze_opinions(kg_dir: Path) -> Optional[Dict[str, Any]]:
    """Build per-participant opinion profiles + the shared-opinion table."""
    ttls = sorted(kg_dir.glob("*/*.ttl"))
    if not ttls:
        logger.warning(f"No knowledge graphs under {kg_dir}; run --kg first.")
        return None

    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    affect = ConceptAffect()

    # 1. collect raw opinion records across all interviews
    records: List[Dict[str, Any]] = []
    all_lemmas: List[str] = []
    for ttl in ttls:
        interview = extract_interview(ttl)
        if not interview:
            continue
        sent = _load_utterance_sentiment(
            interview["vm"], interview["base_filename"], interview["participant"])
        logger.info(f"{interview['vm']}/{interview['base_filename']}: "
                    f"{len(interview['utterances'])} utterances")
        for u in interview["utterances"].values():
            ctx = sent.get(round(u["start"], 2))
            for topic in u["topics"]:
                all_lemmas.append(topic)
                aff = affect.lookup(topic)
                records.append({
                    "vm": interview["vm"], "interview": interview["interview"],
                    # each VM is one participant (its two interviews may carry
                    # different diarization labels), so identity = VM
                    "participant": interview["vm"],
                    "lemma": topic,
                    "contextual_score": ctx,
                    "concept_polarity": aff["polarity"] if aff else None,
                    "moodtags": aff["moodtags"] if aff else [],
                })

    if not records:
        logger.warning("No opinion records extracted.")
        return None

    # 2. canonicalize topics and attach
    canon = canonicalize(all_lemmas)
    for r in records:
        r["topic"] = canon[r["lemma"]]
        # opinion score = contextual stance if available, else concept polarity
        r["score"] = r["contextual_score"] if r["contextual_score"] is not None else r["concept_polarity"]

    return {
        "profiles": _participant_profiles(records),
        "shared": _shared_opinions(records),
        "n_records": len(records),
        "n_topics": len(set(r["topic"] for r in records)),
    }


def _agg_moodtags(records) -> List[Tuple[str, int]]:
    c = Counter(m for r in records for m in r["moodtags"])
    return c.most_common(3)


def _participant_profiles(records) -> List[Dict[str, Any]]:
    """Per participant: topic scores + an overall Individual Sentiment Score."""
    by_part = defaultdict(list)
    for r in records:
        by_part[r["participant"]].append(r)

    profiles = []
    for participant, recs in sorted(by_part.items()):
        by_topic = defaultdict(list)
        for r in recs:
            by_topic[r["topic"]].append(r)
        topics = []
        for topic, trecs in by_topic.items():
            scored = [t["score"] for t in trecs if t["score"] is not None]
            topics.append({
                "topic": topic,
                "n_mentions": len(trecs),
                "score": round(mean(scored), 4) if scored else None,
                "top_moods": _agg_moodtags(trecs),
            })
        topics.sort(key=lambda t: t["n_mentions"], reverse=True)
        overall = [t["score"] for t in topics if t["score"] is not None]
        profiles.append({
            "participant": participant,
            "interviews": sorted(set(r["interview"] for r in recs)),
            "individual_sentiment_score": round(mean(overall), 4) if overall else None,
            "n_topics": len(topics),
            "topics": topics,
        })
    return profiles


def _shared_opinions(records) -> List[Dict[str, Any]]:
    """Per canonical topic: coverage, consensus vs contested, Sync/Async contrast."""
    by_topic = defaultdict(list)
    for r in records:
        by_topic[r["topic"]].append(r)

    shared = []
    for topic, trecs in by_topic.items():
        # per-participant mean score
        by_part = defaultdict(list)
        for r in trecs:
            if r["score"] is not None:
                by_part[r["participant"]].append(r["score"])
        part_scores = {p: round(mean(v), 4) for p, v in by_part.items() if v}
        if not part_scores:
            continue
        vals = list(part_scores.values())
        consensus = round(mean(vals), 4)
        spread = round(pstdev(vals), 4) if len(vals) > 1 else 0.0

        pos = sum(1 for v in vals if v > 0.05)
        neg = sum(1 for v in vals if v < -0.05)
        if len(vals) < 2:
            label = "individual"
        elif pos and neg:
            label = "contested"
        elif consensus > 0.05:
            label = "consensus_positive"
        elif consensus < -0.05:
            label = "consensus_negative"
        else:
            label = "neutral"

        def cond_mean(cond):
            xs = [r["score"] for r in trecs if r["interview"] == cond and r["score"] is not None]
            return round(mean(xs), 4) if xs else None

        shared.append({
            "topic": topic,
            "n_participants": len(part_scores),
            "n_mentions": len(trecs),
            "consensus_score": consensus,
            "spread": spread,
            "label": label,
            "participant_scores": part_scores,
            "sync_score": cond_mean("Sync"),
            "async_score": cond_mean("Async"),
            "top_moods": _agg_moodtags(trecs),
        })

    # most discussed & most shared first
    shared.sort(key=lambda s: (s["n_participants"], s["n_mentions"]), reverse=True)
    return shared


def save_opinions(result: Dict[str, Any], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "opinions.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved opinion analysis to {path} "
                f"({result['n_topics']} topics, {result['n_records']} records)")


def create_opinions_output_dir() -> Path:
    output_dir = Path(config.get("paths.processed_dir")) / "opinions"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
