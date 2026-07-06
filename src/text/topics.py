#!/usr/bin/env python3
"""
Topic modeling for interview participants using BERTopic.

Operates on the per-speaker transcription files. As in the sentiment stage, only
the *participant* (the speaker who is not the first to talk) is analyzed. Each
document is a participant utterance embedded together with its `previous_other`
context, so short replies cluster by what they respond to.

Topics are learned over the whole corpus (all interviews), then reported per
interview so Sync/Async conditions can be compared later.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from src.config import config, logger
from src.text.sentiment import (
    group_speaker_transcriptions, identify_participant, _interview_type,
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def collect_participant_documents(dataset, trans_dir: Path) -> List[Dict[str, Any]]:
    """Gather every participant utterance across the dataset as a topic document.

    Each document carries its metadata plus `doc_text` = previous_other + text.
    """
    documents: List[Dict[str, Any]] = []
    for vm_name in dataset:
        vm_trans = trans_dir / vm_name
        if not vm_trans.is_dir():
            logger.warning(f"No transcriptions for {vm_name}; run --transcribe first. Skipping.")
            continue

        for audio_filename, speakers in group_speaker_transcriptions(vm_trans).items():
            participant, _ = identify_participant(speakers)
            if participant is None:
                logger.warning(f"{vm_name}/{audio_filename}: cannot identify participant; skipping")
                continue

            for seg in speakers[participant].get("segments", []):
                text = seg.get("text", "").strip()
                if not text:
                    continue
                previous_other = seg.get("previous_other", "")
                documents.append({
                    "vm": vm_name,
                    "audio_filename": audio_filename,
                    "interview": _interview_type(audio_filename),
                    "participant_speaker": participant,
                    "start": seg["start"],
                    "end": seg["end"],
                    "previous_other": previous_other,
                    "text": text,
                    "doc_text": (previous_other + " " + text).strip(),
                })
    return documents


def _build_topic_model(n_docs: int):
    """Construct a BERTopic model with parameters adapted to the corpus size."""
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN

    device = config.get("processing.device", "cpu")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device if device == "cuda" else "cpu")

    # Strip English stopwords from the c-TF-IDF keywords so topic labels are
    # meaningful (embeddings still use the full text, including previous_other).
    vectorizer_model = CountVectorizer(stop_words="english")

    # Small-corpus-safe UMAP/HDBSCAN settings (defaults assume thousands of docs)
    n_neighbors = max(2, min(15, n_docs - 1))
    n_components = max(2, min(5, n_docs - 2))
    min_cluster_size = max(2, min(10, n_docs // 10))

    umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components,
                      min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean",
                            cluster_selection_method="eom", prediction_data=True)

    return BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=False,
    )


def fit_topics(documents: List[Dict[str, Any]]):
    """Fit BERTopic over the corpus. Returns (topic_model, topic_ids) or (None, None)."""
    doc_texts = [d["doc_text"] for d in documents]
    if len(doc_texts) < 5:
        logger.warning(
            f"Only {len(doc_texts)} participant documents; too few for reliable topic "
            f"modeling. Add more interviews for meaningful topics."
        )
        return None, None

    logger.info(f"Fitting BERTopic on {len(doc_texts)} participant documents ({EMBEDDING_MODEL})")
    topic_model = _build_topic_model(len(doc_texts))
    topic_ids, _ = topic_model.fit_transform(doc_texts)
    n_topics = len([t for t in set(topic_ids) if t != -1])
    logger.info(f"Discovered {n_topics} topic(s) (+ outliers)")
    return topic_model, topic_ids


def _topic_keywords(topic_model, topic_id: int, top_n: int = 10) -> List[str]:
    return [word for word, _ in (topic_model.get_topic(topic_id) or [])[:top_n]]


def build_global_summary(topic_model, documents: List[Dict[str, Any]],
                         topic_ids: List[int]) -> Dict[str, Any]:
    """Build the corpus-level topic table."""
    info = topic_model.get_topic_info()
    topics = []
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        topics.append({
            "topic_id": tid,
            "label": "outliers" if tid == -1 else row.get("Name", str(tid)),
            "keywords": _topic_keywords(topic_model, tid),
            "size": int(row["Count"]),
            "representative_docs": list(row.get("Representative_Docs", []))[:3],
        })

    return {
        "engine": "BERTopic",
        "embedding_model": EMBEDDING_MODEL,
        "document_unit": "previous_other + text (participant utterances)",
        "n_documents": len(documents),
        "n_topics": len([t for t in topics if t["topic_id"] != -1]),
        "topics": topics,
    }


def build_interview_results(topic_model, documents: List[Dict[str, Any]],
                            topic_ids: List[int]) -> Dict[tuple, Dict[str, Any]]:
    """Group per-document topic assignments back into per-interview results.

    Returns {(vm, audio_filename): result_dict}.
    """
    results: Dict[tuple, Dict[str, Any]] = {}
    for doc, tid in zip(documents, topic_ids):
        key = (doc["vm"], doc["audio_filename"])
        result = results.setdefault(key, {
            "vm": doc["vm"],
            "interview": doc["interview"],
            "base_filename": doc["audio_filename"][:-4]
            if doc["audio_filename"].lower().endswith(".wav") else doc["audio_filename"],
            "audio_filename": doc["audio_filename"],
            "participant_speaker": doc["participant_speaker"],
            "topic_distribution": {},
            "segments": [],
        })
        tid = int(tid)
        dist = result["topic_distribution"]
        dist[str(tid)] = dist.get(str(tid), 0) + 1
        result["segments"].append({
            "start": doc["start"],
            "end": doc["end"],
            "previous_other": doc["previous_other"],
            "text": doc["text"],
            "topic_id": tid,
            "topic_keywords": _topic_keywords(topic_model, tid, top_n=5),
        })
    return results


def save_global_summary(summary: Dict[str, Any], output_dir: Path):
    output_path = output_dir / "topics_model.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved global topic model to {output_path}")


def save_interview_topics(result: Dict[str, Any], output_dir: Path):
    vm_output_dir = output_dir / result["vm"]
    vm_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = vm_output_dir / f"{result['base_filename']}.{result['participant_speaker']}.topics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved interview topics to {output_path}")


def create_topics_output_dir() -> Path:
    output_dir = Path(config.get("paths.processed_dir")) / "topics"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
