#!/usr/bin/env python3
"""
Sentiment analysis for interview participants.

Operates on the per-speaker transcription JSON files produced by the transcriber.
For each interview the experimenter is the first person to speak; sentiment is
computed on the *participant* only (the other speaker).

Two signals are produced per utterance:
- a transformer 3-class sentiment (negative / neutral / positive) with scores
- a VADER lexicon compound score in [-1, 1] as a lightweight secondary signal
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from src.config import config, logger

# Parse "<base>.wav.<SPEAKER_xx>.transcription.json"
_TRANSCRIPT_RE = re.compile(
    r"^(?P<audio>.+\.wav)\.(?P<speaker>SPEAKER_\d+|UNKNOWN)\.transcription\.json$",
    re.IGNORECASE,
)


class SentimentAnalyzer:
    """Transformer (3-class) + VADER sentiment scoring."""

    TRANSFORMER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    BATCH_SIZE = 16
    MAX_TOKENS = 256

    def __init__(self):
        self.device = config.get("processing.device", "cpu")
        self._load()

    def _load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        logger.info(f"Loading sentiment model {self.TRANSFORMER_MODEL} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.TRANSFORMER_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.TRANSFORMER_MODEL)
        self.model.to(self.device)
        self.model.eval()
        # e.g. {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.id2label = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.vader = SentimentIntensityAnalyzer()
        logger.info("Sentiment model loaded successfully")

    def score_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Score a list of utterances; returns one result dict per input text."""
        import torch

        results: List[Dict[str, Any]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=self.MAX_TOKENS,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

            for prob, text in zip(probs, batch):
                scores = {self.id2label[j]: float(p) for j, p in enumerate(prob)}
                label = max(scores, key=scores.get)
                results.append({
                    "label": label,
                    "score": round(scores[label], 4),
                    "scores": {k: round(v, 4) for k, v in scores.items()},
                    # continuous sentiment in [-1, 1], comparable to VADER
                    "signed_score": round(scores.get("positive", 0.0) - scores.get("negative", 0.0), 4),
                    "vader_compound": round(self.vader.polarity_scores(text)["compound"], 4),
                })
        return results


def _interview_type(audio_filename: str) -> str:
    """Classify an interview as Sync/Async from its filename ('async' contains 'sync')."""
    name = audio_filename.lower()
    if "async" in name:
        return "Async"
    if "sync" in name:
        return "Sync"
    return "unknown"


def group_speaker_transcriptions(vm_trans_dir: Path) -> Dict[str, Dict[str, dict]]:
    """Group a VM's per-speaker transcripts by interview.

    Returns {audio_filename: {speaker: transcription_dict}}.
    """
    interviews: Dict[str, Dict[str, dict]] = {}
    for path in sorted(vm_trans_dir.glob("*.transcription.json")):
        match = _TRANSCRIPT_RE.match(path.name)
        if not match:
            continue  # e.g. a combined "<base>.wav.transcription.json" with no speaker
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read transcript {path}: {e}")
            continue
        interviews.setdefault(match.group("audio"), {})[match.group("speaker")] = data
    return interviews


def _first_start(transcript: dict) -> float:
    """Earliest utterance start time for a speaker (inf if none)."""
    starts = [s["start"] for s in transcript.get("segments", []) if "start" in s]
    return min(starts) if starts else float("inf")


def _total_speech(transcript: dict) -> float:
    return sum(s.get("end", 0) - s.get("start", 0) for s in transcript.get("segments", []))


def identify_participant(speakers: Dict[str, dict]) -> Tuple[Optional[str], Optional[str]]:
    """Return (participant_speaker, experimenter_speaker).

    The experimenter is whoever speaks first; the participant is the other speaker.
    If more than two speakers are present, the participant is the most talkative
    among the non-first speakers.
    """
    labelled = {s: t for s, t in speakers.items() if s.upper().startswith("SPEAKER_")}
    if len(labelled) < 2:
        return None, None

    experimenter = min(labelled, key=lambda s: _first_start(labelled[s]))
    others = [s for s in labelled if s != experimenter]
    participant = max(others, key=lambda s: _total_speech(labelled[s]))
    return participant, experimenter


def analyze_interview(analyzer: SentimentAnalyzer, vm_name: str,
                      audio_filename: str, speakers: Dict[str, dict]) -> Optional[Dict[str, Any]]:
    """Run participant-only sentiment analysis for a single interview."""
    participant, experimenter = identify_participant(speakers)
    if participant is None:
        logger.warning(
            f"{vm_name}/{audio_filename}: need 2 speakers to identify the participant, "
            f"found {sorted(speakers)}; skipping"
        )
        return None

    transcript = speakers[participant]
    segments = [s for s in transcript.get("segments", []) if s.get("text", "").strip()]
    if not segments:
        logger.warning(f"{vm_name}/{audio_filename}: participant {participant} has no speech; skipping")
        return None

    logger.info(
        f"{vm_name}/{audio_filename}: analyzing participant {participant} "
        f"(experimenter {experimenter}), {len(segments)} utterances"
    )
    scored = analyzer.score_texts([s["text"] for s in segments])

    seg_out = []
    counts = {"negative": 0, "neutral": 0, "positive": 0}
    for seg, sc in zip(segments, scored):
        counts[sc["label"]] = counts.get(sc["label"], 0) + 1
        seg_out.append({
            "start": seg["start"], "end": seg["end"], "text": seg["text"],
            **sc,
        })

    n = len(seg_out)
    overall = {
        "distribution": {k: round(v / n, 4) for k, v in counts.items()},
        "counts": counts,
        "mean_signed_score": round(sum(s["signed_score"] for s in seg_out) / n, 4),
        "mean_vader_compound": round(sum(s["vader_compound"] for s in seg_out) / n, 4),
        "label": max(counts, key=counts.get),
    }

    return {
        "vm": vm_name,
        "interview": _interview_type(audio_filename),
        "base_filename": audio_filename[:-4] if audio_filename.lower().endswith(".wav") else audio_filename,
        "audio_filename": audio_filename,
        "participant_speaker": participant,
        "experimenter_speaker": experimenter,
        "engine": {"transformer": SentimentAnalyzer.TRANSFORMER_MODEL, "lexicon": "vader"},
        "n_utterances": n,
        "overall": overall,
        "most_positive": max(seg_out, key=lambda s: s["signed_score"]),
        "most_negative": min(seg_out, key=lambda s: s["signed_score"]),
        "segments": seg_out,
    }


def save_sentiment(result: Dict[str, Any], output_dir: Path, vm_name: str):
    """Save a participant's sentiment result to JSON file."""
    vm_output_dir = output_dir / vm_name
    vm_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = vm_output_dir / f"{result['base_filename']}.{result['participant_speaker']}.sentiment.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved sentiment to {output_path}")


def create_sentiment_output_dir() -> Path:
    output_dir = Path(config.get("paths.processed_dir")) / "sentiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
