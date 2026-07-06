#!/usr/bin/env python3
"""
Knowledge-graph construction for interview participants via Text2AMR2FRED (local).

For each participant utterance (the participant's own `text` only — see option (a)
below — so extracted topics are not contaminated by the experimenter's question):

    text  ->  amrlib local parse  ->  AMR (Penman)
          ->  py_amr2fred          ->  FRED RDF/OWL graph

The per-utterance FRED graphs are merged into a single RDF graph per interview.
To keep utterances distinct (a fresh AMR parse reuses individual names like
`fred:person_1`), FRED *domain individuals* are namespaced per utterance
(`fred:u{idx}_person_1`), while shared vocabulary (PropBank rolesets, DUL,
VerbNet roles) is kept common so it can be queried across utterances. Every
individual is linked back to its source utterance for provenance.

As in the other stages, only the *participant* is analyzed (experimenter = first
speaker). The Sync/Async tag and utterance timing are stored as provenance.

Wikidata post-processing is intentionally disabled (see README / kg_smoke_test):
on the `amr=` path it only adds Wikidata linking, which needs a large external DB
and is meaningless for subjective speech; skipping it keeps everything local.
"""

import re
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from rdflib import Graph, URIRef, Namespace, Literal, RDF
from rdflib.namespace import XSD

from src.config import config, logger
from src.text.sentiment import (
    group_speaker_transcriptions, identify_participant, _interview_type,
)

# FRED emits domain individuals under this namespace; shared vocab uses others.
FRED_DOMAIN = "http://www.ontologydesignpatterns.org/ont/fred/domain.owl#"
PROPBANK_MARK = "propbank"  # rolesets live under .../framester/data/propbank-*/

# Our small provenance vocabulary.
IA = Namespace("http://interview-analysis.local/ontology#")

# AMR parsing batch size (also the progress-report granularity)
AMR_CHUNK = 16


class KnowledgeGraphBuilder:
    """Local Text2AMR2FRED: amrlib parse + py_amr2fred AMR->RDF."""

    def __init__(self):
        self.device = config.get("processing.device", "cpu")
        self._load()

    def _load(self):
        import amrlib
        from py_amr2fred import Amr2fred

        logger.info(f"Loading amrlib parse model (local) on {self.device}")
        self.stog = amrlib.load_stog_model(
            device=self.device if self.device == "cuda" else "cpu"
        )
        self.amr2fred = Amr2fred()
        logger.info("Knowledge-graph models loaded successfully")

    def parse_to_amr(self, sentences: List[str]) -> List[Optional[str]]:
        """Batch text -> AMR (Penman strings)."""
        return self.stog.parse_sents(sentences)

    def amr_to_graph(self, amr: str) -> Graph:
        """AMR -> FRED RDF graph (rdflib Graph), fully local."""
        return self.amr2fred.translate(amr=amr, serialize=False, post_processing=False)


# Characters that make a URI invalid (would break Turtle serialization).
_INVALID_URI = re.compile(r'[\s<>"{}|\\^`]')


def _valid_uri(term) -> bool:
    return not (isinstance(term, URIRef) and _INVALID_URI.search(str(term)))


def _remap(term, idx: int):
    """Prefix FRED-domain individuals with an utterance id to avoid cross-utterance clashes."""
    if isinstance(term, URIRef) and str(term).startswith(FRED_DOMAIN):
        local = str(term)[len(FRED_DOMAIN):]
        return URIRef(f"{FRED_DOMAIN}u{idx}_{local}")
    return term


def _merge_utterance(master: Graph, g: Graph, idx: int, utter_uri: URIRef):
    """Add an utterance's FRED graph into master, namespaced and linked to provenance."""
    linked = set()
    for s, p, o in g:
        s2, p2, o2 = _remap(s, idx), _remap(p, idx), _remap(o, idx)
        # Skip malformed-URI triples (rare py_amr2fred artifacts, e.g. "amr/x->y")
        if not (_valid_uri(s2) and _valid_uri(p2) and _valid_uri(o2)):
            continue
        master.add((s2, p2, o2))
        # Link each FRED-domain individual back to its source utterance
        if isinstance(s2, URIRef) and str(s2).startswith(FRED_DOMAIN) and s2 not in linked:
            master.add((s2, IA.fromUtterance, utter_uri))
            linked.add(s2)


def build_interview_graph(builder: KnowledgeGraphBuilder, vm_name: str,
                          audio_filename: str,
                          speakers: Dict[str, dict]) -> Optional[Tuple[Graph, Dict[str, Any]]]:
    """Build the merged RDF graph + stats for one interview's participant."""
    participant, experimenter = identify_participant(speakers)
    if participant is None:
        logger.warning(f"{vm_name}/{audio_filename}: cannot identify participant; skipping")
        return None

    segments = [s for s in speakers[participant].get("segments", []) if s.get("text", "").strip()]
    if not segments:
        logger.warning(f"{vm_name}/{audio_filename}: participant has no speech; skipping")
        return None

    base_filename = audio_filename[:-4] if audio_filename.lower().endswith(".wav") else audio_filename
    interview_type = _interview_type(audio_filename)
    base_ns = Namespace(f"http://interview-analysis.local/{vm_name}/{base_filename}/{participant}#")

    logger.info(
        f"{vm_name}/{audio_filename}: building KG for participant {participant} "
        f"(experimenter {experimenter}), {len(segments)} utterances"
    )

    # Option (a): parse the participant's own `text` only, so the extracted topics
    # are not contaminated by the experimenter's question (`previous_other` is still
    # kept as metadata on each utterance below). To revert to context-aware parsing
    # of `previous_other + text`, swap in the commented line instead.
    # Parse in chunks so the (slow) AMR step reports a completion ratio.
    # sentences = [(s.get("previous_other", "") + " " + s["text"]).strip() for s in segments]
    sentences = [s["text"].strip() for s in segments]
    total = len(sentences)
    amrs: List[Optional[str]] = []
    for start in range(0, total, AMR_CHUNK):
        amrs.extend(builder.parse_to_amr(sentences[start:start + AMR_CHUNK]))
        logger.info(f"  AMR parse {min(start + AMR_CHUNK, total)}/{total}")

    master = Graph()
    master.bind("ia", IA)
    master.bind("fred", FRED_DOMAIN)

    # Interview-level provenance node
    interview_uri = URIRef(str(base_ns).rstrip("#"))
    master.add((interview_uri, RDF.type, IA.Interview))
    master.add((interview_uri, IA.vm, Literal(vm_name)))
    master.add((interview_uri, IA.interviewType, Literal(interview_type)))
    master.add((interview_uri, IA.baseFilename, Literal(base_filename)))
    master.add((interview_uri, IA.participant, Literal(participant)))

    n_ok, n_failed = 0, 0
    for idx, (seg, amr) in enumerate(zip(segments, amrs)):
        if idx and idx % 20 == 0:
            logger.info(f"  AMR->RDF {idx}/{total}")
        if not amr:
            n_failed += 1
            continue
        utter_uri = URIRef(f"{base_ns}utterance_{idx}")
        master.add((utter_uri, RDF.type, IA.Utterance))
        master.add((utter_uri, IA.partOf, interview_uri))
        master.add((utter_uri, IA["index"], Literal(idx, datatype=XSD.integer)))
        master.add((utter_uri, IA.startTime, Literal(seg["start"], datatype=XSD.decimal)))
        master.add((utter_uri, IA.endTime, Literal(seg["end"], datatype=XSD.decimal)))
        master.add((utter_uri, IA.text, Literal(seg["text"])))
        master.add((utter_uri, IA.previousOther, Literal(seg.get("previous_other", ""))))

        try:
            g = builder.amr_to_graph(amr)
            # keep readable prefixes from the FRED graph
            for prefix, ns in g.namespaces():
                master.bind(prefix, ns, replace=False)
            _merge_utterance(master, g, idx, utter_uri)
            n_ok += 1
        except Exception as e:
            logger.warning(f"{vm_name}/{audio_filename} u{idx}: AMR->RDF failed: {e}")
            n_failed += 1

    stats = _compute_stats(master, vm_name, interview_type, base_filename,
                           participant, experimenter, len(segments), n_ok, n_failed)
    return master, stats


def _local_name(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def _compute_stats(master: Graph, vm_name, interview_type, base_filename,
                   participant, experimenter, n_utterances, n_ok, n_failed) -> Dict[str, Any]:
    """Summarize the graph: frame (roleset) and predicate frequencies."""
    frames = Counter()
    for _s, _p, o in master.triples((None, RDF.type, None)):
        if PROPBANK_MARK in str(o):
            frames[_local_name(str(o))] += 1

    predicates = Counter()
    for _s, p, _o in master:
        p_str = str(p)
        if p_str.startswith(FRED_DOMAIN) or "interview-analysis" in p_str:
            continue  # skip provenance/instance-level plumbing
        predicates[_local_name(p_str)] += 1

    return {
        "vm": vm_name,
        "interview": interview_type,
        "base_filename": base_filename,
        "participant_speaker": participant,
        "experimenter_speaker": experimenter,
        "engine": {"amr_parser": "amrlib/parse_xfm_bart_base", "amr2rdf": "py_amr2fred"},
        "n_utterances": n_utterances,
        "n_utterances_ok": n_ok,
        "n_utterances_failed": n_failed,
        "n_triples": len(master),
        "n_frame_instances": sum(frames.values()),
        "n_distinct_frames": len(frames),
        "top_frames": frames.most_common(20),
        "top_predicates": predicates.most_common(20),
    }


def save_graph(master: Graph, stats: Dict[str, Any], output_dir: Path):
    """Write the RDF (Turtle) graph and its JSON stats sidecar."""
    vm_output_dir = output_dir / stats["vm"]
    vm_output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{stats['base_filename']}.{stats['participant_speaker']}"

    ttl_path = vm_output_dir / f"{stem}.ttl"
    master.serialize(destination=str(ttl_path), format="turtle")
    logger.info(f"Saved knowledge graph to {ttl_path} ({stats['n_triples']} triples)")

    json_path = vm_output_dir / f"{stem}.kg.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved KG stats to {json_path}")


def create_kg_output_dir() -> Path:
    output_dir = Path(config.get("paths.processed_dir")) / "knowledge_graph"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
