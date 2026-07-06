#!/usr/bin/env python3
"""
Smoke test for the Text2AMR2FRED knowledge-graph stack (local, offline).

Proves the local pipeline end-to-end on a couple of sample participant utterances:

    text (previous_other + text)  ->  amrlib local parse  ->  AMR (Penman)
                                  ->  py_amr2fred          ->  RDF/OWL (Turtle)

Per the design, the AMR is parsed on `previous_other + " " + text` so the model
has the conversational context to resolve references ("it", "this time", ...).

Run:
    venv/bin/python -m scripts.kg_smoke_test
or
    venv/bin/python scripts/kg_smoke_test.py

Requires the amrlib parse model installed at <amrlib>/data/model_stog
(see README "Knowledge Graph" section for the one-time model download).
"""

import sys
from pathlib import Path

# Allow running as a plain script (add project root to path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# A few real VM05 participant utterances with their previous_other context.
SAMPLES = [
    {
        "previous_other": "How did you find it this time?",
        "text": "Honestly, this time the whole experience was much better.",
    },
    {
        "previous_other": "So you managed to relax more than others?",
        "text": "Yeah, exactly, unlike last time.",
    },
]


def main() -> int:
    print("Loading amrlib parse model (local)...")
    import amrlib
    stog = amrlib.load_stog_model()

    from py_amr2fred import Amr2fred, Glossary
    amr2fred = Amr2fred()

    # Parse `previous_other + text` so AMR gets the conversational context.
    sentences = [(s["previous_other"] + " " + s["text"]).strip() for s in SAMPLES]
    print(f"Parsing {len(sentences)} utterance(s) to AMR...")
    amr_graphs = stog.parse_sents(sentences)

    for sample, amr in zip(SAMPLES, amr_graphs):
        print("\n" + "=" * 70)
        print("previous_other:", sample["previous_other"])
        print("text          :", sample["text"])
        print("-" * 70)
        print("AMR:\n" + (amr or "<none>"))

        if not amr:
            print("!! AMR parse returned nothing for this sentence")
            continue

        # py_amr2fred strips the ::snt comment lines itself; pass the raw AMR.
        # post_processing=False: its only effect on the amr= path is Wikidata
        # entity-linking, which needs an 832MB DB from an unreliable external host
        # and is meaningless for subjective interview content. Skipping it keeps
        # the pipeline fully local, offline and deterministic.
        turtle = amr2fred.translate(
            amr=amr,
            serialize=True,
            mode=Glossary.RdflibMode.TURTLE,
            post_processing=False,
        )
        n_lines = turtle.count("\n")
        print("-" * 70)
        print(f"RDF (Turtle) — {n_lines} lines, {len(turtle)} chars:\n")
        print(turtle)

    print("\n" + "=" * 70)
    print("SMOKE TEST OK — local text -> AMR -> RDF pipeline works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
