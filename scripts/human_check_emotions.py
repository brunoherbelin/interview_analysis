#!/usr/bin/env python3
"""
Human verification of the Hourglass-model emotions (text only).

Each emotion in the corpus profile comes from a *concept* in a sentence: the FRED
topic concept is looked up in SenticNet, which returns Hourglass moodtags
(joy, calmness, fear, ...). This script inverts that mapping — for every emotion it
shows representative sentences whose concepts triggered it (with the trigger word) —
so you can judge whether the sentence really conveys that emotion.

Usage:
  venv/bin/python scripts/human_check_emotions.py               # interactive verify
  venv/bin/python scripts/human_check_emotions.py --list        # just print examples
  venv/bin/python scripts/human_check_emotions.py --per-emotion 8
  venv/bin/python scripts/human_check_emotions.py --emotion joy

Interactive controls per sentence: y = fits, n = does not, s = skip, q = quit.
Reports per-emotion agreement and saves to data/results/emotion_check_<ts>.json.
"""

import sys
import json
import argparse
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import config, logger
from src.text.opinions import extract_interview, ConceptAffect


def collect(affect, min_words=4) -> dict:
    """emotion -> list of {text, trigger, vm, interview} candidates."""
    kg_root = Path(config.get("paths.processed_dir")) / "knowledge_graph"
    ttls = sorted(kg_root.glob("*/*.ttl"))
    per_emotion = defaultdict(list)
    for ttl in ttls:
        interview = extract_interview(ttl)
        if not interview:
            continue
        for u in interview["utterances"].values():
            text = u["text"].strip()
            if len(text.split()) < min_words:      # skip trivial sentences ("Yeah.")
                continue
            for concept in u["topics"]:
                a = affect.lookup(concept)
                if not a:
                    continue
                for mood in a["moodtags"]:
                    per_emotion[mood].append({
                        "text": text, "trigger": concept,
                        "vm": interview["vm"], "interview": interview["interview"],
                    })
    return per_emotion


def representative(cands, k):
    """Pick k sentences favouring diverse trigger concepts, distinct sentences."""
    by_trig = defaultdict(list)
    for c in cands:
        by_trig[c["trigger"]].append(c)
    trig_order = sorted(by_trig, key=lambda t: -len(by_trig[t]))  # common triggers first
    idx = {t: 0 for t in trig_order}
    picks, seen = [], set()
    while len(picks) < k and any(idx[t] < len(by_trig[t]) for t in trig_order):
        for t in trig_order:
            if len(picks) >= k:
                break
            while idx[t] < len(by_trig[t]):
                c = by_trig[t][idx[t]]; idx[t] += 1
                if c["text"] not in seen:
                    seen.add(c["text"]); picks.append(c); break
    return picks


def ask(prompt, valid):
    while True:
        a = input(prompt).strip().lower()
        if a in valid:
            return a
        print(f"  please answer one of: {', '.join(valid)}")


class _Quit(Exception):
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Human check of Hourglass emotions (text)")
    ap.add_argument("--per-emotion", type=int, default=5, help="sentences per emotion (1-20)")
    ap.add_argument("--emotion", type=str, default=None, help="restrict to one emotion (e.g. joy)")
    ap.add_argument("--min-words", type=int, default=4, help="skip sentences shorter than this")
    ap.add_argument("--list", action="store_true", help="just print examples, no prompts")
    args = ap.parse_args()
    k = max(1, min(20, args.per_emotion))

    affect = ConceptAffect()
    per_emotion = collect(affect, args.min_words)
    if not per_emotion:
        logger.error("No emotion-bearing concepts found; run --kg and --opinions first.")
        return 1

    # order emotions by how often they occur in the corpus
    emotions = sorted(per_emotion, key=lambda e: -len(per_emotion[e]))
    if args.emotion:
        emotions = [e for e in emotions if e == args.emotion.lower()]
        if not emotions:
            logger.error(f"Emotion '{args.emotion}' not found. Available: "
                         f"{', '.join(sorted(per_emotion))}")
            return 1

    print(f"Emotions in corpus (by frequency): "
          + ", ".join(f"{e}({len(per_emotion[e])})" for e in emotions) + "\n")

    results = defaultdict(list)
    try:
        for emotion in emotions:
            picks = representative(per_emotion[emotion], k)
            print(f"\n=== {emotion.upper()} — {len(picks)} representative sentence(s) ===")
            for c in picks:
                print(f"  · \"{c['text']}\"")
                print(f"      (trigger: '{c['trigger']}' → {emotion};  {c['vm']}/{c['interview']})")
                if args.list:
                    continue
                a = ask("    Does this convey the emotion?  [y]es / [n]o / [s]kip / [q]uit: ",
                        {"y", "n", "s", "q"})
                if a == "q":
                    raise _Quit
                if a == "s":
                    continue
                results[emotion].append({**c, "fits": a == "y"})
    except (_Quit, KeyboardInterrupt):
        print("\n(quit — saving partial report)")

    if args.list:
        print("\nList mode — no verification recorded.")
        return 0
    if not results:
        print("No sentences verified.")
        return 0

    print("\n" + "=" * 60)
    print("EMOTION VERIFICATION REPORT")
    for emotion in sorted(results, key=lambda e: -len(results[e])):
        rs = results[emotion]
        agree = sum(r["fits"] for r in rs)
        print(f"  {emotion:14s} {agree}/{len(rs)} fit  ({100*agree/len(rs):.0f}%)")
        for r in rs:
            if not r["fits"]:
                print(f"      MISFIT (trigger '{r['trigger']}'): \"{r['text'][:60]}\"")
    print("=" * 60)

    out = Path(config.get("paths.results_dir", "data/results"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"emotion_check_{datetime.now():%Y%m%d_%H%M%S}.json"
    summary = {e: {"checked": len(rs), "fit": sum(r["fits"] for r in rs), "results": rs}
               for e, rs in results.items()}
    json.dump(summary, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved report to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
