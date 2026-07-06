#!/usr/bin/env python3
"""
Human verification of the Hourglass-model emotions (text only).

Each emotion in the corpus profile comes from a *concept* in a sentence: the FRED
topic concept is looked up in SenticNet, which returns Hourglass moodtags
(joy, calmness, fear, ...). This script inverts that mapping — for every emotion it
shows representative sentences whose concepts triggered it (with the trigger word) —
so you can judge whether the sentence really conveys that emotion.

Before any verification it runs a knowledge-graph integrity check: it guards
against the silent parse-model-collapse failure mode where an interview's
utterances exist but carry no FRED individuals (0 fromUtterance links), which
makes every downstream stage report "0 utterances". If any interview has no
analysable content the script stops (re-run --kg) unless --force is given.

Usage:
  venv/bin/python scripts/human_check_emotions.py               # check KG, then interactive verify
  venv/bin/python scripts/human_check_emotions.py --check-only  # only run the KG integrity check
  venv/bin/python scripts/human_check_emotions.py --list        # just print examples
  venv/bin/python scripts/human_check_emotions.py --per-emotion 8
  venv/bin/python scripts/human_check_emotions.py --emotion joy
  venv/bin/python scripts/human_check_emotions.py --force       # verify even if KG check fails

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


def check_kg_integrity(min_ok_fraction: float = 0.5) -> tuple[bool, dict]:
    """Verify the knowledge graphs actually carry analysable content.

    For each interview it compares the build stats (kg.json: n_utterances_ok) with
    what the analysis query really retrieves (extract_interview), and flags:
      - BROKEN  : no utterance carries any FRED individual (the analysis sees 0),
                  i.e. the parse-model-collapse failure mode.
      - degraded: <min_ok_fraction of utterances converted (a partial collapse).

    Returns (ok, report); ok is False iff at least one interview is BROKEN.
    """
    kg_root = Path(config.get("paths.processed_dir")) / "knowledge_graph"
    ttls = sorted(kg_root.glob("*/*.ttl"))
    report = {"n_interviews": len(ttls), "usable_utterances": 0,
              "broken": [], "degraded": [], "error": None}
    if not ttls:
        report["error"] = f"No knowledge graphs under {kg_root}; run --kg first."
        return False, report

    for ttl in ttls:
        # build stats sidecar (e.g. VM11-1-Sync.SPEAKER_01.ttl -> ....kg.json)
        n_utt = n_ok = None
        stats_path = ttl.parent / (ttl.stem + ".kg.json")
        if stats_path.exists():
            try:
                s = json.load(open(stats_path, encoding="utf-8"))
                n_utt, n_ok = s.get("n_utterances"), s.get("n_utterances_ok")
            except Exception:
                pass

        interview = extract_interview(ttl)
        n_usable = len(interview["utterances"]) if interview else 0
        report["usable_utterances"] += n_usable

        name = f"{ttl.parent.name}/{ttl.stem}"
        if n_usable == 0:
            report["broken"].append({"interview": name, "n_utterances": n_utt, "n_ok": n_ok})
        elif n_ok and n_utt and (n_ok / n_utt) < min_ok_fraction:
            report["degraded"].append({"interview": name, "n_ok": n_ok, "n_utterances": n_utt})

    return (not report["broken"]), report


def print_kg_report(report: dict) -> None:
    print("=" * 60)
    print("KNOWLEDGE-GRAPH INTEGRITY CHECK")
    print(f"  interviews: {report['n_interviews']}   "
          f"usable utterances: {report['usable_utterances']}")
    if report["error"]:
        print(f"  ERROR: {report['error']}")
    if report["degraded"]:
        print(f"  degraded ({len(report['degraded'])}, low conversion rate):")
        for d in report["degraded"]:
            print(f"      {d['interview']}  ok={d['n_ok']}/{d['n_utterances']}")
    if report["broken"]:
        print(f"  BROKEN ({len(report['broken'])}) — no analysable content:")
        for b in report["broken"]:
            print(f"      {b['interview']}  build ok={b['n_ok']}/{b['n_utterances']}")
        print("  -> re-run:  python -m src.pipeline.run --kg "
              "(silent AMR parse-model collapse; see src/text/knowledge_graph.py)")
    elif not report["error"]:
        print("  OK — every interview has analysable content.")
    print("=" * 60)


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
    ap.add_argument("--check-only", action="store_true",
                    help="only run the knowledge-graph integrity check, then exit")
    ap.add_argument("--force", action="store_true",
                    help="run the emotion check even if the KG integrity check fails")
    args = ap.parse_args()
    k = max(1, min(20, args.per_emotion))

    # 1. Verify the knowledge graphs are analysable before checking emotions.
    kg_ok, kg_report = check_kg_integrity()
    print_kg_report(kg_report)
    if args.check_only:
        return 0 if kg_ok else 1
    if not kg_ok and not args.force:
        logger.error("Knowledge-graph check failed; re-run --kg (or pass --force to continue).")
        return 1

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
