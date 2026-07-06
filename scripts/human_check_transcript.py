#!/usr/bin/env python3
"""
Human verification of diarization + transcription.

Run AFTER the pipeline (e.g. `--all`). For every interview it samples a few short
audio clips — spread across the timeline and across both speakers — plays each to
you, and asks:

  1. who is speaking:  [e]xperimenter  or  [p]articipant   (asked blind)
  2. is the shown transcription text correct:  [y]/[n]

It then reports what matched the pipeline and what did not, and saves a JSON report
under data/results/.

Usage:
  venv/bin/python scripts/human_check_transcript.py            # 3 clips / interview
  venv/bin/python scripts/human_check_transcript.py --per-file 5
  venv/bin/python scripts/human_check_transcript.py --vm VM05
  venv/bin/python scripts/human_check_transcript.py --dry-run  # select+extract, no prompts

Controls during a clip: r = replay, s = skip, q = quit (saves partial report).
"""

import sys
import json
import shutil
import argparse
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import config, logger
from src.text.sentiment import group_speaker_transcriptions, identify_participant

PAD = 0.3          # seconds of context padding around a clip
MIN_WORDS = 3      # prefer clips with at least this many words
MIN_DUR = 1.0      # and at least this many seconds


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _candidates(transcript):
    segs = [s for s in transcript.get("segments", [])
            if len(s.get("text", "").split()) >= MIN_WORDS and (s["end"] - s["start"]) >= MIN_DUR]
    if not segs:  # relax if the speaker has only short turns
        segs = [s for s in transcript.get("segments", []) if s.get("text", "").strip()]
    return sorted(segs, key=lambda s: s["start"])


def _pick_spread(cands, m):
    """Pick m segments spread across time, preferring the wordiest in each bin."""
    if m <= 0 or not cands:
        return []
    if len(cands) <= m:
        return cands
    picks, step = [], len(cands) / m
    for b in range(m):
        lo, hi = int(b * step), int((b + 1) * step)
        chunk = cands[lo:hi] or [cands[min(lo, len(cands) - 1)]]
        picks.append(max(chunk, key=lambda s: len(s["text"])))
    return picks


def select_clips(speakers, experimenter, participant, per_file):
    """Allocate per_file clips across speakers (participant gets the remainder)."""
    present = [s for s in (experimenter, participant) if s in speakers]
    if not present:
        return []
    n = len(present)
    base, rem = per_file // n, per_file % n
    alloc = {s: base for s in present}
    rem_order = [s for s in (participant, experimenter) if s in alloc]
    for i in range(rem):
        alloc[rem_order[i % len(rem_order)]] += 1

    clips = []
    for spk in present:
        role = "experimenter" if spk == experimenter else "participant"
        for seg in _pick_spread(_candidates(speakers[spk]), alloc[spk]):
            clips.append({"speaker": spk, "role": role, "start": seg["start"],
                          "end": seg["end"], "text": seg["text"].strip()})
    clips.sort(key=lambda c: c["start"])
    return clips


def load_interviews(vm_filter=None):
    """Yield (vm, audio_filename, wav_path, speakers, experimenter, participant)."""
    trans_root = Path(config.get("paths.processed_dir")) / "transcriptions"
    raw_root = Path(config.get("paths.raw_audio_dir"))
    for vm_dir in sorted(p for p in trans_root.glob("*") if p.is_dir()):
        vm = vm_dir.name
        if vm_filter and vm != vm_filter:
            continue
        for audio_filename, speakers in group_speaker_transcriptions(vm_dir).items():
            participant, experimenter = identify_participant(speakers)
            if experimenter is None:
                logger.warning(f"{vm}/{audio_filename}: <2 speakers; skipping")
                continue
            wav = raw_root / vm / audio_filename
            if not wav.exists():
                logger.warning(f"Raw audio not found: {wav}; skipping")
                continue
            yield vm, audio_filename, wav, speakers, experimenter, participant


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def extract_clip(wav, start, end, tmp):
    s = max(0.0, start - PAD)
    d = (end - start) + 2 * PAD
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{s:.2f}",
                    "-t", f"{d:.2f}", "-i", str(wav), "-ar", "16000", "-ac", "1", str(tmp)],
                   check=True)


def play(tmp) -> bool:
    if shutil.which("ffplay"):
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(tmp)])
        return True
    for player in ("paplay", "aplay"):
        if shutil.which(player):
            subprocess.run([player, str(tmp)] if player == "paplay" else [player, "-q", str(tmp)])
            return True
    return False


# --------------------------------------------------------------------------- #
# Interaction
# --------------------------------------------------------------------------- #
def ask(prompt, valid):
    while True:
        a = input(prompt).strip().lower()
        if a in valid:
            return a
        print(f"  please answer one of: {', '.join(valid)}")


def check_clip(clip, wav, tmp):
    """Return a result dict, or None if skipped. Raises KeyboardInterrupt-like on quit."""
    extract_clip(wav, clip["start"], clip["end"], tmp)
    while True:
        if not play(tmp):
            print("  !! no audio player available — cannot verify audio.")
            return None
        a = ask("  Who is speaking?  [e]xperimenter / [p]articipant / [r]eplay / [s]kip / [q]uit: ",
                {"e", "p", "r", "s", "q"})
        if a == "r":
            continue
        if a == "s":
            return None
        if a == "q":
            raise _Quit
        human_role = "experimenter" if a == "e" else "participant"
        break

    print(f"    transcription: \"{clip['text']}\"")
    t = ask("  Is the transcription correct?  [y]es / [n]o / [r]eplay / [q]uit: ",
            {"y", "n", "r", "q"})
    while t == "r":
        play(tmp)
        t = ask("  Is the transcription correct?  [y]es / [n]o / [q]uit: ", {"y", "n", "q"})
    if t == "q":
        raise _Quit

    speaker_ok = (human_role == clip["role"])
    text_ok = (t == "y")
    if not speaker_ok:
        print(f"    -> speaker MISMATCH (pipeline said {clip['role']})")
    return {**{k: clip[k] for k in ("role", "start", "end", "text")},
            "human_role": human_role, "speaker_ok": speaker_ok, "text_ok": text_ok}


class _Quit(Exception):
    pass


# --------------------------------------------------------------------------- #
def report(results, path):
    checked = len(results)
    spk_bad = [r for r in results if not r["speaker_ok"]]
    txt_bad = [r for r in results if not r["text_ok"]]
    print("\n" + "=" * 66)
    print(f"VERIFICATION REPORT — {checked} clips checked")
    print(f"  speaker attribution : {checked - len(spk_bad)}/{checked} OK, {len(spk_bad)} wrong")
    print(f"  transcription text  : {checked - len(txt_bad)}/{checked} OK, {len(txt_bad)} wrong")
    if spk_bad:
        print("\n  Speaker mismatches:")
        for r in spk_bad:
            print(f"    {r['vm']}/{r['interview']} [{r['start']:.1f}s] "
                  f"pipeline={r['role']} human={r['human_role']}  \"{r['text'][:50]}\"")
    if txt_bad:
        print("\n  Transcription errors:")
        for r in txt_bad:
            print(f"    {r['vm']}/{r['interview']} [{r['start']:.1f}s]  \"{r['text'][:60]}\"")
    print("=" * 66)

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"checked": checked, "speaker_wrong": len(spk_bad),
                   "text_wrong": len(txt_bad), "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved report to {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Human check of diarization + transcription")
    ap.add_argument("--per-file", type=int, default=3, help="clips per interview (1-10)")
    ap.add_argument("--vm", type=str, default=None, help="restrict to one VM (e.g. VM05)")
    ap.add_argument("--dry-run", action="store_true", help="select+extract clips, no playback/prompts")
    args = ap.parse_args()
    per_file = max(1, min(10, args.per_file))

    interviews = list(load_interviews(args.vm))
    if not interviews:
        logger.error("No per-speaker transcriptions found; run --diarize + --transcribe first.")
        return 1
    total = sum(len(select_clips(sp, e, p, per_file)) for _, _, _, sp, e, p in interviews)
    print(f"{len(interviews)} interview(s), {per_file} clip(s) each → ~{total} checks.\n")

    results = []
    tmpdir = Path(tempfile.mkdtemp(prefix="human-check-"))
    try:
        for vm, audio, wav, speakers, exp, part in interviews:
            clips = select_clips(speakers, exp, part, per_file)
            print(f"--- {vm} / {audio}  (experimenter={exp}, participant={part}) ---")
            for i, clip in enumerate(clips, 1):
                clip.update(vm=vm, interview=audio)
                if args.dry_run:
                    tmp = tmpdir / f"{vm}_{audio}_{i}.wav"
                    extract_clip(wav, clip["start"], clip["end"], tmp)
                    print(f"  [{i}] {clip['role']:12s} {clip['start']:6.1f}-{clip['end']:.1f}s "
                          f"({tmp.stat().st_size} B)  \"{clip['text'][:55]}\"")
                    continue
                print(f"  [clip {i}/{len(clips)}] {clip['start']:.1f}-{clip['end']:.1f}s")
                res = check_clip(clip, wav, tmpdir / "clip.wav")
                if res:
                    res.update(vm=vm, interview=audio)
                    results.append(res)
    except (_Quit, KeyboardInterrupt):
        print("\n(quit — saving partial report)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if args.dry_run:
        print("\nDry run complete — clip selection and extraction OK.")
        return 0
    if results:
        out = Path(config.get("paths.results_dir", "data/results"))
        out.mkdir(parents=True, exist_ok=True)
        report(results, out / f"human_check_{datetime.now():%Y%m%d_%H%M%S}.json")
    else:
        print("No clips verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
