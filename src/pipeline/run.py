#!/usr/bin/env python3
"""
Main pipeline entry point and CLI interface.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
from src.config import config, logger
from src.audio.scanner import AudioScanner

def create_parser():
    """Create argument parser for CLI"""
    parser = argparse.ArgumentParser(
        description="Interview Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan dataset structure and exit"
    )
    
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Transcribe audio files using Whisper"
    )
    
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="small",
        help="Whisper model size to use"
    )
    
    parser.add_argument(
        "--language",
        choices=["auto", "en", "fr", "es", "de", "it", "pt", "ru", "zh", "ja"],
        default="en",
        help="Force language for transcription (default: en for English)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="This is an interview with non-native English speakers.",
        help="Initial prompt to provide context for better transcription"
    )
    
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Perform speaker diarization on audio files"
    )

    parser.add_argument(
        "--sentiment",
        action="store_true",
        help="Run participant-only sentiment analysis on per-speaker transcriptions"
    )

    parser.add_argument(
        "--topics",
        action="store_true",
        help="Run participant-only topic modeling (BERTopic) across all interviews"
    )

    parser.add_argument(
        "--kg",
        action="store_true",
        help="Build participant-only knowledge graphs (Text2AMR2FRED) per interview"
    )

    parser.add_argument(
        "--opinions",
        action="store_true",
        help="Frame-based shared-opinion analysis across participants (needs --kg + --sentiment)"
    )

    parser.add_argument(
        "--vm",
        nargs="+",
        metavar="VM",
        default=None,
        help="Restrict every step to specific VM(s), e.g. --vm VM11 or --vm 11 12 (default: all)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every analysis step in order: diarize, transcribe, sentiment, topics, kg, opinions"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    return parser


def run_diarize(scanner, dataset) -> int:
    """Speaker diarization for every WAV file in the dataset."""
    from src.audio.diarizer import SpeakerDiarizer, save_diarization, create_diarization_output_dir

    diarizer = SpeakerDiarizer()
    output_dir = create_diarization_output_dir()

    n_vms = len(dataset)
    for v, (vm_name, files) in enumerate(dataset.items(), 1):
        vm_dir = scanner.raw_dir / vm_name

        logger.info(f"[VM {v}/{n_vms}] Diarizing {vm_name}...")
        audio_files = []
        for ext in ['*.wav', '*.WAV']:
            audio_files.extend(vm_dir.glob(ext))

        if audio_files:
            diarizations = diarizer.diarize_directory(vm_dir)
            for filename, diarization in diarizations.items():
                save_diarization(diarization, output_dir / vm_name / f"{filename}.diarization.json")
                logger.info(f"Saved diarization for {vm_name}/{filename}")
        else:
            logger.warning(f"No WAV files to diarize in {vm_name}")

    logger.info("Diarization complete!")
    return 0


def run_transcribe(scanner, dataset, args) -> int:
    """Transcribe every file, splitting per speaker when diarization is available."""
    from src.audio.transcriber import (
        AudioTranscriber, save_transcription, save_speaker_transcription,
        load_diarization, split_by_speaker, create_transcription_output_dir
    )

    transcriber = AudioTranscriber(
        model_size=args.model,
        language=args.language,
        initial_prompt=args.prompt
    )
    output_dir = create_transcription_output_dir()

    n_vms = len(dataset)
    for v, (vm_name, files) in enumerate(dataset.items(), 1):
        vm_dir = scanner.raw_dir / vm_name

        logger.info(f"[VM {v}/{n_vms}] Processing {vm_name}...")
        transcriptions = transcriber.transcribe_directory(vm_dir)

        for filename, transcription in transcriptions.items():
            if transcription.get("error"):
                save_transcription(transcription, output_dir, vm_name)
                continue

            diarization = load_diarization(vm_name, filename)
            if diarization:
                per_speaker = split_by_speaker(transcription, diarization)
                for speaker_result in per_speaker.values():
                    save_speaker_transcription(speaker_result, output_dir, vm_name)
                logger.info(
                    f"Saved {len(per_speaker)} per-speaker transcription(s) for {vm_name}/{filename}"
                )
            else:
                logger.warning(
                    f"No diarization found for {vm_name}/{filename}; run --diarize first "
                    f"to split transcription by speaker. Saving combined transcription."
                )
                save_transcription(transcription, output_dir, vm_name)

    logger.info("Transcription complete!")
    return 0


def run_sentiment(dataset) -> int:
    """Participant-only sentiment analysis over the per-speaker transcriptions."""
    from src.audio.transcriber import create_transcription_output_dir
    from src.text.sentiment import (
        SentimentAnalyzer, group_speaker_transcriptions, analyze_interview,
        save_sentiment, create_sentiment_output_dir
    )

    trans_dir = create_transcription_output_dir()
    output_dir = create_sentiment_output_dir()
    analyzer = SentimentAnalyzer()

    n_vms = len(dataset)
    for v, vm_name in enumerate(dataset, 1):
        vm_trans_dir = trans_dir / vm_name
        if not vm_trans_dir.is_dir():
            logger.warning(f"No transcriptions for {vm_name}; run --transcribe first. Skipping.")
            continue

        logger.info(f"[VM {v}/{n_vms}] Analyzing sentiment for {vm_name}...")
        interviews = group_speaker_transcriptions(vm_trans_dir)
        if not interviews:
            logger.warning(f"No per-speaker transcriptions found for {vm_name}; skipping")
            continue

        for audio_filename, speakers in interviews.items():
            result = analyze_interview(analyzer, vm_name, audio_filename, speakers)
            if result:
                save_sentiment(result, output_dir, vm_name)

    logger.info("Sentiment analysis complete!")
    return 0


def run_topics(dataset) -> int:
    """Participant-only topic modeling (BERTopic) across all interviews."""
    from src.audio.transcriber import create_transcription_output_dir
    from src.text.topics import (
        collect_participant_documents, fit_topics, build_global_summary,
        build_interview_results, save_global_summary, save_interview_topics,
        create_topics_output_dir
    )

    trans_dir = create_transcription_output_dir()
    output_dir = create_topics_output_dir()

    documents = collect_participant_documents(dataset, trans_dir)
    if not documents:
        logger.warning("No participant documents found; run --transcribe first.")
        return 1

    topic_model, topic_ids = fit_topics(documents)
    if topic_model is None:
        logger.warning("Topic modeling skipped (insufficient data).")
        return 1

    save_global_summary(build_global_summary(topic_model, documents, topic_ids), output_dir)
    for result in build_interview_results(topic_model, documents, topic_ids).values():
        save_interview_topics(result, output_dir)

    logger.info("Topic modeling complete!")
    return 0


def run_kg(dataset) -> int:
    """Participant-only knowledge-graph construction (Text2AMR2FRED) per interview."""
    from src.audio.transcriber import create_transcription_output_dir
    from src.text.knowledge_graph import (
        KnowledgeGraphBuilder, build_interview_graph, save_graph, create_kg_output_dir
    )
    from src.text.sentiment import group_speaker_transcriptions

    trans_dir = create_transcription_output_dir()
    output_dir = create_kg_output_dir()
    builder = KnowledgeGraphBuilder()

    n_vms = len(dataset)
    for v, vm_name in enumerate(dataset, 1):
        vm_trans_dir = trans_dir / vm_name
        if not vm_trans_dir.is_dir():
            logger.warning(f"No transcriptions for {vm_name}; run --transcribe first. Skipping.")
            continue

        logger.info(f"[VM {v}/{n_vms}] Building knowledge graphs for {vm_name}...")
        interviews = group_speaker_transcriptions(vm_trans_dir)
        if not interviews:
            logger.warning(f"No per-speaker transcriptions found for {vm_name}; skipping")
            continue

        for audio_filename, speakers in interviews.items():
            result = build_interview_graph(builder, vm_name, audio_filename, speakers)
            if result:
                master, stats = result
                save_graph(master, stats, output_dir)

    logger.info("Knowledge-graph construction complete!")
    return 0


def run_opinions(dataset) -> int:
    """Frame-based shared-opinion analysis across participants (Sentilo-inspired)."""
    from src.text.opinions import analyze_opinions, save_opinions, create_opinions_output_dir
    from src.text.knowledge_graph import create_kg_output_dir

    kg_dir = create_kg_output_dir()
    result = analyze_opinions(kg_dir)
    if result is None:
        logger.warning("Opinion analysis skipped (no knowledge graphs).")
        return 1

    save_opinions(result, create_opinions_output_dir())
    logger.info("Opinion analysis complete!")
    return 0


def main() -> int:
    """Main entry point for the pipeline"""
    parser = create_parser()
    args = parser.parse_args()

    if args.debug:
        logger.info("Debug mode enabled")

    # Scan dataset first
    scanner = AudioScanner()
    dataset = scanner.scan_dataset()

    if not dataset:
        logger.error("No audio files found to process")
        return 1

    # Optionally restrict to specific VM(s) (e.g. --vm VM11, or --vm 11 12)
    if args.vm:
        wanted = {(f"VM{v}" if v.strip().isdigit() else v.strip()).upper() for v in args.vm}
        filtered = {k: v for k, v in dataset.items() if k.upper() in wanted}
        missing = wanted - {k.upper() for k in dataset}
        if missing:
            logger.warning(f"Requested VM(s) not found and skipped: {', '.join(sorted(missing))}")
        if not filtered:
            logger.error(f"None of the requested VM(s) match the dataset. "
                         f"Available: {', '.join(dataset)}")
            return 1
        dataset = filtered
        logger.info(f"Restricted to {len(dataset)} VM(s): {', '.join(dataset)}")

    # Scan only mode
    if args.scan:
        logger.info("Dataset Structure:")
        for vm_name, files in dataset.items():
            logger.info(f"  {vm_name}: {len(files)} files")
        return 0

    # --all runs every step in dependency order
    if args.all:
        args.diarize = args.transcribe = args.sentiment = args.topics = args.kg = args.opinions = True

    # Run requested steps in dependency order (diarize -> transcribe -> analyses)
    steps = [
        ("diarize", args.diarize, lambda: run_diarize(scanner, dataset)),
        ("transcribe", args.transcribe, lambda: run_transcribe(scanner, dataset, args)),
        ("sentiment", args.sentiment, lambda: run_sentiment(dataset)),
        ("topics", args.topics, lambda: run_topics(dataset)),
        ("kg", args.kg, lambda: run_kg(dataset)),
        ("opinions", args.opinions, lambda: run_opinions(dataset)),
    ]

    exit_code = 0
    ran_any = False
    for name, enabled, run_step in steps:
        if not enabled:
            continue
        ran_any = True
        rc = run_step()
        if rc:
            logger.error(f"Step '{name}' returned exit code {rc}")
            exit_code = rc

    if not ran_any:
        logger.info("Pipeline initialization complete")

    return exit_code

if __name__ == "__main__":
    sys.exit(main())