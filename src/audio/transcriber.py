#!/usr/bin/env python3
"""
Audio transcription module using Whisper for high-accuracy transcription.
Supports WAV files with automatic language detection and translation.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import json

from src.config import config, logger

# Global model cache
_model_cache = {}

class AudioTranscriber:
    """Handle audio transcription using Whisper model"""
    
    def __init__(self, model_size="small", language="en", initial_prompt=None):
        self.device = config.get("processing.device", "cpu")
        self.model_size = model_size
        self.language = language
        self.initial_prompt = initial_prompt or "This is an interview with non-native English speakers."
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model with proper device handling and caching"""
        try:
            logger.info(f"Loading Whisper {self.model_size} model on {self.device}")
            
            # Check global cache first
            cache_key = f"{self.model_size}_{self.device}"
            if cache_key in _model_cache:
                self.model = _model_cache[cache_key]
                logger.info("Using cached model")
                return
            
            # Import here to avoid slow imports when not needed
            import whisper
            
            # Load model
            self.model = whisper.load_model(self.model_size)
            
            # Cache the model
            _model_cache[cache_key] = self.model
            
            logger.info("Whisper model loaded successfully")
            
        except ImportError:
            logger.error("Whisper not installed. Install with: pip install whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribe audio file with language detection and translation"""
        try:
            # Only accept WAV files - M4A conversion is handled by scanner
            if audio_path.suffix.lower() != '.wav':
                raise ValueError(f"Transcriber only accepts WAV files, got {audio_path.suffix}")
            
            logger.info(f"Transcribing {audio_path.name}...")
            
            # Use specified language and prompt
            language_param = self.language if self.language != "auto" else None
            
            result = self.model.transcribe(
                str(audio_path),
                language=language_param,
                fp16=False if self.device == "cpu" else True,
                word_timestamps=True,
                initial_prompt=self.initial_prompt
            )
            
            # Extract transcription details
            transcription = {
                'filename': audio_path.name,
                'language': self.language if self.language != "auto" else result.get('language', 'unknown'),
                'text': result['text'].strip(),
                'duration': result.get('duration', 0),
                'segments': []
            }
            
            # Add segment-level details (speaker context is added later, during the
            # speaker split, where each segment's speaker is known).
            for segment in result.get('segments', []):
                transcription['segments'].append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip()
                })
            
            logger.info(f"Transcription completed for {audio_path.name}")
            
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path.name}: {e}")
            return {
                'filename': audio_path.name,
                'error': str(e),
                'text': ''
            }
    
    def transcribe_directory(self, vm_dir: Path) -> Dict[str, Any]:
        """Transcribe all audio files in a VM directory"""
        results = {}
        audio_files = []
        
        # Find all WAV files (M4A should be converted by scanner)
        for ext in ['*.wav', '*.WAV']:
            audio_files.extend(vm_dir.glob(ext))
        
        if not audio_files:
            logger.warning(f"No WAV audio files found in {vm_dir.name}")
            return results
        
        total = len(audio_files)
        logger.info(f"Found {total} WAV audio files in {vm_dir.name}")

        # Process each audio file
        for i, audio_file in enumerate(sorted(audio_files), 1):
            try:
                logger.info(f"[{i}/{total}] Transcribing {audio_file.name}")
                transcription = self.transcribe_audio(audio_file)
                results[audio_file.name] = transcription
                
            except Exception as e:
                logger.error(f"Failed to process {audio_file.name}: {e}")
                results[audio_file.name] = {
                    'filename': audio_file.name,
                    'error': str(e),
                    'text': ''
                }
        
        return results


def save_transcription(result: Dict[str, Any], output_dir: Path, vm_name: str):
    """Save transcription result to JSON file"""
    try:
        vm_output_dir = output_dir / vm_name
        vm_output_dir.mkdir(parents=True, exist_ok=True)

        filename = result['filename']
        output_path = vm_output_dir / f"{filename}.transcription.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved transcription to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save transcription: {e}")
        raise


def load_diarization(vm_name: str, filename: str) -> Optional[Dict[str, Any]]:
    """Load a previously computed diarization result for a WAV file, if available."""
    path = (Path(config.get("paths.processed_dir")) / "diarization"
            / vm_name / f"{filename}.diarization.json")
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            diarization = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read diarization {path}: {e}")
        return None
    # A diarization with no speakers (e.g. it errored) is unusable for splitting
    return diarization if diarization.get("speakers") else None


def _speaker_intervals(diarization: Dict[str, Any]) -> list:
    """Flatten a diarization result into (start, end, speaker) intervals."""
    intervals = []
    for speaker, segments in diarization.get("speakers", {}).items():
        for seg in segments:
            if "start" in seg and "end" in seg:
                intervals.append((seg["start"], seg["end"], speaker))
    return intervals


def split_by_speaker(transcription: Dict[str, Any],
                     diarization: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Split a transcription into one transcription per speaker.

    Each transcription segment is attributed to the diarization speaker with the
    greatest temporal overlap. Every segment also carries `previous_other`: the
    text of the most recent preceding utterance from a *different* speaker, giving
    context to short replies like "Okay, cool.".
    """
    intervals = _speaker_intervals(diarization)

    # First pass: attribute each non-empty segment to a speaker, keeping order
    assigned = []  # (segment, speaker), chronological
    for segment in transcription.get("segments", []):
        if not segment.get("text", "").strip():
            continue  # empty/whitespace-only segments carry no content
        best_speaker, best_overlap = "UNKNOWN", 0.0
        for start, end, speaker in intervals:
            overlap = min(segment["end"], end) - max(segment["start"], start)
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, speaker
        assigned.append((segment, best_speaker))

    # Second pass: bucket per speaker, filling previous_other from another speaker
    per_speaker: Dict[str, Dict[str, Any]] = {}
    last_seen: Dict[str, tuple] = {}  # speaker -> (order_index, text)
    for idx, (segment, speaker) in enumerate(assigned):
        others = [pos for spk, pos in last_seen.items() if spk != speaker]
        previous_other = max(others)[1] if others else ""

        entry = per_speaker.setdefault(speaker, {
            "filename": transcription["filename"],
            "speaker": speaker,
            "language": transcription.get("language"),
            "segments": [],
        })
        entry["segments"].append({
            "start": segment["start"],
            "end": segment["end"],
            "previous_other": previous_other,
            "text": segment["text"],
        })
        last_seen[speaker] = (idx, segment["text"])

    # Concatenate each speaker's text in chronological order
    for entry in per_speaker.values():
        entry["text"] = " ".join(s["text"] for s in entry["segments"]).strip()

    return per_speaker


def save_speaker_transcription(result: Dict[str, Any], output_dir: Path, vm_name: str):
    """Save a single speaker's transcription to JSON file"""
    vm_output_dir = output_dir / vm_name
    vm_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = vm_output_dir / f"{result['filename']}.{result['speaker']}.transcription.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {result['speaker']} transcription to {output_path}")


def create_transcription_output_dir() -> Path:
    """Create transcription output directory"""
    output_dir = Path(config.get("paths.processed_dir")) / "transcriptions"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir