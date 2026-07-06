#!/usr/bin/env python3
"""
Speaker diarization module using PyAnnote for identifying different speakers in audio.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import json

import torch

from src.config import config, logger


class SpeakerDiarizer:
    """Handle speaker diarization using PyAnnote"""

    def __init__(self):
        self.device = config.get("processing.device", "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load PyAnnote diarization model; leave model=None if unavailable"""
        try:
            logger.info("Loading PyAnnote speaker diarization model")

            # Set offline mode by default for privacy and performance
            os.environ["HF_HUB_OFFLINE"] = "1"

            # Set Hugging Face token if available
            huggingface_token = config.get("huggingface.token", None)
            if huggingface_token:
                os.environ["HF_TOKEN"] = huggingface_token

            # Import here to avoid slow imports when not needed
            from pyannote.audio import Pipeline

            # Load pre-trained pipeline
            self.model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )

            # Move to GPU if available
            if self.device == "cuda":
                self.model.to(torch.device("cuda"))

            logger.info("Speaker diarization model loaded successfully in offline mode")

        except Exception as e:
            logger.error(f"Failed to load diarization model: {e}")
            self.model = None

    def diarize_audio(self, audio_path: Path) -> Dict[str, Any]:
        """Perform speaker diarization on audio file"""
        try:
            # Only accept WAV files
            if audio_path.suffix.lower() != '.wav':
                raise ValueError(f"Diarizer only accepts WAV files, got {audio_path.suffix}")

            # Diarization requires the PyAnnote model — no fallback
            if self.model is None:
                logger.warning(
                    f"Speaker diarization unavailable (PyAnnote model not loaded); "
                    f"skipping {audio_path.name}"
                )
                return {
                    'filename': audio_path.name,
                    'error': 'diarization unavailable: PyAnnote model not loaded',
                    'speakers': {}
                }

            logger.info(f"Diarizing {audio_path.name}...")
            return self._diarize_with_pyannote(audio_path)

        except Exception as e:
            logger.error(f"Diarization failed for {audio_path.name}: {e}")
            return {
                'filename': audio_path.name,
                'error': str(e),
                'speakers': {}
            }
    
    def _diarize_with_pyannote(self, audio_path: Path) -> Dict[str, Any]:
        """Perform diarization using PyAnnote"""
        try:
            # Run diarization with a progress hook so the (slow) internal stages
            # — segmentation, embeddings, clustering — report a live ratio.
            from pyannote.audio.pipelines.utils.hook import ProgressHook
            with ProgressHook() as hook:
                output = self.model(audio_path, hook=hook)

            # PyAnnote >= 4.x returns a DiarizeOutput wrapping the Annotation in
            # `.speaker_diarization`; 3.x returns the Annotation directly.
            annotation = getattr(output, "speaker_diarization", output)

            # Extract speaker segments (labels are e.g. "SPEAKER_00")
            speakers: Dict[str, List[Dict[str, float]]] = {}
            for segment, _track, speaker in annotation.itertracks(yield_label=True):
                speakers.setdefault(speaker, []).append({
                    'start': segment.start,
                    'end': segment.end,
                    'duration': segment.end - segment.start,
                })

            # Append total speech time per speaker
            for segments in speakers.values():
                segments.append({
                    'total_speech_time': sum(seg['duration'] for seg in segments)
                })

            result = {
                'filename': audio_path.name,
                'speakers': speakers,
                'speaker_count': len(speakers),
                'total_duration': self._audio_duration(audio_path),
                'method': 'pyannote'
            }

            logger.info(f"Diarization completed for {audio_path.name}: {len(speakers)} speakers found")

            return result

        except Exception as e:
            logger.error(f"PyAnnote diarization failed: {e}")
            raise

    @staticmethod
    def _audio_duration(audio_path: Path) -> float:
        """Return audio duration in seconds, falling back to a 16kHz/16-bit estimate."""
        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            return info.frames / info.samplerate
        except Exception:
            return audio_path.stat().st_size / (2 * 16000)
    
    def diarize_directory(self, vm_dir: Path) -> Dict[str, Any]:
        """Diarize all audio files in a VM directory"""
        results = {}
        audio_files = []
        
        # Find all WAV files
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
                logger.info(f"[{i}/{total}] Diarizing {audio_file.name}")
                diarization = self.diarize_audio(audio_file)
                results[audio_file.name] = diarization
                
            except Exception as e:
                logger.error(f"Failed to diarize {audio_file.name}: {e}")
                results[audio_file.name] = {
                    'filename': audio_file.name,
                    'error': str(e),
                    'speakers': {}
                }
        
        return results

def save_diarization(result: Dict[str, Any], output_path: Path):
    """Save diarization result to JSON file"""
    try:
        import json
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved diarization to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save diarization: {e}")
        raise


def create_diarization_output_dir() -> Path:
    """Create diarization output directory"""
    output_dir = Path(config.get("paths.processed_dir")) / "diarization"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir