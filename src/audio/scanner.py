#!/usr/bin/env python3
"""
Audio file scanner and validator for interview dataset.
Scans the raw audio directory and validates file structure.
"""

import os
from pathlib import Path
from typing import List, Dict
import subprocess
from src.config import config, logger

class AudioScanner:
    """Scan and validate audio file structure"""
    
    def __init__(self):
        self.raw_dir = Path(config.get("paths.raw_audio_dir"))
        self.processed_dir = Path(config.get("paths.processed_dir"))
    
    def scan_dataset(self) -> Dict[str, List[str]]:
        """Scan the dataset structure and return file mapping"""
        logger.info(f"Scanning dataset in {self.raw_dir}")
        
        if not self.raw_dir.exists():
            logger.error(f"Raw data directory not found: {self.raw_dir}")
            return {}
        
        dataset = {}
        
        # Find all VMXX directories (case-insensitive)
        vm_dirs = sorted([d for d in self.raw_dir.iterdir() 
                         if d.is_dir() and d.name.upper().startswith("VM")])
        
        if not vm_dirs:
            logger.warning(f"No VMXX directories found in {self.raw_dir}")
            return {}
        
        # Scan each VM directory for WAV and M4A files (case-insensitive)
        audio_extensions = ("*.wav", "*.m4a", "*.WAV", "*.M4A")
        
        for vm_dir in vm_dirs:
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(vm_dir.glob(ext))
            
            audio_files = sorted(audio_files)
            
            if audio_files:
                # Convert M4A files to WAV
                processed_files = []
                for audio_file in audio_files:
                    if audio_file.suffix.lower() == '.m4a':
                        # Convert to WAV
                        wav_file = self._convert_m4a_to_wav(audio_file)
                        processed_files.append(wav_file.name)
                    else:
                        processed_files.append(audio_file.name)
                
                dataset[vm_dir.name] = processed_files
                logger.info(f"Found {len(audio_files)} audio files in {vm_dir.name}")
            else:
                logger.warning(f"No audio files found in {vm_dir.name}")
        
        logger.info(f"Dataset scan complete. Found {len(dataset)} VM directories with audio files.")
        return dataset
    
    def _convert_m4a_to_wav(self, m4a_path: Path) -> Path:
        """Convert M4A to WAV using ffmpeg"""
        try:
            wav_path = m4a_path.with_suffix('.wav')
            
            # Check if WAV already exists
            if wav_path.exists():
                logger.info(f"WAV file already exists: {wav_path.name}")
                return wav_path
            
            cmd = [
                'ffmpeg',
                '-i', str(m4a_path),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                str(wav_path),
                '-y',  # Overwrite without asking
                '-loglevel', 'error'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and wav_path.exists():
                logger.info(f"Converted {m4a_path.name} to WAV")
                return wav_path
            else:
                logger.error(f"FFmpeg conversion failed: {result.stderr}")
                return m4a_path  # Return original if conversion fails
                
        except Exception as e:
            logger.error(f"M4A conversion error: {e}")
            return m4a_path
    
    def validate_structure(self) -> bool:
        """Validate the dataset structure meets requirements"""
        dataset = self.scan_dataset()
        
        if not dataset:
            logger.error("No valid dataset structure found")
            return False
        
        # Check for expected VM range (VM01-VM43)
        expected_vms = set([f"VM{i:02d}" for i in range(1, 44)])
        found_vms = set(dataset.keys())
        missing_vms = expected_vms - found_vms
        
        if missing_vms:
            logger.warning(f"Missing VM directories: {', '.join(sorted(missing_vms))}")
        
        # Create processed directory structure
        self._create_processed_structure(dataset)
        
        return True
    
    def _create_processed_structure(self, dataset: Dict[str, List[str]]):
        """Create mirrored structure in processed directory"""
        logger.info(f"Creating processed directory structure in {self.processed_dir}")
        
        for vm_dir, files in dataset.items():
            vm_processed_dir = self.processed_dir / vm_dir
            vm_processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Create empty files to mirror structure
            for file in files:
                (vm_processed_dir / file).touch()
        
        logger.info("Processed directory structure created")

if __name__ == "__main__":
    scanner = AudioScanner()
    
    # Validate dataset structure
    if scanner.validate_structure():
        logger.info("Dataset validation successful")
    else:
        logger.error("Dataset validation failed")
        exit(1)