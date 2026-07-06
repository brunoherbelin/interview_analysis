#!/usr/bin/env python3
"""
Configuration management for the interview analysis pipeline.
Handles JSON config loading, validation, and environment setup.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import torch
import platform

from loguru import logger

# Single console format, applied from the very first log line for consistency.
_CONSOLE_FORMAT = (
    "<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:"
    "<cyan>{line}</cyan> - <level>{message}</level>"
)

class ConfigManager:
    """Centralized configuration management"""

    def __init__(self, config_path: str = "configs/config.json"):
        self.config_path = Path(config_path)
        self._install_console_handler("INFO")  # consistent console before any logging
        self.config = self._load_config()
        self._validate_config()
        self._setup_environment()
        self._setup_device()
        self._setup_logging()

    def _install_console_handler(self, level: str):
        """Install the console handler, replacing loguru's default (timestamped) one."""
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level=level, format=_CONSOLE_FORMAT)

    def _setup_environment(self):
        """Authenticate HF and quiet known-noisy third-party warnings for a clean console."""
        import os
        import warnings

        # Use the configured HF token for every model download: removes the
        # "unauthenticated requests" notices and raises HF Hub rate limits.
        token = self.config.get("huggingface", {}).get("token")
        if token:
            os.environ.setdefault("HF_TOKEN", token)

        # Expected, non-actionable warnings — keep them out of the console.
        # (TF32 stays disabled: pyannote's reproducible default, which we keep.)
        # Patterns are anchored at the start of the message, so prefix with ".*".
        for pattern in (".*TensorFloat-32", ".*degrees of freedom is <= 0"):
            warnings.filterwarnings("ignore", message=pattern)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and parse JSON configuration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in config file: {self.config_path}")
            raise
    
    def _validate_config(self):
        """Validate configuration structure"""
        required_sections = ["project", "paths", "processing", "logging"]
        
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing config section: {section}")
                self.config[section] = {}
    
    def _setup_device(self):
        """Automatic device detection (CUDA/MPS/CPU)"""
        if self.config["processing"].get("device") == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Using CUDA (NVIDIA GPU)")
            elif torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using MPS (Apple Silicon)")
            else:
                device = "cpu"
                logger.warning("No GPU detected, using CPU")
            
            self.config["processing"]["device"] = device
            
            # Platform-specific optimizations
            if platform.system() == "Darwin":
                torch.set_default_dtype(torch.float32)
                logger.info("Applied macOS optimizations")
            elif platform.system() == "Linux":
                torch.backends.cudnn.benchmark = True
                logger.info("Applied Linux optimizations")
    
    def _setup_logging(self):
        """Configure logging system"""
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO").upper()
        log_file = log_config.get("file", "logs/pipeline.log")
        
        # Create logs directory if it doesn't exist
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        # Reconfigure: drop the early console handler, add file + console at the
        # configured level. Console keeps the same format as the startup handler
        # (no timestamp); the file log records timestamps.
        logger.remove()
        logger.add(
            log_file,
            level=log_level,
            rotation=log_config.get("rotation", "10 MB"),
            retention=log_config.get("retention", "30 days"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        logger.add(lambda msg: print(msg, end=""), level=log_level, format=_CONSOLE_FORMAT)

        logger.info("Logging system initialized")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation"""
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                logger.warning(f"Config key not found: {key}")
                return default
        
        return value
    
    def update(self, key: str, value: Any):
        """Update configuration value"""
        keys = key.split(".")
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
        self._save_config()
    
    def _save_config(self):
        """Save configuration back to file"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        logger.info(f"Configuration saved to {self.config_path}")

# Global config instance
config = ConfigManager()