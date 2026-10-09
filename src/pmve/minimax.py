"""
MiniMax Music 3 - Artist Catalog Port & Fine-Tune Pipeline
Artist: Aries Hilton / Frozen In Flames
Repository: MiniMax-Music-03 (Open Source Fine-Tuning Extension)

This pipeline is designed to bridge artist catalogs into the MiniMax Music 3
open-source training ecosystem. It handles the complete workflow from source
material acquisition to training-ready dataset generation.

Key Features:
- 32kHz stereo WAV extraction matching MiniMax Music 3 native format
- Structured lyric mapping with section awareness for semantic training
- Comprehensive metadata generation with musical feature extraction
- Checkpoint/resume capability for large catalog processing
- GitHub Actions ready with complete environment documentation

Author: Aries Hilton
License: Apache 2.0 (Matching MiniMax Music 3 base repository)
"""

import os
import sys
import json
import wave
import time
import shutil
import logging
import subprocess
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

@dataclass
class PipelineConfig:
    """Central configuration for the MiniMax Music 3 fine-tuning pipeline."""
    # Artist metadata
    artist_name: str = "Aries Hilton"
    project_moniker: str = "Frozen In Flames"

    # Repository references
    base_repo_url: str = "https://github.com/MiniMax-Al/MiniMax-Music-03"
    dataset_version: str = "1.0.0"

    # Audio specifications (MiniMax Music 3 requirements)
    sample_rate: int = 32000  # 32kHz as required by MiniMax Music 3
    channels: int = 2          # Stereo
    sample_width: int = 2      # 16-bit
    audio_format: str = "wav"

    # Processing parameters
    batch_size: int = 5
    delay_between_batches: int = 10
    max_retries: int = 3
    retry_delay: int = 5

    # Paths
    workspace: str = "minimax_artist_workspace"
    catalog_file: str = "catalog/youtube_catalog.txt"
    config_file: str = "config/pipeline_config.yaml"

    # Feature extraction
    enable_audio_analysis: bool = True
    enable_loudness_normalization: bool = True
    target_loudness: float = -14.0  # LUFS for streaming standard

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PipelineConfig':
        valid_fields = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)


class MiniMaxDatasetBuilder:
    """
    Complete dataset preparation pipeline for MiniMax Music 3 fine-tuning.
    This class handles the end-to-end process of converting an artist's catalog
    into the structured format required for training MiniMax Music 3 models.
    """

    def __init__(self, config: Optional[PipelineConfig] = None, config_path: Optional[str] = None):
        if config:
            self.config = config
        elif config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            self.config = PipelineConfig.from_dict(config_dict)
        else:
            self.config = PipelineConfig()

        # Setup paths
        self.workspace = Path(self.config.workspace)
        self.audio_dir = self.workspace / "dataset" / "audio"
        self.text_dir = self.workspace / "dataset" / "text"
        self.metadata_dir = self.workspace / "dataset" / "metadata"
        self.checkpoint_file = self.workspace / "checkpoints" / "processed_tracks.json"
        self.failed_file = self.workspace / "checkpoints" / "failed_tracks.txt"

        # Setup logging
        self._setup_logging()

        # Initialize state
        self.processed_tracks = set()
        self.failed_tracks = []
        self.metadata_records = []

        self.logger.info(f"Initialized MiniMax Music 3 Dataset Builder for {self.config.artist_name}")

    def _setup_logging(self):
        """Configure comprehensive logging for pipeline monitoring."""
        log_dir = self.workspace / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(f"MiniMax-{self.config.artist_name}")

    def setup_workspace(self):
        """Create necessary directory structure for dataset preparation."""
        self.logger.info(f"Setting up workspace: {self.workspace}")
        directories = [
            self.audio_dir,
            self.text_dir,
            self.metadata_dir,
            self.workspace / "checkpoints",
            self.workspace / "logs",
            self.workspace / "config"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        default_config_path = self.workspace / "config" / "pipeline_config.yaml"
        if not default_config_path.exists():
            with open(default_config_path, 'w') as f:
                yaml.dump(self.config.to_dict(), f, default_flow_style=False)

    def load_checkpoints(self):
        """Load processing state for resume capability."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                self.processed_tracks = set(json.load(f))
            self.logger.info(f"Loaded {len(self.processed_tracks)} processed tracks from checkpoint")

        if self.failed_file.exists():
            with open(self.failed_file, 'r') as f:
                self.failed_tracks = [line.strip() for line in f if line.strip()]
            self.logger.info(f"Loaded {len(self.failed_tracks)} failed tracks from previous runs")

    def save_checkpoint(self, track_id: str):
        """Save processing state after each track."""
        self.processed_tracks.add(track_id)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(self.processed_tracks), f, indent=2)

    def parse_catalog(self) -> List[Dict]:
        """Parse the artist catalog from various sources."""
        catalog_path = Path(self.config.catalog_file)
        tracks = []

        if catalog_path.exists():
            self.logger.info(f"Loading catalog from: {catalog_path}")
            with open(catalog_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('|')
                    url = parts[0].strip()
                    custom_title = parts[1].strip() if len(parts) > 1 else None
                    tags = parts[2].strip() if len(parts) > 2 else ""
                    tracks.append({
                        'url': url,
                        'custom_title': custom_title,
                        'tags': tags,
                        'catalog_line': line_num
                    })
        else:
            self.logger.warning(f"Catalog file not found: {catalog_path}")
            if self.config.catalog_file.startswith(('http://', 'https://')):
                tracks.append({
                    'url': self.config.catalog_file,
                    'custom_title': None,
                    'tags': "",
                    'catalog_line': 1
                })

        self.logger.info(f"Parsed {len(tracks)} tracks from catalog")
        return tracks

    def extract_audio(self, url: str, track_id: str, custom_title: Optional[str] = None) -> Optional[Path]:
        """Extract and process audio from source URL."""
        output_path = self.audio_dir / f"{track_id}.{self.config.audio_format}"

        if output_path.exists() and self._validate_audio_file(output_path):
            self.logger.info(f"Audio already exists and valid: {output_path.name}")
            return output_path

        self.logger.info(f"Extracting audio for {track_id}")
        cmd = [
            "yt-dlp",
            "-x",
            "--audio-format", self.config.audio_format,
            "--audio-quality", "0",
            "--postprocessor-args", f"-ar {self.config.sample_rate} -ac {self.config.channels} -sample_fmt s16",
            "-o", str(self.audio_dir / f"{track_id}.%(ext)s"),
            "--no-playlist",
            "--retries", str(self.config.max_retries),
            "--retry-sleep", str(self.config.retry_delay),
            url
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.debug(f"yt-dlp output: {result.stdout}")
            if output_path.exists() and self._validate_audio_file(output_path):
                self.logger.info(f"Successfully extracted: {output_path.name}")
                return output_path
            else:
                self.logger.error(f"Extraction failed validation: {track_id}")
                return None
        except subprocess.CalledProcessError as e:
            self.logger.error(f"yt-dlp failed for {track_id}: {e.stderr}")
            return None

    def _validate_audio_file(self, audio_path: Path) -> bool:
        """Validate WAV file against MiniMax Music 3 specifications."""
        if not audio_path.exists():
            return False
        if audio_path.stat().st_size < 10000:
            self.logger.warning(f"Audio file too small: {audio_path.name}")
            return False

        try:
            with wave.open(str(audio_path), 'rb') as wav_file:
                channels = wav_file.getnchannels()
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                num_frames = wav_file.getnframes()

            is_valid = (
                channels == self.config.channels and
                sample_rate == self.config.sample_rate and
                sample_width == self.config.sample_width and
                num_frames > 0
            )
            if not is_valid:
                self.logger.warning(
                    f"Audio validation failed for {audio_path.name}: "
                    f"channels={channels}, rate={sample_rate}, width={sample_width}, frames={num_frames}"
                )
            return is_valid
        except Exception as e:
            self.logger.error(f"Error validating {audio_path.name}: {e}")
            return False

    def normalize_audio(self, audio_path: Path) -> bool:
        """Normalize audio loudness for consistent training data."""
        if not self.config.enable_loudness_normalization:
            return True

        self.logger.info(f"Normalizing loudness: {audio_path.name}")
        try:
            result = subprocess.run(["ffmpeg-normalize", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.warning("ffmpeg-normalize not available, skipping normalization")
                return False

            cmd = [
                "ffmpeg-normalize",
                str(audio_path),
                "-o", str(audio_path),
                "--sample-rate", str(self.config.sample_rate),
                "--audio-codec", "pcm_s16le",
                "--audio-bitrate", "16",
                "--target-level", str(self.config.target_loudness),
                "--keep-loudness-range-target",
                "--print-stats"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.debug(f"Normalization output: {result.stdout}")
            return self._validate_audio_file(audio_path)
        except Exception as e:
            self.logger.error(f"Normalization failed for {audio_path.name}: {e}")
            return False

    def generate_lyrics_template(self, track_id: str, raw_title: str) -> Path:
        """Generate structured lyric template with section markers."""
        lyric_path = self.text_dir / f"{track_id}.txt"
        if lyric_path.exists():
            return lyric_path

        template = f"""# MiniMax Music 3 Training Template
# Artist: {self.config.artist_name} ({self.config.project_moniker})
# Track: {raw_title}
# Dataset Version: {self.config.dataset_version}
# Generated: {datetime.now().isoformat()}

[INTRO]
# Instrumental introduction
# Mood: Atmospheric
# Energy: Building

[VERSE 1]
# Main lyrical content
# Theme: Personal expression
# Vocal style: Signature delivery

[PRE-CHORUS]
# Transitional section
# Energy: Rising

[CHORUS]
# Hook section
# Theme: Central message
# Repetition: Key phrases

[VERSE 2]
# Secondary verse
# Theme: Development of ideas

[BRIDGE]
# Contrast section
# Energy: Reflective

[OUTRO]
# Closing section
# Mood: Resolution
"""
        with open(lyric_path, 'w', encoding='utf-8') as f:
            f.write(template)

        self.logger.info(f"Created lyric template: {lyric_path.name}")
        return lyric_path

    def extract_audio_features(self, audio_path: Path) -> Dict:
        """Extract musical features for enhanced metadata."""
        if not self.config.enable_audio_analysis:
            return {}

        features = {
            'bpm': 90,
            'key': 'minor',
            'energy': 'medium',
            'mood': 'atmospheric'
        }

        try:
            import librosa
            self.logger.info(f"Extracting audio features: {audio_path.name}")
            y, sr = librosa.load(str(audio_path), sr=self.config.sample_rate, mono=True)

            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['bpm'] = round(float(tempo), 1)

            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_mean = chroma.mean(axis=1)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_index = np.argmax(chroma_mean)
            features['key'] = keys[key_index]

            rms = librosa.feature.rms(y=y)[0]
            features['energy'] = 'high' if rms.mean() > 0.1 else 'medium' if rms.mean() > 0.05 else 'low'

            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
            if spectral_centroid < 1500:
                features['mood'] = 'dark'
            elif spectral_centroid < 3000:
                features['mood'] = 'atmospheric'
            else:
                features['mood'] = 'bright'
        except ImportError:
            self.logger.warning("librosa not installed, using default audio features")
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")

        return features

    def generate_metadata(self, track_id: str, raw_title: str, audio_path: Path, tags: str = "") -> Dict:
        """Generate comprehensive metadata for training."""
        features = self.extract_audio_features(audio_path)
        caption = (
            f"Global Metadata: Artist: {self.config.artist_name}, "
            f"Title: {raw_title}, "
            f"Genre: hip-hop/alternative, "
            f"BPM: {features.get('bpm', 90)}, "
            f"Key: {features.get('key', 'minor')}, "
            f"Energy: {features.get('energy', 'medium')}, "
            f"Mood: {features.get('mood', 'atmospheric')}, "
            "Emotional Progression: Atmospheric production and signature vocal styling. "
            "Vocal Details: Distinctive lead timbre with layered harmonies. "
            "Arrangement: Professional studio master instrumentation with dynamic range."
        )
        if tags:
            caption += f" Additional Tags: {tags}"

        metadata = {
            'file_name': f"audio/{track_id}.{self.config.audio_format}",
            'caption': caption,
            'lyrics_path': f"text/{track_id}.txt",
            'artist': self.config.artist_name,
            'project': self.config.project_moniker,
            'title': raw_title,
            'track_id': track_id,
            'sample_rate': self.config.sample_rate,
            'channels': self.config.channels,
            'sample_width': self.config.sample_width,
            'dataset_version': self.config.dataset_version,
            'features': json.dumps(features),
            'tags': tags,
            'processed_date': datetime.now().isoformat()
        }
        return metadata

    def process_track(self, track_info: Dict, index: int) -> bool:
        """Process a single track through the complete pipeline."""
        url = track_info['url']
        custom_title = track_info.get('custom_title')
        tags = track_info.get('tags', "")

        raw_title = custom_title or self._get_track_title(url, index)
        track_id = f"{index:04d}_{self._sanitize_filename(raw_title[:40])}"

        if track_id in self.processed_tracks:
            self.logger.info(f"Skipping already processed: {track_id}")
            return True

        self.logger.info(f"Processing track {index}: {raw_title}")
        try:
            audio_path = self.extract_audio(url, track_id, custom_title)
            if not audio_path:
                self.logger.error(f"Audio extraction failed for {track_id}")
                self.failed_tracks.append(url)
                return False

            if self.config.enable_loudness_normalization:
                self.normalize_audio(audio_path)

            self.generate_lyrics_template(track_id, raw_title)
            metadata = self.generate_metadata(track_id, raw_title, audio_path, tags)
            self.metadata_records.append(metadata)

            self.save_checkpoint(track_id)
            self.logger.info(f"Successfully processed: {track_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to process {track_id}: {e}")
            self.failed_tracks.append(url)
            return False

    def _get_track_title(self, url: str, index: int) -> str:
        """Retrieve track title from source URL."""
        try:
            result = subprocess.run(
                ["yt-dlp", "--get-title", "--no-playlist", url],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            self.logger.warning(f"Failed to get title for track {index}: {e}")
        return f"track_{index:04d}"

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        """Sanitize string for safe filename usage."""
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', text)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized.lower()

    def save_dataset(self):
        """Save all metadata records to CSV and generate documentation."""
        if not self.metadata_records:
            self.logger.warning("No metadata records to save")
            return

        df = pd.DataFrame(self.metadata_records)
        csv_path = self.metadata_dir / "metadata.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved metadata to: {csv_path}")

        json_path = self.metadata_dir / "metadata.json"
        with open(json_path, 'w') as f:
            json.dump(self.metadata_records, f, indent=2)
        self.logger.info(f"Saved JSON metadata to: {json_path}")

        self.generate_dataset_documentation(df)

        if self.failed_tracks:
            self.failed_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.failed_file, 'w') as f:
                f.write("\n".join(self.failed_tracks))
            self.logger.warning(f"Saved {len(self.failed_tracks)} failed tracks to: {self.failed_file}")

    def generate_dataset_documentation(self, df: pd.DataFrame):
        """Generate comprehensive dataset documentation."""
        doc_path = self.metadata_dir / "README.md"
        total_files = len(df)
        total_size = sum(
            (self.audio_dir / f"{row['track_id']}.{self.config.audio_format}").stat().st_size
            for _, row in df.iterrows()
            if (self.audio_dir / f"{row['track_id']}.{self.config.audio_format}").exists()
        )

        documentation = f"""# {self.config.artist_name} ({self.config.project_moniker}) Dataset

## Overview
This dataset contains {total_files} tracks from {self.config.artist_name}, processed for fine-tuning MiniMax Music 3 models.

## Dataset Statistics
- **Total Tracks:** {total_files}
- **Total Size:** {total_size / (1024**3):.2f} GB
- **Format:** {self.config.audio_format.upper()} ({self.config.sample_rate}Hz, {self.config.channels}-channel, {self.config.sample_width * 8}-bit)
- **Dataset Version:** {self.config.dataset_version}
- **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Directory Structure
```text
dataset/
├── audio/          # WAV files ({self.config.sample_rate}Hz stereo 16-bit)
│   └── 0001_track_name.wav
├── text/           # Lyric files with section markers
│   └── 0001_track_name.txt
└── metadata/       # Training metadata
    ├── metadata.csv
    ├── metadata.json
    └── README.md
"""
        with open(doc_path, 'w') as fh:
            fh.write(documentation)
        self.logger.info(f"Saved dataset documentation to: {doc_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a MiniMax Music 3 fine-tune dataset from an artist catalog")
    p.add_argument("--workspace", default="minimax_artist_workspace")
    p.add_argument("--catalog", default="catalog/youtube_catalog.txt",
                   help="catalog file: one 'url | title | tags' per line")
    p.add_argument("--config", default=None,
                   help="optional pipeline_config.yaml")
    p.add_argument("--init-only", action="store_true",
                   help="only create the workspace and default config")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = PipelineConfig(workspace=args.workspace, catalog_file=args.catalog)
    if args.config:
        cfg = PipelineConfig.from_dict(
            yaml.safe_load(open(args.config)))
        cfg.workspace = args.workspace
        cfg.catalog_file = args.catalog
    builder = MiniMaxDatasetBuilder(config=cfg)
    builder.setup_workspace()
    builder.load_checkpoints()
    if args.init_only:
        print(f"Workspace ready: {builder.workspace}")
        return 0
    tracks = builder.parse_catalog()
    for i, track in enumerate(tracks, 1):
        builder.process_track(track, i)
        if i % cfg.batch_size == 0:
            time.sleep(cfg.delay_between_batches)
    builder.save_dataset()
    print(f"Done. {len(builder.metadata_records)} tracks processed, "
          f"{len(builder.failed_tracks)} failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
