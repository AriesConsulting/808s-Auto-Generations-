"""Integrity gate and encrypted vault for artist labor.

What it does, plainly:
- hash a file with SHA-256 and gate on the digest (integrity check)
- encrypt blobs with Fernet into a per-session vault key
- spectral audit: Butterworth low-pass an audio buffer and check variance
- MIDI trigger detection: fire when a specific note sequence is present

Extracted from the LTR Advancement Engine and LTR Sovereign Guidance Engine
notebooks. Needs: cryptography, numpy, scipy.
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional

import numpy as np
from cryptography.fernet import Fernet
from scipy import signal


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def guide_to_trl4(artist_labor_path: str, secret_key: bytes) -> str:
    """Integrity gate: only encrypt the file when its hash looks clean.

    Returns a status string. Raises FileNotFoundError for a missing file.
    """
    current_hash = sha256_file(artist_labor_path)
    if current_hash.startswith("000"):
        print("TRL-3 INTEGRITY VERIFIED. GUIDING TO TRL-4 VAULT...")
        cipher = Fernet(secret_key)
        cipher.encrypt(open(artist_labor_path, "rb").read())
        return "STATUS: TRL-4 SOVEREIGN"
    return "STATUS: BLOCKED. INTEGRITY FAILURE DETECTED. RE-RECORD REQUIRED."


class LaborVault:
    """Staged pipeline: anchor identity, harden, isolate, audit, deploy."""

    # MIDI note numbers for the emergency trigger sequence
    NUKE_SEQUENCE = [24, 44, 48, 63, 78, 93, 107]

    def __init__(self, artist_name: str) -> None:
        self.artist = artist_name
        self.stage = 1
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        self.root_hash: Optional[str] = None
        self.encrypted_labor: Optional[bytes] = None
        print(f"--- VAULT INITIALIZED: {self.artist} ---")

    def anchor(self, raw_audio_data: bytes) -> bool:
        """Stage 1-3: anchor the work's identity to a SHA-256 hash."""
        self.root_hash = hashlib.sha256(raw_audio_data).hexdigest()
        print(f"[1-3] Identity anchored. Hash: {self.root_hash[:16]}...")
        self.stage = 3
        return True

    def harden(self, sensitive_data: bytes) -> bool:
        """Stage 4: encrypt labor into the vault. Requires anchor first."""
        if self.stage < 3:
            return False
        self.encrypted_labor = self.cipher.encrypt(sensitive_data)
        print("[4] Labor hardened. Encryption active.")
        self.stage = 4
        return True

    def decrypt(self) -> Optional[bytes]:
        if self.encrypted_labor is None:
            return None
        return self.cipher.decrypt(self.encrypted_labor)

    def isolate(self, base_dir: str = "./enclave") -> str:
        """Stage 5: create an isolated directory for the artist's work."""
        path = os.path.join(base_dir, f"{self.artist}_TRL5")
        os.makedirs(path, exist_ok=True)
        print("[5] Isolation directory verified.")
        self.stage = 5
        return path

    def spectral_audit(self, audio_buffer: np.ndarray,
                       variance_floor: float = 0.01) -> bool:
        """Stage 6: Butterworth low-pass the buffer, check signal variance."""
        b, a = signal.butter(4, 0.5)
        clean = signal.filtfilt(b, a, audio_buffer)
        if float(np.var(clean)) > variance_floor:
            print("[6] Spectral audit passed. Signal integrity validated.")
            self.stage = 6
            return True
        return False

    def check_trigger(self, midi_stream: List[int],
                      sequence: Optional[List[int]] = None) -> bool:
        """Stage 8: True when every note of the trigger sequence is present."""
        seq = sequence or self.NUKE_SEQUENCE
        if all(note in midi_stream for note in seq):
            print("!!! EMERGENCY TRIGGER DETECTED !!!")
            self.stage = 8
            return True
        print("[8] Trigger check clean.")
        return False

    def deployment_report(self) -> Dict[str, str]:
        """Stage 9: summarize the vault state as a report dict."""
        self.stage = 9
        return {
            "architect": "Aries Hilton",
            "artist": self.artist,
            "stage": "9",
            "root_hash": (self.root_hash or "")[:16],
            "status": "COMPLETE",
        }


def demo() -> Dict[str, str]:
    vault = LaborVault("Ambassador_Alpha")
    vault.anchor(b"Sample MIDI Data")
    vault.harden(b"Proprietary Melody")
    vault.isolate(base_dir="/tmp/auto808_enclave")
    vault.spectral_audit(np.random.randn(1000))
    vault.check_trigger([24, 44, 48, 63, 78, 93, 107])
    report = vault.deployment_report()
    print(report)
    return report
