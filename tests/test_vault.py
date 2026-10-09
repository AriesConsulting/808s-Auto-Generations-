import hashlib

import numpy as np
import pytest

cryptography = pytest.importorskip("cryptography")

from pmve import vault
from cryptography.fernet import Fernet


def test_sha256_file(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello")
    assert vault.sha256_file(f) == hashlib.sha256(b"hello").hexdigest()


def test_guide_to_trl4_blocked(tmp_path):
    f = tmp_path / "labor.bin"
    f.write_bytes(b"not a clean hash")
    key = Fernet.generate_key()
    assert vault.guide_to_trl4(str(f), key).startswith("STATUS: BLOCKED")


def test_guide_to_trl4_missing_file():
    with pytest.raises(FileNotFoundError):
        vault.guide_to_trl4("/nope/missing.bin", Fernet.generate_key())


def test_vault_roundtrip(tmp_path):
    v = vault.LaborVault("Tester")
    assert v.stage == 1
    assert not v.harden(b"x")  # must anchor first
    assert v.anchor(b"audio bytes")
    assert v.harden(b"secret melody")
    assert v.decrypt() == b"secret melody"
    path = v.isolate(base_dir=str(tmp_path / "enclave"))
    assert path.endswith("Tester_TRL5")
    assert v.stage == 5


def test_spectral_audit():
    v = vault.LaborVault("Tester")
    assert v.spectral_audit(np.random.randn(1000))
    assert not v.spectral_audit(np.zeros(1000))


def test_check_trigger():
    v = vault.LaborVault("Tester")
    assert v.check_trigger([24, 44, 48, 63, 78, 93, 107])
    assert not v.check_trigger([60, 62, 64])


def test_deployment_report():
    v = vault.LaborVault("Tester")
    v.anchor(b"data")
    report = v.deployment_report()
    assert report["artist"] == "Tester"
    assert report["status"] == "COMPLETE"
