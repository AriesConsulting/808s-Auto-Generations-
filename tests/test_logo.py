import pytest

svgwrite = pytest.importorskip("svgwrite")

from auto808 import logo


def test_logo_renders(tmp_path):
    out = str(tmp_path / "logo.svg")
    result = logo.create_lucid_triangulation_logo(out)
    assert result == out
    text = open(out).read()
    assert text.count("<polygon") >= 4  # 3 triangles + white core
    assert "LUCID TRIANGULATION" in text
    assert "RECORDS" in text
    assert len(text) > 1000
