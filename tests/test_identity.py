"""Content-addressed identity: stability under edits, sensitivity to content."""
from __future__ import annotations

from dwago.identity import content_hash, node_key


def test_key_is_stable_across_body_edits():
    """The whole point: editing a function must not invalidate its identity."""
    a = node_key("src/a.py", "handler", "function")
    b = node_key("src/a.py", "handler", "function")
    assert a == b


def test_key_separates_same_name_in_different_files():
    assert node_key("a.py", "run", "function") != node_key("b.py", "run", "function")


def test_key_separates_kinds():
    assert node_key("a.py", "Thing", "class") != node_key("a.py", "Thing", "function")


def test_length_prefixing_prevents_boundary_collisions():
    """('ab','c') and ('a','bc') must not hash alike."""
    assert node_key("ab", "c", "") != node_key("a", "bc", "")


def test_start_line_disambiguates_overloads():
    base = node_key("a.py", "f", "function")
    assert node_key("a.py", "f", "function", 10) != base
    assert node_key("a.py", "f", "function", 10) != node_key("a.py", "f", "function", 20)


def test_content_hash_tracks_text():
    assert content_hash("x = 1") == content_hash("x = 1")
    assert content_hash("x = 1") != content_hash("x = 2")
