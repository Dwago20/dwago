"""Content-addressed node identity.

graphify's node IDs are *not* stable across rebuilds. Its own ``ids.py`` docstring
lists a recurring bug class (#811 Unicode collapse, #550 same-filename collisions,
#1033 AST-vs-LLM mismatch, #1104, #2614 Turkish casefold), and ``export.py`` warns
that fuzzy dedup can collapse same-named symbols across files during an ``--update``.
Keying an expensive artifact (a vector, an LLM summary) on a graphify ID therefore
means it rots silently the first time upstream changes its recipe or dedup fires.

So dwago never uses a graphify ID as a primary key. It computes its own key from
*content* — where the symbol lives and what it is called — and keeps a remap table
from graphify ID to that key. When graphify renames a node but the code did not
move, the content key is unchanged and the cached vector survives.

Two different keys are needed and they are deliberately distinct:

``node_key``
    Identity. "Is this the same symbol as before?" Derives from location + name
    only, so a node keeps its key while its body is edited. Vectors and summaries
    are keyed on this.

``content_hash``
    Freshness. "Has this symbol's text changed?" Derives from the source text.
    A changed hash marks the node dirty and schedules re-embedding.

Separating them is what makes incremental refresh work: an edit changes the
content hash but not the key, so we re-embed exactly one row instead of
rebuilding the table.
"""
from __future__ import annotations

import hashlib

__all__ = ["node_key", "content_hash", "file_key"]

# Truncated BLAKE2b. 16 bytes / 128 bits: collision probability stays negligible
# far past any realistic node count (~1e-14 at 10M nodes) while keeping the key
# short enough to sit in an index without bloating it.
_DIGEST_BYTES = 16


def _h(*parts: str) -> str:
    d = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    for p in parts:
        # Length-prefix each part. Without it, ("ab", "c") and ("a", "bc") hash
        # identically, which would silently merge two distinct symbols.
        d.update(str(len(p)).encode("utf-8"))
        d.update(b"\x1f")
        d.update(p.encode("utf-8", errors="replace"))
        d.update(b"\x1e")
    return d.hexdigest()


def node_key(source_file: str, label: str, kind: str = "", start_line: int | None = None) -> str:
    """Stable identity for one graph node.

    ``start_line`` is deliberately excluded from the hash by default. Adding a
    function above another shifts every line below it; including the line number
    would invalidate the entire file's cache on a one-line insert. Location is
    represented by the file path only, with ``label`` and ``kind`` disambiguating
    symbols inside it.

    Overloads and same-named nested symbols in one file collide by construction.
    That is accepted: the alternative (line numbers in the key) trades a rare
    collision for guaranteed cache invalidation on every edit. Callers that need
    to split a collision pass ``start_line`` explicitly.
    """
    parts = [source_file or "", label or "", kind or ""]
    if start_line is not None:
        parts.append(f"L{start_line}")
    return _h(*parts)


def content_hash(text: str) -> str:
    """Freshness fingerprint for a node's source text."""
    return _h(text or "")


def file_key(source_file: str) -> str:
    """Identity for a file-level node."""
    return _h(source_file or "", "", "file")
