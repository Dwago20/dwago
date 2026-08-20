"""Turn each graph node into a retrieval document.

This is the least glamorous module here and the one that decides whether
retrieval works at all. A node's label alone — ``_pick_seeds``, ``IdempotencyKey``
— is a terrible search target: BM25 sees one rare token and an embedding model
sees an out-of-vocabulary blob. The document assembled here surrounds that label
with everything that gives it meaning: how it is spelled in prose, what file it
lives in, its signature, its docstring, the head of its body, and the names of
its immediate neighbours.

Identifier splitting is the single highest-value transformation. ``parseURLPath``
is one token to a tokenizer and three concepts to a human; emitting
``parseURLPath parse URL Path`` lets a query for "url parsing" match lexically
*and* gives the encoder real words instead of a rare compound. graphify's search
never does this, which is a large part of why its retrieval needs the
hand-written vocabulary workaround in its own skill.

Neighbour labels are included because a symbol is partly defined by what it
touches — a function called only by ``authenticate`` and ``verify_token`` is
about authentication even if its own name says nothing. They are capped and
placed last so they enrich rather than dominate the term statistics.
"""
from __future__ import annotations

import re
from collections import defaultdict

__all__ = ["split_identifier", "build_documents", "node_text"]

# camelCase / PascalCase / snake_case / kebab-case / SCREAMING_CASE, plus the
# ACRONYMWord boundary that a naive regex gets wrong (HTTPServer -> HTTP Server).
_SPLIT_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_SEP_RE = re.compile(r"[_\-./:\\]+")

# Tokens that appear in nearly every identifier and carry no discriminative
# power. Left in the raw label (exact match still works) but not re-emitted as
# split terms, where they would only flatten IDF.
_STOP = frozenset({
    "get", "set", "the", "and", "for", "with", "from", "into", "self", "cls",
    "init", "main", "new", "obj", "val", "tmp", "var", "func", "def",
})


def split_identifier(name: str, *, keep_stop: bool = False) -> list[str]:
    """Split an identifier into its constituent words, lowercased.

    >>> split_identifier("parseURLPath")
    ['parse', 'url', 'path']
    >>> split_identifier("MAX_RETRY_COUNT")
    ['max', 'retry', 'count']
    """
    if not name:
        return []
    out: list[str] = []
    for chunk in _SEP_RE.split(name):
        for part in _SPLIT_RE.findall(chunk):
            low = part.lower()
            if len(low) < 2:
                continue
            if not keep_stop and low in _STOP:
                continue
            out.append(low)
    # Preserve first-seen order while dropping repeats.
    return list(dict.fromkeys(out))


def _path_terms(source_file: str) -> list[str]:
    """Directory and filename components as searchable words.

    Path is a strong topical signal in most codebases: anything under
    ``auth/`` is about auth, and a query saying "auth" should reach it even when
    no symbol name says so.
    """
    if not source_file:
        return []
    parts = [p for p in re.split(r"[/\\]", source_file) if p and p not in {".", ".."}]
    terms: list[str] = []
    for p in parts:
        p = re.sub(r"\.\w+$", "", p)          # strip extension
        terms.extend(split_identifier(p))
    return list(dict.fromkeys(terms))


def node_text(row: dict, neighbours: list[str] | None = None,
              *, max_body_lines: int = 15) -> str:
    """Assemble one node's retrieval document."""
    label = row.get("label") or ""
    clean = label.rstrip("()").lstrip(".")
    parts: list[str] = []

    # 1. The label verbatim, so exact-identifier queries hit hardest.
    parts.append(clean)

    # 2. The label split into words — the transformation that makes natural
    #    language queries able to reach code identifiers at all.
    split = split_identifier(clean)
    if split:
        parts.append(" ".join(split))

    # 3. Kind and path context.
    kind = row.get("kind") or ""
    if kind:
        parts.append(kind)
    pt = _path_terms(row.get("source_file") or "")
    if pt:
        parts.append(" ".join(pt))

    # 4. Signature — types and parameter names are dense with meaning.
    sig = (row.get("signature") or "").strip()
    if sig:
        parts.append(sig[:300])

    # 5. Docstring / leading comment: the author's own description, usually the
    #    single best match for a natural-language question.
    doc = (row.get("docstring") or "").strip()
    if doc:
        parts.append(doc[:600])

    # 6. Head of the body, when there is no docstring to carry the meaning.
    body = (row.get("_body") or "").strip()
    if body and not doc:
        lines = [l.strip() for l in body.splitlines()[:max_body_lines] if l.strip()]
        if lines:
            parts.append(" ".join(lines)[:500])

    # 7. Neighbour labels, capped and last.
    if neighbours:
        nb: list[str] = []
        for n in neighbours[:12]:
            nb.extend(split_identifier(n.rstrip("()").lstrip(".")))
        if nb:
            parts.append(" ".join(list(dict.fromkeys(nb))[:24]))

    return "\n".join(p for p in parts if p)


def build_documents(store, *, neighbour_cap: int = 12) -> int:
    """Build and persist a retrieval document for every node.

    Neighbours are read from the structural channel only. Temporal neighbours
    are deliberately excluded: co-change says two things move together, not that
    they are *about* the same thing, and letting it leak into the text index
    would blur topics that the graph walk can relate later anyway.
    """
    labels: dict[int, str] = {}
    rows: list[dict] = []
    for r in store.conn.execute(
        "SELECT idx, label, kind, source_file, start_line, end_line, "
        "signature, docstring FROM nodes ORDER BY idx"
    ):
        d = dict(r)
        rows.append(d)
        labels[d["idx"]] = d["label"] or ""

    # Adjacency, capped per node so a god node with 3,000 edges does not
    # produce a document that is 95% neighbour names.
    neigh: dict[int, list[str]] = defaultdict(list)
    for e in store.conn.execute(
        "SELECT src, dst FROM edges WHERE channel = 'structural'"
    ):
        s, d = e["src"], e["dst"]
        if len(neigh[s]) < neighbour_cap:
            neigh[s].append(labels.get(d, ""))
        if len(neigh[d]) < neighbour_cap:
            neigh[d].append(labels.get(s, ""))

    updates = []
    for d in rows:
        text = node_text(d, [n for n in neigh.get(d["idx"], []) if n])
        updates.append((text, d["idx"]))

    store.conn.executemany("UPDATE nodes SET doc = ? WHERE idx = ?", updates)
    store.commit()
    return len(updates)
