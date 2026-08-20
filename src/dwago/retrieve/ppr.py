"""Personalized PageRank over the graph, run as two independent channels.

The mechanism is HippoRAG-2's: rather than expanding uniformly from a query hit
(graphify's BFS/DFS, which treats a god node's 3,000th edge exactly like a
function's only caller), seed a random walk with the retrieval scores and let
the graph's own structure decide what else is relevant. Nodes that many strong
seeds reach by short paths accumulate mass; nodes that one weak seed touches
once do not.

**Why two channels rather than one diffusion.** Structural edges (calls,
imports, inheritance) and temporal edges (co-change) answer different questions
and have wildly different degree distributions. Co-change is dominated by hubs —
a settings file that moves with everything — so mixing the two lets those hubs
absorb walk mass that belongs to the call graph. Running them separately, each
row-normalized in its own channel, and combining the resulting distributions
afterwards keeps both signals intact and, just as usefully, keeps them
*separable*: `impact_of` can report "this is here because it is called" versus
"this is here because it always changes with you", which a single blended score
could never explain.

Direction is applied at query time. Context retrieval walks undirected (a caller
explains a callee and vice versa); impact analysis walks against the edges, from
a symbol to the things that depend on it.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["ppr", "combine_channels", "PPRResult"]

DEFAULT_ALPHA = 0.85
DEFAULT_ITERS = 30
CONVERGENCE_TOL = 1e-6


def _row_normalize(mat):
    """Scale each row to sum to 1, leaving dangling rows at zero.

    Normalizing per channel is what stops a dense channel from dominating a
    sparse one when the two are combined: after this, every node distributes
    exactly one unit of mass regardless of how many edges it happens to have.
    """
    from scipy import sparse

    deg = np.asarray(mat.sum(axis=1)).ravel()
    inv = np.zeros_like(deg, dtype=np.float32)
    nz = deg > 0
    inv[nz] = 1.0 / deg[nz]
    return sparse.diags(inv, dtype=np.float32) @ mat


class PPRResult:
    __slots__ = ("scores", "iterations", "converged")

    def __init__(self, scores: np.ndarray, iterations: int, converged: bool):
        self.scores = scores
        self.iterations = iterations
        self.converged = converged

    def top(self, k: int, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        s = self.scores
        if exclude:
            s = s.copy()
            s[list(exclude)] = 0.0
        # argpartition beats a full sort when k << n, which it always is here.
        k = min(k, int((s > 0).sum()))
        if k <= 0:
            return []
        part = np.argpartition(-s, k - 1)[:k]
        order = part[np.argsort(-s[part])]
        return [(int(i), float(s[i])) for i in order]


def ppr(
    mat,
    seeds: dict[int, float],
    *,
    alpha: float = DEFAULT_ALPHA,
    iters: int = DEFAULT_ITERS,
    reverse: bool = False,
) -> PPRResult:
    """Personalized PageRank by power iteration.

    ``seeds`` maps node index to preference mass; it is normalized internally so
    callers can pass raw retrieval scores. ``reverse`` transposes the matrix,
    turning "what does this reach" into "what reaches this" — the orientation
    impact analysis needs.

    A seed is not guaranteed to come out on top. Mass spreads along edges, so a
    low-degree seed next to a hub ends up with less of it than the hub does.
    That is correct behaviour for a diffusion, and it is why the retrieval
    pipeline combines these scores with the direct retrieval score rather than
    ranking on them alone.
    """
    n = mat.shape[0]
    if n == 0 or not seeds:
        return PPRResult(np.zeros(n, dtype=np.float32), 0, True)

    m = mat.T if reverse else mat
    m = _row_normalize(m).T.tocsr()  # column-stochastic for the matvec below

    p0 = np.zeros(n, dtype=np.float32)
    total = sum(max(v, 0.0) for v in seeds.values())
    if total <= 0:
        return PPRResult(p0, 0, True)
    for i, v in seeds.items():
        if 0 <= i < n and v > 0:
            p0[i] = v / total

    p = p0.copy()
    converged = False
    it = 0
    for it in range(1, iters + 1):
        nxt = alpha * (m @ p) + (1.0 - alpha) * p0
        # Mass lands on dangling nodes and vanishes; return it to the seeds so
        # the vector stays a distribution and the ranking stays comparable.
        leaked = 1.0 - float(nxt.sum())
        if leaked > 0:
            nxt += leaked * p0
        delta = float(np.abs(nxt - p).sum())
        p = nxt
        if delta < CONVERGENCE_TOL:
            converged = True
            break

    return PPRResult(p, it, converged)


def combine_channels(
    structural: np.ndarray | None,
    temporal: np.ndarray | None,
    *,
    temporal_weight: float = 0.3,
) -> np.ndarray:
    """Blend the two channel distributions.

    Each channel is max-normalized before mixing. Raw PPR masses are not
    comparable across channels — they depend on that channel's density — so
    combining them directly would let whichever graph happens to be denser set
    the scale. ``temporal_weight`` is a tunable the eval harness fits per repo
    rather than a constant asserted here.
    """
    if structural is None and temporal is None:
        raise ValueError("no channels to combine")
    if temporal is None:
        return structural
    if structural is None:
        return temporal

    def _norm(v: np.ndarray) -> np.ndarray:
        peak = float(v.max())
        return v / peak if peak > 0 else v

    w = float(np.clip(temporal_weight, 0.0, 1.0))
    return (1.0 - w) * _norm(structural) + w * _norm(temporal)
