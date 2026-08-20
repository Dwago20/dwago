"""Repo-specific retrieval benchmark mined from the project's own git history.

The claim "this retrieves better than graphify" is worthless unless it is a
measured number on the user's own code, so this harness exists to produce that
number — including when the answer is "it doesn't".

**Ground truth.** For each historical change, the query is what the author wrote
about the work (commit subject and body) and the correct answer is the set of
files they actually touched. That is a real retrieval task with a real label,
available in any repository, at any size, with no annotation effort.

**Leakage control.** The co-change layer is mined from commit history. If the
evaluation commits are inside that history, the temporal channel has memorized
the answer key and every number involving it is inflated. So history is split by
time: temporal edges come only from commits *before* a cutoff, and only commits
*after* the cutoff are scored. The structural graph is still built from HEAD,
which is correct — a user queries the code as it exists now — but nothing about
the evaluated changes reaches the index.

**Known biases**, stated because they affect how the numbers should be read:

- Commit subjects are written *after* the work, so they use the vocabulary of
  the implementation. That systematically favours lexical retrieval over
  semantic retrieval, meaning the BM25 rung is flattered relative to how it
  performs on the questions a person actually asks mid-task. Improvements over
  BM25 here are therefore conservative.
- Files renamed or deleted since the cutoff are dropped from ground truth rather
  than counted as misses, which would blame retrieval for history.
- Lockfiles and generated artifacts are excluded; they co-occur with everything
  and would inflate every rung equally.
"""
from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ["EvalItem", "RungResult", "build_eval_set", "run_ladder", "format_report"]

# A change touching more files than this is a refactor, a reformat, or a
# vendored-dependency bump. Its "topic" is not a retrievable thing.
MAX_FILES_PER_ITEM = 12
MIN_FILES_PER_ITEM = 1

# Subjects that describe process rather than code. Retrieval cannot be expected
# to localize "bump version to 1.2.3" and scoring it only adds noise.
_SKIP_SUBJECT = re.compile(
    r"^\s*(merge|revert|bump|release|v?\d+\.\d+\.\d+|chore\(deps\)|"
    r"update changelog|typo|formatting|lint|rename)\b",
    re.IGNORECASE,
)

_NOISE_PATH = re.compile(
    r"(^|/)(package-lock\.json|yarn\.lock|poetry\.lock|Cargo\.lock|uv\.lock|go\.sum|"
    r"CHANGELOG(\.md)?)$|\.(min\.js|min\.css|lock)$|(^|/)(dist|build|vendor)/",
    re.IGNORECASE,
)


@dataclass
class EvalItem:
    sha: str
    ts: int
    query: str
    files: list[str]


@dataclass
class RungResult:
    name: str
    recall_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    seconds_per_query: float = 0.0
    per_item_recall: list[float] = field(default_factory=list)
    # Per-item recall at each k. A single k hides where a change acts: a
    # retriever already near ceiling at R@20 can still improve a lot at R@5,
    # which is the regime that matters when packing a context window.
    per_item_by_k: dict = field(default_factory=dict)
    n: int = 0
    note: str = ""


def _git(root: Path, *args: str) -> str:
    import subprocess

    proc = subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {proc.stderr[:200]}")
    return proc.stdout


def split_commit(root: Path, fraction: float = 0.8) -> tuple[str, int]:
    """Return (cutoff_sha, cutoff_timestamp) at ``fraction`` through history."""
    shas = _git(root, "log", "--no-merges", "--format=%H %at").strip().splitlines()
    if len(shas) < 20:
        raise RuntimeError(f"only {len(shas)} commits — too little history to evaluate")
    # git log is newest-first; the cutoff sits `fraction` of the way back.
    idx = int(len(shas) * (1.0 - fraction))
    sha, ts = shas[idx].split()
    return sha, int(ts)


def build_eval_set(
    root: Path,
    *,
    after_ts: int,
    limit: int = 200,
    seed: int = 0,
) -> list[EvalItem]:
    """Mine scoreable changes newer than ``after_ts``."""
    sep = chr(30)
    fs = chr(31)
    raw = _git(
        root, "log", "--no-merges", "--name-only",
        f"--format={sep}%H{fs}%at{fs}%s{fs}%b", f"--since={after_ts}",
    )

    items: list[EvalItem] = []
    cur: EvalItem | None = None
    # split on newline only — splitlines() would eat the \x1e separator itself.
    for line in raw.split(chr(10)):
        if line.startswith(sep):
            parts = line[1:].split(fs)
            if len(parts) >= 3:
                subject = parts[2].strip()
                body = parts[3].strip() if len(parts) > 3 else ""
                query = f"{subject}. {body}".strip()[:600]
                cur = EvalItem(sha=parts[0], ts=int(parts[1] or 0),
                               query=query, files=[])
                items.append(cur)
            continue
        line = line.strip()
        if not line or cur is None:
            continue
        if not _NOISE_PATH.search(line):
            cur.files.append(line)

    existing = set(_git(root, "ls-files").split(chr(10)))
    keep: list[EvalItem] = []
    for it in items:
        if it.ts <= after_ts:
            continue
        if _SKIP_SUBJECT.match(it.query):
            continue
        # Files gone since the cutoff are dropped: a rename is history moving,
        # not retrieval failing.
        files = [f for f in dict.fromkeys(it.files) if f in existing]
        if not (MIN_FILES_PER_ITEM <= len(files) <= MAX_FILES_PER_ITEM):
            continue
        if len(it.query) < 15:
            continue
        it.files = files
        keep.append(it)

    rng = random.Random(seed)
    rng.shuffle(keep)
    return keep[:limit]


def _files_from_hits(hits, k: int) -> list[str]:
    """Collapse ranked node hits to a ranked, deduplicated file list."""
    out: list[str] = []
    seen: set[str] = set()
    for h in hits:
        f = h.source_file
        if f and f not in seen:
            seen.add(f)
            out.append(f)
            if len(out) >= k:
                break
    return out


def _score(predicted: list[str], truth: list[str], ks: tuple[int, ...]) -> tuple[dict, float]:
    truth_set = set(truth)
    recalls = {}
    for k in ks:
        hit = len(truth_set & set(predicted[:k]))
        recalls[k] = hit / len(truth_set) if truth_set else 0.0
    rr = 0.0
    for rank, f in enumerate(predicted, start=1):
        if f in truth_set:
            rr = 1.0 / rank
            break
    return recalls, rr


def run_ladder(
    store,
    root: Path,
    items: list[EvalItem],
    *,
    rungs: list[str] | None = None,
    ks: tuple[int, ...] = (1, 5, 10, 20),
    temporal_weight: float = 0.3,
) -> list[RungResult]:
    """Evaluate each retrieval configuration on the same items."""
    from ..retrieve.hybrid import Retriever

    rungs = rungs or ["bm25", "dense", "hybrid", "hybrid+ppr"]
    results: list[RungResult] = []
    max_k = max(ks)

    for rung in rungs:
        r = Retriever(store, temporal_weight=temporal_weight)
        if rung == "bm25":
            r.dense.vectors = None          # force lexical-only
        elif rung == "dense":
            r.lexical = type(r.lexical)()   # empty index, disabled
            if not r.dense.available:
                results.append(RungResult(name=rung, note="no dense index — skipped"))
                continue

        use_ppr = rung.endswith("+ppr")
        rerank = "rerank" in rung

        rec_tot = {k: 0.0 for k in ks}
        rr_tot = 0.0
        per_item: list[float] = []
        per_k: dict[int, list[float]] = {k: [] for k in ks}
        t0 = time.time()
        n = 0

        for it in items:
            res = r.search(it.query, k=max_k * 3, use_ppr=use_ppr, rerank=rerank)
            pred = _files_from_hits(res.hits, max_k)
            recalls, rr = _score(pred, it.files, ks)
            for k in ks:
                rec_tot[k] += recalls[k]
                per_k[k].append(recalls[k])
            rr_tot += rr
            per_item.append(recalls[max(ks)])
            n += 1

        if n == 0:
            results.append(RungResult(name=rung, note="no items"))
            continue

        results.append(RungResult(
            name=rung,
            recall_at={k: rec_tot[k] / n for k in ks},
            mrr=rr_tot / n,
            seconds_per_query=(time.time() - t0) / n,
            per_item_recall=per_item,
            per_item_by_k=per_k,
            n=n,
        ))

    return results


def paired_bootstrap(a: list[float], b: list[float], *, iters: int = 2000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap of mean(b) - mean(a).

    Paired, because both rungs answer the *same* queries: the variance between
    queries dwarfs the difference between systems, and an unpaired test would
    drown a real improvement in it. Returns (delta, ci_low, ci_high).
    """
    if not a or not b or len(a) != len(b):
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(a)
    diffs = [b[i] - a[i] for i in range(n)]
    observed = sum(diffs) / n
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters) - 1]
    return observed, lo, hi


def format_report(results: list[RungResult], ks: tuple[int, ...] = (1, 5, 10, 20),
                  *, baseline: str | None = None) -> str:
    live = [r for r in results if r.n]
    if not live:
        return "no rungs produced results"

    out = ["", f"Retrieval ladder — {live[0].n} held-out changes", ""]
    header = f"  {'rung':<16}" + "".join(f"{'R@'+str(k):>9}" for k in ks) + \
             f"{'MRR':>9}{'s/query':>10}"
    out.append(header)
    out.append("  " + "-" * (len(header) - 2))
    for r in results:
        if not r.n:
            out.append(f"  {r.name:<16} {r.note}")
            continue
        row = f"  {r.name:<16}" + "".join(f"{r.recall_at[k]:>9.3f}" for k in ks)
        row += f"{r.mrr:>9.3f}{r.seconds_per_query:>10.3f}"
        out.append(row)

    # Consecutive comparisons with a paired CI, which is what decides whether a
    # tier ships on by default.
    out.append("")
    out.append(f"  Pairwise gain (paired bootstrap, 95% CI, on R@{max(ks)}):")
    for prev, cur in zip(live, live[1:]):
        d, lo, hi = paired_bootstrap(prev.per_item_recall, cur.per_item_recall)
        verdict = "significant" if lo > 0 else ("regression" if hi < 0 else "not significant")
        out.append(f"    {prev.name:>14} -> {cur.name:<16} "
                   f"{d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}")

    # The consecutive chain can hide the number that actually matters: whether
    # the full stack beats the cheapest baseline. Report it explicitly.
    if len(live) > 1:
        first, last = live[0], live[-1]
        out.append("")
        out.append(f"  End to end: {first.name} -> {last.name}")
        for k in sorted(ks):
            a = first.per_item_by_k.get(k) or []
            b = last.per_item_by_k.get(k) or []
            if not a:
                continue
            d, lo, hi = paired_bootstrap(a, b)
            verdict = ("significant" if lo > 0
                       else ("regression" if hi < 0 else "not significant"))
            ceiling = " (baseline near ceiling)" if first.recall_at[k] > 0.8 else ""
            out.append(f"    R@{k:<3} {first.recall_at[k]:.3f} -> "
                       f"{last.recall_at[k]:.3f}   {d:+.3f} "
                       f"[{lo:+.3f}, {hi:+.3f}]  {verdict}{ceiling}")
    out.append("")
    return "\n".join(out)
