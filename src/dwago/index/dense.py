"""Dense embeddings over node documents, with a tiered backend.

Three tiers, resolved at runtime, best-available first:

1. ``sentence-transformers`` + a real encoder (Qwen3-Embedding-0.6B by default).
   Best quality; needs torch and a multi-GB download.
2. ``model2vec`` static embeddings (potion-base-8M). Pure numpy at inference,
   ~30MB, milliseconds per thousand documents. Far better than nothing and it is
   what ``--fast`` uses.
3. Nothing. Retrieval falls back to BM25 alone, which still works.

Two facts drive the design. First, embedding is the only genuinely slow step in
a build: 100k documents through a 0.6B encoder on CPU is hours, not minutes, so
the caller must be told before it starts and the work must never be repeated for
unchanged nodes. Second, ranking only needs relative order, so vectors are
stored float16 — half the bytes, no measurable ranking difference, and it
matters because the matrix is memmapped.

No vector database. Below ~200k nodes a brute-force cosine against a normalized
float16 matrix is a single BLAS call taking a few milliseconds; an index would
add a dependency, a build step, and recall loss to save time that is not being
spent. ``usearch`` is used above that threshold, where the tradeoff flips.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["Embedder", "DenseIndex", "resolve_backend", "BRUTE_FORCE_LIMIT"]

BRUTE_FORCE_LIMIT = 200_000

# Apache-2.0, strong general retrieval, widely used. Chosen as the default over
# jina-code-embeddings because several Jina checkpoints ship CC-BY-NC, which is
# a licensing trap for anyone using this at work.
DEFAULT_ST_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_STATIC_MODEL = "minishlab/potion-base-8M"

CACHE_DIR = os.path.expanduser("~/.cache/dwago/models")


@dataclass
class BackendInfo:
    kind: str          # "sentence-transformers" | "model2vec" | "none"
    model: str
    device: str
    dim: int


def _pick_device() -> str:
    """Prefer an accelerator; a CPU-only run is the slow path worth warning about."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def resolve_backend(prefer: str = "auto", model: str | None = None) -> BackendInfo:
    """Decide which embedding backend to use without loading it."""
    if prefer in ("none", "off"):
        return BackendInfo("none", "", "cpu", 0)

    if prefer in ("auto", "st", "sentence-transformers"):
        try:
            import sentence_transformers  # noqa: F401, PLC0415

            return BackendInfo("sentence-transformers", model or DEFAULT_ST_MODEL,
                               _pick_device(), 0)
        except ImportError:
            if prefer != "auto":
                log.warning("sentence-transformers not installed; falling back")

    if prefer in ("auto", "fast", "model2vec", "static"):
        try:
            import model2vec  # noqa: F401, PLC0415

            return BackendInfo("model2vec", model or DEFAULT_STATIC_MODEL, "cpu", 0)
        except ImportError:
            if prefer not in ("auto",):
                log.warning("model2vec not installed; dense retrieval disabled")

    return BackendInfo("none", "", "cpu", 0)


class Embedder:
    """Lazily-loaded encoder. Load and release explicitly.

    The encoder and the cross-encoder reranker are never held at once: together
    they do not fit comfortably on an 8-16GB machine, and the build only needs
    one of them at a time.
    """

    def __init__(self, info: BackendInfo):
        self.info = info
        self._model = None

    def load(self) -> "Embedder":
        if self._model is not None or self.info.kind == "none":
            return self
        t0 = time.time()
        if self.info.kind == "sentence-transformers":
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(
                self.info.model, device=self.info.device, cache_folder=CACHE_DIR
            )
            self.info.dim = self._model.get_sentence_embedding_dimension()
        elif self.info.kind == "model2vec":
            from model2vec import StaticModel  # noqa: PLC0415

            self._model = StaticModel.from_pretrained(self.info.model)
            self.info.dim = int(self._model.dim)
        log.info("loaded %s (%s, dim=%d) in %.1fs",
                 self.info.model, self.info.device, self.info.dim, time.time() - t0)
        return self

    def release(self) -> None:
        self._model = None
        try:
            import gc

            import torch  # noqa: PLC0415

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self.info.kind != "none"

    def encode(self, texts: list[str], batch_size: int = 64,
               progress: bool = False) -> np.ndarray:
        """Encode to L2-normalized float32. Normalizing here makes cosine a dot product."""
        if not self.available or not texts:
            return np.zeros((len(texts), 0), dtype=np.float32)
        self.load()

        if self.info.kind == "sentence-transformers":
            vecs = self._model.encode(
                texts, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=progress,
            )
            return vecs.astype(np.float32)

        vecs = np.asarray(self._model.encode(texts), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        np.divide(vecs, norms, out=vecs, where=norms > 0)
        return vecs

    def estimate_seconds(self, n_docs: int) -> float:
        """Rough wall-clock estimate, used to warn before a long build.

        Deliberately conservative. The point is to prevent someone starting an
        overnight job believing it takes a minute, not to be accurate.
        """
        if self.info.kind == "model2vec":
            return n_docs / 20_000.0
        rate = {"cuda": 400.0, "mps": 120.0, "cpu": 18.0}.get(self.info.device, 18.0)
        return n_docs / rate


class DenseIndex:
    """Vectors plus the key->row map, with incremental updates."""

    def __init__(self, vectors: np.ndarray | None, keys: dict[str, int],
                 node_of_key: dict[str, int] | None = None):
        self.vectors = vectors
        self.keys = keys
        self.node_of_key = node_of_key or {}
        self._rows_to_node: np.ndarray | None = None

    @property
    def available(self) -> bool:
        return self.vectors is not None and len(self.keys) > 0

    @classmethod
    def load(cls, store) -> "DenseIndex":
        vecs = store.load_vectors()
        keys = store.load_vector_keys()
        node_of_key = {
            r["node_key"]: r["idx"]
            for r in store.conn.execute("SELECT node_key, idx FROM nodes")
        }
        return cls(vecs, keys, node_of_key)

    def _row_to_node(self) -> np.ndarray:
        if self._rows_to_node is None:
            arr = np.full(len(self.keys), -1, dtype=np.int32)
            for key, row in self.keys.items():
                node = self.node_of_key.get(key)
                if node is not None and 0 <= row < len(arr):
                    arr[row] = node
            self._rows_to_node = arr
        return self._rows_to_node

    def search(self, qvec: np.ndarray, k: int = 100) -> list[tuple[int, float]]:
        """Cosine search. Vectors are pre-normalized, so this is one dot product."""
        if not self.available or qvec.size == 0:
            return []
        mat = self.vectors
        if mat.shape[1] != qvec.shape[-1]:
            log.warning("query dim %d != index dim %d; dense search skipped",
                        qvec.shape[-1], mat.shape[1])
            return []

        # float16 storage, float32 math: accumulating thousands of terms in
        # half precision loses real accuracy, and the upcast is cheap.
        sims = (mat.astype(np.float32) @ qvec.astype(np.float32).ravel())
        r2n = self._row_to_node()
        k = min(k, sims.shape[0])
        if k <= 0:
            return []
        part = np.argpartition(-sims, k - 1)[:k]
        order = part[np.argsort(-sims[part])]

        out: list[tuple[int, float]] = []
        for row in order:
            node = int(r2n[row])
            if node >= 0:
                out.append((node, float(sims[row])))
        return out


def build_dense_index(store, embedder: Embedder, *, force: bool = False,
                      progress: bool = True) -> dict:
    """Embed every node document, reusing vectors for nodes that did not change.

    This is the incremental path that makes a refresh survivable: only nodes
    whose content hash moved are re-encoded, and the rest are copied forward
    from the inherited epoch by row.
    """
    if not embedder.available:
        return {"embedded": 0, "reused": 0, "skipped": "no backend"}

    rows = list(store.conn.execute(
        "SELECT node_key, doc, content_hash FROM nodes "
        "WHERE doc IS NOT NULL AND doc != '' ORDER BY idx"
    ))
    if not rows:
        return {"embedded": 0, "reused": 0}

    new_hashes = {r["node_key"]: r["content_hash"] for r in rows}
    if force:
        dirty, reusable = list(new_hashes), []
    else:
        dirty, reusable = store.dirty_keys(new_hashes)

    old_vecs = store.load_vectors()
    old_keys = store.load_vector_keys()

    doc_of = {r["node_key"]: r["doc"] for r in rows}
    order = [r["node_key"] for r in rows]

    est = embedder.estimate_seconds(len(dirty))
    if dirty and est > 60:
        log.warning("embedding %d documents with %s on %s — roughly %.0f min",
                    len(dirty), embedder.info.model, embedder.info.device, est / 60)

    t0 = time.time()
    new_vecs: dict[str, np.ndarray] = {}
    if dirty:
        embedder.load()
        texts = [doc_of[k] for k in dirty]
        mat = embedder.encode(texts, progress=progress)
        for key, v in zip(dirty, mat):
            new_vecs[key] = v

    dim = embedder.info.dim or (old_vecs.shape[1] if old_vecs is not None else 0)
    if dim == 0:
        return {"embedded": 0, "reused": 0, "skipped": "unknown dim"}

    out = np.zeros((len(order), dim), dtype=np.float16)
    reused = 0
    for i, key in enumerate(order):
        if key in new_vecs:
            out[i] = new_vecs[key].astype(np.float16)
        elif old_vecs is not None and key in old_keys and old_keys[key] < len(old_vecs):
            out[i] = old_vecs[old_keys[key]]
            reused += 1

    store.write_vectors(order, out)
    store.set_meta("embedding_model", embedder.info.model)
    store.set_meta("embedding_dim", dim)
    store.set_meta("embedding_backend", embedder.info.kind)
    store.commit()

    return {
        "embedded": len(dirty),
        "reused": reused,
        "dim": dim,
        "seconds": round(time.time() - t0, 1),
        "model": embedder.info.model,
        "device": embedder.info.device,
    }
