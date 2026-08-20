"""Document construction, PPR, fusion and context packing."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from dwago.index.docs import node_text, split_identifier
from dwago.index.lexical import tokenize
from dwago.retrieve.pack import ContextPack, PackItem, estimate_tokens
from dwago.retrieve.ppr import combine_channels, ppr


class TestIdentifierSplitting:
    def test_camel_case(self):
        assert split_identifier("parseURLPath") == ["parse", "url", "path"]

    def test_acronym_boundary(self):
        assert split_identifier("HTTPServer") == ["http", "server"]

    def test_snake_and_screaming(self):
        assert split_identifier("MAX_RETRY_COUNT") == ["max", "retry", "count"]

    def test_leading_underscore(self):
        assert split_identifier("_pick_seeds") == ["pick", "seeds"]

    def test_stopwords_dropped_but_available(self):
        assert "get" not in split_identifier("get_user")
        assert "get" in split_identifier("get_user", keep_stop=True)

    def test_empty(self):
        assert split_identifier("") == []


def test_tokenize_emits_both_whole_and_split_forms():
    toks = tokenize("parseURLPath")
    assert "parseurlpath" in toks           # exact identifier still matches
    assert {"parse", "url", "path"} <= set(toks)


def test_document_includes_signature_docstring_and_neighbours():
    doc = node_text(
        {"label": "handleRetry()", "kind": "function", "source_file": "src/net/http.py",
         "signature": "def handleRetry(attempt: int)", "docstring": "Retries a request."},
        neighbours=["sendRequest()", "BackoffPolicy"],
    )
    assert "handleRetry" in doc
    assert "handle retry" in doc            # split form
    assert "net" in doc and "http" in doc   # path terms
    assert "Retries a request." in doc
    assert "backoff" in doc                 # neighbour context


def test_document_prefers_docstring_over_body():
    d = node_text({"label": "f", "docstring": "Does a thing.", "_body": "x = 1"})
    assert "Does a thing." in d
    assert "x = 1" not in d


class TestPPR:
    def _chain(self, n=5):
        """0 - 1 - 2 - 3 - 4"""
        rows = list(range(n - 1)) + list(range(1, n))
        cols = list(range(1, n)) + list(range(n - 1))
        return sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))

    def test_mass_decays_with_distance_from_the_seed(self):
        """Scores fall off with graph distance — the property PPR guarantees.

        Note it does NOT guarantee the seed itself ranks first: on an
        undirected chain the seed has degree 1 and its neighbour degree 2, so
        the neighbour can legitimately hold more mass. That is why the
        retrieval pipeline blends diffusion mass with the direct retrieval
        score instead of ranking on PPR alone.
        """
        m = self._chain()
        r = ppr(m, {0: 1.0})
        assert r.scores[1] > r.scores[2] > r.scores[3] > r.scores[4]
        assert r.scores[0] > r.scores[3], "the seed must still outrank distant nodes"

    def test_scores_form_a_distribution(self):
        r = ppr(self._chain(), {0: 1.0})
        assert r.scores.sum() == pytest.approx(1.0, abs=1e-3)

    def test_empty_seeds_return_zeros(self):
        r = ppr(self._chain(), {})
        assert r.scores.sum() == 0.0

    def test_top_excludes_requested_nodes(self):
        r = ppr(self._chain(), {0: 1.0})
        assert 0 not in [i for i, _ in r.top(3, exclude={0})]

    def test_reverse_orientation_changes_direction(self):
        m = sparse.csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]],
                                       dtype=np.float32))
        fwd = ppr(m, {0: 1.0}).scores
        rev = ppr(m, {0: 1.0}, reverse=True).scores
        assert fwd[2] > rev[2], "forward should reach downstream; reverse should not"

    def test_disconnected_graph_does_not_leak_mass(self):
        m = sparse.csr_matrix((4, 4), dtype=np.float32)   # no edges at all
        r = ppr(m, {1: 1.0})
        assert r.scores[1] == pytest.approx(1.0, abs=1e-3)


class TestChannelCombination:
    def test_normalizes_before_mixing(self):
        """A denser channel must not win purely by having larger raw masses."""
        s = np.array([0.01, 0.02], dtype=np.float32)      # small scale
        t = np.array([100.0, 1.0], dtype=np.float32)      # large scale
        out = combine_channels(s, t, temporal_weight=0.5)
        assert out[0] == pytest.approx(0.5 * 0.5 + 0.5 * 1.0)

    def test_weight_zero_ignores_temporal(self):
        s = np.array([1.0, 0.0], dtype=np.float32)
        t = np.array([0.0, 1.0], dtype=np.float32)
        assert combine_channels(s, t, temporal_weight=0.0)[1] == 0.0

    def test_single_channel_passes_through(self):
        s = np.array([1.0, 2.0], dtype=np.float32)
        assert np.array_equal(combine_channels(s, None), s)

    def test_no_channels_raises(self):
        with pytest.raises(ValueError):
            combine_channels(None, None)


class TestPacking:
    def test_token_estimate_scales_with_length(self):
        assert estimate_tokens("x" * 320) > estimate_tokens("x" * 32)

    def test_render_reports_omissions(self):
        p = ContextPack(query="q", budget=100, used=90, truncated=7, communities=2)
        p.items.append(PackItem("a", "f.py:1", "function", "matched", "body", 10, 1))
        out = p.render()
        assert "7 further results omitted" in out
        assert "f.py:1" in out

    def test_empty_pack_is_explicit(self):
        p = ContextPack(query="q")
        p.warnings.append("retrieval returned no results")
        assert "No context found" in p.render()
