"""Store lifecycle (epochs, atomic publish) and graph.json ingestion."""
from __future__ import annotations

import json

import pytest

from dwago.ingest import GraphJsonError, ingest, load_graph_json
from dwago.store import Store


def _graph(nodes, links) -> dict:
    return {"directed": False, "multigraph": False, "graph": {},
            "nodes": nodes, "links": links}


def _write_graph(root, data):
    d = root / "graphify-out"
    d.mkdir(parents=True, exist_ok=True)
    (d / "graph.json").write_text(json.dumps(data))
    return d / "graph.json"


def test_publish_is_visible_and_resolves(tmp_path):
    """Regression: the symlink was written relative to the wrong base.

    `current -> 000001` resolved to out/000001 while the epoch lived at
    out/epochs/000001, so every read after a successful build failed.
    """
    _write_graph(tmp_path, _graph(
        [{"id": "a", "label": "a.py", "file_type": "code",
          "source_file": "a.py", "source_location": "L1"}], []))
    assert not Store.exists(tmp_path)
    with Store.begin(tmp_path) as st:
        ingest(tmp_path, st)
    assert Store.exists(tmp_path)
    st = Store.open(tmp_path)
    assert st.node_count() == 1
    assert st.paths.root.exists()


def test_failed_build_leaves_previous_epoch_intact(tmp_path):
    _write_graph(tmp_path, _graph(
        [{"id": "a", "label": "a.py", "file_type": "code",
          "source_file": "a.py", "source_location": "L1"}], []))
    with Store.begin(tmp_path) as st:
        ingest(tmp_path, st)
    first = Store.open(tmp_path).paths.root.name

    with pytest.raises(RuntimeError):
        with Store.begin(tmp_path) as st:
            raise RuntimeError("build blew up")

    assert Store.open(tmp_path).paths.root.name == first, \
        "a failed build must not change what `current` points at"


def test_missing_graph_json_explains_how_to_fix(tmp_path):
    with pytest.raises(GraphJsonError) as e:
        load_graph_json(tmp_path / "graphify-out" / "graph.json")
    assert "graphify" in str(e.value)


def test_wrong_shape_is_rejected_loudly(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"something": "else"}))
    with pytest.raises(GraphJsonError):
        load_graph_json(p)


def test_edges_accept_either_links_or_edges_key(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"nodes": [], "edges": []}))
    assert load_graph_json(p)["edges"] == []


def test_leading_dot_method_labels_match_spans(tmp_path):
    """Regression: graphify emits `.update()` for receiver-less methods.

    Leaving the dot on meant those labels never matched a parsed span, which
    cost 18 points of span coverage on a real repository.
    """
    (tmp_path / "m.py").write_text("class A:\n    def update(self):\n        return 1\n")
    _write_graph(tmp_path, _graph([
        {"id": "m", "label": "m.py", "file_type": "code",
         "source_file": "m.py", "source_location": "L1"},
        {"id": "m_update", "label": ".update()", "file_type": "code",
         "source_file": "m.py", "source_location": "L2"},
    ], []))
    with Store.begin(tmp_path) as st:
        res = ingest(tmp_path, st)
    st = Store.open(tmp_path)
    row = st.conn.execute(
        "SELECT signature, start_line, end_line FROM nodes WHERE label = '.update()'"
    ).fetchone()
    assert row["signature"], "leading-dot label should still resolve to its span"
    assert row["end_line"] >= row["start_line"]


def test_file_nodes_get_whole_file_ranges(tmp_path):
    (tmp_path / "m.py").write_text("a = 1\nb = 2\nc = 3\n")
    _write_graph(tmp_path, _graph([
        {"id": "m", "label": "m.py", "file_type": "code",
         "source_file": "m.py", "source_location": "L1"}], []))
    with Store.begin(tmp_path) as st:
        ingest(tmp_path, st)
    row = Store.open(tmp_path).conn.execute(
        "SELECT start_line, end_line, kind FROM nodes WHERE label='m.py'").fetchone()
    assert row["kind"] == "file"
    assert row["start_line"] == 1 and row["end_line"] == 3


def test_dangling_and_self_edges_are_counted_not_crashed(tmp_path):
    _write_graph(tmp_path, _graph(
        [{"id": "a", "label": "a.py", "file_type": "code",
          "source_file": "a.py", "source_location": "L1"}],
        [{"source": "a", "target": "ghost", "relation": "calls"},
         {"source": "a", "target": "a", "relation": "calls"}]))
    with Store.begin(tmp_path) as st:
        res = ingest(tmp_path, st)
    assert res.edges == 0
    assert res.dropped_edges == 2


def test_dirty_keys_reuses_unchanged_vectors(tmp_path):
    _write_graph(tmp_path, _graph(
        [{"id": "a", "label": "a.py", "file_type": "code",
          "source_file": "a.py", "source_location": "L1"}], []))
    with Store.begin(tmp_path) as st:
        ingest(tmp_path, st)
        st.set_meta("content_hashes", {"k1": "h1", "k2": "h2"})
        import numpy as np
        st.write_vectors(["k1", "k2"], np.zeros((2, 4), dtype=np.float16))
        dirty, reusable = st.dirty_keys({"k1": "h1", "k2": "CHANGED", "k3": "new"})
    assert set(dirty) == {"k2", "k3"}
    assert reusable == ["k1"]
