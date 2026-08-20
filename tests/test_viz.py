"""The viz builder: layout sanity and the self-contained output contract."""
from __future__ import annotations

import numpy as np

import math

from dwago.viz.build_html import (DISC_R, _galaxy_layout,
                                   _merge_communities, _star_colors)


def _toy():
    # two well-connected communities + one isolated micro-community
    comms = np.array([0]*6 + [1]*6 + [2]*3, dtype=np.int32)
    src = np.array([0,1,2, 6,7,8, 0, 13], dtype=np.int32)
    dst = np.array([1,2,3, 7,8,9, 6, 14], dtype=np.int32)
    return comms, src, dst


def test_merge_keeps_connected_and_buckets_islands():
    comms, src, dst = _toy()
    merged, misc = _merge_communities(comms, src, dst, top_k=2)
    # the two big communities survive as distinct groups
    assert merged[0] != merged[6]
    # the island (nodes 12..14, community 2, no external edges) is the misc group
    assert merged[12] == misc
    assert misc == merged.max()


def test_merge_multipass_resolves_chains():
    # micro A links only to micro B; micro B links to kept community 0.
    # single-pass would dump A into misc; multi-pass must pull it through B.
    comms = np.array([0]*8 + [1]*8 + [5]*2 + [6]*2, dtype=np.int32)
    src = np.array([0,1,2, 8,9, 16,   18], dtype=np.int32)   # A(18,19)->B(16,17)->kept
    dst = np.array([1,2,3, 9,10, 0,   16], dtype=np.int32)
    merged, misc = _merge_communities(comms, src, dst, top_k=2)
    assert merged[18] == merged[16] == merged[0], "chain must resolve to the kept group"
    assert merged[18] != misc


def test_galaxy_layout_core_arms_and_halo():
    comms, src, dst = _toy()
    merged, misc = _merge_communities(comms, src, dst, top_k=2)
    rng = np.random.RandomState(0)
    deg = np.ones(len(comms), dtype=np.float32)
    pos = _galaxy_layout(merged, deg, rng, misc)
    assert pos.shape == (len(comms), 3)
    # biggest community forms the core bulge near the origin
    core = pos[merged == merged[0]]
    assert np.linalg.norm(core.mean(axis=0)) < 80
    # second community sits out on an arm, away from the core
    arm = pos[merged == merged[6]]
    assert np.linalg.norm(arm.mean(axis=0)[[0, 2]]) > 90
    # halo (misc) stars orbit beyond the disc material
    halo = pos[merged == misc]
    assert np.linalg.norm(halo, axis=1).min() > 250
    # disc is thin: in-plane extent dwarfs vertical extent
    disc = pos[merged != misc]
    assert np.abs(disc[:, 1]).max() < np.hypot(disc[:, 0], disc[:, 2]).max()


def test_star_colors_run_warm_core_to_blue_arms():
    comms, src, dst = _toy()
    merged, misc = _merge_communities(comms, src, dst, top_k=2)
    rng = np.random.RandomState(0)
    pos = np.zeros((len(comms), 3), dtype=np.float32)
    pos[0] = (10, 0, 10)                       # core star
    pos[6] = (DISC_R * 0.95, 0, 0)             # far arm star
    cols = _star_colors(pos, merged, misc, np.random.RandomState(1))
    r_core, b_core = int(cols[0][0]), int(cols[0][2])
    r_arm, b_arm = int(cols[6][0]), int(cols[6][2])
    assert r_core > b_core, "core is warm (red over blue)"
    assert b_arm > r_arm, "outer arm is blue-white"


def test_write_html_is_self_contained(tmp_path):
    import json

    from dwago.ingest import ingest
    from dwago.store import Store

    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    d = tmp_path / "graphify-out"; d.mkdir()
    (d / "graph.json").write_text(json.dumps({
        "nodes": [
            {"id": "m", "label": "m.py", "file_type": "code",
             "source_file": "m.py", "source_location": "L1"},
            {"id": "m_f", "label": "f()", "file_type": "code",
             "source_file": "m.py", "source_location": "L1"},
        ],
        "links": [{"source": "m", "target": "m_f", "relation": "contains",
                   "confidence": "EXTRACTED"}]}))
    with Store.begin(tmp_path) as st:
        ingest(tmp_path, st)

    from dwago.viz.build_html import write_html
    out = tmp_path / "map.html"
    n = write_html(Store.open(tmp_path), out)
    html = out.read_text()
    assert n == 2
    assert "http://" not in html.replace("http://www.w3.org", "")   # no external refs
    assert "https://" not in html
    assert "dwago" in html
    assert "uYaw" in html, "the WebGL scene must be inlined"
