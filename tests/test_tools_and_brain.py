"""The graph tools (path/cycles/diff_impact/tests_for), summaries and the
brain payload, exercised against a small synthetic store."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dwago.store import Store
from dwago.summarize import summarize_communities, get_summaries
from dwago.viz.build_brain import build_payload, write_brain


@pytest.fixture()
def toy_store(tmp_path: Path) -> Store:
    with Store.begin(tmp_path, inherit=False) as st:
        rows = []
        files = ["src/a.ts", "src/b.ts", "src/c.ts", "test/a.test.ts"]
        for i, f in enumerate(files):
            rows.append(dict(idx=i, node_key=f"k{i}", graphify_id=str(i),
                             label=f.split("/")[-1],
                             kind="file", file_type="code", source_file=f,
                             start_line=1, end_line=10, signature=None,
                             docstring=None, community=i % 2,
                             community_name=f"comm{i % 2}", content_hash=f"h{i}",
                             doc=f"doc {f}"))
        st.write_nodes(rows)
        st.write_edges([
            dict(src=0, dst=1, relation="imports", confidence="high",
                 confidence_score=1.0, weight=1.0, source_file=None,
                 source_location=None),
            dict(src=1, dst=2, relation="imports", confidence="high",
                 confidence_score=1.0, weight=1.0, source_file=None,
                 source_location=None),
            dict(src=2, dst=0, relation="imports", confidence="high",
                 confidence_score=1.0, weight=1.0, source_file=None,
                 source_location=None),
            dict(src=3, dst=0, relation="imports", confidence="high",
                 confidence_score=1.0, weight=1.0, source_file=None,
                 source_location=None),
        ], channel="structural")
        st.build_csr("structural", len(rows))
    return Store.open(tmp_path)


def test_brain_payload(toy_store):
    p = build_payload(toy_store)
    assert len(p["files"]) == 4
    assert len(p["regions"]) == 3          # 2 communities + everything-else
    assert p["edges"], "file edges collapsed"
    assert all(0 <= f["r"] < len(p["regions"]) for f in p["files"])


def test_brain_html(toy_store, tmp_path):
    out = tmp_path / "brain.html"
    n = write_brain(toy_store, out, title="toy")
    assert n == 4
    html = out.read_text()
    assert "window.DWAGO_DATA=" in html
    payload = html.split("window.DWAGO_DATA=")[1].split(";\n")[0].split(";const")[0]
    data = json.loads(payload.rstrip(";"))
    assert data["files"][0]["p"].endswith(".ts")


def test_summaries_cached(toy_store):
    calls = []

    def stub(prompt):
        calls.append(prompt)
        return "Does X. Entry point is a.ts."

    r1 = summarize_communities(toy_store, top=2, caller=stub)
    assert r1["written"] == 2 and not r1["errors"]
    r2 = summarize_communities(toy_store, top=2, caller=stub)
    assert r2["cached"] == 2 and r2["written"] == 0
    assert len(calls) == 2, "cache prevented re-calls"
    assert len(get_summaries(toy_store)) == 2


@pytest.mark.parametrize("tool,kwargs,expect", [
    ("path", dict(from_symbol="a.ts", to_symbol="c.ts"), "hops"),
    ("cycles", dict(min_size=2), "cycle"),
    ("tests_for", dict(symbol="src/a.ts"), "a.test.ts"),
])
def test_graph_tools(toy_store, tmp_path, tool, kwargs, expect):
    mcp = pytest.importorskip("mcp")  # noqa: F841
    import asyncio
    from dwago.serve import build_server

    server = build_server(toy_store.out_dir.parent)

    async def run():
        r = await server.call_tool(tool, kwargs)
        content = getattr(r, "content", r)
        if isinstance(content, list):
            return "\n".join(getattr(c, "text", str(c)) for c in content)
        return str(content)

    text = asyncio.run(run())
    assert expect in text
