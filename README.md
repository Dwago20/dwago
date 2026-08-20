# dwago

Ask your codebase anything. dwago builds a knowledge graph of your repository
— every file, symbol and import, parsed with tree-sitter — layers your git
history on top of it, and answers through whichever surface fits: a CLI, an
MCP server any coding agent can call, an agent skill, or a living-brain map
you can fly through.

No API key, no cloud, no extraction service. `dwago build` is a local parse.

## What it answers

| question | command |
|---|---|
| where is X / how does Y work | `dwago ask "how do we prevent double charging?"` |
| everything an agent needs for a task, token-budgeted, cited | `dwago pack "migrate the session store" --budget 8000` |
| what breaks if I touch this | `dwago impact src/auth/session.ts` |
| what else always changes with this file | `dwago co-change src/db/store.ts` |
| where does change concentrate; who owns it | `dwago hotspots`, bus factor included |
| what did the last N commits ripple into | `diff_impact` (MCP) |
| which tests cover this | `tests_for` (MCP) |
| how are A and B connected; where are the cycles | `path`, `cycles` (MCP) |
| show me the whole thing | `dwago map` — the brain |

Under the hood: BM25 + dense embeddings fused with reciprocal-rank fusion, an
exact-symbol fast path, then Personalized PageRank over two graph channels —
structure (imports, containment) and history (statistically significant
co-change). Retrieval is *measured*, not promised: `dwago eval` mines your own
git history into a leak-free, time-split benchmark with paired bootstrap CIs.
Run it before trusting us.

## Install

```bash
uv tool install "dwago[lexical,fast,mcp] @ git+https://github.com/Dwago20/dwago"
# or, from a clone:  pip install -e ".[lexical,fast,mcp]"
# best retrieval quality (torch + a real encoder):  ".[lexical,dense,mcp]"
```

## Use

```bash
dwago build . --fast     # parse + index; minutes on a large repo
dwago ask "where is the OIDC issuer configured?"
dwago map                # writes dwago-out/brain.html — open it
dwago eval               # benchmark retrieval on your own history
```

`dwago refresh` after a commit re-embeds only what changed; readers never see
a torn index (atomic epoch swap).

## Hook it into your agent

**Any MCP client** (Cursor, Claude Code, Codex CLI, Cline, Windsurf, Zed, …):

```bash
dwago serve /path/to/repo        # stdio MCP server, 13 tools
```

e.g. Claude Code: `claude mcp add dwago -- dwago serve .` · Cursor: add the
same command under `mcpServers` in `.cursor/mcp.json`.

**As an agent skill** — `SKILL.md` is plain markdown instructions with a
frontmatter description; it works anywhere skills or rules files do:

```bash
# Claude Code
mkdir -p ~/.claude/skills/dwago && cp SKILL.md ~/.claude/skills/dwago/ && cp -r references ~/.claude/skills/dwago/
# Codex / Cursor / other agents: point your rules at SKILL.md, or paste it into
# AGENTS.md — it is instructions + a command table, nothing Claude-specific.
```

**Plain CLI** — no agent, no LLM: everything above works from a terminal.

Community summaries (`dwago summarize`) are the only feature that touches an
LLM, and the backend is pluggable: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, a
local `claude` CLI, or skip it entirely.

## The brain

`dwago map` renders your repository as a living brain: files are neurons
placed inside an anatomical cortex, the largest communities are the lobes
(named from your real directory structure), and dependencies are the axons.
Click a file and electric strikes fire along its real edges while everything
else dims; the mini-brain in the corner works like a CAD view-cube — click a
lobe to open it. Searchable file panel, key-file overlay from git hotspots,
x-ray views. One self-contained HTML file, fully offline.

`dwago map --galaxy` renders the symbol-level map instead.

## Languages

Python, TypeScript/TSX, JavaScript, Go, Rust, Java, C, C++, C#, Ruby, PHP —
plus Markdown/RST/text as document nodes. Import-edge resolution is exact for
Python and TS/JS; other languages connect through containment and co-change.
An existing node-link `graph.json` from another extractor can be ingested
with `dwago build --graph path/to/graph.json`.

## Status

Tested (67 tests): extraction, hybrid retrieval, PPR, temporal layer, eval
harness, brain + galaxy maps, MCP server, incremental refresh, cached
summaries. Open: SCIP precision edges, coverage-backed `tests_for` (currently
a labelled heuristic), symbol-level co-change, ANN above 200k nodes.

## License

Apache-2.0.
