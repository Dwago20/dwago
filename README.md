# dwago

Semantic retrieval, git-history intelligence and a living-brain visualization,
layered over [graphify](https://github.com/Graphify-Labs/graphify) knowledge
graphs.

graphify extracts the graph — 37 tree-sitter language extractors, Leiden
communities, multi-backend LLM extraction. That part is mature and dwago does
not reimplement it. dwago adds everything *behind* the graph:

| | graphify | dwago |
|---|---|---|
| Retrieval | case-folded substring + IDF | BM25 + embeddings fused by RRF, exact-symbol fast path, then Personalized PageRank |
| Traversal | uniform BFS/DFS | relevance-weighted diffusion over structural + temporal channels |
| Git history | — | co-change coupling (G-test), hotspots, ownership, bus factor |
| Visualization | vis-network 2D, CDN-loaded, 5,000-node cap | the codebase as a living brain: files are neurons inside an anatomical cortex, communities are lobes, dependencies fire as electric strikes; offline, single file |
| Agent surface | 10 MCP tools over lexical search | 13 tools over semantic search + history (`context_pack`, `diff_impact`, `path`, `cycles`, `tests_for`, …) |
| Identity | node ids change across updates | content-addressed keys, atomic epoch swaps, dirty-set re-embedding |

Measured, not promised — leak-free PR-mined eval (time-split, paired bootstrap):

- repo 1 (graphify itself): file R@5 0.583 → **0.715** end-to-end, significant
- repo 2 (a private TS/IaC monorepo): R@20 +3.7pt significant; R@5 regressed
  with the `--fast` embedder — recorded in `references/eval.md`, full-size
  encoder evaluation still open. Run `dwago eval` on your own repo; the
  harness is the point.

## Install

Prerequisite: [graphify](https://github.com/Graphify-Labs/graphify)
(`uv tool install graphifyy`).

```bash
# CLI — from GitHub
uv tool install "dwago[lexical,fast,mcp] @ git+https://github.com/Dwago20/dwago"

# or from a clone, editable, with the full dense tier
pip install -e ".[dense,lexical,mcp]"
```

### The Claude Code skill (`/dwago`)

```bash
git clone https://github.com/Dwago20/dwago
mkdir -p ~/.claude/skills/dwago
cp dwago/SKILL.md ~/.claude/skills/dwago/
cp -r dwago/references ~/.claude/skills/dwago/
```

If the graphify skill is also installed, read `references/coexist.md` —
two skills answering "any question about a codebase" will collide.

### The MCP server

```bash
claude mcp add dwago -- dwago serve /path/to/repo
```

## Use

```bash
graphify update .                    # substrate (code only, no API key)
dwago build . --fast                 # minutes; drop --fast for the full encoder
dwago ask "how do we prevent double charging?"
dwago pack "migrate the session store" --budget 8000
dwago impact src/auth/session.ts
dwago co-change src/db/store.ts
dwago hotspots -n 20
dwago map                            # the brain (brain.html); --galaxy for symbols
dwago summarize                      # community summaries (needs claude CLI or ANTHROPIC_API_KEY)
dwago eval                           # measure it on your repo before trusting it
```

`dwago refresh` chains onto graphify's post-commit hook: only dirty nodes are
re-embedded, and readers never see a torn index.

## Status

Working and tested (65 tests): hybrid retrieval, PPR, temporal layer, eval
harness, brain + galaxy maps, MCP server, incremental refresh, summaries
(cached; needs an LLM backend configured).

Open: SCIP precision edges, coverage-backed `tests_for` (currently a labelled
heuristic), symbol-level co-change, usearch ANN above 200k nodes, full-size
dense-tier evaluation.

## License

Apache-2.0.
