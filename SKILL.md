---
name: dwago
description: "Use for any question about a codebase — how something works, where a thing lives, what a change will break, who owns a file, what else must change with it. Semantic search plus git-history intelligence over a code graph dwago builds itself. When dwago-out/ exists, treat the question as a dwago query first."
---

# dwago

Semantic retrieval, change-impact analysis and a living-brain visualization
over a code knowledge graph dwago builds itself: tree-sitter parsing of every
file and symbol, import resolution, community detection, and a git-history
layer no parser can produce.

## Decide what to do

**An index already exists** (`dwago-out/` is present) → answer from it. Do not
rebuild. Do not read files one by one to answer a question the index covers.

| The user wants | Run |
|---|---|
| to understand or locate something | `dwago ask "<question>"` |
| code to work from, in one paste | `dwago pack "<task>" --budget 8000` |
| to know what a change breaks | `dwago impact "<symbol>"` |
| what else changes with a file | `dwago co-change <path>` |
| where the risk is | `dwago hotspots` |
| to see the graph | `dwago map` — the brain; `--galaxy` for the symbol map |
| to check the index | `dwago stats` |

**No index yet** → `dwago build .`

**Code changed since the build** → `dwago refresh` (re-embeds only what moved).

## Building

```bash
dwago build .            # parse + index the repository (local, no API key)
dwago build . --fast     # static embeddings, no torch — minutes, not hours
```

Extraction is built in: tree-sitter over Python, TS/TSX, JS, Go, Rust, Java,
C, C++, C#, Ruby and PHP, plus Markdown as document nodes. An existing
node-link `graph.json` from another tool can be ingested with `--graph`.

On a large repository with the default encoder, `build` prints a time estimate
and stops rather than silently starting an hours-long job. Pass `--yes` to
proceed or `--fast` to use the pure-numpy encoder.

## Answering with it

Quote `file:line` from the results — every hit carries one. Say which subsystem
a result came from when it matters.

Results are labelled with *why* they were retrieved:

- `matched query` — the text matched directly
- `connected to matches` — reached through calls/imports from something that did
- `co-changes with matches` — **history** says it moves with the match, though
  nothing in the code links them. This is the finding no parser can produce; call
  it out explicitly rather than folding it in with the rest.

If retrieval returns nothing, say so. Do not fall back to guessing — an empty
result means the corpus does not cover the question, and that is information.

## Honesty rules

- Never claim a symbol exists because it seems like it should. Cite or say no.
- `--fast` retrieval is measurably weaker than the full encoder. If results look
  thin and the index was built with `--fast`, say that before blaming the graph.
- Co-change is correlation. Report lift and p-value when it drives a
  recommendation; do not present it as a call graph.
- The index reflects the commit it was built from. If the working tree has moved
  on, `dwago refresh` first rather than answering from a stale graph.

## Reference

Load only what the task needs.

- `references/build.md` — pipeline stages, embedding tiers, incremental refresh
- `references/query.md` — retrieval stages, ranking, tuning
- `references/temporal.md` — how co-change, hotspots and ownership are computed
- `references/viz.md` — visualization modes and limits
- `references/mcp.md` — MCP tools for agents
- `references/eval.md` — the benchmark, and what it measured here
