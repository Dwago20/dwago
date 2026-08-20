---
name: dwago
description: "Use for any question about a codebase — how something works, where a thing lives, what a change will break, who owns a file, what else must change with it. Semantic search plus git-history intelligence over a graphify knowledge graph. When dwago-out/ exists, treat the question as a dwago query first."
---

# dwago

Semantic retrieval, change-impact analysis and GPU visualization over a
[graphify](https://github.com/Graphify-Labs/graphify) knowledge graph.

graphify extracts the graph; dwago makes it answerable. It adds embeddings and
graph diffusion (graphify's search is substring matching), a git-history layer
(graphify has none), and a visualization that does not stop at 5,000 nodes.

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
dwago build .            # full: needs graphify-out/graph.json to exist
dwago build . --fast     # static embeddings, no torch — minutes, not hours
```

`build` requires graphify's graph first. If `graphify-out/graph.json` is missing,
create it:

```bash
graphify update .          # code only, no LLM, no API key
graphify extract .         # includes docs/PDFs, needs an LLM backend
```

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
- `references/coexist.md` — running alongside graphify's own skill
