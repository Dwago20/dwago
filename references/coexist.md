# Running alongside graphify

graphify's skill description triggers on "any question about a codebase". So
does dwago's. With both installed, the agent has two tools for every job and
picks the weaker one roughly half the time — and neither skill will tell you
that is happening.

Decide explicitly.

## Recommended: dwago in front, graphify as the extractor

dwago depends on graphify and calls it. Keep the *package*, retire the
*skill*.

1. **Narrow or remove the graphify skill.** Either delete
   `~/.claude/skills/graphify/`, or edit its front-matter `description` so it
   only claims graph *construction*, not question answering — e.g. "Build a
   knowledge graph from a folder of code, docs, papers or video." Leave the
   answering to dwago.

2. **Replace a configured graphify MCP server** with dwago's. Both expose
   `graph_stats`; an agent seeing two will guess.

3. **Keep `graphify update` / `graphify extract`.** dwago needs
   `graphify-out/graph.json` and does not reimplement extraction. graphify's
   post-commit hook stays useful — chain `dwago refresh` onto it.

## The other way round

If you would rather keep graphify's skill as the front door, install dwago
without its skill and use it as a CLI. You lose automatic invocation but keep
`dwago ask`, `dwago map` and the MCP tools when you ask for them by name.

## Rollback

dwago writes only to `dwago-out/`. Deleting that directory and its skill
returns you to plain graphify with nothing else changed. Reinstate the graphify
skill's original `description` if you edited it.

## Version pinning

dwago reads `graphify-out/graph.json` and pins a compatible graphify range in
`pyproject.toml`. That format has changed across releases — direction carriers
appeared in 0.9.x, community labels are present in some versions and not others,
and a future multigraph mode will change link multiplicity. Ingest validates the
shape and fails with a clear message rather than producing a half-empty index.
