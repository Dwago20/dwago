# The git-history layer

No parser can see time. This is the part of dwago with no static
equivalent, and it answers a question static analysis structurally cannot: *what
else will I have to change?*

Two files with no import between them that changed together in 40 of the last 50
commits are coupled. No parser will ever see it.

## Co-change

One `git log --no-merges --numstat` pass, cached and extended by commit range
rather than re-mined from scratch.

**Commits are weighted by inverse size, not filtered by it.** The textbook move
is to discard commits touching more than N files. That also discards the single
strongest evidence in the repository — an API change plus every call site it
forced — and in a squash-merge repo it deletes most of the history, since a
whole feature arrives as one commit. Instead each commit contributes `1/C(n,2)`
to every pair it implies, so its total contribution is exactly 1.0 however many
files it touched. Large commits still count, in proportion to how much they
actually say.

**Significance is a G-test, not lift.** Lift is unstable at low support: two
files touched twice, together both times, score infinite lift on no evidence.
The G-test asks whether co-occurrence exceeds chance *given* how often each file
changes independently, which discounts both the rare pair and the
everybody-touches-it file. Only positive association counts — co-occurring less
than chance is not evidence of coupling.

Lockfiles, changelogs, minified bundles and generated code are excluded; they
move with everything and would dominate the table.

### Thresholds

Defaults were calibrated on a 1,419-commit OSS repository (920 files,
53,021 candidate pairs). `min_support` is the binding filter by a wide margin:

| min_support | pairs kept (p ≤ 0.01, ≥3 co-occurrences) |
|---|---|
| 1.0 | 88 |
| 0.5 | 123 |
| **0.3** (default) | **176** |
| 0.1 | 264 |

The significance gate is what guarantees quality; support just sets depth.

## In the graph

Co-change is projected onto **file nodes**, one edge per significant pair, and
each symbol links to its own file node.

The obvious alternative — link every symbol in file A to every symbol in file B
— is quadratic and produced 518,793 edges from 88 pairs on this repository,
because a single pair of large files contributes |A|×|B| of them. That density
buries the signal it came from. Keeping coupling at the granularity it was
measured at gives the same reachability in three hops at O(nodes + pairs) edges.

## Hotspots, ownership, age

- **Hotspot** = time-decayed churn (90-day half-life) × a size-based complexity
  proxy. A file edited constantly two years ago is not a hotspot today.
- **Bus factor** = the smallest set of authors accounting for half the commits
  touching a file. A value of 1 is flagged everywhere it appears.
- **Age** = first and last touch per file, which drives the visualization's time
  scrubber.

Per-symbol dating and symbol-level co-change need a parse of every historical
revision; both are gated behind a flag and a history window rather than being on
by default. Intersecting historical diff hunks with *today's* line ranges — the
cheap version — is wrong beyond a few hundred commits because lines drift.
