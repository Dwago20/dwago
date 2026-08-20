# Retrieval

Plain lexical search matches queries by substring over node labels.
There is no stemming, no synonyms, no semantic match — which is why its own
skill ships a manual workaround instructing the model to read a vocabulary file
and hand-pick up to twelve tokens before searching. dwago removes that step.

## Documents

Retrieval quality is decided here, before any model runs. A bare label is a
terrible search target: one rare token to BM25, an out-of-vocabulary blob to an
encoder. Each node's document is:

```
identifier, verbatim            → exact-symbol queries hit hardest
identifier, split into words    → parseURLPath -> parse url path
kind                            → function | class | file | ...
path components, split          → src/auth/session.py -> auth session
signature                       → parameter names and types
docstring or leading comment    → usually the best natural-language match
first ~15 body lines            → only when there is no docstring
1-hop neighbour labels, capped  → a symbol is partly defined by what it touches
```

Identifier splitting is the highest-value transformation and handles the
acronym boundary correctly (`HTTPServer` → `http server`, not `h t t p server`).
CJK text is segmented with jieba when installed, matching common extractor
handling.

## Stages

```
query
 ├─ 0. exact / prefix symbol match        (a named symbol resolves immediately)
 ├─ 1. BM25 top-100  +  dense top-100  →  Reciprocal Rank Fusion  →  top-50
 │      (RRF, not score addition: BM25 scores and cosine similarities live on
 │       incomparable scales, and RRF needs no per-corpus calibration)
 ├─    optional cross-encoder rerank      (off by default — seconds on CPU)
 ├─ 2. seeds = top-15, kind prior applied
 └─ 3. Personalized PageRank from those seeds, two channels, blended
```

Final score is `0.6 × retrieval + 0.4 × diffusion`, each max-normalized. Giving
seeds an absolute bonus instead — the intuitive choice — silently makes the
diffusion a no-op whenever `seed_k` exceeds the result count.

## Two channels

Structural edges (calls, imports, inheritance) and temporal edges (co-change)
are diffused **separately** and combined afterwards, each row-normalized in its
own channel.

Mixing them in one walk lets co-change hubs — a settings file that moves with
everything — absorb mass belonging to the call graph. Separation also keeps the
signals *distinguishable*, which is what lets every result say whether it is
here because it is called or because it always changes with you.

`--temporal-weight` (default 0.3) sets the blend. It is a fitted parameter, not
a claim; `dwago eval` can tune it per repository.

## The test-file prior

Test names are literally descriptive sentences, so lexical search ranks
`test_exact_duplicates_merged` far above the implementation it exercises.
Someone asking how something works wants the implementation. Tests are
discounted (×0.55), not hidden — they do document behaviour. `--no-tests`
removes them.

The prior is applied **before seeds are chosen**, not only at final ranking.
Seeds decide where diffusion starts, so a test-heavy seed set walks outward
through the test suite and never reaches the implementation.

## Duplicate suppression

At most two results may share a label. Real repositories vendor the same file
into many places — some tools copy one reference doc into fourteen per-platform
skill directories — and without this a single document occupies most of the
result list.

## Tuning

| Flag | Effect |
|---|---|
| `--no-ppr` | retrieval only, no diffusion — faster, more literal |
| `--rerank` | cross-encoder over the top 50; seconds on CPU |
| `--temporal-weight 0` | ignore history entirely |
| `--no-tests` | exclude test files |
| `-k` | results returned |
