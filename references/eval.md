# The benchmark

Any claim that this retrieves better than graphify is worthless unless it is a
measured number on your own repository — including when the answer is "it
doesn't". `dwago eval` produces that number.

```bash
dwago eval .              # ladder over 200 held-out changes
dwago eval . -n 500 --split 0.7
```

## Ground truth

For each historical change, the query is what the author wrote about the work
(commit subject and body) and the correct answer is the set of files they
actually touched. Real task, real label, available in any repository with no
annotation effort.

## Leakage control

Co-change is mined from commit history. If evaluation commits sit inside that
history, the temporal channel has memorized the answer key. So history is split
by time: temporal edges come only from commits **before** a cutoff, and only
commits **after** it are scored. The structural graph is still built from HEAD,
which is correct — you query the code as it exists now.

`--no-rebuild` skips the leak-free rebuild and is labelled optimistic for that
reason.

## Known biases

Stated because they change how the numbers should be read.

- Commit subjects are written *after* the work, in the vocabulary of the
  implementation. That flatters lexical retrieval relative to how it performs on
  the questions people actually ask mid-task, so **improvements over BM25 here
  are conservative**.
- Files renamed or deleted since the cutoff are dropped rather than counted as
  misses; that is history moving, not retrieval failing.
- Lockfiles and generated artifacts are excluded — they co-occur with everything.
- Version bumps, merges and pure-formatting commits are skipped; retrieval
  cannot localize "bump to 1.2.3".

## Reading the output

Rungs are compared with a **paired** bootstrap: both configurations answer the
same queries, and variance between queries dwarfs the difference between
systems, so an unpaired test would drown a real effect.

Measured on graphify's own repository — 150 held-out changes, leak-free split,
`--fast` (potion-base-8M) embeddings:

| rung | R@1 | R@5 | R@10 | R@20 | MRR | s/query |
|---|---|---|---|---|---|---|
| bm25 | 0.275 | 0.583 | 0.789 | 0.827 | 0.689 | 0.000 |
| dense | 0.256 | 0.501 | 0.644 | 0.790 | 0.624 | 0.004 |
| hybrid | 0.289 | 0.697 | 0.812 | 0.834 | 0.698 | 0.003 |
| hybrid+ppr | 0.283 | **0.715** | 0.825 | **0.849** | 0.691 | 0.014 |

**Pairwise, on R@20:**

- `bm25 → dense` −0.036 [−0.072, +0.000] — not significant
- `dense → hybrid` +0.043 [+0.015, +0.071] — significant
- `hybrid → hybrid+ppr` +0.016 [+0.003, +0.030] — significant

**End to end, bm25 → hybrid+ppr, tested at every k:**

| | baseline | full stack | gain | 95% CI | |
|---|---|---|---|---|---|
| R@1 | 0.275 | 0.283 | +0.009 | [−0.029, +0.051] | not significant |
| R@5 | 0.583 | **0.715** | **+0.132** | [+0.084, +0.182] | **significant** |
| R@10 | 0.789 | 0.825 | +0.036 | [+0.009, +0.064] | significant |
| R@20 | 0.827 | 0.849 | +0.023 | [−0.000, +0.048] | not significant |

Read this carefully rather than quoting one number.

**The gain is real and it is concentrated at R@5.** That is the regime that
matters: a context pack shows a handful of results, not twenty. Getting the
right file into the top five 71.5% of the time instead of 58.3% is the
difference between the agent reading the right code first and reading around it.

**At R@20 the improvement does not clear its confidence interval**, because BM25
is already at 0.827 and there is almost no headroom left. Reporting only R@20
would understate the system; reporting only R@5 would overstate it. Both are
here.

**Dense alone is worse than BM25** (−0.036, not significant). A real result, not
a bug: static 256-dimension embeddings are weak, and commit subjects favour
lexical matching. Fusion still helps substantially — the two retrievers fail on
*different* queries, which is exactly what RRF exploits — but re-run with the
full encoder (`pip install 'dwago[dense]'`) before concluding anything about
the dense tier on your own repository.

## Decision rule

A tier ships on by default only if its rung beats the previous rung's confidence
interval. Otherwise it ships off by default and this file says so.
