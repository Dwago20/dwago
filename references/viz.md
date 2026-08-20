# Visualization

```bash
dwago map                     # writes dwago-out/map.html
dwago map --out ~/galaxy.html --title "my project"
```

One self-contained HTML file (~3.4MB at 14k nodes), zero external requests, no
webfonts, no vendored libraries — the entire scene is ~400 lines of first-party
WebGL. It opens with the network off.

## The design: the codebase as a spiral galaxy

3D only, photographic-realism direction (Hubble reference):

- **Core bulge** — the largest connected community, warm K-giant amber.
- **Spiral arms** — remaining communities strung along two logarithmic arms as
  blue-white star clusters, elongated along the arm tangent like real
  star-forming regions. Each data node is one star, sized by degree.
- **Stellar halo** — genuinely disconnected material (isolated doc clusters)
  orbits as sparse dim stars beyond the disc.
- **Photographic dressing** (decoration, never pickable): ~17k unresolved
  filler stars tracing arms and bulge, arm-glow haze, dark dust lanes biting
  the arms' inner edges, ~4k twinkling field stars, faint distant galaxies,
  diffraction spikes on the brightest hubs, slow comet motes along real edges.

**Edges are annotation, not scenery.** A photograph of a galaxy contains no
lines. Links appear as a mission-control overlay — thin cyan constellation
lines (amber dashed = co-change) — only when a star is selected or blast is
armed. Realism at rest, data on demand.

## Performance model

The earlier 2D renderer cached layers and re-rendered on a debounce; its zoom
seam ("glitchy, delayed") was structural. Here nothing is cached to invalidate:
every frame is drawn fresh by the GPU, camera position/zoom are uniforms, and
all input edits *targets* that the camera eases toward with inertia — zoom and
orbit can never stall or snap. Star recolors (spectral modes) and the history
scrubber update one interleaved buffer field, debounced at 90ms.

## Controls

- **drag** orbit · **scroll / pinch** zoom · **click** select a star
- **search** (`/`) — Enter flies the camera to the star, front-facing
- **spectral view** (keys 1-6) — natural · community · hotspot · age · bus
  factor · owner
- **blast** — 3-hop impact web across structural + co-change channels
- **history** — scrub the galaxy back to any date; stars vanish by file age
- click a neighbour in the detail panel to chain-fly through the graph

## Limits, stated plainly

- In-page search is lexical; semantic search lives in `dwago ask`.
- The constellation overlay caps at 60 direct links and 900 blast edges per
  frame — legibility, not capability.
- History uses file-level dates; per-symbol dating would require parsing every
  historical revision.


## The brain map (default since v0.2)

`dwago map` now writes **brain.html**: the codebase as a living brain.
Files are neurons placed inside an anatomical cortex; the seven largest
communities are the lobes (real community names label the regions), and
everything else pools in the cerebellum slot. Structural + co-change edges
are the axons. Selecting a file — by clicking its neuron, or from the
searchable file panel on the right — fires electric strikes along its real
edges, dims the rest of the scene, and locks reticles + labels on every
connected file. The mini-brain in the top-left corner works like a CAD
view cube: hover names a lobe, click opens it (x-ray) and swings the
camera. KEY FILES overlays the top hotspot files from the git layer.

The old symbol-level galaxy map is still available: `dwago map --galaxy`.

Layout is deterministic per file path, so the same repo renders the same
brain across rebuilds. Payload is plain JSON injected into the template;
559 files ≈ 0.1MB of data on top of the ~120KB engine.
