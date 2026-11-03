# WeightLang Viewer

Dependency-free Web Component for model architecture and parameter inspection.
This directory is the reusable frontend library. The native CLI development
integration embeds `viewer.js` and `viewer.css` into standalone offline HTML;
this npm package does not publish or update the native CLI. There is currently no npm publishing workflow; releases are published explicitly
after local validation.

## Install

```sh
npm install weightlang-viewer
```

```js
import 'weightlang-viewer';
import 'weightlang-viewer/viewer.css';
```

Use the CSS import with a bundler that supports CSS imports, or copy the two
published files and load the stylesheet with a `<link>`.

## Embed the viewer

```html
<link rel="stylesheet" href="./viewer/viewer.css">
<weightlang-viewer id="model"></weightlang-viewer>
<script type="module">
  import './viewer/viewer.js';
  const viewer = document.querySelector('#model');
  // graph is the JSON object emitted by `wgt graph MODEL --format json --stdout`.
  viewer.configure({theme: 'auto'}).load(graph);
</script>
```

`graph` is supplied by the embedding application, not downloaded by the component.
Serve ES modules over HTTP when embedding; the CLI's standalone HTML works offline.
The viewer is a full-bleed stage (default `height: 100vh`); set the element's
`height` to embed it in a smaller area.

## Layout of the stage

The whole viewer is one dotted canvas. The model title floats top-left; search,
expand and view settings float top-right; the legend floats bottom-left; zoom and
export float bottom-right. Floating layers let clicks through except on their
controls. Nothing is framed: nodes are borderless cards lifted by shadow.

Clicking a node opens a **floating details popup**. Drag it by its top bar; it
remembers its position until reload. × or Escape closes it and clears the
selection. `load()` and switching between overview and expanded mode close it
when the selected node no longer exists. On screens up to 760px it becomes a
bottom sheet.

## Control flow

Expanded graphs make the four kinds of control explicit:

- **Flow:** solid edges with arrowheads; edges of the selected node animate in the
  direction data travels (disabled with reduced motion).
- **Loops** (`scan`, `recur`, `ponder`, `scan_state`): a green tinted region
  encloses the loop head and body, labeled “↻ … · weights shared across steps”. The
  feedback edge is green and dotted, labeled *next step*. The scan state node sits
  directly under its loop instead of at the top.
- **Gates:** `mix` inputs are labeled `g`, `× g` and `× (1−g)`; a sigmoid into
  `mul` is labeled *gate*; a ponder halting edge is labeled *halt probability*.
  Gate signals are amber and dashed.
- **Routes:** `graph.rs` returns each branch to its route with a `conditional`
  edge. The viewer adds a virtual `route_merge` node (`<route id>::merge`),
  so branches fan out from the route (*Branch 0*, *Branch 1*, …) and rejoin before
  the next layer. A teal region marks the route; only the selected branch runs.

Small operators (add, mul, mix, concat, activations, state, results) render as
compact pills with their formula. The legend shows only the edge roles present.
`normalizeGraph`, `annotateEdges`, `graphRegions` and `junctionOps` are
exported for reuse.

## What is visualized

- Architecture cells show **element positions and tensor shape**, not runtime
  activations, token IDs or fabricated numerical values. Equal fills carry no value encoding.
- Rank 0: one scalar; rank 1: a vector; rank 2: rows × columns; rank 3+: stacked
  matrices. At most three original axes are visible, defaulting to axes 0, 1, 2.
  Hidden axes are fixed at index 0, never flattened or averaged. With three visible
  axes, the first selects planes and the next two select rows and columns.
  The node inspector lets users uncheck an axis and select another. Full original
  dimensions and cell coordinates are retained.
- Default shape limits: first 6 rows, 8 columns and 3 planes. An ellipsis explicitly
  marks omitted elements. These are representative positions, **not tile averages**.
- Hover a cell to inspect its coordinates. Shape glyphs have accessible labels.
  Original named axes remain in node details.
- The separately opened **weight inspector** displays recorded parameter values.
  Its dense-tensor previews explicitly use tile means; full weights enable exact slices.
- The overview collapses modules; expanded mode retains all graph nodes and edges.
  Solid edges carry data, dashed edges are residual skips, dotted edges are recurrence.
  Skip, recurrent and long forward edges run in separate side lanes instead of
  crossing behind nodes. Each node carries a category color (input/output,
  projection, attention, normalization, activation, pooling, merge, structure,
  recurrent, gate, route, other); the legend lists only the categories present.

## Navigate the canvas

- **Zoom:** toolbar buttons, `+` / `-` / `0` (fit) while the canvas has focus, or
  Ctrl/⌘ + wheel (trackpad pinch). Clicking the percentage restores 100%. Fit shows
  the whole graph when it stays at least 75% legible; otherwise it fits the cross
  axis and scrolls along the flow.
- **Pan:** plain wheel/trackpad scrolling, or drag empty canvas space with the mouse.
- **Keyboard:** Tab to a node; arrow keys along the flow follow edges, arrow keys
  across it move between nodes of the same rank; Enter/Space selects.
- **Search:** matches node paths, operators, titles, std blocks, shapes and parameter
  tensor names (`qkv`). In the overview, a module matches when any collapsed inner
  node or tensor matches. Enter / Shift+Enter cycles matches; Escape clears.
- **Details:** the popup lists shape, parameters, block, operators,
  configuration, and clickable upstream/downstream neighbours, then the shape-axis
  controls and the optional weight inspector.
- **Help:** the ? button in the legend shows the reading and shape notes.
- **Layout:** top-to-bottom or left-to-right.
- **Export:** SVG (computed theme colors inlined, no stylesheet needed) or 2× PNG.
  Search dimming and selection emphasis are not exported.

## Configure a viewer

```js
viewer.configure({
  theme: 'dark',                       // 'light' | 'dark' | 'auto'
  presentation: 'auto',                // 'auto' (tensor cards in overview, compact
                                       // when expanded) | 'tensors' | 'compact'
  direction: 'TB',                     // 'TB' top-to-bottom | 'LR' left-to-right
  layout: null,                        // custom layout function; null = built-in
  tensorLimits: {rows: 6, columns: 8, planes: 3},
  nodeRenderer: null,                  // null restores built-in nodes
  tensorRenderer: null,                // null restores built-in tensor glyphs
});
viewer.setLanguage('en');              // 'zh-CN' | 'en'
viewer.setTheme('light');
viewer.setTensorAxes('f32[2,3,4,5,6]', [0, 2, 4]); // at most 3 original axes
viewer.setExpanded(true);              // overview ⇄ all internal nodes
viewer.selectNode(id);                 // selects, opens the popup, expands if internal
viewer.closePopup();
viewer.setZoom(1.5); viewer.fitView();
const svgText = viewer.exportSVG();    // standalone SVG string
const pngBlob = await viewer.exportPNG(2);
```

`configure()` merges options and returns the viewer. Limits are bounded to 12 rows,
16 columns and 4 planes to avoid unbounded DOM growth. A renderer returning null
or undefined uses the built-in renderer. Renderer functions are trusted application
code and should return SVG elements in the provided bounds. Paths and values are
passed as data; use `svg(tag, attributes, text)` to create escaped text nodes.

`configure({theme})` sets this instance without changing saved preferences.
`setTheme(theme, persist = true)` and the selector remember user choices when local
storage is available. `auto` follows system changes. Language preferences likewise
fall back gracefully when storage is disabled. Browser-native file inputs follow
browser/OS localization. Theme changes redraw canvas colors without altering data.

### Register operators, blocks and categories

Titles and categories come from module-level registries, so new operators need no
viewer changes. A title or description is a translatable string, a
`{'zh-CN': …, en: …}` object, or (descriptions only) a `(node, language) => string`.
Category names use lowercase letters, digits and hyphens. Built-in category
colors come from `--cat-<name>`; a registered `color` overrides that.

```js
import {registerOperator, registerBlock, registerCategory, addTranslations} from 'weightlang-viewer';

registerCategory('quant', {title: {'zh-CN': '量化', en: 'Quantization'}, color: '#6d7fa6'});
registerOperator('fake_quant', {
  title: {'zh-CN': '伪量化', en: 'Fake quantization'},
  category: 'quant',
  description: (node, language) => language === 'en' ? `Simulate ${node.args} quantization.` : `模拟 ${node.args} 量化。`,
});
addTranslations([['专家混合', 'Mixture of experts']]);
registerBlock('acme/moe', {title: '专家混合', category: 'dense'});
```

Registrations are global to the module; call them before `load()` or reload the
graph to re-render. `nodeTitle`, `nodeDescription`, `nodeCategory`,
`categoryTitle`, `layoutGraph` and `routeEdges` are exported for reuse.

### Provide a layout

```js
viewer.configure({
  // Receives {nodes, edges, width, height, gap, direction}; returns
  // {width, height, positions: Map(id → {x, y, width, height})}.
  // Returning null falls back to the built-in layered layout.
  layout: ({nodes, width, height}) => ({
    width: nodes.length * (width + 20) + 40, height: height + 60,
    positions: new Map(nodes.map((n, i) => [n.id, {x: 20 + i * (width + 20), y: 30, width, height}])),
  }),
});
```

Edges are routed by `routeEdges` for any layout.

### Replace or decorate a node

The wrapper retains selection, keyboard behavior, graph edges and `data-op` /
`data-category` attributes. The renderer context also includes `category`.
The renderer controls only its content.

```js
viewer.configure({
  nodeRenderer({node, width, language, svg, defaultRender}) {
    const content = defaultRender();
    if (node.op === 'self_attention') {
      content.append(svg('text', {
        x: width - 16, y: 25, 'text-anchor': 'end', class: 'attention-badge',
      }, language === 'en' ? 'Attention' : '注意力'));
    }
    return content;
  },
});
```

### Replace tensor glyphs

```js
viewer.configure({
  tensorRenderer(context) {
    const {geometry, width, height, svg, defaultRender} = context;
    if (!geometry || geometry.rank !== 1) return defaultRender(context);
    const group = svg('g', {'aria-label': `Vector with ${geometry.shape[0]} elements`});
    const count = geometry.shownCols;
    for (let i = 0; i < count; i++) {
      group.append(svg('circle', {
        cx: (i + 1) * width / (count + 1), cy: height / 2, r: 5,
        fill: 'var(--tensor-cell)',
      }));
    }
    group.append(svg('text', {x: 8, y: height - 4, fill: 'var(--text)'},
      `${geometry.shape[0]} elements${geometry.truncated ? ' …' : ''}`));
    return group;
  },
});
```

Tensor context also includes `node`, `side` (`input`/`output`), raw `label`,
`language`, `limits`, `visibleAxes`, and parsed axes. Geometry exposes
`displayedAxes`, `displayedShape`, and `hiddenAxes` alongside the full `shape`.
Axis choices are scoped to a shape label within this instance and reset on `load()`. The exported `tensorGeometry()` and
`renderTensorShape()` are reusable separately. Unknown shapes fall back to text.

## Style without replacing renderers

```css
/* Load after viewer.css. Instance-scoped variables override both themes. */
#model {
  --tensor-cell: #b67b56;
  --tensor-node-width: 460; /* SVG units, numeric */
  --tensor-node-height: 190;
}
#model[data-theme="dark"] {
  --page-bg: #1b1a17;
  --panel-bg: #24231f;
  --module-fill: #302b24;
}
#model [data-op="self_attention"] { --node-fill: var(--io-fill); }
#model .attention-badge { fill: var(--selection); font: 12px system-ui; }
```

Color tokens: `--page-bg`, `--panel-bg`, `--text`, `--muted`, `--line`,
`--node-fill`, `--node-border`, `--module-fill`, `--io-fill`, `--io-border`,
`--selection`, `--edge-color`, `--skip-color`, `--recurrent-color`,
`--tensor-cell`, `--tensor-plane`, `--tensor-outline`, `--tensor-positive`,
`--tensor-negative`, `--tensor-zero`, `--heatmap-contrast`, `--error-color`, `--shadow`,
`--dot` (canvas grid), `--glass` (floating legend/select backing), `--control-bg`,
`--control-hover`, `--node-shadow`, `--node-shadow-raised`.
Category tokens: `--cat-io`, `--cat-dense`, `--cat-attention`, `--cat-norm`,
`--cat-activation`, `--cat-pool`, `--cat-merge`, `--cat-structure`,
`--cat-recurrent`, `--cat-gate`, `--cat-route`, `--cat-other`; per node, override `--accent`.
Loop, gate and route edges, labels and regions use the matching category tokens.
Font tokens: `--font-ui`, `--font-mono`.
Geometry tokens: `--tensor-node-width`, `--tensor-node-height`,
`--node-width`, `--node-height` (the latter two apply to compact mode).
The model entry convention is `model.wgt`, with its companion `model.css`.
For HTML, the CLI selects an explicit `--style custom.css` first, then
`model.css` in the artifact directory, then the source's same-name CSS in the
current project (or `-C` directory). The source is `model.source` from `wgt.toml`,
defaulting to `model.wgt`, and must exist for project discovery. Missing CSS uses
the built-in theme; unreadable CSS is an error. Other export formats do not
auto-load CSS. Artifact `source.wgt` remains unchanged; CSS is not automatically
included in published packages.
Use `wgt graph MODEL --open` for discovery or
`wgt graph MODEL --style custom.css --open` for an explicit override.

## Events and verification

Bubbling events: `node-select` (node), `tensor-select` (tensor),
`theme-change` (`{theme, resolved}`), `graph-render` (`{expanded, direction, nodes}`), `popup-change` (`{open}`). `load(graph)` validates and replaces model data.
Local tests: `npm test`; browser tests in `test/*.browser.mjs` plus `test/browser.mjs`
use `WEIGHTLANG_PLAYWRIGHT` pointing to an installed Playwright module, with Chromium.
No browser dependency or network request is needed by the viewer itself.
