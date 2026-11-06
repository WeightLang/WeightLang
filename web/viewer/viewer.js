const injectedLocales = globalThis.__WEIGHTLANG_LOCALES__;
async function loadLocale() {
  if (injectedLocales?.['zh-CN']) return injectedLocales['zh-CN'];
  const url = new URL('./locales/zh-CN.json', import.meta.url);
  try {
    if (url.protocol === 'file:' && globalThis.process?.versions?.node) {
      const {readFile} = await import('node:fs/promises');
      return JSON.parse(await readFile(url, 'utf8'));
    }
    const response = await fetch(url);
    if (!response.ok) throw Error('Unable to load viewer translations');
    return await response.json();
  } catch {
    return null;
  }
}
const zhLocale = await loadLocale();
const translationIndex = new Map();
for (const [zh, en] of Object.entries(zhLocale || {})) {
  translationIndex.set(zh, [zh, en]);
  translationIndex.set(en, [zh, en]);
}
// Existing embedders may supply additional pairs in the reverse direction.
for (const [en, zh] of Object.entries(injectedLocales?.en || {})) {
  if (!translationIndex.has(en)) translationIndex.set(en, [zh, en]);
  if (!translationIndex.has(zh)) translationIndex.set(zh, [zh, en]);
}
export function translate(text, language = 'zh-CN') {
  const pair = translationIndex.get(text);
  return pair ? pair[language === 'en' ? 1 : 0] : text;
}

/** Generic graph/tensor explorer. No model-name templates or dependencies. */
export function validateGraph(data) {
  if (!data || !Array.isArray(data.nodes) || !Array.isArray(data.edges))
    throw Error("Expected nodes and edges");
  const ids = new Set();
  for (const node of data.nodes) {
    if (typeof node.id !== "string" || ids.has(node.id))
      throw Error("Node IDs must be unique strings");
    ids.add(node.id);
  }
  for (const edge of data.edges)
    if (!ids.has(edge.from) || !ids.has(edge.to))
      throw Error("Missing edge endpoint");
  for (const t of data.tensor_data || []) {
    if (
      !Array.isArray(t.shape) ||
      t.shape.some((n) => !Number.isSafeInteger(n) || n <= 0)
    )
      throw Error("Invalid tensor shape");
    const count = t.shape.reduce((a, b) => a * b, 1);
    if (!Number.isSafeInteger(count))
      throw Error("Tensor too large for exact indexing");
    for (const key of ["values", "baseline_values"]) {
      if (
        t[key] != null &&
        (t[key].length !== count || t[key].some((v) => !Number.isFinite(v)))
      )
        throw Error("Invalid tensor values");
    }
  }
  return data;
}
/** Collapse implementation nodes into their outer module without inventing edges. */
export function overviewGraph(data) {
  const roots = data.nodes.filter((n) => !n.parent && !n.path?.includes('.$'));
  const owners = new Map(data.nodes.map((n) => [n.id,
    roots.find((r) => n.id === r.id || n.path?.startsWith(r.path + '.'))?.id || n.id]));
  const nodes = roots.map((n) => {
    const children = data.nodes.filter((c) => owners.get(c.id) === n.id);
    return {...n, parameters: children.reduce((s, c) => s + (c.parameters || 0), 0),
      members: children.length, operations: [...new Set(children.map((c) => c.op))]};
  });
  // Unknown external graph nodes remain visible even without module metadata.
  for (const n of data.nodes)
    if (owners.get(n.id) === n.id && !nodes.some((r) => r.id === n.id)) nodes.push(n);
  const seen = new Set(), edges = [];
  for (const e of data.edges) {
    const from = owners.get(e.from), to = owners.get(e.to);
    const key = JSON.stringify([from, to, e.kind]);
    if (from !== to && !seen.has(key)) { seen.add(key); edges.push({...e, from, to}); }
  }
  return {nodes, edges};
}
/**
 * Make control flow acyclic for display. `graph.rs` returns each route branch to the
 * route node with a `conditional` edge; a virtual `route_merge` node receives those
 * edges and takes over the route's downstream edges, so branches visibly fan out
 * and rejoin. Graphs without routes are returned unchanged.
 */
export function normalizeGraph(data) {
  const routes = data.nodes.filter((n) => n.op === 'route');
  if (!routes.length) return data;
  const nodes = [...data.nodes], merges = new Map();
  for (const route of routes) {
    const path = route.path || route.id;
    const merge = {id: `${route.id}::merge`, path: `${path}.$merge`, op: 'route_merge', parent: path,
      input_shape: route.output_shape, output_shape: route.output_shape, parameters: 0, virtual: true};
    nodes.push(merge);
    merges.set(route.id, {merge, prefix: path + '.'});
  }
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const edges = data.edges.map((edge) => {
    const target = merges.get(edge.to), source = merges.get(edge.from);
    if (target && edge.kind === 'conditional') return {...edge, to: target.merge.id};
    if (source && !(byId.get(edge.to)?.path || '').startsWith(source.prefix)) return {...edge, from: source.merge.id};
    return edge;
  });
  return {...data, nodes, edges};
}
/** Operators drawn as small junction pills instead of full tensor cards. */
export const junctionOps = new Set(['add', 'mul', 'mix', 'where', 'concat', 'route_merge', 'state', 'argument', 'result',
  'loop_output', 'last', 'relu', 'gelu', 'tanh', 'sigmoid', 'silu', 'identity']);
const loopOps = new Set(['scan', 'recur', 'ponder', 'scan_state']);
const within = (node, prefix) => (node?.path || node?.id || '').startsWith(prefix);
/**
 * Explain control signals on edges: gate inputs of mix/mul, loop feedback, halting
 * probabilities, route branches and concat order. Returns {role, label} per edge.
 */
export function annotateEdges(nodes, edges) {
  const byId = new Map(nodes.map((n) => [n.id, n])), incoming = new Map();
  for (const edge of edges) {
    if (!incoming.has(edge.to)) incoming.set(edge.to, []);
    incoming.get(edge.to).push(edge);
  }
  return edges.map((edge) => {
    const from = byId.get(edge.from), to = byId.get(edge.to), inputs = incoming.get(edge.to), index = inputs.indexOf(edge);
    if (edge.kind === 'recurrent') return {role: 'loop', label: '下一步'};
    if (edge.kind === 'halt') return {role: 'gate', label: '停止概率'};
    if (edge.kind === 'conditional') return {role: 'route'};
    if (from?.op === 'route' && within(to, (from.path || from.id) + '.'))
      return {role: 'route', label: '分支', index: edges.filter((e) => e.from === edge.from && within(byId.get(e.to), (from.path || from.id) + '.')).indexOf(edge)};
    if (to?.op === 'mix') return [{role: 'gate', label: 'g'}, {label: '× g'}, {label: '× (1−g)'}][index] || {};
    if (to?.op === 'mul' && from?.op === 'sigmoid') return {role: 'gate', label: '门'};
    if (to?.op === 'concat' && inputs.length > 1) return {label: String(index + 1)};
    return {};
  });
}
/**
 * Background regions for expanded control flow: loop bodies (scan/recur/ponder/
 * scan_state), whole routes and each route branch. Coordinates come from the layout.
 */
export function graphRegions(nodes, edges, positions) {
  const regions = [];
  const bounds = (members, pad, top) => {
    const boxes = members.map((n) => positions.get(n.id)).filter(Boolean);
    if (!boxes.length) return null;
    const x = Math.min(...boxes.map((p) => p.x)) - pad, y = Math.min(...boxes.map((p) => p.y)) - top;
    return {x, y, width: Math.max(...boxes.map((p) => p.x + p.width)) + pad - x, height: Math.max(...boxes.map((p) => p.y + p.height)) + pad - y};
  };
  for (const head of nodes) {
    const prefix = (head.path || head.id) + '.';
    if (loopOps.has(head.op)) {
      const box = bounds(nodes.filter((n) => n === head || (within(n, prefix) && !n.path?.endsWith('.$output'))), 22, 34);
      if (box) regions.push({kind: 'loop', node: head, ...box});
    } else if (head.op === 'route') {
      const box = bounds(nodes.filter((n) => n === head || within(n, prefix)), 22, 34);
      if (box) regions.push({kind: 'route', node: head, ...box});
    }
  }
  return regions.sort((a, b) => b.width * b.height - a.width * a.height);
}
/** Split a line into code and a trailing `# comment` (a `#` after whitespace, outside quotes). */
function splitComment(line) {
  let quote = null;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quote) { if (c === quote) quote = null; continue; }
    if (c === '"' || c === "'") quote = c;
    else if (c === '#' && (i === 0 || /\s/.test(line[i - 1]))) return [line.slice(0, i).trimEnd(), line.slice(i + 1).trim()];
  }
  return [line.trimEnd(), ''];
}
/**
 * Comments attached to declarations in one WGT module. A run of `#` lines directly
 * above a declaration (no blank line between) and a trailing `# …` on the same line
 * both belong to it. Comments before the first declaration form the module header.
 */
export function parseSourceComments(text, module = 'model') {
  const result = {module, header: '', model: null, aliases: {}, interface: {input: [], output: []}, layers: {}, blocks: {}};
  const header = [];
  let run = [], block = null, declared = false;
  String(text).split(/\r?\n/).forEach((raw, index) => {
    const trimmed = raw.trim();
    if (!trimmed) { if (!declared) header.push(...run); run = []; return; }
    if (trimmed.startsWith('#')) { run.push(trimmed.replace(/^#\s?/, '')); return; }
    const [code, trailing] = splitComment(raw);
    const comment = [...run, trailing].filter(Boolean).join('\n');
    run = [];
    const indent = raw.length - raw.trimStart().length, statement = code.trim();
    if (!indent) block = null;
    let match;
    if (/^wgt\s/.test(statement) || (match = /^use\s+(\S+)(?:\s+as\s+(\w+))?/.exec(statement))) {
      if (match) result.aliases[match[2] || match[1].split('/').pop()] = match[1];
      if (!declared && comment) header.push(comment);
      return;
    }
    const entry = {comment, code: statement, line: index + 1};
    if ((match = /^model\s+(\w+)/.exec(statement))) { declared = true; result.model = {...entry, name: match[1]}; }
    else if (!indent && (match = /^(input|output)\s+(\w+)/.exec(statement))) { declared = true; result.interface[match[1]].push({...entry, name: match[2]}); }
    else if ((match = /^block\s+(\w+)/.exec(statement))) { declared = true; block = match[1]; result.blocks[block] = {...entry, layers: {}}; }
    else if ((match = /^layer\s+(\w+)\s*:\s*(.*)$/.exec(statement))) {
      declared = true;
      (block && indent ? result.blocks[block].layers : result.layers)[match[1]] = {...entry, expression: match[2]};
    }
  });
  result.header = header.join('\n');
  return result;
}
/**
 * Map source comments and code lines onto graph nodes. `graph.sources` holds the model
 * source (`model`) and the std modules it uses. Top-level layers match node paths;
 * a layer that calls a block (`transformer.decoder(...)`, `residual(attention(...))`)
 * passes that block's layer comments down to its child paths. Returns
 * `{notes: Map(nodeId → {comment, code, source, line, block, blockComment}), model}`.
 */
export function sourceNotes(graph) {
  const notes = new Map();
  if (!graph?.sources?.model) return {notes, model: null};
  const modules = Object.entries(graph.sources).map(([name, text]) => parseSourceComments(text, name));
  const blocks = new Map();
  for (const owner of modules)
    for (const [name, block] of Object.entries(owner.blocks)) blocks.set(`${owner.module === 'model' ? 'model' : owner.module}.${name}`, {...block, owner});
  const resolve = (expression, owner) => {
    for (const [, call] of String(expression).matchAll(/([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\s*\(/g)) {
      const [head, tail] = call.includes('.') ? call.split('.') : [null, call];
      const key = head ? `${owner.aliases[head]}.${tail}` : `${owner.module}.${tail}`;
      if (blocks.has(key)) return key;
    }
    return null;
  };
  const byPath = new Map(graph.nodes.map((n) => [n.path || n.id, n]));
  const visit = (prefix, layers, owner, depth, parent) => {
    if (depth > 32) return;
    for (const [name, layer] of Object.entries(layers)) {
      const path = prefix ? `${prefix}.${name}` : name, node = byPath.get(path);
      const call = resolve(layer.expression, owner), block = call && blocks.get(call);
      if (node) notes.set(node.id, {comment: layer.comment, code: layer.code, source: owner.module, line: layer.line,
        block: call, blockComment: block?.comment || '', parentBlock: parent?.key || null, parentComment: parent?.comment || ''});
      if (block) visit(path, block.layers, block.owner, depth + 1, {key: call, comment: block.comment});
    }
  };
  const model = modules.find((m) => m.module === 'model');
  visit('', model.layers, model, 0, null);
  for (const kind of ['input', 'output']) {
    const nodes = graph.nodes.filter((n) => n.op === kind && !n.parent), declarations = model.interface[kind];
    declarations.forEach((declaration, i) => {
      const node = nodes.find((n) => n.path === declaration.name) || (nodes.length === declarations.length ? nodes[i] : null);
      if (node) notes.set(node.id, {comment: declaration.comment, code: declaration.code, source: 'model', line: declaration.line});
    });
  }
  return {notes, model: {name: model.model?.name, comment: model.model?.comment || model.header, header: model.header,
    code: model.model?.code || '', line: model.model?.line, interface: model.interface}};
}
/** Rough SVG text width for truncation: CJK glyphs are about twice as wide. */
function fitText(text, width, size = 12) {
  let used = 0, out = '';
  for (const ch of String(text)) {
    used += (ch.codePointAt(0) > 0x2e80 ? 1 : 0.58) * size;
    if (used > width) return out.trimEnd() + '…';
    out += ch;
  }
  return out;
}
/* Extensible vocabulary: operators, std blocks and visual categories are data, not
   hard-coded branches. Embedders register their own entries; unknown operators
   still render with the generic category. Titles and descriptions may be a
   translatable string, a {'zh-CN', en} object, or (description only) a function. */
const operatorRegistry = new Map();
const blockRegistry = new Map();
const categoryRegistry = new Map();
function localized(value, language = 'zh-CN', ...args) {
  if (value == null) return null;
  if (typeof value === 'function') return value(...args, language);
  if (typeof value === 'object') return value[language] ?? value.en ?? value['zh-CN'] ?? null;
  return translate(String(value), language);
}
function checkedSpec(kind, name, spec) {
  if (typeof name !== 'string' || !name) throw Error(`${kind} name must be a non-empty string`);
  if (!spec || typeof spec !== 'object') throw Error(`${kind} spec must be an object`);
  if (spec.category != null && !/^[a-z][a-z0-9-]*$/.test(spec.category)) throw Error('Category names use lowercase letters, digits and hyphens');
  return spec;
}
/** Register or extend an operator: {title, description, category}. */
export function registerOperator(op, spec) {
  operatorRegistry.set(op, {...operatorRegistry.get(op), ...checkedSpec('Operator', op, spec)});
}
/** Register or extend a module block such as `std/transformer.decoder`. */
export function registerBlock(block, spec) {
  blockRegistry.set(block, {...blockRegistry.get(block), ...checkedSpec('Block', block, spec)});
}
/** Register a visual category: {title, color?}. Built-in colors come from `--cat-<name>`. */
export function registerCategory(name, spec) {
  checkedSpec('Category', name, {...spec, category: name});
  categoryRegistry.set(name, {...categoryRegistry.get(name), ...spec});
}
/** Add [zh-CN, en] pairs used by string titles and descriptions. */
export function addTranslations(pairs) {
  for (const pair of pairs) {
    if (!Array.isArray(pair) || pair.length !== 2 || pair.some(s => typeof s !== 'string')) throw Error('Translations are [zh-CN, en] string pairs');
    translationIndex.set(pair[0], pair).set(pair[1], pair);
  }
}
for (const [name, title] of Object.entries({
  io: '输入输出', dense: '映射', attention: '注意力', norm: '归一化', activation: '激活',
  pool: '汇聚', merge: '合并', structure: '组合', recurrent: '循环', gate: '门控', route: '路由', other: '其他',
})) registerCategory(name, {title});
for (const [op, [title, category, description]] of Object.entries({
  input: ['输入', 'io', '模型接收的数据。形状不包含批次维度。'],
  output: ['输出', 'io', '模型返回的结果。分类得分不一定是概率。'],
  linear: ['全连接', 'dense', '对最后一维做线性映射；其他轴保持不变。'],
  conv2d: ['二维卷积', 'dense', '用共享卷积核提取图片中的局部特征。'],
  embedding: ['词嵌入', 'dense'],
  position_embedding: ['位置嵌入', 'dense'],
  self_attention: ['自注意力', 'attention', '让序列位置之间交换信息；是否允许看后面的位置由 causal 决定。'],
  layernorm: ['层归一化', 'norm'],
  globalavgpool2d: ['全局平均池化', 'pool', '对每个通道的空间位置取平均，保留通道维度。'],
  maxpool2d: ['最大池化', 'pool'],
  last: ['取最后一步', 'pool', '取序列最后一个时间步的特征。'],
  residual: ['残差模块', 'structure', '输入绕过内部计算，与内部结果相加。'],
  seq: ['组合模块', 'structure', '按连接顺序执行内部层；展开内部层可查看具体计算。'],
  scan: ['逐步扫描', 'recurrent', '沿时间轴逐步执行，传递隐藏状态，各步共享同一套权重。'],
  recur: ['共享权重循环', 'recurrent', '重复执行内部模块，各轮共享同一套权重。'],
  state: ['上一步记忆', 'recurrent', '保存上一时间步的隐藏状态，初始值为零。'],
  loop_output: ['循环输出', 'recurrent'],
  add: ['逐元素相加', 'merge'], mul: ['逐元素相乘', 'merge'],
  mix: ['门控混合', 'gate', '按门值 g 混合两个输入：g × a + (1 − g) × b。'],
  where: ['条件选择', 'gate'],
  route: ['路由', 'route', '按选择器只执行其中一个分支；所有分支保持相同的形状和类型。'],
  route_merge: ['路由汇合', 'route', '被选中分支的结果从这里继续向后传递。'],
  ponder: ['自适应循环', 'recurrent', '重复执行内部模块，各轮共享权重；停止概率决定各步结果的加权。'],
  scan_state: ['状态扫描', 'recurrent', '沿时间轴逐步执行，显式传递状态字段，各步共享同一套权重。'],
  argument: ['块参数', 'structure', '把外部张量或上一步状态送入内部模块。'],
  result: ['结果字段', 'structure', '模块额外导出的命名结果。'],
  silu: ['SiLU 激活', 'activation'], identity: ['恒等', 'structure'],
  concat: ['拼接', 'merge'],
  relu: ['ReLU 激活', 'activation'], gelu: ['GELU 激活', 'activation'],
  tanh: ['Tanh 激活', 'activation'], sigmoid: ['Sigmoid 激活', 'activation'],
})) registerOperator(op, {title, category, ...(description && {description})});
for (const [block, [title, category, description]] of Object.entries({
  'std/transformer.tokens': ['符号嵌入 + 位置编码', 'dense', '把符号编号转换为特征向量，再加入位置信息。'],
  'std/transformer.decoder': ['Transformer 解码器', 'attention', '先用因果自注意力汇聚前面的信息，再用前馈网络处理各位置；两部分都有归一化和残差连接。'],
  'std/transformer.encoder': ['Transformer 编码器', 'attention', '先用双向自注意力交换位置信息，再用前馈网络处理各位置；两部分都有残差连接。'],
  'std/transformer.gated_decoder': ['门控 Transformer 解码器', 'attention'],
  'std/recurrent.gru': ['GRU 序列记忆', 'recurrent', '逐步读取序列，用重置门和更新门控制记忆。所有时间步共享权重，输出包含每一步的隐藏状态。'],
  'std/vision.conv_relu': ['卷积 + ReLU', 'dense', '先提取局部图案，再用 ReLU 保留正响应。'],
  'std/vision.classifier': ['空间汇聚 + 分类', 'pool', '先对各通道的空间位置取平均，再映射为分类得分。'],
  'std/heads.classifier': ['分类输出层', 'dense', '把最后一维映射为各类别的原始得分；这里没有 Softmax。'],
})) registerBlock(block, {title, category, ...(description && {description})});

function operatorTitle(op, language) {
  return localized(operatorRegistry.get(op)?.title, language) || op;
}
export function nodeTitle(node, language = 'zh-CN') {
  return localized(blockRegistry.get(node.block)?.title, language) ||
    localized(operatorRegistry.get(node.op)?.title, language) || node.op || translate('模块', language);
}
export function nodeDescription(node, language = 'zh-CN') {
  const block = blockRegistry.get(node.block)?.description, op = operatorRegistry.get(node.op)?.description;
  return localized(block, language, node) || localized(op, language, node) ||
    `${nodeTitle(node, language)}: ${translate('按图中的连线处理输入。', language)}`;
}
/** Visual category for a node: block category, then operator category, then `other`. */
export function nodeCategory(node) {
  return blockRegistry.get(node.block)?.category || operatorRegistry.get(node.op)?.category || 'other';
}
export function categoryTitle(name, language = 'zh-CN') {
  return localized(categoryRegistry.get(name)?.title, language) || name;
}
/**
 * Layered layout. Ranks follow the longest feed-forward path (recurrent edges are
 * ignored so feedback cannot push nodes down); a few barycenter sweeps order each
 * rank to reduce edge crossings. `direction` is `TB` (top to bottom) or `LR`.
 */
export function layoutGraph(nodes, edges, width = 260, height = 76, gap = 54, options = {}) {
  const horizontal = options.direction === 'LR', spacing = 28;
  const incoming = new Map(nodes.map((n) => [n.id, 0])),
    outgoing = new Map(nodes.map((n) => [n.id, []])),
    preds = new Map(nodes.map((n) => [n.id, []]));
  const ranks = new Map(nodes.map((n) => [n.id, 0])),
    done = new Set();
  for (const e of edges) {
    if (e.kind === 'recurrent') continue;
    incoming.set(e.to, incoming.get(e.to) + 1);
    outgoing.get(e.from).push(e.to);
    preds.get(e.to).push(e.from);
  }
  // Entry nodes inside a module (scan state, block arguments) rank just below that module.
  const byPath = new Map(nodes.map((n) => [n.path || n.id, n.id]));
  for (const n of nodes) {
    const owner = n.parent && byPath.get(n.parent);
    if (incoming.get(n.id) === 0 && owner && owner !== n.id) {
      incoming.set(n.id, 1);
      outgoing.get(owner).push(n.id);
    }
  }
  const queue = nodes.filter((n) => incoming.get(n.id) === 0).map((n) => n.id);
  for (let i = 0; i < queue.length; i++) {
    const id = queue[i];
    done.add(id);
    for (const target of outgoing.get(id)) {
      ranks.set(target, Math.max(ranks.get(target), ranks.get(id) + 1));
      incoming.set(target, incoming.get(target) - 1);
      if (incoming.get(target) === 0) queue.push(target);
    }
  }
  let last = Math.max(0, ...ranks.values());
  for (const n of nodes) if (!done.has(n.id)) ranks.set(n.id, ++last);
  const rows = new Map();
  for (const n of nodes) {
    const r = ranks.get(n.id);
    if (!rows.has(r)) rows.set(r, []);
    rows.get(r).push(n);
  }
  const columns = Math.max(1, ...[...rows.values()].map((r) => r.length));
  const slot = new Map();
  const place = () => { for (const row of rows.values()) row.forEach((n, c) => slot.set(n.id, (columns - row.length) / 2 + c)); };
  place();
  const keys = [...rows.keys()].sort((a, b) => a - b);
  for (let sweep = 0; sweep < 4; sweep++) {
    const down = sweep % 2 === 0, neighbours = down ? preds : outgoing;
    for (const r of down ? keys : [...keys].reverse()) {
      const row = rows.get(r);
      const score = new Map(row.map((n, i) => {
        const linked = neighbours.get(n.id).filter((id) => ranks.get(id) !== r);
        return [n.id, linked.length ? linked.reduce((s, id) => s + slot.get(id), 0) / linked.length : slot.get(n.id) + i * 1e-6];
      }));
      row.sort((a, b) => score.get(a.id) - score.get(b.id));
      place();
    }
  }
  const positions = new Map();
  for (const [r, row] of rows)
    for (const n of row) {
      const across = 24 + slot.get(n.id) * ((horizontal ? height : width) + spacing),
        along = 28 + r * ((horizontal ? width : height) + gap);
      const size = options.size?.(n) || {width, height};
      const x = (horizontal ? along : across) + (width - size.width) / 2, y = (horizontal ? across : along) + (height - size.height) / 2;
      positions.set(n.id, {x, y, width: size.width, height: size.height, rank: r});
    }
  const acrossSize = columns * ((horizontal ? height : width) + spacing) + 40,
    alongSize = (last + 1) * ((horizontal ? width : height) + gap) + 32;
  return {positions, direction: horizontal ? 'LR' : 'TB',
    width: horizontal ? alongSize : acrossSize, height: horizontal ? acrossSize : alongSize};
}
function roundedPath(points, radius = 10) {
  let d = `M${points[0][0]},${points[0][1]}`;
  for (let i = 1; i < points.length - 1; i++) {
    const [px, py] = points[i], [ax, ay] = points[i - 1], [bx, by] = points[i + 1];
    const l1 = Math.hypot(px - ax, py - ay), l2 = Math.hypot(bx - px, by - py);
    const r = Math.min(radius, l1 / 2, l2 / 2);
    if (!r) { d += ` L${px},${py}`; continue; }
    d += ` L${px - (px - ax) / l1 * r},${py - (py - ay) / l1 * r} Q${px},${py} ${px + (bx - px) / l2 * r},${py + (by - py) / l2 * r}`;
  }
  const end = points.at(-1);
  return d + ` L${end[0]},${end[1]}`;
}
/**
 * Route edges for a layout. Forward edges are smooth curves between facing sides;
 * skip, recurrent and backward edges run in separate lanes beside the nodes they
 * pass so parallel bypasses never overlap. Returns SVG path data and the extent.
 */
export function routeEdges(edges, positions, direction = 'TB') {
  const lr = direction === 'LR';
  const along = (p) => lr ? p.x : p.y, alongSize = (p) => lr ? p.width : p.height;
  const across = (p) => lr ? p.y : p.x, acrossSize = (p) => lr ? p.height : p.width;
  const boxes = [...positions.values()];
  const toSvg = ([a, b]) => lr ? [b, a] : [a, b];
  let extent = Math.max(0, ...boxes.map((p) => across(p) + acrossSize(p)));
  const lanes = {[-1]: [], 1: []}, bypass = [];
  let start = Math.min(...boxes.map(across));
  const routes = edges.map((edge) => {
    const a = positions.get(edge.from), b = positions.get(edge.to);
    const route = {edge, kind: edge.kind || 'flow'};
    const x1 = across(a) + acrossSize(a) / 2, y1 = along(a) + alongSize(a);
    const x2 = across(b) + acrossSize(b) / 2, y2 = along(b), m = (y1 + y2) / 2;
    // A forward edge that would pass behind an intermediate node takes a side lane instead.
    const blocked = boxes.some((p) => p !== a && p !== b && along(p) < y2 && along(p) + alongSize(p) > y1 &&
      across(p) < Math.max(x1, x2) + 8 && across(p) + acrossSize(p) > Math.min(x1, x2) - 8);
    if (route.kind !== 'skip' && route.kind !== 'recurrent' && y2 >= y1 && !blocked) {
      const [p0, p1, p2, p3] = [[x1, y1], [x1, m], [x2, m], [x2, y2]].map(toSvg);
      route.d = `M${p0} C${p1} ${p2} ${p3}`;
      const t = 0.72, u = 1 - t;
      route.label = [0, 1].map((k) => u * u * u * p0[k] + 3 * u * u * t * p1[k] + 3 * u * t * t * p2[k] + t * t * t * p3[k]);
    } else bypass.push({route, a, b, lo: Math.min(along(a), along(b)), hi: Math.max(along(a) + alongSize(a), along(b) + alongSize(b))});
    return route;
  });
  bypass.sort((m, n) => (m.hi - m.lo) - (n.hi - n.lo));
  for (const item of bypass) {
    const {route, a, b, lo, hi} = item;
    const passed = boxes.filter((p) => along(p) < hi && along(p) + alongSize(p) > lo);
    const near = Math.min(...passed.map(across)), far = Math.max(...passed.map((p) => across(p) + acrossSize(p)));
    const middle = (across(a) + acrossSize(a) / 2 + across(b) + acrossSize(b) / 2) / 2;
    const side = middle < (near + far) / 2 - 1 ? -1 : 1, pool = lanes[side];
    let lane = pool.findIndex((used) => used.every(([l, h]) => h <= lo || l >= hi));
    if (lane < 0) { lane = pool.length; pool.push([]); }
    pool[lane].push([lo, hi]);
    const c = side > 0 ? far + 22 + lane * 14 : near - 22 - lane * 14;
    extent = Math.max(extent, c);
    start = Math.min(start, c);
    const y1 = along(a) + alongSize(a) / 2, y2 = along(b) + alongSize(b) / 2;
    const edgeOf = (p) => side > 0 ? across(p) + acrossSize(p) : across(p);
    route.lane = lane;
    route.side = side;
    route.d = roundedPath([[edgeOf(a), y1], [c, y1], [c, y2], [edgeOf(b), y2]].map(toSvg));
    route.label = toSvg([c, (y1 + y2) / 2]);
  }
  return {routes, extent, start};
}
export function flatIndex(shape, coordinates) {
  if (shape.length !== coordinates.length)
    throw Error("Coordinate rank mismatch");
  return shape.reduce((index, size, axis) => {
    const v = coordinates[axis];
    if (!Number.isInteger(v) || v < 0 || v >= size)
      throw Error("Coordinate out of bounds");
    return index * size + v;
  }, 0);
}
/** Two display axes; fixed indices for all other axes; tile means for dense planes. */
export function sliceTensor(
  shape,
  getValue,
  rowAxis,
  colAxis,
  fixed,
  limit = 96,
) {
  if (!shape.length)
    return {
      rows: 1,
      cols: 1,
      values: [getValue(0)],
      tiles: [{ coordinates: [], index: 0 }],
      aggregated: false,
    };
  if (shape.length === 1) {
    rowAxis = -1;
    colAxis = 0;
  } else if (
    rowAxis === colAxis ||
    rowAxis < 0 ||
    colAxis < 0 ||
    rowAxis >= shape.length ||
    colAxis >= shape.length
  )
    throw Error("Choose two different axes");
  if (!Number.isInteger(limit) || limit < 1) throw Error("Invalid tile limit");
  const rows = rowAxis < 0 ? 1 : shape[rowAxis],
    cols = shape[colAxis];
  const rs = Math.max(1, Math.ceil(rows / limit)),
    cs = Math.max(1, Math.ceil(cols / limit)),
    values = [],
    tiles = [];
  for (let r = 0; r < rows; r += rs)
    for (let c = 0; c < cols; c += cs) {
      let sum = 0,
        count = 0;
      const coord = shape.map((_, a) => fixed[a] || 0);
      for (let rr = r; rr < Math.min(rows, r + rs); rr++)
        for (let cc = c; cc < Math.min(cols, c + cs); cc++) {
          if (rowAxis >= 0) coord[rowAxis] = rr;
          coord[colAxis] = cc;
          sum += getValue(flatIndex(shape, coord));
          count++;
        }
      if (rowAxis >= 0) coord[rowAxis] = r;
      coord[colAxis] = c;
      values.push(sum / count);
      tiles.push({
        coordinates: [...coord],
        index: flatIndex(shape, coord),
        row: [r, Math.min(rows, r + rs) - 1],
        col: [c, Math.min(cols, c + cs) - 1],
        count,
      });
    }
  return {
    rows: Math.ceil(rows / rs),
    cols: Math.ceil(cols / cs),
    values,
    tiles,
    aggregated: rs > 1 || cs > 1,
  };
}
const ns = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, text) {
  const e = document.createElementNS(ns, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}
/** Parse contract labels without assuming semantic axis names. */
export function tensorGeometry(label, limits = {}, visibleAxes) {
  const match = /^(\w+)\[([^\]]*)\]$/.exec(label || '');
  if (!match) return null;
  const axes = match[2].trim() ? match[2].split(',').map(part => {
    const fields = part.trim().split('=');
    const size = Number(fields.at(-1));
    return {name: fields.length === 2 ? fields[0] : null, size};
  }) : [];
  if (axes.some(a => !Number.isSafeInteger(a.size) || a.size < 1)) return null;
  const shape = axes.map(a => a.size), rank = shape.length;
  const displayedAxes = visibleAxes === undefined ? axes.slice(0, 3).map((_, i) => i) : [...visibleAxes];
  if (displayedAxes.length > 3 || new Set(displayedAxes).size !== displayedAxes.length ||
      displayedAxes.some(i => !Number.isInteger(i) || i < 0 || i >= rank)) throw Error('Select at most three valid, distinct axes');
  const displayedShape = displayedAxes.map(i => shape[i]), displayedRank = displayedAxes.length;
  const hiddenAxes = axes.map((_, i) => i).filter(i => !displayedAxes.includes(i));
  const rows = displayedRank > 1 ? displayedShape.at(-2) : 1, cols = displayedShape.at(-1) || 1;
  const planes = displayedRank === 3 ? displayedShape[0] : 1;
  const cap = (value, fallback, max) => Number.isInteger(value) && value > 0 ? Math.min(value, max) : fallback;
  const shownRows = Math.min(rows, cap(limits.rows, 6, 12));
  const shownCols = Math.min(cols, cap(limits.columns, 8, 16));
  const shownPlanes = Math.min(planes, cap(limits.planes, 3, 4));
  return {dtype: match[1], axes, shape, rank, displayedAxes, displayedShape, displayedRank, hiddenAxes, rows, cols, planes,
    shownRows, shownCols, shownPlanes,
    truncated: rows > shownRows || cols > shownCols || planes > shownPlanes};
}

/** Default shape-only renderer: cells represent elements, never invented values. */
export function renderTensorShape({label, width = 180, height = 100, language = 'zh-CN', limits, visibleAxes}) {
  const geometry = tensorGeometry(label, limits, visibleAxes), group = svg('g', {class: 'wl-tensor-glyph'});
  if (!geometry) {
    group.append(svg('text', {x: 0, y: 25, class: 'wl-node-shape'}, label || '—'));
    return group;
  }
  const g = geometry, en = language === 'en';
  group.setAttribute('data-rank', g.displayedRank);
  group.setAttribute('data-visible-axes', g.displayedAxes.join(','));
  group.setAttribute('aria-label', `${label}; ${en ? 'shape only, no activation values' : '仅形状，不含运行值'}`);
  const margin = 10, offset = 5, labelHeight = 25;
  const cell = Math.min(15, (width - margin * 2 - (g.shownPlanes - 1) * offset) / g.shownCols,
    (height - labelHeight - 10 - (g.shownPlanes - 1) * offset) / g.shownRows);
  const left = (width - g.shownCols * cell - (g.shownPlanes - 1) * offset) / 2;
  const top = Math.max(6, (height - labelHeight - g.shownRows * cell - (g.shownPlanes - 1) * offset) / 2);
  for (let plane = g.shownPlanes - 1; plane >= 0; plane--) {
    const layer = svg('g', {transform: `translate(${left + plane * offset} ${top + plane * offset})`, 'data-plane': plane});
    layer.append(svg('rect', {x: -2, y: -2, width: g.shownCols * cell + 3, height: g.shownRows * cell + 3, rx: 3, class: 'wl-tensor-plane'}));
    for (let row = 0; row < g.shownRows; row++) for (let col = 0; col < g.shownCols; col++) {
      const coordinates = g.shape.map(() => 0);
      if (g.displayedRank === 3) coordinates[g.displayedAxes[0]] = plane;
      if (g.displayedRank > 1) coordinates[g.displayedAxes.at(-2)] = row;
      if (g.displayedRank > 0) coordinates[g.displayedAxes.at(-1)] = col;
      const rect = svg('rect', {x: col * cell, y: row * cell, width: Math.max(1, cell - 2), height: Math.max(1, cell - 2), rx: 2,
        class: 'wl-tensor-cell', 'data-coordinate': coordinates.join(',')});
      rect.append(svg('title', {}, `[${coordinates.join(', ')}] · ${en ? 'element position, value not recorded' : '元素位置，未记录运行值'}`));
      layer.append(rect);
    }
    group.append(layer);
  }
  const dimensions = g.displayedShape.length ? g.displayedShape.join(' × ') : (en ? 'scalar slice' : '标量切片');
  group.append(svg('title', {}, `${en ? 'Visible axes' : '显示轴'}: ${g.displayedAxes.join(', ')}; ${en ? 'Hidden axes fixed at 0' : '隐藏轴固定索引 0'}: ${g.hiddenAxes.join(', ') || '—'}`));
  group.append(svg('text', {x: width / 2, y: height - 7, 'text-anchor': 'middle', class: 'wl-tensor-dimensions'}, dimensions + (g.truncated ? ' …' : '')));
  if (g.truncated) group.append(svg('title', {}, en ? 'Leading elements shown; full dimensions are labeled. No averaging.' : '仅展示各轴前部元素，标签为完整尺寸；未进行均值聚合。'));
  return group;
}

function num(v) {
  return v == null
    ? "—"
    : Math.abs(v) < 0.001 && v !== 0
      ? v.toExponential(3)
      : Number(v.toPrecision(5)).toString();
}
const template = "<header class=\"wl-header\"><div class=\"wl-title\"><h1 class=\"wl-model-title\" tabindex=\"0\" role=\"button\"></h1><div class=\"wl-summary\"></div></div></header><div class=\"wl-hud\"><div class=\"wl-actions\"><label class=\"wl-search-box\"><input class=\"wl-search\" type=\"search\" data-label=\"搜索节点\" autocomplete=\"off\"><span class=\"wl-search-count\" aria-live=\"polite\"></span></label><button class=\"wl-expand\" aria-pressed=\"false\">展开内部层</button></div><div class=\"wl-header-tools\"><select class=\"wl-presentation\" aria-label=\"Display / 显示\"><option value=\"auto\">自动视图</option><option value=\"tensors\">张量视图</option><option value=\"compact\">紧凑视图</option></select><select class=\"wl-direction\" aria-label=\"Layout / 布局\"><option value=\"TB\">纵向布局</option><option value=\"LR\">横向布局</option></select><select class=\"wl-theme\" aria-label=\"Theme / 主题\"><option value=\"auto\">跟随系统</option><option value=\"light\">亮色</option><option value=\"dark\">暗色</option></select><select class=\"wl-language\" aria-label=\"Language / 语言\"><option value=\"zh-CN\">中文</option><option value=\"en\">English</option></select></div></div><div class=\"wl-workspace\"><div class=\"wl-flow\"></div><div class=\"wl-legend-row\"><div class=\"wl-key\"><span data-kind=\"flow\">数据流</span><span data-kind=\"skip\">残差旁路</span><span data-kind=\"loop\">循环反馈</span><span data-kind=\"gate\">门控信号</span><span data-kind=\"route\">路由分支</span></div><div class=\"wl-categories\"></div><button class=\"wl-help\" aria-expanded=\"false\" data-label=\"说明\">?</button><div class=\"wl-footnotes\" hidden><p class=\"wl-note\">沿箭头阅读：输入 → 特征处理 → 输出。滚轮缩放，拖动画布平移，拖动节点调整位置，双击空白处适应画布；点击模块查看说明和源码注释。形状不含批次维度。</p><p class=\"wl-shape-note\">格子表示元素位置，不是运行值；省略号表示只展示部分元素。</p></div></div><div class=\"wl-zoom\" role=\"toolbar\"><button class=\"wl-zoom-out\" data-label=\"缩小\">−</button><button class=\"wl-zoom-level\" data-label=\"实际大小\">100%</button><button class=\"wl-zoom-in\" data-label=\"放大\">+</button><button class=\"wl-fit\" data-label=\"适应画布\">⤢</button><button class=\"wl-reset-layout\" data-label=\"恢复自动布局\" hidden>↺</button><button class=\"wl-export-svg\" data-label=\"导出 SVG\">SVG</button><button class=\"wl-export-png\" data-label=\"导出 PNG\">PNG</button></div><section class=\"wl-explorer wl-popup\" role=\"dialog\" hidden><div class=\"wl-popup-bar\"><span class=\"wl-popup-grip\"></span><button class=\"wl-popup-close\" data-label=\"关闭\">×</button></div><div class=\"wl-popup-body\"><div class=\"wl-node-info\" aria-live=\"polite\"><h2>从输入开始看</h2><p>选择左侧模块，查看它的作用和形状变化。</p></div><button class=\"wl-inspect-toggle\" aria-expanded=\"false\">查看参数值</button><div class=\"wl-weight-tools\" hidden><div class=\"wl-panel-heading\"><h2>权重检查</h2><span class=\"wl-origin\"></span></div><label class=\"wl-tensor-label\">参数张量<select class=\"wl-tensor\"></select></label><div class=\"wl-statistics\"></div><div class=\"wl-controls\"><label>View <select class=\"wl-mode\"></select></label><label class=\"wl-row-control\">Rows <select class=\"wl-row\"></select></label><label class=\"wl-column-control\">Columns <select class=\"wl-column\"></select></label></div><div class=\"wl-fixed\"></div><div class=\"wl-presets\"></div><p class=\"wl-description\"></p><div class=\"wl-plot\"><canvas tabindex=\"0\" aria-label=\"Tensor values\"></canvas></div><div class=\"wl-legend\"><span class=\"wl-low\"></span><span class=\"wl-gradient\"></span><span class=\"wl-high\"></span></div><output class=\"wl-readout\" aria-live=\"polite\">Point at a cell to inspect its coordinates and value.</output><div class=\"wl-files\"><label>Full weights for exact large-tensor slices<input class=\"wl-current-file\" type=\"file\" accept=\".safetensors\"></label><label class=\"wl-baseline-file-label\">Baseline weights<input class=\"wl-baseline-file\" type=\"file\" accept=\".safetensors\"></label></div><p class=\"wl-note\">下图是参数值，不是模型运行时的激活或注意力概率；不能据此判断训练效果。</p></div></div></section></div><p class=\"wl-error\" role=\"alert\"></p>";
let viewerCount = 0;
const edgeKinds = ['flow', 'skip', 'recurrent', 'conditional', 'halt', 'gate', 'active'];
const kindRoles = {skip: 'skip', recurrent: 'loop', conditional: 'route', halt: 'gate'};
const exportedStyles = ['fill', 'fill-opacity', 'stroke', 'stroke-width', 'stroke-dasharray', 'stroke-linecap',
  'stroke-opacity', 'stroke-linejoin', 'paint-order', 'opacity', 'font-family', 'font-size', 'font-weight', 'visibility'];
export class WeightLangViewer extends (globalThis.HTMLElement || class {}) {
  constructor() {
    super();
    this.language = globalThis.navigator?.language?.startsWith('zh') ? 'zh-CN' : 'en';
    try { const saved = localStorage.getItem('weightlang.language'); if (['zh-CN', 'en'].includes(saved)) this.language = saved; } catch {}
    this.options = {presentation: 'auto', direction: 'TB', layout: null, tensorLimits: {rows: 6, columns: 8, planes: 3}};
    this.tensorAxes = new Map();
    this.theme = 'auto';
    try { const saved = localStorage.getItem('weightlang.theme'); if (['auto', 'light', 'dark'].includes(saved)) this.theme = saved; } catch {}
    this.buffers = {};
    this.fixed = [];
    this.range = null;
    this.uid = `wl${++viewerCount}`;
    this.zoom = 1;
    this.pendingFit = true;
    this.searchMatches = [];
    this.searchIndex = -1;
  }
  connectedCallback() {
    if (!this.ready) {
      this.innerHTML = template;
      this.ready = true;
      this.staticText = [];
      const walker = document.createTreeWalker(this, NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) {
        const node = walker.currentNode;
        if (translationIndex.has(node.textContent.trim())) this.staticText.push([node, node.textContent]);
      }
      this.q('.wl-language').onchange = () => this.setLanguage(this.q('.wl-language').value);
      this.q('.wl-theme').onchange = () => this.setTheme(this.q('.wl-theme').value);
      this.q('.wl-presentation').onchange = () => this.configure({presentation: this.q('.wl-presentation').value});
      this.q('.wl-direction').onchange = () => this.configure({direction: this.q('.wl-direction').value});
      this.q('.wl-presentation').value = this.options.presentation;
      this.q('.wl-direction').value = this.options.direction;
      this.media = matchMedia('(prefers-color-scheme: dark)');
      this.themeChanged = () => this.applyTheme();
      this.setTheme(this.theme, false);
      this.localizeStatic();
      this.q(".wl-inspect-toggle").onclick = () => {
        const panel = this.q(".wl-weight-tools");
        panel.hidden = !panel.hidden;
        this.q(".wl-inspect-toggle").setAttribute("aria-expanded", String(!panel.hidden));
        this.q(".wl-inspect-toggle").textContent = panel.hidden ? this.t("查看参数值") : this.t("收起参数值");
        if (!panel.hidden && this.selected) this.draw();
      };
      this.q(".wl-tensor").onchange = () =>
        this.selectTensor(this.q(".wl-tensor").value);
      this.q(".wl-mode").onchange = () => this.draw();
      for (const k of ["row", "column"])
        this.q(".wl-" + k).onchange = () => {
          this.range = null;
          this.axisControls();
          this.draw();
        };
      this.q(".wl-expand").onclick = () => this.setExpanded(!this.expanded);
      this.bindCanvas();
      this.bindPopup();
      this.q("canvas").onpointermove = (e) => this.inspectCell(e);
      for (const kind of ["current", "baseline"])
        this.q(".wl-" + kind + "-file").onchange = (e) =>
          this.loadWeights(e.target.files[0], kind).catch((error) =>
            this.fail(error),
          );
      this.observer = new ResizeObserver(() => {
        if (this.selected) this.draw();
      });
    }
    this.media.addEventListener("change", this.themeChanged);
    this.observer.observe(this.q(".wl-plot"));
    if (this.data) this.render();
  }
  disconnectedCallback() {
    this.observer?.disconnect();
    this.media?.removeEventListener("change", this.themeChanged);
  }
  /** Floating details popup: opened by clicking a node, dragged by its bar, closed with × or Escape. */
  bindPopup() {
    const popup = this.q('.wl-popup'), bar = this.q('.wl-popup-bar');
    this.q('.wl-popup-close').onclick = () => this.closePopup();
    popup.addEventListener('keydown', (e) => { if (e.key === 'Escape') { e.stopPropagation(); this.closePopup(); } });
    let drag = null;
    bar.onpointerdown = (e) => {
      if (e.button !== 0 || e.target.closest('button') || matchMedia('(max-width: 640px)').matches) return;
      const box = popup.getBoundingClientRect();
      drag = {dx: e.clientX - box.left, dy: e.clientY - box.top};
      bar.setPointerCapture(e.pointerId);
      popup.classList.add('wl-dragging');
      e.preventDefault();
    };
    bar.onpointermove = (e) => {
      if (!drag) return;
      const host = this.getBoundingClientRect();
      const left = Math.min(Math.max(0, e.clientX - host.left - drag.dx), host.width - popup.offsetWidth);
      const top = Math.min(Math.max(0, e.clientY - host.top - drag.dy), host.height - 48);
      this.popupPosition = {left, top};
      Object.assign(popup.style, {left: left + 'px', top: top + 'px', right: 'auto'});
    };
    bar.onpointerup = bar.onpointercancel = () => { drag = null; popup.classList.remove('wl-dragging'); };
    const help = this.q('.wl-help');
    help.onclick = () => {
      const notes = this.q('.wl-footnotes');
      notes.hidden = !notes.hidden;
      help.setAttribute('aria-expanded', String(!notes.hidden));
    };
  }
  openPopup() {
    const popup = this.q('.wl-popup');
    if (!popup.hidden) return;
    popup.hidden = false;
    if (this.popupPosition) Object.assign(popup.style, {left: this.popupPosition.left + 'px', top: this.popupPosition.top + 'px', right: 'auto'});
    this.dispatchEvent(new CustomEvent('popup-change', {detail: {open: true}, bubbles: true}));
  }
  closePopup() {
    const popup = this.q('.wl-popup');
    if (popup.hidden) return this;
    popup.hidden = true;
    const selected = this.q('.wl-selected');
    this.querySelectorAll('.wl-selected').forEach((n) => n.classList.remove('wl-selected'));
    this.markSelection(null);
    selected?.focus({preventScroll: true});
    this.dispatchEvent(new CustomEvent('popup-change', {detail: {open: false}, bubbles: true}));
    return this;
  }
  /** Canvas interaction: Ctrl/⌘ + wheel zoom, mouse drag pan, keyboard shortcuts, search and export. */
  bindCanvas() {
    const flow = this.q('.wl-flow');
    // A mouse wheel zooms around the cursor; pinch (Ctrl/⌘ + wheel) zooms; trackpad
    // two-finger scrolling and Shift + wheel keep panning natively.
    flow.addEventListener('wheel', (e) => {
      const mouseWheel = e.deltaMode !== 0 || (e.deltaX === 0 && Number.isInteger(e.deltaY) && Math.abs(e.deltaY) >= 50);
      if (e.shiftKey || !(e.ctrlKey || e.metaKey || mouseWheel)) return;
      e.preventDefault();
      const bounds = flow.getBoundingClientRect(), speed = mouseWheel ? 0.0015 : 0.01;
      this.setZoom(this.zoom * Math.exp(-e.deltaY * speed), {x: e.clientX - bounds.left, y: e.clientY - bounds.top});
    }, {passive: false});
    // Drag empty canvas to pan, drag a node to move it. Pointer capture starts only
    // after a few pixels so a plain click still reaches the node.
    let drag = null;
    flow.onpointerdown = (e) => {
      if (e.pointerType === 'touch' || e.button !== 0) return;
      const node = e.target.closest('.wl-node'), position = node && this.layout?.positions.get(node.dataset.id);
      drag = {x: e.clientX, y: e.clientY, left: flow.scrollLeft, top: flow.scrollTop, pointer: e.pointerId, moved: false,
        node: position ? node.dataset.id : null, origin: position && {x: position.x, y: position.y}};
    };
    flow.onpointermove = (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!drag.moved) {
        if (Math.hypot(dx, dy) < 4) return;
        drag.moved = true;
        flow.setPointerCapture(drag.pointer);
        flow.classList.add(drag.node ? 'wl-moving' : 'wl-panning');
      }
      if (drag.node) this.moveNode(drag.node, drag.origin.x + dx / this.zoom, drag.origin.y + dy / this.zoom);
      else { flow.scrollLeft = drag.left - dx; flow.scrollTop = drag.top - dy; }
    };
    flow.onpointerup = flow.onpointercancel = () => {
      if (drag?.moved) { this.suppressClick = true; setTimeout(() => { this.suppressClick = false; }); }
      drag = null;
      flow.classList.remove('wl-panning', 'wl-moving');
    };
    flow.addEventListener('click', (e) => { if (this.suppressClick) { e.stopPropagation(); this.suppressClick = false; } }, true);
    flow.ondblclick = (e) => { if (!e.target.closest('.wl-node')) this.fitView(); };
    this.q('.wl-reset-layout').onclick = () => this.resetLayout();
    const title = this.q('.wl-model-title');
    title.onclick = () => this.showModelNotes();
    title.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); this.showModelNotes(); } };
    flow.onkeydown = (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === 'Escape') { this.closePopup(); return; }
      if (e.key === '+' || e.key === '=') this.setZoom(this.zoom * 1.2);
      else if (e.key === '-') this.setZoom(this.zoom / 1.2);
      else if (e.key === '0') this.fitView();
      else return;
      e.preventDefault();
    };
    this.q('.wl-zoom-in').onclick = () => this.setZoom(this.zoom * 1.2);
    this.q('.wl-zoom-out').onclick = () => this.setZoom(this.zoom / 1.2);
    this.q('.wl-zoom-level').onclick = () => this.setZoom(1);
    this.q('.wl-fit').onclick = () => this.fitView();
    this.q('.wl-export-svg').onclick = () => this.download(new Blob([this.exportSVG()], {type: 'image/svg+xml'}), 'svg');
    this.q('.wl-export-png').onclick = () => this.exportPNG().then((blob) => this.download(blob, 'png')).catch((error) => this.fail(error));
    const search = this.q('.wl-search');
    search.oninput = () => { this.searchIndex = -1; this.applySearch(); };
    search.onkeydown = (e) => {
      if (e.key === 'Escape') { search.value = ''; this.applySearch(); return; }
      if (e.key !== 'Enter' || !this.searchMatches?.length) return;
      e.preventDefault();
      this.searchIndex = (this.searchIndex + (e.shiftKey ? -1 : 1) + this.searchMatches.length) % this.searchMatches.length;
      this.selectNode(this.searchMatches[this.searchIndex], {focus: false});
      this.applySearch();
    };
  }
  configure(options = {}) {
    if (options.presentation && !['auto', 'tensors', 'compact'].includes(options.presentation)) throw Error('Invalid presentation');
    if (options.direction && !['TB', 'LR'].includes(options.direction)) throw Error('Invalid direction');
    for (const key of ['nodeRenderer', 'tensorRenderer', 'layout'])
      if (options[key] != null && typeof options[key] !== 'function') throw Error(`${key} must be a function or null`);
    if (['presentation', 'direction', 'layout'].some((key) => key in options && options[key] !== this.options[key])) this.pendingFit = true;
    this.options = {...this.options, ...options, tensorLimits: {...this.options.tensorLimits, ...options.tensorLimits}};
    if (options.theme) this.setTheme(options.theme, false);
    if (this.ready) {
      this.q('.wl-presentation').value = this.options.presentation;
      this.q('.wl-direction').value = this.options.direction;
      if (this.data) this.renderGraph();
    }
    return this;
  }
  setExpanded(expanded) {
    this.expanded = !!expanded;
    this.pendingFit = true;
    if (!this.ready) return this;
    this.q(".wl-expand").setAttribute("aria-pressed", String(this.expanded));
    this.q(".wl-expand").textContent = this.expanded ? this.t("返回模型总览") : this.t("展开内部层");
    if (this.data) this.renderGraph();
    return this;
  }
  setTheme(theme, persist = true) {
    if (!['auto', 'light', 'dark'].includes(theme)) throw Error('Invalid theme');
    this.theme = theme;
    if (persist) { try { localStorage.setItem('weightlang.theme', theme); } catch {} }
    if (this.ready) { this.q('.wl-theme').value = theme; this.applyTheme(); }
    return this;
  }
  applyTheme() {
    this.dataset.theme = this.theme === 'auto' ? (this.media.matches ? 'dark' : 'light') : this.theme;
    if (this.selected) this.draw();
    this.dispatchEvent(new CustomEvent('theme-change', {detail: {theme: this.theme, resolved: this.dataset.theme}, bubbles: true}));
  }
  setTensorAxes(label, axes) {
    if (!tensorGeometry(label, this.options.tensorLimits, axes)) throw Error('Invalid tensor shape');
    this.tensorAxes.set(label, [...axes]);
    if (this.ready && this.data) this.renderGraph();
    return this;
  }
  shapeAxisControls(node) {
    const container = document.createElement('div'); container.className = 'wl-shape-axes';
    const seen = new Set();
    for (const [side, label] of [['input', node.input_shape], ['output', node.output_shape]]) {
      if (seen.has(label)) continue;
      seen.add(label);
      const geometry = tensorGeometry(label, this.options.tensorLimits, this.tensorAxes.get(label));
      if (!geometry || geometry.rank < 2) continue;
      const fieldset = document.createElement('fieldset'); fieldset.dataset.shape = label;
      const legend = document.createElement('legend');
      legend.textContent = (node.input_shape === node.output_shape ? (this.language === 'en' ? 'Input / output' : '输入 / 输出') : this.t(side === 'input' ? '输入' : '输出')) +
        (this.language === 'en' ? ' · up to 3 axes' : ' · 最多显示 3 轴');
      fieldset.append(legend);
      geometry.axes.forEach((axis, index) => {
        const item = document.createElement('label'), input = document.createElement('input');
        input.type = 'checkbox'; input.checked = geometry.displayedAxes.includes(index);
        input.disabled = !input.checked && geometry.displayedAxes.length === 3;
        input.dataset.axis = index;
        input.onchange = () => {
          const selected = geometry.displayedAxes.filter(i => i !== index);
          if (input.checked) selected.push(index);
          this.setTensorAxes(label, selected.sort((a,b) => a-b));
          const replacement = [...this.querySelectorAll('.wl-shape-axes fieldset')].find(e => e.dataset.shape === label);
          replacement?.querySelector(`[data-axis="${index}"]`)?.focus();
        };
        item.append(input, document.createTextNode(` ${axis.name || (this.language === 'en' ? 'Axis ' : '轴 ') + index} · ${axis.size}`));
        fieldset.append(item);
      });
      const note = document.createElement('p'); note.className = 'wl-note';
      note.textContent = this.language === 'en' ? 'Unchecked axes use index 0. Uncheck one to select another.' : '隐藏轴取索引 0；取消一个轴后可选择另一个。';
      fieldset.append(note); container.append(fieldset);
    }
    return container;
  }
  /** Tensor cards in the overview; compact cards for expanded graphs unless chosen explicitly. */
  tensorView() {
    const mode = this.options.presentation;
    return mode === 'tensors' || (mode === 'auto' && !this.expanded);
  }
  renderTensor(label, width, height, side, node) {
    const visibleAxes = this.tensorAxes.get(label);
    const context = {label, visibleAxes, geometry: tensorGeometry(label, this.options.tensorLimits, visibleAxes), width, height, side, node,
      language: this.language, limits: this.options.tensorLimits, svg, defaultRender: renderTensorShape};
    return this.options.tensorRenderer?.(context) || renderTensorShape(context);
  }
  renderNodeContent(node, width, height) {
    const group = svg('g', {class: 'wl-node-content'}), path = node.path || node.id, category = nodeCategory(node);
    const junction = junctionOps.has(node.op), note = this.sourceNotes?.notes.get(node.id);
    const remark = note?.comment?.split('\n')[0] || '';
    group.append(svg('rect', {width, height, rx: junction ? height / 2 : 14, class: 'wl-node-background'}));
    group.append(svg('title', {}, `${categoryTitle(category, this.language)} · ${path}\n${node.input_shape || ''} → ${node.output_shape || ''}${note?.comment ? '\n\n' + note.comment : ''}`));
    if (note?.comment) group.setAttribute('data-commented', 'true');
    // Loops and routes carry a glyph; other nodes a category dot. No borders or badges.
    const glyph = {recurrent: '↻', route: '⑂'}[category];
    const cy = junction ? height / 2 - 7 : 20.5, cx = junction ? 22 : 20;
    if (glyph) group.append(svg('text', {x: cx, y: cy + 5, 'text-anchor': 'middle', class: 'wl-node-glyph'}, glyph));
    else group.append(svg('circle', {cx, cy, r: 4, class: 'wl-node-dot'}));
    group.append(svg('text', {x: cx + 13, y: cy + 5, class: 'wl-node-title'}, nodeTitle(node, this.language)));
    if (junction) {
      const formula = {add: 'a + b', mul: 'a ⊙ g', mix: 'g·a + (1−g)·b', concat: '[ a ‖ b ]', where: 'c ? a : b'}[node.op];
      group.append(svg('text', {x: cx + 13, y: cy + 23, class: 'wl-node-shape'}, formula ? `${formula}  ·  ${node.output_shape || ''}` : node.output_shape || ''));
      return group;
    }
    if (!this.tensorView()) {
      // The source comment replaces the path line when present; the path stays in the tooltip and popup.
      group.append(remark ? svg('text', {x: 16, y: 47, class: 'wl-node-remark'}, fitText(remark, width - 32, 11.5))
        : svg('text', {x: 16, y: 47, class: 'wl-node-op'}, path.length > 36 ? '…' + path.slice(-35) : path));
      group.append(svg('text', {x: 16, y: 68, class: 'wl-node-shape'}, `${node.input_shape || ''} → ${node.output_shape || ''}`));
      if (node.parameters > 0)
        group.append(svg('text', {x: width - 16, y: 25, 'text-anchor': 'end', class: 'wl-node-meta'},
          `${node.parameters.toLocaleString()} ${this.language === 'en' ? 'params' : '参数'}`));
      return group;
    } else {
      const single = node.op === 'input' || node.op === 'output';
      const glyphWidth = single ? width - 40 : (width - 80) / 2;
      const positions = single ? [{x:20,side:node.op,label:node.output_shape}] :
        [{x:16,side:'input',label:node.input_shape},{x:width / 2 + 24,side:'output',label:node.output_shape}];
      for (const {x,side,label} of positions) {
        group.append(svg('text', {x:x + glyphWidth / 2, y:45,'text-anchor':'middle',class:'wl-node-op'},
          this.t(side === 'input' ? '输入' : '输出') + ' · ' + (tensorGeometry(label)?.dtype || '')));
        const visual = this.renderTensor(label, glyphWidth, height - 76, side, node);
        const holder = svg('g', {transform:`translate(${x} 50)`, 'data-side':side});
        holder.append(visual); group.append(holder);
      }
      if (!single) group.append(svg('path', {d:`M${width/2-14},${height/2+8}h26m-6,-5l6,5l-6,5`,class:'wl-transform-arrow'}));
    }
    const meta = `${(node.parameters || 0).toLocaleString()} ${this.language === 'en' ? 'parameters' : '参数'}` +
      (node.members > 1 ? ` · ${node.members - 1} ${this.language === 'en' ? 'inner nodes' : '个内部节点'}` : '');
    if (remark) {
      group.append(svg('text', {x: width - 16, y: 25, 'text-anchor': 'end', class: 'wl-node-meta'}, meta));
      group.append(svg('text', {x: 16, y: height - 10, class: 'wl-node-remark'}, fitText(remark, width - 32, 11.5)));
    } else group.append(svg('text', {x:16,y:height-10,class:'wl-node-meta'}, meta));
    return group;
  }
  t(text) { return translate(text, this.language); }
  localizeStatic() {
    this.lang = this.language;
    this.q('.wl-language').value = this.language;
    for (const [node, original] of this.staticText) {
      node.textContent = original.replace(original.trim(), this.t(original.trim()));
    }
    for (const element of this.querySelectorAll('[data-label]')) {
      const label = this.t(element.dataset.label);
      element.setAttribute('aria-label', label);
      element.setAttribute(element.tagName === 'INPUT' ? 'placeholder' : 'title', label);
    }
    this.q('.wl-zoom').setAttribute('aria-label', this.t('缩放与导出'));
    this.q('.wl-categories').setAttribute('aria-label', this.t('节点类别'));
    this.q('canvas').setAttribute('aria-label', this.t('Tensor values'));
  }
  setLanguage(language) {
    if (!['zh-CN', 'en'].includes(language)) return;
    const tensor = this.selected?.name, fixed = [...this.fixed], range = this.range;
    const row = this.q('.wl-row').value, col = this.q('.wl-column').value, mode = this.q('.wl-mode').value;
    const nodeId = this.q('.wl-selected')?.dataset.id;
    const flow = this.q('.wl-flow'), top = flow.scrollTop, left = flow.scrollLeft;
    this.language = language;
    try { localStorage.setItem('weightlang.language', language); } catch {}
    this.localizeStatic();
    if (!this.data) return;
    this.render();
    const selectedNode = this.nodeElement(nodeId);
    this.reselect(nodeId);
    const hasParameters = !this.q('.wl-plot').hidden;
    if (tensor) {
      this.selectTensor(tensor);
      this.q('.wl-row').value = row; this.q('.wl-column').value = col;
      this.q('.wl-mode').value = mode;
      this.fixed = fixed; this.range = range;
      this.axisControls();
      const preset = this.q('.wl-presets select');
      if (preset) preset.value = range ? String(this.selected.views.indexOf(range)) : '';
      this.draw();
      if (!hasParameters) this.reselect(nodeId);
    }
    this.q('.wl-inspect-toggle').textContent = this.t(this.q('.wl-weight-tools').hidden ? '查看参数值' : '收起参数值');
    flow.scrollTop = top; flow.scrollLeft = left;
  }
  q(s) {
    return this.querySelector(s);
  }
  reselect(id) {
    const element = this.nodeElement(id);
    if (element) this.activateNode(this.nodeIndex.get(id), element, false);
  }
  nodeElement(id) {
    return [...this.querySelectorAll('.wl-node')].find((n) => n.dataset.id === id) || null;
  }
  fail(error) {
    this.q(".wl-error").textContent = this.t(error.message);
  }
  load(data) {
    this.data = validateGraph(data);
    this.tensorAxes.clear();
    this.buffers = {};
    this.selected = null;
    this.expanded = false;
    this.pendingFit = true;
    if (this.ready) this.render();
    return this;
  }
  render() {
    this.q(".wl-error").textContent = "";
    this.q("h1").textContent =
      this.data.model || this.data.title || this.t("Neural network");
    const summary = this.q(".wl-summary"), en = this.language === 'en';
    summary.replaceChildren();
    const brand = document.createElement('span');
    brand.className = 'wl-eyebrow';
    brand.textContent = 'WeightLang';
    summary.append(brand);
    this.sourceNotes = sourceNotes(this.data);
    this.q('.wl-model-title').title = this.sourceNotes.model?.comment || '';
    const facts = [
      `${(this.data.parameters || 0).toLocaleString()}${en ? ' parameters' : ' 个参数'}`,
      `${this.data.nodes.length}${en ? ' internal nodes' : ' 个内部节点'}`,
      (this.data.tensor_data || []).length && this.data.tensor_data.every(t => t.origin === this.t('random-init')) ? (en ? 'initialized, untrained' : '仅初始化，未训练') : '',
      this.data.baseline_sha256 ? (en ? 'baseline comparison' : '基线比较') : '',
    ];
    for (const fact of facts.filter(Boolean)) {
      const item = document.createElement('span'); item.textContent = fact; summary.append(item);
    }
    this.q(".wl-baseline-file-label").hidden = !this.data.baseline_sha256;
    const mode = this.q(".wl-mode");
    mode.replaceChildren();
    mode.add(new Option(this.t("Current weights"), "values"));
    if (this.data.baseline_sha256) {
      mode.add(new Option(this.t("Baseline weights"), "baseline_values"));
      mode.add(new Option(this.t("Change (current − baseline)"), "delta"));
    }
    this.q(".wl-expand").textContent = this.expanded ? this.t("返回模型总览") : this.t("展开内部层");
    this.q(".wl-expand").setAttribute("aria-pressed", String(this.expanded));
    this.renderGraph();
    const tensors = this.data.tensor_data || [],
      select = this.q(".wl-tensor");
    select.replaceChildren();
    for (const t of tensors) select.add(new Option(t.name, t.name));
    if (tensors.length) this.selectTensor(tensors[0].name);
    else {
      this.q(".wl-description").textContent =
        this.t("No embedded parameter data. All nodes and connections remain available.");
      this.q(".wl-controls").hidden = true;
      this.q(".wl-files").hidden = true;
    }
  }
  renderGraph() {
    const selectedId = this.q('.wl-selected')?.dataset.id;
    const style = getComputedStyle(this), tensorView = this.tensorView(),
      width = parseFloat(style.getPropertyValue(tensorView ? '--tensor-node-width' : '--node-width')) || (tensorView ? 420 : 320),
      height = parseFloat(style.getPropertyValue(tensorView ? '--tensor-node-height' : '--node-height')) || (tensorView ? 184 : 88);
    this.q('.wl-shape-note').hidden = !tensorView;
    const direction = this.options.direction;
    const graph = normalizeGraph(this.data);
    const visible = this.expanded ? graph : overviewGraph(graph);
    const pill = {width: Math.min(width, 236), height: tensorView ? 62 : 56};
    const gap = this.expanded ? 54 : direction === 'LR' ? 56 : 28;
    const layout = this.options.layout?.({nodes: visible.nodes, edges: visible.edges, width, height, gap, direction}) ||
      layoutGraph(visible.nodes, visible.edges, width, height, gap, {direction, size: (n) => junctionOps.has(n.op) ? pill : null});
    const viewKey = [this.expanded, direction, tensorView, width, height].join('|');
    if (this.movedNodes?.key !== viewKey) this.movedNodes = {key: viewKey, positions: new Map()};
    for (const [id, moved] of this.movedNodes.positions) {
      const position = layout.positions.get(id);
      if (position) Object.assign(position, moved);
    }
    this.q('.wl-reset-layout').hidden = !this.movedNodes.positions.size;
    const {routes, extent, start} = routeEdges(visible.edges, layout.positions, direction);
    const shift = Math.max(0, 16 - start);
    const canvasWidth = direction === 'LR' ? layout.width : Math.max(layout.width, extent + 24) + shift,
      canvasHeight = direction === 'LR' ? Math.max(layout.height, extent + 24) + shift : layout.height;
    this.visible = visible;
    this.layout = layout;
    this.canvasSize = {width: canvasWidth, height: canvasHeight};
    this.nodeIndex = new Map(visible.nodes.map((n) => [n.id, n]));
    const diagram = svg("svg", {
      viewBox: `0 0 ${canvasWidth} ${canvasHeight}`,
      width: canvasWidth * this.zoom,
      height: canvasHeight * this.zoom,
      role: "group",
      "aria-label": this.t("Model architecture"),
      "data-direction": direction,
    });
    const defs = svg('defs');
    for (const kind of edgeKinds) {
      const marker = svg('marker', {id: `${this.uid}-arrow-${kind}`, viewBox: '0 0 10 10', refX: 9, refY: 5,
        markerWidth: 8, markerHeight: 8, markerUnits: 'userSpaceOnUse', orient: 'auto-start-reverse'});
      marker.append(svg('path', {d: 'M1,1.5 L9,5 L1,8.5 z', class: 'wl-edge-head', 'data-kind': kind}));
      defs.append(marker);
    }
    diagram.append(defs);
    // Control-flow regions sit beneath everything: loop bodies, routes and their branches.
    const regionLayer = svg('g', {class: 'wl-regions'});
    this.regionLayer = regionLayer;
    this.drawRegions();
    const content = svg('g', {class: 'wl-content', transform: direction === 'LR' ? `translate(0 ${shift})` : `translate(${shift} 0)`});
    diagram.append(content);
    content.append(regionLayer);
    const edgeLayer = svg('g', {class: 'wl-edges'}), labelLayer = svg('g', {class: 'wl-edge-labels'});
    const notes = annotateEdges(visible.nodes, visible.edges), roles = new Set();
    this.edgeElements = [];
    this.labelElements = [];
    routes.forEach((route, i) => {
      const role = notes[i].role || kindRoles[route.kind] || 'flow';
      roles.add(role);
      const marker = role === 'gate' ? 'gate' : edgeKinds.includes(route.kind) ? route.kind : 'flow';
      this.edgeElements[i] = edgeLayer.appendChild(svg("path", {
        d: route.d, class: "wl-edge", "data-kind": route.kind, "data-role": role, "data-marker": marker,
        "data-from": route.edge.from, "data-to": route.edge.to, "marker-end": `url(#${this.uid}-arrow-${marker})`,
      }));
      if (notes[i].label && route.label)
        this.labelElements[i] = labelLayer.appendChild(svg('text', {x: route.label[0] + 6, y: route.label[1] + 4, class: 'wl-edge-label', 'data-role': role,
          'data-from': route.edge.from, 'data-to': route.edge.to}, this.t(notes[i].label) + (notes[i].index != null ? ' ' + notes[i].index : '')));
    });
    content.append(edgeLayer);
    for (const item of this.querySelectorAll('.wl-key [data-kind]')) item.hidden = !roles.has(item.dataset.kind);
    const present = new Set();
    for (const node of visible.nodes) {
      const p = layout.positions.get(node.id),
        path = node.path || node.id, category = nodeCategory(node);
      present.add(category);
      const group = svg("g", {
        transform: "translate(" + p.x + " " + p.y + ")",
        class: "wl-node",
        tabindex: "0",
        role: "button",
        "aria-label": `${nodeTitle(node, this.language)} — ${path}`,
        "data-id": node.id,
        "data-op": node.op || "unknown",
        "data-category": category,
        "data-path": path,
        "data-origin": node.origin || "",
        "data-changed": node.changed_parameters > 0 ? "true" : "false",
      });
      const color = categoryRegistry.get(category)?.color;
      if (color) group.style.setProperty('--accent', color);
      const context = {node, width: p.width, height: p.height, language: this.language, svg, category,
        defaultRender: () => this.renderNodeContent(node, p.width, p.height)};
      group.append(this.options.nodeRenderer?.(context) || context.defaultRender());
      group.onclick = () => this.activateNode(node, group, true);
      group.onkeydown = (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          this.activateNode(node, group, true);
        } else if (e.key.startsWith('Arrow')) {
          e.preventDefault();
          this.navigate(node.id, e.key);
        }
      };
      content.append(group);
    }
    // Labels last so they read above nodes and edges.
    content.append(labelLayer);
    const legend = this.q('.wl-categories');
    legend.replaceChildren();
    for (const category of [...categoryRegistry.keys(), ...present].filter((c, i, all) => present.has(c) && all.indexOf(c) === i)) {
      const item = document.createElement('span');
      item.dataset.category = category;
      item.textContent = categoryTitle(category, this.language);
      const color = categoryRegistry.get(category)?.color;
      if (color) item.style.setProperty('--accent', color);
      legend.append(item);
    }
    this.q(".wl-flow").replaceChildren(diagram);
    if (this.pendingFit) this.fitView();
    else this.setZoom(this.zoom);
    this.applySearch();
    if (this.nodeElement(selectedId)) this.reselect(selectedId);
    else this.closePopup();
    this.dispatchEvent(new CustomEvent('graph-render', {detail: {expanded: !!this.expanded, direction, nodes: visible.nodes.length}, bubbles: true}));
  }
  activateNode(node, group, open = false) {
    if (open) this.openPopup();
    const path = node.path || node.id, en = this.language === 'en';
    this.querySelectorAll(".wl-node").forEach((n) => n.classList.toggle("wl-selected", n === group));
    this.markSelection(node.id);
    const info = this.q(".wl-node-info");
    info.replaceChildren();
    const heading = document.createElement('h2');
    heading.textContent = nodeTitle(node, this.language);
    const chip = document.createElement('span');
    chip.className = 'wl-category-chip';
    chip.dataset.category = nodeCategory(node);
    chip.textContent = categoryTitle(chip.dataset.category, this.language);
    const color = categoryRegistry.get(chip.dataset.category)?.color;
    if (color) chip.style.setProperty('--accent', color);
    const pathLine = document.createElement('p');
    pathLine.className = 'wl-node-path'; pathLine.textContent = path;
    const description = document.createElement('p');
    description.className = 'wl-node-description'; description.textContent = nodeDescription(node, this.language);
    info.append(chip, heading, pathLine);
    const note = this.sourceNotes?.notes.get(node.id);
    if (note) info.append(this.sourcePanel(note));
    info.append(description);
    const facts = document.createElement('dl');
    facts.className = 'wl-facts';
    const fact = (label, value) => {
      if (!value) return;
      const term = document.createElement('dt'), detail = document.createElement('dd');
      term.textContent = this.t(label);
      if (value instanceof Node) detail.append(value); else detail.textContent = value;
      facts.append(term, detail);
    };
    fact('形状', `${this.t('输入')} ${node.input_shape || '—'} → ${this.t('输出')} ${node.output_shape || '—'}`);
    fact('参数', `${(node.parameters || 0).toLocaleString()}` + (node.members > 1 ? (en ? ' (including inner layers)' : '（包含内部层）') : ''));
    fact('模块来源', node.block);
    fact('内部算子', node.operations?.length > 1 ? node.operations.map(op => operatorTitle(op, this.language)).join(en ? ', ' : '、') : '');
    fact('配置', node.args && node.args !== '{}' ? node.args : '');
    for (const [label, ids] of [['上游', this.visible.edges.filter(e => e.to === node.id).map(e => e.from)],
      ['下游', this.visible.edges.filter(e => e.from === node.id).map(e => e.to)]]) {
      const unique = [...new Set(ids)];
      if (!unique.length) continue;
      const links = document.createElement('span');
      links.className = 'wl-links';
      for (const id of unique) {
        const target = this.nodeIndex.get(id), button = document.createElement('button');
        button.type = 'button';
        button.className = 'wl-link';
        button.textContent = nodeTitle(target, this.language);
        button.title = target.path || target.id;
        button.onclick = () => this.selectNode(id);
        links.append(button);
      }
      fact(label, links);
    }
    info.append(facts);
    info.append(this.shapeAxisControls(node));
    const matches = (this.data.tensor_data || []).filter(
      (t) => t.owner === path || t.name.startsWith(path + "."),
    );
    if (matches.length) this.selectTensor(matches[0].name);
    for (const selector of [".wl-plot", ".wl-controls", ".wl-fixed", ".wl-presets", ".wl-legend"])
      this.q(selector).hidden = !matches.length;
    if (!matches.length) {
      this.q(".wl-statistics").textContent = this.t("此节点没有可查看的参数张量。");
      this.q(".wl-origin").textContent = "";
      this.q(".wl-description").textContent = "";
      this.q(".wl-readout").textContent = path + ": " + (node.op || "unknown") + " · " + (node.parameters || 0) +
        (en ? ' owned parameters. ' : ' 个直接参数。') + (node.block || "");
    }
    this.dispatchEvent(new CustomEvent("node-select", { detail: node, bubbles: true }));
  }
  /** Comment, block comment and the declaring source line, as written in the WGT code. */
  sourcePanel(note) {
    const panel = document.createElement('div');
    panel.className = 'wl-source';
    // Own comment first; then the block this layer calls; then the block it belongs to.
    const seen = new Set();
    for (const [text, kind, label] of [[note.comment, 'comment', ''],
      [note.blockComment, 'block', note.block ? `${this.t('调用模块')} ${note.block}` : ''],
      [note.parentComment, 'parent', note.parentBlock ? `${this.t('所在模块')} ${note.parentBlock}` : '']]) {
      if (!text || seen.has(text)) continue;
      seen.add(text);
      const quote = document.createElement('p');
      quote.className = 'wl-source-comment';
      quote.dataset.kind = kind;
      if (label) {
        const caption = document.createElement('span');
        caption.className = 'wl-source-caption';
        caption.textContent = label;
        quote.append(caption);
      }
      quote.append(document.createTextNode(text));
      panel.append(quote);
    }
    if (note.code) {
      const code = document.createElement('pre'), where = document.createElement('span');
      code.className = 'wl-source-code';
      where.className = 'wl-source-where';
      where.textContent = `${note.source === 'model' ? this.t('模型源码') : note.source + '.wgt'}${note.line ? ' · L' + note.line : ''}`;
      code.append(where, document.createTextNode(note.code));
      panel.append(code);
    }
    return panel;
  }
  /** Model-level notes: header comment and commented interface declarations. */
  showModelNotes() {
    const model = this.sourceNotes?.model, info = this.q('.wl-node-info');
    this.querySelectorAll('.wl-selected').forEach((n) => n.classList.remove('wl-selected'));
    this.markSelection(null);
    info.replaceChildren();
    const heading = document.createElement('h2');
    heading.textContent = this.data.model || this.data.title || this.t('Neural network');
    info.append(heading);
    if (model) {
      info.append(this.sourcePanel({comment: model.comment, code: model.code, source: 'model', line: model.line}));
      for (const declaration of [...model.interface.input, ...model.interface.output])
        info.append(this.sourcePanel({comment: declaration.comment, code: declaration.code, source: 'model', line: declaration.line}));
    } else {
      const empty = document.createElement('p');
      empty.className = 'wl-note';
      empty.textContent = this.t('此图没有附带源码注释。');
      info.append(empty);
    }
    for (const selector of ['.wl-plot', '.wl-controls', '.wl-fixed', '.wl-presets', '.wl-legend']) this.q(selector).hidden = true;
    this.openPopup();
  }
  drawRegions() {
    const layer = this.regionLayer;
    if (!layer) return;
    layer.replaceChildren();
    if (!this.expanded) return;
    for (const region of graphRegions(this.visible.nodes, this.visible.edges, this.layout.positions)) {
      const group = svg('g', {class: 'wl-region', 'data-kind': region.kind});
      group.append(svg('rect', {x: region.x, y: region.y, width: region.width, height: region.height, rx: 22}));
      const label = region.kind === 'loop' ? `↻  ${nodeTitle(region.node, this.language)} · ${this.t('各步共享权重')}`
        : `⑂  ${this.t('路由')} · ${this.t('只执行被选中的分支')}`;
      group.append(svg('text', {x: region.x + 14, y: region.y + 21, class: 'wl-region-label'}, label));
      layer.append(group);
    }
  }
  /** Discard dragged node positions and re-fit the automatic layout. */
  resetLayout() {
    this.movedNodes = null;
    this.pendingFit = true;
    if (this.ready && this.data) this.renderGraph();
    return this;
  }
  /** Move a node in layout units; its edges, labels and regions follow. */
  moveNode(id, x, y) {
    const position = this.layout?.positions.get(id);
    if (!position) return this;
    Object.assign(position, {x, y});
    this.movedNodes?.positions.set(id, {x, y});
    this.q('.wl-reset-layout').hidden = false;
    this.nodeElement(id)?.setAttribute('transform', `translate(${x} ${y})`);
    const {routes} = routeEdges(this.visible.edges, this.layout.positions, this.options.direction);
    routes.forEach((route, i) => {
      this.edgeElements[i]?.setAttribute('d', route.d);
      const label = this.labelElements[i];
      if (label && route.label) { label.setAttribute('x', route.label[0] + 6); label.setAttribute('y', route.label[1] + 4); }
    });
    this.drawRegions();
    return this;
  }
  /** Emphasize the edges connected to the selected node. */
  markSelection(id) {
    this.q('.wl-flow').classList.toggle('wl-has-selection', id != null);
    for (const edge of this.querySelectorAll('.wl-edge')) {
      const active = edge.dataset.from === id || edge.dataset.to === id;
      edge.classList.toggle('wl-edge-active', active);
      edge.setAttribute('marker-end', `url(#${this.uid}-arrow-${active ? 'active' : edge.dataset.marker || 'flow'})`);
    }
  }
  /** Select a node by id, expanding the graph when the node is internal. */
  selectNode(id, {focus = true} = {}) {
    if (!this.data) return this;
    if (!this.nodeElement(id) && !this.expanded && this.data.nodes.some((n) => n.id === id)) this.setExpanded(true);
    const element = this.nodeElement(id);
    if (!element) return this;
    this.activateNode(this.nodeIndex.get(id), element, true);
    if (focus) element.focus({preventScroll: true});
    this.revealNode(element);
    return this;
  }
  /** Scroll only the canvas (never the page) until the node is fully visible. */
  revealNode(element) {
    const flow = this.q('.wl-flow'), box = element.getBoundingClientRect(), view = flow.getBoundingClientRect(), margin = 16;
    if (box.top < view.top + margin) flow.scrollTop -= view.top + margin - box.top;
    else if (box.bottom > view.bottom - margin) flow.scrollTop += Math.min(box.bottom - view.bottom + margin, box.top - view.top - margin);
    if (box.left < view.left + margin) flow.scrollLeft -= view.left + margin - box.left;
    else if (box.right > view.right - margin) flow.scrollLeft += Math.min(box.right - view.right + margin, box.left - view.left - margin);
  }
  /** Arrow keys follow edges along the flow direction and move between siblings across it. */
  navigate(id, key) {
    const lr = this.options.direction === 'LR', positions = this.layout.positions, p = positions.get(id);
    const main = lr ? 'x' : 'y', cross = lr ? 'y' : 'x';
    const forward = lr ? 'ArrowRight' : 'ArrowDown', backward = lr ? 'ArrowLeft' : 'ArrowUp';
    let candidates;
    if (key === forward) candidates = this.visible.edges.filter((e) => e.from === id && e.kind !== 'recurrent').map((e) => e.to);
    else if (key === backward) candidates = this.visible.edges.filter((e) => e.to === id && e.kind !== 'recurrent').map((e) => e.from);
    else {
      const sign = key === 'ArrowRight' || key === 'ArrowDown' ? 1 : -1;
      candidates = [...positions].filter(([other, q]) => other !== id && q[main] === p[main] && (q[cross] - p[cross]) * sign > 0).map(([other]) => other);
    }
    const target = candidates.filter((c) => c !== id && positions.has(c))
      .sort((a, b) => Math.abs(positions.get(a)[cross] - p[cross]) - Math.abs(positions.get(b)[cross] - p[cross]))[0];
    if (target) this.selectNode(target);
  }
  /** Top-left of the diagram in the canvas scroll space (SVG has no offsetLeft). */
  diagramOffset(flow, diagram) {
    const a = diagram.getBoundingClientRect(), b = flow.getBoundingClientRect();
    return {left: a.left - b.left + flow.scrollLeft, top: a.top - b.top + flow.scrollTop};
  }
  setZoom(zoom, anchor) {
    const flow = this.q('.wl-flow'), diagram = flow?.querySelector(':scope > svg');
    const next = Math.min(3, Math.max(0.1, Number.isFinite(zoom) ? zoom : 1));
    if (!diagram || !this.canvasSize) { this.zoom = next; return this; }
    const previous = parseFloat(diagram.getAttribute('width')) / this.canvasSize.width || next;
    const ax = anchor?.x ?? flow.clientWidth / 2, ay = anchor?.y ?? flow.clientHeight / 2;
    const before = this.diagramOffset(flow, diagram);
    const cx = (flow.scrollLeft + ax - before.left) / previous, cy = (flow.scrollTop + ay - before.top) / previous;
    this.zoom = next;
    diagram.setAttribute('width', this.canvasSize.width * next);
    diagram.setAttribute('height', this.canvasSize.height * next);
    const after = this.diagramOffset(flow, diagram);
    flow.scrollLeft = cx * next + after.left - ax;
    flow.scrollTop = cy * next + after.top - ay;
    this.q('.wl-zoom-level').textContent = Math.round(next * 100) + '%';
    return this;
  }
  /** Fit the whole graph when it stays legible; otherwise fit its width and start at the top. */
  fitView() {
    this.pendingFit = false;
    const flow = this.q('.wl-flow');
    if (!this.canvasSize || !flow?.clientWidth) return this.setZoom(1);
    const top = parseFloat(getComputedStyle(this).getPropertyValue('--stage-top')) || 112, bottom = 64;
    const width = (flow.clientWidth - 48) / this.canvasSize.width, height = (flow.clientHeight - top - bottom) / this.canvasSize.height;
    const whole = Math.min(width, height), across = this.options.direction === 'LR' ? height : width;
    // Below 75% node labels become hard to read; fit the cross axis and scroll along the flow instead.
    this.setZoom(Math.min(1, whole >= 0.75 ? whole : across));
    const diagram = flow.querySelector(':scope > svg'), w = this.canvasSize.width * this.zoom, h = this.canvasSize.height * this.zoom;
    const room = flow.clientHeight - top - bottom;
    const offset = this.diagramOffset(flow, diagram);
    flow.scrollLeft = offset.left + w / 2 - flow.clientWidth / 2;
    flow.scrollTop = offset.top - top - Math.max(0, (room - h) / 2);
    return this;
  }
  applySearch() {
    const input = this.q('.wl-search'), terms = input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const flow = this.q('.wl-flow'), count = this.q('.wl-search-count');
    flow.classList.toggle('wl-searching', terms.length > 0);
    this.searchMatches = [];
    for (const element of this.querySelectorAll('.wl-node')) {
      const node = this.nodeIndex?.get(element.dataset.id);
      if (!node) continue;
      const prefix = (node.path || node.id) + '.';
      // Overview modules also match on their collapsed inner nodes.
      const members = this.expanded ? [node] : [node, ...this.data.nodes.filter((n) => n.path?.startsWith(prefix))];
      // Parameter tensor names are searchable too: an expanded node owns its tensors, a module owns its subtree.
      const path = node.path || node.id;
      const tensors = (this.data.tensor_data || []).filter((t) => this.expanded
        ? t.owner === path || (!t.owner && t.name.startsWith(prefix))
        : t.owner === path || t.owner?.startsWith(prefix) || t.name.startsWith(prefix)).map((t) => t.name);
      const text = [...members.flatMap((n) => [n.id, n.path, n.op, n.block, nodeTitle(n, this.language), n.input_shape, n.output_shape,
        this.sourceNotes?.notes.get(n.id)?.comment]), ...tensors].join(' ').toLowerCase();
      const match = terms.length > 0 && terms.every((term) => text.includes(term));
      element.classList.toggle('wl-match', match);
      element.classList.toggle('wl-current-match', match && this.searchMatches.length === this.searchIndex);
      if (match) this.searchMatches.push(node.id);
    }
    count.textContent = terms.length ? (this.searchMatches.length ? `${Math.max(0, this.searchIndex ?? -1) + 1}/${this.searchMatches.length}` : '0') : '';
  }
  /** Standalone SVG with computed theme colors inlined, independent of viewer.css. */
  exportSVG() {
    const source = this.q('.wl-flow > svg');
    if (!source) return '';
    // Export the plain diagram: search dimming and selection emphasis are view state.
    const flow = this.q('.wl-flow'), transient = ['wl-searching', 'wl-has-selection'].filter((c) => flow.classList.contains(c));
    flow.classList.add('wl-exporting');
    flow.classList.remove(...transient);
    const marked = ['wl-selected', 'wl-edge-active', 'wl-match', 'wl-current-match'].flatMap((name) =>
      [...source.querySelectorAll('.' + name)].map((element) => [element, name]));
    for (const [element, name] of marked) element.classList.remove(name);
    const clone = source.cloneNode(true), originals = [source, ...source.querySelectorAll('*')], copies = [clone, ...clone.querySelectorAll('*')];
    originals.forEach((element, i) => {
      const computed = getComputedStyle(element);
      copies[i].setAttribute('style', exportedStyles.map((name) => `${name}:${computed.getPropertyValue(name)}`).join(';'));
      copies[i].removeAttribute('tabindex');
    });
    flow.classList.add(...transient);
    for (const [element, name] of marked) element.classList.add(name);
    flow.classList.remove('wl-exporting');
    for (const edge of clone.querySelectorAll('.wl-edge'))
      edge.setAttribute('marker-end', `url(#${this.uid}-arrow-${edge.dataset.marker || 'flow'})`);
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    clone.setAttribute('width', this.canvasSize.width);
    clone.setAttribute('height', this.canvasSize.height);
    clone.prepend(svg('rect', {width: '100%', height: '100%', fill: getComputedStyle(this).getPropertyValue('--page-bg').trim() || '#fff'}));
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(clone);
  }
  async exportPNG(scale = 2) {
    const text = this.exportSVG();
    if (!text) throw Error('Nothing to export');
    const {width, height} = this.canvasSize, factor = Math.min(scale, 16000 / Math.max(width, height));
    const url = URL.createObjectURL(new Blob([text], {type: 'image/svg+xml'}));
    try {
      const image = new Image();
      await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(Error('Export failed')); image.src = url; });
      const canvas = document.createElement('canvas');
      canvas.width = Math.ceil(width * factor); canvas.height = Math.ceil(height * factor);
      canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);
      return await new Promise((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(Error('Export failed')), 'image/png'));
    } finally { URL.revokeObjectURL(url); }
  }
  download(blob, extension) {
    const link = document.createElement('a'), name = String(this.data?.model || this.data?.title || 'model').replace(/[^\w.-]+/g, '_');
    link.href = URL.createObjectURL(blob);
    link.download = `${name}-architecture.${extension}`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }
  selectTensor(name) {
    const t = (this.data.tensor_data || []).find((t) => t.name === name);
    if (!t) return;
    for (const selector of ['.wl-plot', '.wl-controls', '.wl-fixed', '.wl-presets', '.wl-legend'])
      this.q(selector).hidden = false;
    this.q(".wl-readout").textContent = this.t("指向图中的单元格，查看坐标和参数值。");
    this.selected = t;
    this.fixed = t.shape.map(() => 0);
    this.range = null;
    this.q(".wl-tensor").value = name;
    this.q(".wl-origin").textContent = this.t(t.origin || "");
    this.q(".wl-statistics").textContent =
      (this.language === 'en' ? 'Shape [' : '形状 [') +
      t.shape.join(" × ") +
      "] · " +
      (t.elements || t.shape.reduce((a, b) => a * b, 1)).toLocaleString() +
      (this.language === 'en' ? ' values · min ' : ' 个值 · 最小 ') +
      num(t.min) +
      (this.language === 'en' ? ' · max ' : ' · 最大 ') +
      num(t.max) +
      (this.language === 'en' ? ' · mean ' : ' · 均值 ') +
      num(t.mean) +
      (t.changed_elements != null
        ? " · " + t.changed_elements + (this.language === 'en' ? ' changed' : ' 个值已变化')
        : "");
    for (const k of ["row", "column"]) {
      const select = this.q(".wl-" + k);
      select.replaceChildren();
      t.shape.forEach((n, i) =>
        select.add(new Option((this.language === 'en' ? 'Axis ' : '轴 ') + i + " (" + n + ")", i)),
      );
      select.value =
        k === "row"
          ? Math.max(0, t.shape.length - 2)
          : Math.max(0, t.shape.length - 1);
    }
    this.q(".wl-controls").hidden = false;
    this.q(".wl-files").hidden = false;
    this.axisControls();
    this.presets();
    this.draw();
    this.dispatchEvent(
      new CustomEvent("tensor-select", { detail: t, bubbles: true }),
    );
  }
  axisControls() {
    const t = this.selected,
      row = +this.q(".wl-row").value,
      col = +this.q(".wl-column").value,
      container = this.q(".wl-fixed");
    container.replaceChildren();
    this.q(".wl-row-control").hidden = t.shape.length < 2;
    this.q(".wl-column-control").hidden = t.shape.length < 2;
    t.shape.forEach((size, axis) => {
      if (axis === row || axis === col) return;
      const label = document.createElement("label");
      label.textContent = (this.language === 'en' ? 'Axis ' : '轴 ') + axis + (this.language === 'en' ? ' slice ' : ' 切片 ');
      const input = document.createElement("input");
      Object.assign(input, {
        type: "number",
        min: 0,
        max: size - 1,
        value: this.fixed[axis],
      });
      input.setAttribute("aria-label", (this.language === 'en' ? 'Axis ' : '轴 ') + axis + (this.language === 'en' ? ' slice' : ' 切片'));
      input.onchange = () => {
        this.fixed[axis] = Math.max(
          0,
          Math.min(size - 1, Math.trunc(+input.value || 0)),
        );
        input.value = this.fixed[axis];
        this.draw();
      };
      label.append(input);
      container.append(label);
    });
  }
  viewLabel(label) {
    if (this.language === 'en') return label;
    const names = {Query: '查询', Key: '键', Value: '值'};
    if (names[label]) return names[label];
    return label.replace(/^(Query|Key|Value) · head (\d+)$/, (_, kind, head) =>
      names[kind] + ' · 注意力头 ' + head);
  }
  presets() {
    const holder = this.q(".wl-presets");
    holder.replaceChildren();
    if (!this.selected.views?.length) return;
    const label = document.createElement("label");
    label.textContent = this.t("Named view ");
    const select = document.createElement("select");
    select.add(new Option(this.t("All values"), ""));
    this.selected.views.forEach((v, i) => select.add(new Option(this.viewLabel(v.label), i)));
    select.onchange = () => {
      this.range =
        select.value === "" ? null : this.selected.views[+select.value];
      if (this.range) {
        this.q(".wl-row").value = "0";
        this.q(".wl-column").value = "1";
        this.axisControls();
      }
      this.draw();
    };
    label.append(select);
    holder.append(label);
  }
  accessor(t, baseline = false) {
    const values = t[baseline ? "baseline_values" : "values"];
    if (values) return (i) => values[i];
    const b = this.buffers[baseline ? "baseline" : "current"],
      entry = b?.header[t.name];
    return entry
      ? (i) => b.view.getFloat32(b.start + entry.data_offsets[0] + i * 4, true)
      : null;
  }
  async loadWeights(file, kind) {
    if (!file) return;
    const bytes = await file.arrayBuffer(),
      hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
        .map((n) => n.toString(16).padStart(2, "0"))
        .join("");
    if (
      hash !==
      (kind === "baseline"
        ? this.data.baseline_sha256
        : this.data.weights_sha256)
    )
      throw Error(this.t("Weights file SHA-256 does not match this model"));
    const view = new DataView(bytes),
      length = Number(view.getBigUint64(0, true)),
      start = 8 + length;
    if (!Number.isSafeInteger(length) || start > bytes.byteLength)
      throw Error(this.t("Invalid safetensors header"));
    const header = JSON.parse(new TextDecoder().decode(bytes.slice(8, start)));
    for (const t of this.data.tensor_data || []) {
      const e = header[t.name];
      if (
        !e ||
        e.dtype !== "F32" ||
        JSON.stringify(e.shape) !== JSON.stringify(t.shape) ||
        e.data_offsets[1] - e.data_offsets[0] !==
          t.shape.reduce((a, b) => a * b, 1) * 4 ||
        start + e.data_offsets[1] > bytes.byteLength
      )
        throw Error(this.t("Weights tensor contract mismatch"));
    }
    this.buffers[kind] = { view, header, start };
    this.q(".wl-error").textContent = "";
    this.draw();
  }
  draw() {
    if (!this.selected || !this.ready) return;
    try {
      const t = this.selected,
        mode = this.q(".wl-mode").value,
        current = this.accessor(t),
        baseline = this.accessor(t, true);
      let get = mode === "baseline_values" ? baseline : current;
      if (mode === "delta")
        get = current && baseline ? (i) => current(i) - baseline(i) : null;
      const row = +this.q(".wl-row").value,
        col = +this.q(".wl-column").value;
      if (get) {
        let shape = t.shape,
          reader = get;
        if (this.range && shape.length === 2) {
          const { start, end } = this.range;
          if (
            !Number.isInteger(start) ||
            !Number.isInteger(end) ||
            start < 0 ||
            end <= start ||
            end > shape[0]
          )
            throw Error(this.t("Invalid named view range"));
          shape = [end - start, shape[1]];
          reader = (i) => get(start * t.shape[1] + i);
        }
        this.plane = sliceTensor(shape, reader, row, col, this.fixed);
        this.plane.offset =
          this.range && t.shape.length === 2
            ? this.range.start * t.shape[1]
            : 0;
        this.q(".wl-description").textContent =
          (t.shape.length < 2
            ? this.t("Vector / scalar")
            : (this.language === 'en' ? 'Axes ' : '轴 ') + row + " × " + col) +
          (this.range ? " · " + this.viewLabel(this.range.label) : "") +
          " · " +
          (this.plane.aggregated
            ? this.t("Each cell is the mean of its coordinate range.")
            : this.t("Each cell is one original float32 parameter."));
      } else {
        const preview =
          mode === "delta"
            ? t.delta_overview
            : mode === "baseline_values"
              ? t.baseline_overview
              : t.overview;
        if (!preview)
          throw Error(this.t("Load matching weights to inspect these values"));
        this.plane = { ...preview, aggregated: true, tiles: [] };
        this.q(".wl-description").textContent =
          this.t("Whole-tensor tile means: axis 0 × flattened remaining axes. Load weights for exact values and arbitrary slices.");
      }
      this.getCurrent = current;
      this.getBaseline = baseline;
      const values = this.plane.values;
      if (!values.length || values.some((v) => !Number.isFinite(v)))
        throw Error(this.t("Non-finite tensor view"));
      this.scale =
        mode === "delta"
          ? Math.max(Math.abs(t.max_abs_delta || 0), ...values.map(Math.abs))
          : Math.max(
              Math.abs(t.min || 0),
              Math.abs(t.max || 0),
              Math.abs(t.baseline_min || 0),
              Math.abs(t.baseline_max || 0),
              ...values.map(Math.abs),
            );
      this.paint();
      this.q(".wl-error").textContent = "";
    } catch (error) {
      this.fail(error);
    }
  }
  paint() {
    const canvas = this.q("canvas"),
      { rows, cols, values } = this.plane,
      width = Math.max(240, this.q(".wl-plot").clientWidth - 24),
      vector = this.selected.shape.length <= 1;
    const height = vector
        ? 240
        : Math.max(220, Math.min(600, (width * rows) / cols)),
      ratio = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.ceil(width * ratio);
    canvas.height = Math.ceil(height * ratio);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);
    const style = getComputedStyle(this);
    const positive =
        style.getPropertyValue("--tensor-positive").trim() || "#b75b35",
      negative =
        style.getPropertyValue("--tensor-negative").trim() || "#7b8554",
      neutral = style.getPropertyValue("--tensor-zero").trim() || "#f4efe5";
    ctx.fillStyle = neutral;
    ctx.fillRect(0, 0, width, height);
    const cw = width / cols,
      ch = height / rows;
    values.forEach((v, i) => {
      const r = Math.floor(i / cols),
        c = i % cols,
        intensity = this.scale ? Math.abs(v) / this.scale : 0;
      ctx.globalAlpha = Math.max(0.035, intensity);
      ctx.fillStyle = v < 0 ? negative : positive;
      if (vector) {
        const h = intensity * (height / 2 - 12);
        ctx.fillRect(
          c * cw + 1,
          v >= 0 ? height / 2 - h : height / 2,
          Math.max(1, cw - 2),
          h,
        );
      } else ctx.fillRect(c * cw, r * ch, cw + 0.5, ch + 0.5);
      ctx.globalAlpha = 1;
      if (!vector && !this.plane.aggregated && cw > 46 && ch > 25) {
        ctx.fillStyle = style.getPropertyValue(intensity > 0.65 ? "--heatmap-contrast" : "--text").trim();
        ctx.font = "11px ui-monospace,monospace";
        ctx.textAlign = "center";
        ctx.fillText(num(v), (c + 0.5) * cw, (r + 0.5) * ch + 4);
      }
    });
    if (vector) {
      ctx.strokeStyle = style.getPropertyValue("--line").trim();
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
    }
    this.q(".wl-low").textContent = num(-this.scale);
    this.q(".wl-high").textContent = num(this.scale);
  }
  inspectCell(event) {
    if (!this.plane) return;
    const bounds = event.currentTarget.getBoundingClientRect(),
      { rows, cols } = this.plane;
    const c = Math.min(
      cols - 1,
      Math.max(
        0,
        Math.floor(((event.clientX - bounds.left) / bounds.width) * cols),
      ),
    );
    const r =
        this.selected.shape.length <= 1
          ? 0
          : Math.min(
              rows - 1,
              Math.max(
                0,
                Math.floor(
                  ((event.clientY - bounds.top) / bounds.height) * rows,
                ),
              ),
            ),
      i = r * cols + c,
      tile = this.plane.tiles?.[i];
    let text = tile
      ? (this.language === 'en' ? 'Coordinates [' : '坐标 [') + tile.coordinates.join(", ") + "]"
      : (this.language === 'en' ? 'Overview tile [' : '总览分块 [') + r + ", " + c + "]";
    if (this.plane.aggregated)
      text +=
        (this.language === 'en' ? ' · mean ' : ' · 均值 ') +
        num(this.plane.values[i]) +
        (tile
          ? (this.language === 'en' ? ' · rows ' : ' · 行 ') +
            tile.row.join("…") +
            (this.language === 'en' ? ', columns ' : '，列 ') +
            tile.col.join("…") +
            " · " +
            tile.count +
            (this.language === 'en' ? ' values' : ' 个值')
          : "");
    else {
      const offset = tile.index + (this.plane.offset || 0);
      text +=
        (this.language === 'en' ? ' · flat index ' : ' · 展平索引 ') +
        offset +
        (this.language === 'en' ? ' · current ' : ' · 当前 ') +
        num(this.getCurrent?.(offset));
      if (this.getBaseline)
        text +=
          (this.language === 'en' ? ' · baseline ' : ' · 基线 ') +
          num(this.getBaseline(offset)) +
          " · Δ " +
          num(this.getCurrent(offset) - this.getBaseline(offset));
    }
    this.q(".wl-readout").textContent = text;
  }
}
if (globalThis.customElements && !customElements.get("weightlang-viewer"))
  customElements.define("weightlang-viewer", WeightLangViewer);
