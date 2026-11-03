const translations = [
  [
    "模型架构",
    "Architecture"
  ],
  [
    "展开内部层",
    "Expand layers"
  ],
  [
    "返回模型总览",
    "Module overview"
  ],
  [
    "适应宽度",
    "Fit width"
  ],
  [
    "从上往下读：输入 → 特征处理 → 输出。点击模块查看说明。形状不含批次维度。",
    "Read top to bottom: input → processing → output. Select a module for details. Shapes omit the batch axis."
  ],
  [
    "实线 → 数据流",
    "Solid → data flow"
  ],
  [
    "虚线 → 残差旁路",
    "Dashed → residual skip"
  ],
  [
    "点线 → 循环记忆",
    "Dotted → recurrent state"
  ],
  [
    "从输入开始看",
    "Start with the input"
  ],
  [
    "选择左侧模块，查看它的作用和形状变化。",
    "Select a module to see its purpose and shape changes."
  ],
  [
    "查看参数值",
    "Inspect weights"
  ],
  [
    "收起参数值",
    "Hide weights"
  ],
  [
    "权重检查",
    "Weight inspector"
  ],
  [
    "参数张量",
    "Parameter"
  ],
  [
    "下图是参数值，不是模型运行时的激活或注意力概率；不能据此判断训练效果。",
    "These are parameters, not runtime activations or attention probabilities. They do not establish model quality."
  ],
  [
    "视图",
    "View"
  ],
  [
    "行",
    "Rows"
  ],
  [
    "列",
    "Columns"
  ],
  [
    "张量值",
    "Tensor values"
  ],
  [
    "指向图中的单元格，查看坐标和参数值。",
    "Point at a cell to inspect its coordinates and value."
  ],
  [
    "完整权重文件（用于精确切片）",
    "Full weights for exact large-tensor slices"
  ],
  [
    "基线权重",
    "Baseline weights"
  ],
  [
    "当前权重",
    "Current weights"
  ],
  [
    "差值（当前 − 基线）",
    "Change (current − baseline)"
  ],
  [
    "此节点没有可查看的参数张量。",
    "This node has no parameter tensors to inspect."
  ],
  [
    "没有内嵌参数数据，仍可查看全部节点与连接。",
    "No embedded parameter data. All nodes and connections remain available."
  ],
  [
    "全部参数值",
    "All values"
  ],
  [
    "命名视图 ",
    "Named view "
  ],
  [
    "向量 / 标量",
    "Vector / scalar"
  ],
  [
    "每格是对应坐标范围的均值。",
    "Each cell is the mean of its coordinate range."
  ],
  [
    "每格是一个原始 float32 参数。",
    "Each cell is one original float32 parameter."
  ],
  [
    "整个张量的分块均值：轴 0 × 其余轴展平。加载权重可查看精确值和任意切片。",
    "Whole-tensor tile means: axis 0 × flattened remaining axes. Load weights for exact values and arbitrary slices."
  ],
  [
    "输入",
    "Input"
  ],
  [
    "输出",
    "Output"
  ],
  [
    "全连接",
    "Linear"
  ],
  [
    "二维卷积",
    "2D convolution"
  ],
  [
    "词嵌入",
    "Token embedding"
  ],
  [
    "位置嵌入",
    "Position embedding"
  ],
  [
    "自注意力",
    "Self-attention"
  ],
  [
    "层归一化",
    "Layer normalization"
  ],
  [
    "全局平均池化",
    "Global average pooling"
  ],
  [
    "最大池化",
    "Max pooling"
  ],
  [
    "残差模块",
    "Residual block"
  ],
  [
    "组合模块",
    "Sequential block"
  ],
  [
    "逐步扫描",
    "Sequence scan"
  ],
  [
    "共享权重循环",
    "Shared-weight loop"
  ],
  [
    "上一步记忆",
    "Previous state"
  ],
  [
    "循环输出",
    "Loop output"
  ],
  [
    "逐元素相加",
    "Elementwise addition"
  ],
  [
    "逐元素相乘",
    "Elementwise multiplication"
  ],
  [
    "门控混合",
    "Gated mix"
  ],
  [
    "拼接",
    "Concatenation"
  ],
  [
    "取最后一步",
    "Last step"
  ],
  [
    "ReLU 激活",
    "ReLU activation"
  ],
  [
    "GELU 激活",
    "GELU activation"
  ],
  [
    "Tanh 激活",
    "Tanh activation"
  ],
  [
    "Sigmoid 激活",
    "Sigmoid activation"
  ],
  [
    "符号嵌入 + 位置编码",
    "Token + position embedding"
  ],
  [
    "Transformer 解码器",
    "Transformer decoder"
  ],
  [
    "Transformer 编码器",
    "Transformer encoder"
  ],
  [
    "门控 Transformer 解码器",
    "Gated Transformer decoder"
  ],
  [
    "GRU 序列记忆",
    "GRU sequence memory"
  ],
  [
    "卷积 + ReLU",
    "Convolution + ReLU"
  ],
  [
    "空间汇聚 + 分类",
    "Pooling + classification"
  ],
  [
    "分类输出层",
    "Classification head"
  ],
  [
    "模块",
    "Module"
  ],
  [
    "把符号编号转换为特征向量，再加入位置信息。",
    "Convert token IDs into feature vectors and add position information."
  ],
  [
    "先用因果自注意力汇聚前面的信息，再用前馈网络处理各位置；两部分都有归一化和残差连接。",
    "Causal self-attention gathers preceding context, then a feed-forward network processes each position. Both stages use normalization and residual connections."
  ],
  [
    "先用双向自注意力交换位置信息，再用前馈网络处理各位置；两部分都有残差连接。",
    "Bidirectional self-attention exchanges information between positions, followed by a feed-forward network. Both stages use residual connections."
  ],
  [
    "逐步读取序列，用重置门和更新门控制记忆。所有时间步共享权重，输出包含每一步的隐藏状态。",
    "Read the sequence step by step, controlling memory with reset and update gates. All steps share weights; the output contains every hidden state."
  ],
  [
    "先提取局部图案，再用 ReLU 保留正响应。",
    "Extract local patterns, then retain positive responses with ReLU."
  ],
  [
    "先对各通道的空间位置取平均，再映射为分类得分。",
    "Average spatial positions in each channel, then project to class scores."
  ],
  [
    "把最后一维映射为各类别的原始得分；这里没有 Softmax。",
    "Map the last dimension to raw class scores. No Softmax is applied here."
  ],
  [
    "对最后一维做线性映射；其他轴保持不变。",
    "Apply a linear map to the last dimension, leaving other axes unchanged."
  ],
  [
    "用共享卷积核提取图片中的局部特征。",
    "Extract local image features using shared convolution kernels."
  ],
  [
    "让序列位置之间交换信息；是否允许看后面的位置由 causal 决定。",
    "Exchange information between sequence positions; causal controls access to later positions."
  ],
  [
    "输入绕过内部计算，与内部结果相加。",
    "Add the input directly to the result of the inner computation."
  ],
  [
    "沿时间轴逐步执行，传递隐藏状态，各步共享同一套权重。",
    "Scan along time, passing hidden state between steps with shared weights."
  ],
  [
    "重复执行内部模块，各轮共享同一套权重。",
    "Repeat the inner module using the same weights each time."
  ],
  [
    "保存上一时间步的隐藏状态，初始值为零。",
    "Hold the previous step's hidden state, initialized to zero."
  ],
  [
    "对每个通道的空间位置取平均，保留通道维度。",
    "Average spatial positions per channel, preserving the channel axis."
  ],
  [
    "取序列最后一个时间步的特征。",
    "Take features from the final sequence step."
  ],
  [
    "按门值 g 混合两个输入：g × a + (1 − g) × b。",
    "Mix two inputs using gate g: g × a + (1 − g) × b."
  ],
  [
    "按连接顺序执行内部层；展开内部层可查看具体计算。",
    "Execute the inner layers in connection order. Expand layers to inspect the computation."
  ],
  [
    "模型接收的数据。形状不包含批次维度。",
    "Data received by the model. The shape omits the batch axis."
  ],
  [
    "模型返回的结果。分类得分不一定是概率。",
    "The model's result. Classification scores are not necessarily probabilities."
  ],
  [
    "按图中的连线处理输入。",
    "Process inputs along the displayed connections."
  ],
  [
    "神经网络",
    "Neural network"
  ],
  [
    "模型架构",
    "Model architecture"
  ],
  [
    "需要节点和连接列表",
    "Expected nodes and edges"
  ],
  [
    "节点 ID 必须是唯一字符串",
    "Node IDs must be unique strings"
  ],
  [
    "连接端点不存在",
    "Missing edge endpoint"
  ],
  [
    "张量形状无效",
    "Invalid tensor shape"
  ],
  [
    "张量过大，无法精确索引",
    "Tensor too large for exact indexing"
  ],
  [
    "张量值无效",
    "Invalid tensor values"
  ],
  [
    "坐标维数不匹配",
    "Coordinate rank mismatch"
  ],
  [
    "坐标超出范围",
    "Coordinate out of bounds"
  ],
  [
    "请选择两个不同的轴",
    "Choose two different axes"
  ],
  [
    "分块上限无效",
    "Invalid tile limit"
  ],
  [
    "权重文件 SHA-256 与模型不符",
    "Weights file SHA-256 does not match this model"
  ],
  [
    "safetensors 文件头无效",
    "Invalid safetensors header"
  ],
  [
    "权重张量契约不匹配",
    "Weights tensor contract mismatch"
  ],
  [
    "命名视图范围无效",
    "Invalid named view range"
  ],
  [
    "请加载匹配的权重以查看这些值",
    "Load matching weights to inspect these values"
  ],
  [
    "张量视图包含非有限值",
    "Non-finite tensor view"
  ],
  [
    "随机初始化",
    "random-init"
  ],
  [
    "已训练",
    "trained"
  ],
  [
    "已导入",
    "imported"
  ]
];
translations.push(['主题', 'Theme'], ['跟随系统', 'System'], ['亮色', 'Light'], ['暗色', 'Dark'],
  ['张量视图', 'Tensor view'], ['紧凑视图', 'Compact view'],
  ['格子表示元素位置，不是运行值；省略号表示只展示部分元素。', 'Cells show element positions, not runtime values; … indicates omitted elements.']);
const translationIndex = new Map(translations.flatMap(([zh, en]) => [[zh, [zh, en]], [en, [zh, en]]]));
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
const operatorNames = {
  input: '输入', output: '输出', linear: '全连接', conv2d: '二维卷积',
  embedding: '词嵌入', position_embedding: '位置嵌入', self_attention: '自注意力',
  layernorm: '层归一化', globalavgpool2d: '全局平均池化', maxpool2d: '最大池化',
  residual: '残差模块', seq: '组合模块', scan: '逐步扫描', recur: '共享权重循环',
  state: '上一步记忆', loop_output: '循环输出', add: '逐元素相加',
  mul: '逐元素相乘', mix: '门控混合', concat: '拼接', last: '取最后一步',
  relu: 'ReLU 激活', gelu: 'GELU 激活', tanh: 'Tanh 激活', sigmoid: 'Sigmoid 激活',
};
function nodeTitle(node, language = 'zh-CN') {
  const blocks = {
    'std/transformer.tokens': '符号嵌入 + 位置编码',
    'std/transformer.decoder': 'Transformer 解码器',
    'std/transformer.encoder': 'Transformer 编码器',
    'std/transformer.gated_decoder': '门控 Transformer 解码器',
    'std/recurrent.gru': 'GRU 序列记忆',
    'std/vision.conv_relu': '卷积 + ReLU',
    'std/vision.classifier': '空间汇聚 + 分类',
    'std/heads.classifier': '分类输出层',
  };
  return translate(blocks[node.block] || operatorNames[node.op] || node.op || '模块', language);
}
export function nodeDescription(node, language = 'zh-CN') {
  const blocks = {
    'std/transformer.tokens': '把符号编号转换为特征向量，再加入位置信息。',
    'std/transformer.decoder': '先用因果自注意力汇聚前面的信息，再用前馈网络处理各位置；两部分都有归一化和残差连接。',
    'std/transformer.encoder': '先用双向自注意力交换位置信息，再用前馈网络处理各位置；两部分都有残差连接。',
    'std/recurrent.gru': '逐步读取序列，用重置门和更新门控制记忆。所有时间步共享权重，输出包含每一步的隐藏状态。',
    'std/vision.conv_relu': '先提取局部图案，再用 ReLU 保留正响应。',
    'std/vision.classifier': '先对各通道的空间位置取平均，再映射为分类得分。',
    'std/heads.classifier': '把最后一维映射为各类别的原始得分；这里没有 Softmax。',
  };
  if (blocks[node.block]) return translate(blocks[node.block], language);
  const descriptions = {
    linear: '对最后一维做线性映射；其他轴保持不变。',
    conv2d: '用共享卷积核提取图片中的局部特征。',
    self_attention: '让序列位置之间交换信息；是否允许看后面的位置由 causal 决定。',
    residual: '输入绕过内部计算，与内部结果相加。',
    scan: '沿时间轴逐步执行，传递隐藏状态，各步共享同一套权重。',
    recur: '重复执行内部模块，各轮共享同一套权重。',
    state: '保存上一时间步的隐藏状态，初始值为零。',
    globalavgpool2d: '对每个通道的空间位置取平均，保留通道维度。',
    last: '取序列最后一个时间步的特征。',
    mix: '按门值 g 混合两个输入：g × a + (1 − g) × b。',
    seq: '按连接顺序执行内部层；展开内部层可查看具体计算。',
    input: '模型接收的数据。形状不包含批次维度。',
    output: '模型返回的结果。分类得分不一定是概率。',
  };
  return descriptions[node.op] ? translate(descriptions[node.op], language) : `${nodeTitle(node, language)}: ${translate('按图中的连线处理输入。', language)}`;
}
export function layoutGraph(nodes, edges, width = 260, height = 76, gap = 54) {
  const incoming = new Map(nodes.map((n) => [n.id, 0])),
    outgoing = new Map(nodes.map((n) => [n.id, []]));
  const ranks = new Map(nodes.map((n) => [n.id, 0])),
    done = new Set();
  for (const e of edges) {
    if (e.kind === 'recurrent') continue;
    incoming.set(e.to, incoming.get(e.to) + 1);
    outgoing.get(e.from).push(e.to);
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
  const columns = Math.max(1, ...[...rows.values()].map((r) => r.length)),
    positions = new Map();
  for (const [r, row] of rows)
    row.forEach((n, c) =>
      positions.set(n.id, {
        x: 24 + ((columns - row.length) / 2 + c) * (width + 28),
        y: 28 + r * (height + gap),
        width,
        height,
      }),
    );
  return {
    positions,
    width: columns * (width + 28) + 40,
    height: (last + 1) * (height + gap) + 32,
  };
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
const template =
  '<header class="wl-header"><div><span class="wl-eyebrow">WEIGHTLANG / MODEL EXPLORER</span><h1></h1></div><div class="wl-header-tools"><label><span>主题</span> <select class="wl-theme" aria-label="Theme / 主题"><option value="auto">跟随系统</option><option value="light">亮色</option><option value="dark">暗色</option></select></label><div class="wl-summary"></div><label>Language / 语言 <select class="wl-language" aria-label="Language / 语言"><option value="zh-CN">中文</option><option value="en">English</option></select></label></div></header><div class="wl-workspace"><section class="wl-architecture"><div class="wl-panel-heading"><h2>模型架构</h2><div class="wl-actions"><select class="wl-presentation" aria-label="Display / 显示"><option value="tensors">张量视图</option><option value="compact">紧凑视图</option></select><button class="wl-expand" aria-pressed="false">展开内部层</button><button class="wl-fit">适应宽度</button></div></div><p class="wl-note">从上往下读：输入 → 特征处理 → 输出。点击模块查看说明。形状不含批次维度。</p><div class="wl-key"><span>实线 → 数据流</span><span>虚线 → 残差旁路</span><span>点线 → 循环记忆</span></div><p class="wl-shape-note">格子表示元素位置，不是运行值；省略号表示只展示部分元素。</p><div class="wl-flow"></div></section><section class="wl-explorer"><div class="wl-node-info" aria-live="polite"><h2>从输入开始看</h2><p>选择左侧模块，查看它的作用和形状变化。</p></div><button class="wl-inspect-toggle" aria-expanded="false">查看参数值</button><div class="wl-weight-tools" hidden><div class="wl-panel-heading"><h2>权重检查</h2><span class="wl-origin"></span></div><label class="wl-tensor-label">参数张量<select class="wl-tensor"></select></label><div class="wl-statistics"></div><div class="wl-controls"><label>View <select class="wl-mode"></select></label><label class="wl-row-control">Rows <select class="wl-row"></select></label><label class="wl-column-control">Columns <select class="wl-column"></select></label></div><div class="wl-fixed"></div><div class="wl-presets"></div><p class="wl-description"></p><div class="wl-plot"><canvas tabindex="0" aria-label="Tensor values"></canvas></div><div class="wl-legend"><span class="wl-low"></span><span class="wl-gradient"></span><span class="wl-high"></span></div><output class="wl-readout" aria-live="polite">Point at a cell to inspect its coordinates and value.</output><div class="wl-files"><label>Full weights for exact large-tensor slices<input class="wl-current-file" type="file" accept=".safetensors"></label><label class="wl-baseline-file-label">Baseline weights<input class="wl-baseline-file" type="file" accept=".safetensors"></label></div><p class="wl-note">下图是参数值，不是模型运行时的激活或注意力概率；不能据此判断训练效果。</p></div></section></div><p class="wl-error" role="alert"></p>';
export class WeightLangViewer extends (globalThis.HTMLElement || class {}) {
  constructor() {
    super();
    this.language = globalThis.navigator?.language?.startsWith('zh') ? 'zh-CN' : 'en';
    try { const saved = localStorage.getItem('weightlang.language'); if (['zh-CN', 'en'].includes(saved)) this.language = saved; } catch {}
    this.options = {presentation: 'tensors', tensorLimits: {rows: 6, columns: 8, planes: 3}};
    this.tensorAxes = new Map();
    this.theme = 'auto';
    try { const saved = localStorage.getItem('weightlang.theme'); if (['auto', 'light', 'dark'].includes(saved)) this.theme = saved; } catch {}
    this.buffers = {};
    this.fixed = [];
    this.range = null;
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
      this.q('.wl-presentation').value = this.options.presentation;
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
      this.q(".wl-expand").onclick = () => {
        this.expanded = !this.expanded;
        this.q(".wl-expand").setAttribute("aria-pressed", String(this.expanded));
        this.q(".wl-expand").textContent = this.expanded ? this.t("返回模型总览") : this.t("展开内部层");
        this.renderGraph();
      };
      this.q(".wl-fit").onclick = () =>
        this.q(".wl-flow").classList.toggle("wl-fitted");
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
  configure(options = {}) {
    if (options.presentation && !['tensors', 'compact'].includes(options.presentation)) throw Error('Invalid presentation');
    for (const key of ['nodeRenderer', 'tensorRenderer'])
      if (options[key] != null && typeof options[key] !== 'function') throw Error(`${key} must be a function or null`);
    this.options = {...this.options, ...options, tensorLimits: {...this.options.tensorLimits, ...options.tensorLimits}};
    if (options.theme) this.setTheme(options.theme, false);
    if (this.ready) {
      this.q('.wl-presentation').value = this.options.presentation;
      if (this.data) this.renderGraph();
    }
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
  renderTensor(label, width, height, side, node) {
    const visibleAxes = this.tensorAxes.get(label);
    const context = {label, visibleAxes, geometry: tensorGeometry(label, this.options.tensorLimits, visibleAxes), width, height, side, node,
      language: this.language, limits: this.options.tensorLimits, svg, defaultRender: renderTensorShape};
    return this.options.tensorRenderer?.(context) || renderTensorShape(context);
  }
  renderNodeContent(node, width, height) {
    const group = svg('g', {class: 'wl-node-content'}), path = node.path || node.id;
    group.append(svg('rect', {width, height, rx: 12, class: 'wl-node-background'}));
    group.append(svg('title', {}, `${path}\n${node.input_shape || ''} → ${node.output_shape || ''}`));
    group.append(svg('text', {x: 16, y: 25, class: 'wl-node-title'}, nodeTitle(node, this.language)));
    if (this.options.presentation === 'compact') {
      group.append(svg('text', {x: 16, y: 46, class: 'wl-node-op'}, path.length > 36 ? '…' + path.slice(-35) : path));
      group.append(svg('text', {x: 16, y: 65, class: 'wl-node-shape'}, `${node.input_shape || ''} → ${node.output_shape || ''}`));
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
    group.append(svg('text', {x:16,y:height-10,class:'wl-node-meta'},
      `${(node.parameters || 0).toLocaleString()} ${this.language === 'en' ? 'parameters' : '参数'}` +
      (node.members > 1 ? ` · ${node.members - 1} ${this.language === 'en' ? 'inner nodes' : '个内部节点'}` : '')));
    return group;
  }
  t(text) { return translate(text, this.language); }
  localizeStatic() {
    this.lang = this.language;
    this.q('.wl-language').value = this.language;
    for (const [node, original] of this.staticText) {
      node.textContent = original.replace(original.trim(), this.t(original.trim()));
    }
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
    const selectedNode = [...this.querySelectorAll('.wl-node')].find(n => n.dataset.id === nodeId);
    selectedNode?.dispatchEvent(new Event('click'));
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
      if (!hasParameters) selectedNode?.dispatchEvent(new Event('click'));
    }
    this.q('.wl-inspect-toggle').textContent = this.t(this.q('.wl-weight-tools').hidden ? '查看参数值' : '收起参数值');
    flow.scrollTop = top; flow.scrollLeft = left;
  }
  q(s) {
    return this.querySelector(s);
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
    if (this.ready) this.render();
    return this;
  }
  render() {
    this.q(".wl-error").textContent = "";
    this.q("h1").textContent =
      this.data.model || this.data.title || this.t("Neural network");
    this.q(".wl-summary").textContent =
      (this.data.parameters || 0).toLocaleString() +
      (this.language === 'en' ? ' parameters · ' : ' 个参数 · ') +
      this.data.nodes.length +
      (this.language === 'en' ? ' internal nodes' : ' 个内部节点') +
      ((this.data.tensor_data || []).length && this.data.tensor_data.every(t => t.origin === this.t('random-init')) ? (this.language === 'en' ? ' · initialized, untrained' : ' · 仅初始化，未训练') : '') +
      (this.data.baseline_sha256 ? (this.language === 'en' ? ' · baseline comparison' : ' · 基线比较') : "");
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
    const style = getComputedStyle(this), tensorView = this.options.presentation === 'tensors',
      width = parseFloat(style.getPropertyValue(tensorView ? '--tensor-node-width' : '--node-width')) || (tensorView ? 420 : 320),
      height = parseFloat(style.getPropertyValue(tensorView ? '--tensor-node-height' : '--node-height')) || (tensorView ? 184 : 88);
    this.q('.wl-shape-note').hidden = !tensorView;
    const visible = this.expanded ? this.data : overviewGraph(this.data);
    const layout = layoutGraph(visible.nodes, visible.edges, width, height, this.expanded ? 54 : 20);
    const diagram = svg("svg", {
      viewBox: "0 0 " + layout.width + " " + layout.height,
      width: layout.width,
      height: layout.height,
      role: "group",
      "aria-label": this.t("Model architecture"),
    });
    for (const edge of visible.edges) {
      const a = layout.positions.get(edge.from),
        b = layout.positions.get(edge.to),
        x = a.x + width / 2,
        y = a.y + height,
        u = b.x + width / 2,
        v = b.y;
      const skip = edge.kind === "skip" || edge.kind === "recurrent" || v <= y;
      const d = skip
        ? "M" +
          (a.x + width) +
          "," +
          (a.y + height / 2) +
          " C" +
          layout.width +
          "," +
          (a.y + height / 2) +
          " " +
          layout.width +
          "," +
          (b.y + height / 2) +
          " " +
          (b.x + width) +
          "," +
          (b.y + height / 2)
        : "M" +
          x +
          "," +
          y +
          " C" +
          x +
          "," +
          (y + v) / 2 +
          " " +
          u +
          "," +
          (y + v) / 2 +
          " " +
          u +
          "," +
          v;
      diagram.append(
        svg("path", {
          d,
          class: "wl-edge",
          "data-kind": edge.kind || "flow",
        }),
      );
      if (!skip)
        diagram.append(
          svg("path", {
            d:
              "M" +
              (u - 4) +
              "," +
              (v - 6) +
              " L" +
              u +
              "," +
              v +
              " L" +
              (u + 4) +
              "," +
              (v - 6),
            class: "wl-edge",
          }),
        );
    }
    for (const node of visible.nodes) {
      const p = layout.positions.get(node.id),
        path = node.path || node.id;
      const group = svg("g", {
        transform: "translate(" + p.x + " " + p.y + ")",
        class: "wl-node",
        tabindex: "0",
        role: "button",
        "aria-label": path,
        "data-id": node.id,
        "data-op": node.op || "unknown",
        "data-path": path,
        "data-origin": node.origin || "",
        "data-changed": node.changed_parameters > 0 ? "true" : "false",
      });
      const context = {node, width, height, language: this.language, svg,
        defaultRender: () => this.renderNodeContent(node, width, height)};
      group.append(this.options.nodeRenderer?.(context) || context.defaultRender());
      const activate = () => {
        this.querySelectorAll(".wl-node").forEach((n) =>
          n.classList.toggle("wl-selected", n === group),
        );
        const info = this.q(".wl-node-info");
        info.replaceChildren();
        const heading = document.createElement('h2');
        heading.textContent = nodeTitle(node, this.language);
        info.append(heading);
        for (const text of [path, node.block ? `${this.language === 'en' ? 'Module: ' : '模块来源：'}${node.block}` : '', nodeDescription(node, this.language),
          `${this.t('输入')} ${node.input_shape || '—'} → ${this.t('输出')} ${node.output_shape || '—'}`,
          `${(node.parameters || 0).toLocaleString()} ${this.language === 'en' ? 'parameters' : '个参数'}` + (node.members > 1 ? (this.language === 'en' ? ' (including inner layers)' : '（包含内部层）') : ''),
          node.operations?.length > 1 ? `${this.language === 'en' ? 'Operators: ' : '内部算子：'}${node.operations.map(op => this.t(operatorNames[op] || op)).join(this.language === 'en' ? ', ' : '、')}` : '',
          node.args && node.args !== '{}' ? `${this.language === 'en' ? 'Configuration: ' : '配置：'}${node.args}` : '']) {
          if (!text) continue;
          const line = document.createElement('p'); line.textContent = text; info.append(line);
        }
        info.append(this.shapeAxisControls(node));
        const matches = (this.data.tensor_data || []).filter(
          (t) =>
            t.owner === path || t.name.startsWith(path + "."),
        );
        if (matches.length) this.selectTensor(matches[0].name);
        this.q(".wl-plot").hidden = !matches.length;
        this.q(".wl-controls").hidden = !matches.length;
        this.q(".wl-fixed").hidden = !matches.length;
        this.q(".wl-presets").hidden = !matches.length;
        this.q(".wl-legend").hidden = !matches.length;
        if (!matches.length) {
          this.q(".wl-statistics").textContent = this.t("此节点没有可查看的参数张量。");
          this.q(".wl-origin").textContent = "";
          this.q(".wl-description").textContent = "";
        }
        if (!matches.length)
          this.q(".wl-readout").textContent =
            path +
            ": " +
            (node.op || "unknown") +
            " · " +
            (node.parameters || 0) +
            (this.language === 'en' ? ' owned parameters. ' : ' 个直接参数。') +
            (node.block || "");
        this.dispatchEvent(
          new CustomEvent("node-select", { detail: node, bubbles: true }),
        );
      };
      group.onclick = activate;
      group.onkeydown = (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          activate();
        }
      };
      diagram.append(group);
    }
    this.q(".wl-flow").replaceChildren(diagram);
    ([...this.querySelectorAll('.wl-node')].find(n => n.dataset.id === selectedId) || this.q('.wl-node'))?.dispatchEvent(new Event('click'));
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
