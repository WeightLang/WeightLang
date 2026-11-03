import test from "node:test";
import assert from "node:assert/strict";
import {
  validateGraph,
  layoutGraph,
  flatIndex,
  sliceTensor,
} from "../viewer.js";
test("unknown operators, branches and cycles retain every node", () => {
  const graph = validateGraph({
    nodes: [
      { id: "a", op: "future_operator" },
      { id: "b" },
      { id: "c" },
      { id: "d" },
    ],
    edges: [
      { from: "a", to: "b" },
      { from: "a", to: "c" },
      { from: "b", to: "d" },
      { from: "c", to: "d" },
    ],
  });
  const { positions } = layoutGraph(graph.nodes, graph.edges);
  assert.equal(positions.size, 4);
  assert.equal(positions.get("b").y, positions.get("c").y);
  assert.notEqual(positions.get("b").x, positions.get("c").x);
  const cycle = layoutGraph(graph.nodes, [
    ...graph.edges,
    { from: "d", to: "a" },
  ]);
  assert.equal(cycle.positions.size, 4);
  assert.equal(
    new Set([...cycle.positions.values()].map((p) => p.x + "," + p.y)).size,
    4,
  );
});
test("bad graphs fail rather than silently losing data", () => {
  assert.throws(
    () => validateGraph({ nodes: [{ id: "a" }, { id: "a" }], edges: [] }),
    /unique/,
  );
  assert.throws(
    () =>
      validateGraph({ nodes: [{ id: "a" }], edges: [{ from: "a", to: "b" }] }),
    /endpoint/,
  );
  assert.throws(
    () =>
      validateGraph({
        nodes: [],
        edges: [],
        tensor_data: [{ shape: [2], values: [1] }],
      }),
    /values/,
  );
});
test("arbitrary rank slicing uses exact row-major coordinates", () => {
  const shape = [2, 3, 4, 5, 2],
    fixed = [1, 0, 2, 0, 1];
  assert.equal(flatIndex(shape, [1, 2, 3, 4, 1]), 239);
  const view = sliceTensor(shape, (i) => i, 1, 3, fixed);
  assert.equal(view.rows, 3);
  assert.equal(view.cols, 5);
  assert.equal(view.values[0], flatIndex(shape, [1, 0, 2, 0, 1]));
  assert.equal(view.values[14], flatIndex(shape, [1, 2, 2, 4, 1]));
  const transpose = sliceTensor([2, 3], (i) => i, 1, 0, []);
  assert.deepEqual(transpose.values, [0, 3, 1, 4, 2, 5]);
});
test("scalar, vector and bounded tile means have explicit semantics", () => {
  assert.deepEqual(sliceTensor([], () => 7, 0, 0, []).values, [7]);
  const vector = sliceTensor([4], (i) => i, 0, 0, [], 2);
  assert.deepEqual(vector.values, [0.5, 2.5]);
  assert.equal(vector.aggregated, true);
  const matrix = sliceTensor([4, 4], (i) => i, 0, 1, [], 2);
  assert.deepEqual(matrix.values, [2.5, 4.5, 10.5, 12.5]);
  assert.equal(matrix.tiles[0].count, 4);
  assert.throws(() => sliceTensor([2, 3], (i) => i, 0, 0, []), /different/);
  assert.throws(() => flatIndex([2, 3], [2, 0]), /bounds/);
});

test('overview preserves branches and aggregates module parameters exactly once', async () => {
  const {overviewGraph} = await import('../viewer.js');
  const graph = {nodes: [
    {id:'i',path:'input',parent:'',parameters:0},
    {id:'m',path:'body',parent:'',op:'residual',parameters:0},
    {id:'a',path:'body.a',parent:'body',parameters:12},
    {id:'b',path:'body.b',parent:'body',parameters:4},
    {id:'sum',path:'body.$add',parent:'',parameters:0},
    {id:'o',path:'output',parent:'',parameters:0}],
    edges: [{from:'i',to:'m'},{from:'m',to:'a'},{from:'a',to:'b'},
      {from:'b',to:'sum'},{from:'m',to:'sum',kind:'skip'},{from:'sum',to:'o'}]};
  const result = overviewGraph(graph);
  assert.deepEqual(result.nodes.map(n=>n.id), ['i','m','o']);
  assert.equal(result.nodes[1].parameters,16);
  assert.deepEqual(result.edges.map(e=>[e.from,e.to]), [['i','m'],['m','o']]);
});
test('recurrent feedback does not destroy feed-forward ranks', () => {
  const nodes = ['input','state','gate','result'].map(id=>({id}));
  const edges = [{from:'input',to:'gate'}, {from:'state',to:'gate'},
    {from:'gate',to:'result'}, {from:'result',to:'state',kind:'recurrent'}];
  const {positions:p} = layoutGraph(nodes,edges);
  assert.equal(p.get('input').y,p.get('state').y);
  assert.ok(p.get('gate').y < p.get('result').y);
});

test('translation preserves identifiers and supports both directions', async () => {
  const {translate,nodeDescription}=await import('../viewer.js');
  assert.equal(translate('展开内部层','en'),'Expand layers');
  assert.equal(translate('Current weights','zh-CN'),'当前权重');
  assert.equal(translate('body.attention.qkv.weight','zh-CN'),'body.attention.qkv.weight');
  assert.match(nodeDescription({op:'seq',block:'std/transformer.decoder'},'en'),/Causal self-attention/);
  assert.match(nodeDescription({op:'unknown'},'en'),/Process inputs/);
});

test('tensor geometry bounds dense views and preserves complete named shapes', async () => {
  const {tensorGeometry:g} = await import('../viewer.js');
  assert.deepEqual(g('i64[4]').shape,[4]);
  assert.equal(g('f32[4,8]').truncated,false);
  const image=g('f32[channel=4,height=32,width=32]');
  assert.deepEqual(image.axes.map(a=>a.name),['channel','height','width']);
  assert.equal(image.shownPlanes,3);
  assert.equal(image.shownRows,6);
  assert.equal(image.shownCols,8);
  assert.equal(image.truncated,true);
  assert.equal(g('f32[]').rank,0);
  assert.equal(g('f32[2,3,4,5,6]').planes,2);
  assert.deepEqual(g('f32[2,3,4,5,6]').displayedAxes,[0,1,2]);
  assert.deepEqual(g('f32[2,3,4,5,6]').hiddenAxes,[3,4]);
  assert.deepEqual(g('f32[2,3,4,5,6]',{},[0,2,4]).displayedShape,[2,4,6]);
  assert.throws(()=>g('f32[2,3,4,5]',{},[0,1,2,3]),/at most three/);
  assert.throws(()=>g('f32[2,3,4]',{},[0,0]),/distinct/);
  assert.equal(g('f32[0,2]'),null);
  assert.equal(g('unavailable'),null);
  assert.equal(g('f32[100,100]',{rows:Infinity,columns:500}).shownCols,16);
});

test('left-to-right layout swaps axes and barycenter ordering removes crossings', async () => {
  const {layoutGraph} = await import('../viewer.js');
  const nodes = ['a','b','x','y'].map(id=>({id}));
  // a→y and b→x cross when ranks keep insertion order; sweeps should untangle them.
  const edges = [{from:'a',to:'y'},{from:'b',to:'x'}];
  const tb = layoutGraph([...nodes], edges, 100, 50, 20);
  assert.ok((tb.positions.get('a').x < tb.positions.get('b').x) === (tb.positions.get('y').x < tb.positions.get('x').x));
  const lr = layoutGraph(nodes, edges, 100, 50, 20, {direction:'LR'});
  assert.equal(lr.direction,'LR');
  assert.equal(lr.positions.get('a').x, lr.positions.get('b').x);
  assert.ok(lr.positions.get('y').x > lr.positions.get('a').x);
});

test('bypass edges get separate lanes and never route through nodes', async () => {
  const {layoutGraph,routeEdges} = await import('../viewer.js');
  const nodes = ['a','b','c','d'].map(id=>({id}));
  const edges = [{from:'a',to:'b'},{from:'b',to:'c'},{from:'c',to:'d'},
    {from:'a',to:'c',kind:'skip'},{from:'b',to:'d',kind:'skip'},{from:'a',to:'d'}];
  const {positions} = layoutGraph(nodes, edges, 100, 50, 20);
  const {routes, extent} = routeEdges(edges, positions);
  const lanes = routes.filter(r=>r.lane!=null);
  // Two overlapping skips plus a long forward edge that would cross b and c.
  assert.equal(lanes.length, 3);
  assert.equal(new Set(lanes.map(r=>r.lane)).size, 3);
  assert.ok(extent > 100 + 24);
  assert.ok(routes.every(r=>/^M[\d.]+,[\d.]+/.test(r.d)));
});

test('registries extend operators, blocks and categories without core changes', async () => {
  const {registerOperator,registerBlock,registerCategory,nodeTitle,nodeDescription,nodeCategory,categoryTitle,addTranslations} = await import('../viewer.js');
  registerCategory('quant', {title:{'zh-CN':'量化',en:'Quantization'}, color:'#123456'});
  registerOperator('fake_quant', {title:{'zh-CN':'伪量化',en:'Fake quant'}, category:'quant',
    description:(node,language)=>language==='en'?`${node.args} bits`:`${node.args} 位`});
  assert.equal(nodeTitle({op:'fake_quant'},'en'),'Fake quant');
  assert.equal(nodeDescription({op:'fake_quant',args:'8'},'zh-CN'),'8 位');
  assert.equal(nodeCategory({op:'fake_quant'}),'quant');
  assert.equal(categoryTitle('quant','en'),'Quantization');
  addTranslations([['专家混合','Mixture of experts']]);
  registerBlock('acme/moe', {title:'专家混合', category:'dense'});
  assert.equal(nodeTitle({op:'seq',block:'acme/moe'},'en'),'Mixture of experts');
  assert.equal(nodeCategory({op:'linear'}),'dense');
  assert.equal(nodeCategory({op:'never_seen'}),'other');
  assert.throws(()=>registerOperator('x',{category:'Bad Name'}),/Category/);
});
