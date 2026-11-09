// A minimal artifact-graph model for exercising the real `wgt graph` renderer.
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');

const output = process.argv[2];
if (!output) throw Error('Provide a new fixture artifact directory');
fs.mkdirSync(output);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const tensors = {
  'fc.bias': {dtype: 'F32', shape: [2], data_offsets: [0, 8]},
  'fc.weight': {dtype: 'F32', shape: [2, 2], data_offsets: [8, 24]},
};
let header = Buffer.from(JSON.stringify(tensors));
header = Buffer.concat([header, Buffer.alloc((8 - header.length % 8) % 8, 32)]);
const length = Buffer.alloc(8);
length.writeBigUInt64LE(BigInt(header.length));
const weights = Buffer.concat([length, header, Buffer.alloc(24)]);
const source = 'wgt 0.5\nmodel demo\ninput x: f32[2]\noutput y: f32[2]\nlayer fc: linear(in=2, out=2)\ninit weights=xavier_uniform(42) bias=zeros()\n';
const manifest = {
  format: 'weightlang.artifact', format_version: 2, model: 'demo',
  source: {file: 'source.wgt', sha256: hash(source)},
  weights: {file: 'weights.safetensors', sha256: hash(weights), tensors: Object.fromEntries(
    Object.entries(tensors).map(([name, value]) => [name, {shape: value.shape, dtype: 'float32', origin: 'random-init'}]))},
  interface: {input: {name: 'x', dtype: 'f32', shape: [2]}, output: {name: 'y', dtype: 'f32', shape: [2]}},
  layers: [{name: 'fc', op: 'linear', args: {in: 2, out: 2}}],
};
for (const [name, bytes] of Object.entries({'manifest.json': JSON.stringify(manifest), 'weights.safetensors': weights, 'source.wgt': source})) {
  fs.writeFileSync(path.join(output, name), bytes, {flag: 'wx'});
}
