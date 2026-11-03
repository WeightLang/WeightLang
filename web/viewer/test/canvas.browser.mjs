import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
const {chromium} = await import(pathToFileURL(process.env.WEIGHTLANG_PLAYWRIGHT));
const browser = await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
const page = await browser.newPage({viewport:{width:1440,height:900}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
const viewer = (fn, arg) => page.evaluate(fn, arg);
const zoom = () => viewer(()=>document.querySelector('weightlang-viewer').zoom);
const scroll = () => page.locator('.wl-flow').evaluate(f=>[f.scrollLeft,f.scrollTop]);
try {
 // Usage: node test/canvas.browser.mjs DIR with CLI-generated tiny_transformer.html and sequence_classifier.html.
 await page.goto(pathToFileURL(resolve(process.argv[2],'tiny_transformer.html')).href);
 await page.waitForSelector('.wl-node');
 await page.selectOption('.wl-language','en');

 // Two-line title: model name, then brand and facts.
 const lines = await page.locator('.wl-title').evaluate(t=>[...t.children].map(c=>Math.round(c.getBoundingClientRect().height)));
 assert.equal(lines.length, 2);
 assert.match(await page.locator('.wl-summary').textContent(), /^WeightLang/);

 // Fit centers the graph horizontally below the title.
 const graph = await page.locator('.wl-flow > svg').boundingBox();
 assert.ok(Math.abs(graph.x + graph.width / 2 - 720) < 4, `centered: ${graph.x}`);
 assert.ok(graph.y >= 100);

 // A plain mouse wheel zooms around the cursor; Shift + wheel does not.
 await page.mouse.move(700, 450);
 const before = await zoom();
 await page.mouse.wheel(0, 240);
 assert.ok(await zoom() < before);
 await page.mouse.wheel(0, -480);
 assert.ok(await zoom() > before);
 const z = await zoom();
 await page.keyboard.down('Shift'); await page.mouse.wheel(0, 240); await page.keyboard.up('Shift');
 assert.equal(await zoom(), z);

 // Drag empty canvas to pan freely, even past the graph's edge.
 await page.locator('.wl-fit').click();
 const [left, top] = await scroll();
 await page.mouse.move(200, 500); await page.mouse.down(); await page.mouse.move(500, 300, {steps: 6}); await page.mouse.up();
 const [left2, top2] = await scroll();
 assert.ok(Math.abs(left - left2 - 300) < 2 && Math.abs(top2 - top - 200) < 2, `${left},${top} → ${left2},${top2}`);
 assert.equal(await page.locator('.wl-popup').isVisible(), false);

 // Double-click empty space re-fits.
 await page.mouse.dblclick(150, 600);
 assert.deepEqual((await scroll()).map(Math.round), [left, top].map(Math.round));

 // Drag a node: it moves, its edges re-route, and no popup opens.
 const node = page.locator('.wl-node[data-path="norm"]');
 const edge = page.locator('.wl-edge[data-to="' + await node.getAttribute('data-id') + '"]').first();
 const d = await edge.getAttribute('d');
 const box = await node.boundingBox();
 await page.mouse.move(box.x + 60, box.y + 30); await page.mouse.down();
 await page.mouse.move(box.x + 260, box.y + 70, {steps: 8}); await page.mouse.up();
 const moved = await node.boundingBox();
 assert.ok(Math.abs(moved.x - box.x - 200) < 3 && Math.abs(moved.y - box.y - 40) < 3);
 assert.notEqual(await edge.getAttribute('d'), d);
 assert.equal(await page.locator('.wl-popup').isVisible(), false);
 // A plain click still opens details; positions survive a language switch; reset restores layout.
 await node.click();
 assert.equal(await page.locator('.wl-popup').isVisible(), true);
 await page.selectOption('.wl-language','zh-CN');
 assert.ok(Math.abs((await node.boundingBox()).x - moved.x) < 3);
 assert.equal(await page.locator('.wl-reset-layout').isVisible(), true);
 await page.locator('.wl-reset-layout').click();
 assert.equal(await page.locator('.wl-reset-layout').isVisible(), false);
 assert.ok(Math.abs((await node.boundingBox()).x - box.x) < 3);
 await page.selectOption('.wl-language','en');

 // Code as comments: the model header comment and the declaring source line.
 await page.locator('.wl-model-title').click();
 assert.match(await page.locator('.wl-node-info').textContent(), /Small next-token smoke test/);
 assert.match(await page.locator('.wl-node-info .wl-source-code').first().textContent(), /model tiny_transformer/);
 // A std block layer shows the comment of the block it calls, with its source line.
 await page.locator('.wl-node[data-path="body"]').click();
 const info = await page.locator('.wl-node-info').textContent();
 assert.match(info, /Decoder-only block: causal self-attention/);
 assert.match(info, /layer body: transformer\.decoder\(dim=8, heads=2, hidden=16\)/);

 // Nested layers carry the comment of the block they belong to.
 await page.goto(pathToFileURL(resolve(process.argv[2],'sequence_classifier.html')).href);
 await page.waitForSelector('.wl-node');
 await page.locator('.wl-expand').click();
 await page.locator('.wl-node[data-path="memory.sequence.joined"]').click();
 const nested = await page.locator('.wl-node-info').textContent();
 assert.match(nested, /A GRU cell with reset applied before the candidate hidden projection/);
 assert.match(nested, /std\/recurrent\.wgt · L\d+/);
 assert.match(nested, /layer joined: concat\(input, state\)/);
 assert.deepEqual(errors, []);
 console.log('PASS: two-line title, centered fit, wheel zoom, free drag pan, double-click fit, node drag with re-routed edges, reset layout, model/block/nested source comments');
} finally {await browser.close();}
