import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
import {readFileSync} from 'node:fs';
const {chromium} = await import(pathToFileURL(process.env.WEIGHTLANG_PLAYWRIGHT));
const browser = await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
const page = await browser.newPage({viewport:{width:1440,height:900}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
const fixture = JSON.parse(readFileSync(new URL('./fixtures/control-flow.json', import.meta.url)));
try {
 // Usage: node test/control-flow.browser.mjs DIR, where DIR holds any CLI-generated tiny_transformer.html.
 await page.goto(pathToFileURL(resolve(process.argv[2],'tiny_transformer.html')).href);
 await page.waitForSelector('.wl-node');
 await page.selectOption('.wl-language','en');

 // The stage is the whole viewer: dotted canvas, floating chrome, no popup until a click.
 assert.equal(await page.locator('.wl-popup').isVisible(), false);
 const stage = await page.locator('.wl-flow').evaluate(f=>({w:f.clientWidth,h:f.clientHeight,bg:getComputedStyle(f).backgroundImage}));
 assert.ok(stage.w >= 1400 && stage.h >= 850 && stage.bg.includes('radial-gradient'));
 assert.equal(await page.locator('.wl-node.wl-selected').count(), 0);

 // Clicking opens the popup; dragging its bar moves it; × closes and clears selection.
 await page.locator('.wl-node[data-path="body"]').click();
 assert.equal(await page.locator('.wl-popup').isVisible(), true);
 assert.match(await page.locator('.wl-node-info').textContent(), /Transformer decoder/);
 const before = await page.locator('.wl-popup').boundingBox();
 const bar = await page.locator('.wl-popup-bar').boundingBox();
 await page.mouse.move(bar.x + 60, bar.y + bar.height / 2);
 await page.mouse.down(); await page.mouse.move(bar.x - 240, bar.y + bar.height / 2 + 140, {steps: 6}); await page.mouse.up();
 const after = await page.locator('.wl-popup').boundingBox();
 assert.ok(Math.abs(after.x - (before.x - 300)) < 2 && Math.abs(after.y - (before.y + 140)) < 2);
 await page.locator('.wl-popup-close').click();
 assert.equal(await page.locator('.wl-popup').isVisible(), false);
 assert.equal(await page.locator('.wl-node.wl-selected').count(), 0);
 // Reopening keeps the dragged position; Escape closes.
 await page.locator('.wl-node[data-op="input"]').click();
 const reopened = await page.locator('.wl-popup').boundingBox();
 assert.ok(Math.abs(reopened.x - after.x) < 2);
 await page.keyboard.press('Escape');
 assert.equal(await page.locator('.wl-popup').isVisible(), false);

 // Control flow: routes, branches, gates, loops and halting are all explicit.
 await page.evaluate(g=>document.querySelector('weightlang-viewer').load(g), fixture);
 await page.locator('.wl-expand').click();
 assert.equal(await page.locator('.wl-node[data-op="route_merge"]').count(), 1);
 assert.deepEqual(await page.locator('.wl-region').evaluateAll(rs=>rs.map(r=>r.dataset.kind).sort()), ['loop','route']);
 const labels = await page.locator('.wl-edge-label').evaluateAll(ls=>ls.map(l=>[l.dataset.role||'', l.textContent]));
 for (const expected of [['gate','g'],['flow','× g'],['flow','× (1−g)'],['route','Branch 0'],['route','Branch 1'],['loop','next step'],['gate','halt probability']])
   assert.ok(labels.some(([role,text])=>role===expected[0] && text===expected[1]), `missing label ${expected}`);
 assert.ok(await page.locator('.wl-edge[data-kind="conditional"][data-to="r::merge"]').count() === 2);
 assert.equal(await page.locator('.wl-edge[data-role="gate"]').count(), 2);
 assert.equal(await page.locator('.wl-edge[data-role="loop"]').count(), 1);
 // The legend only lists roles that appear.
 assert.deepEqual(await page.locator('.wl-key [data-kind]:not([hidden])').evaluateAll(s=>s.map(e=>e.dataset.kind)), ['flow','loop','gate','route']);
 // Expanded graphs default to compact cards; junctions render as pills with formulas.
 assert.equal(await page.locator('.wl-tensor-cell').count(), 0);
 assert.match(await page.locator('.wl-node[data-op="mix"]').textContent(), /g·a \+ \(1−g\)·b/);
 // Nothing renders outside the canvas, and the popup works from the virtual merge node.
 await page.locator('.wl-node[data-op="route_merge"]').click();
 assert.match(await page.locator('.wl-node-info').textContent(), /selected branch result/);
 const exported = await page.evaluate(()=>document.querySelector('weightlang-viewer').exportSVG());
 assert.ok(exported.includes('paint-order') && exported.includes('wl-region'));

 await page.setViewportSize({width:390,height:844});
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 const sheet = await page.locator('.wl-popup').boundingBox();
 assert.ok(sheet.x >= 0 && sheet.x + sheet.width <= 390);
 assert.equal(await page.locator('.wl-error').textContent(),'');
 assert.deepEqual(errors,[]);
 console.log('PASS: full dotted stage, click-to-open draggable popup, close/Escape, route merge, branch/gate/loop/halt labels, regions, compact expanded cards, export, mobile sheet');
} finally {await browser.close();}
