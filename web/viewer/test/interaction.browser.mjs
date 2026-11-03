import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
const {chromium} = await import(pathToFileURL(process.env.WEIGHTLANG_PLAYWRIGHT));
const browser = await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
const page = await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
const viewer = (fn, arg) => page.evaluate(fn, arg);
try {
 // Usage: node test/interaction.browser.mjs DIR, where DIR holds tiny_transformer.html.
 await page.goto(pathToFileURL(resolve(process.argv[2],'tiny_transformer.html')).href);
 await page.waitForSelector('.wl-node');
 await page.selectOption('.wl-language','en');
 // Categories: every node is tagged and the legend lists only categories present.
 const categories = await page.locator('.wl-node').evaluateAll(ns=>ns.map(n=>n.dataset.category));
 assert.ok(categories.every(Boolean));
 assert.deepEqual(new Set(await page.locator('.wl-categories span').evaluateAll(s=>s.map(e=>e.dataset.category))), new Set(categories));

 // Search matches collapsed inner nodes through their overview module.
 await page.fill('.wl-search','qkv');
 assert.equal(await page.locator('.wl-node.wl-match').count(),1);
 assert.equal(await page.locator('.wl-node.wl-match').getAttribute('data-path'),'body');
 await page.press('.wl-search','Enter');
 assert.equal(await page.locator('.wl-selected').getAttribute('data-path'),'body');
 assert.equal(await page.locator('.wl-search-count').textContent(),'1/1');
 await page.press('.wl-search','Escape');
 assert.equal(await page.locator('.wl-node.wl-match').count(),0);

 // Selection emphasizes connected edges; neighbour links jump along the graph.
 const active = await page.locator('.wl-edge.wl-edge-active').evaluateAll(es=>es.map(e=>[e.dataset.from,e.dataset.to]));
 const bodyId = await page.locator('.wl-selected').getAttribute('data-id');
 assert.ok(active.length >= 2 && active.every(pair=>pair.includes(bodyId)));
 await page.locator('.wl-facts .wl-link').last().click();
 assert.notEqual(await page.locator('.wl-selected').getAttribute('data-path'),'body');

 // Arrow keys follow the flow direction.
 await page.locator('.wl-node[data-op="input"]').focus();
 await page.keyboard.press('ArrowDown');
 const next = await page.locator('.wl-selected').getAttribute('data-path');
 assert.equal(next,'embed');
 await page.keyboard.press('ArrowUp');
 assert.equal(await page.locator('.wl-selected').getAttribute('data-op'),'input');

 // Zoom: buttons, keyboard and Ctrl+wheel resize the SVG; fit restores a readable scale.
 const width = () => page.locator('.wl-flow > svg').evaluate(s=>+s.getAttribute('width'));
 const base = await width();
 await page.locator('.wl-zoom-in').click();
 assert.ok(await width() > base);
 await page.locator('.wl-zoom-level').click();
 assert.equal(await page.locator('.wl-zoom-level').textContent(),'100%');
 await page.locator('.wl-flow').hover();
 await page.keyboard.down('Control'); await page.mouse.wheel(0, 400); await page.keyboard.up('Control');
 assert.ok(await viewer(()=>document.querySelector('weightlang-viewer').zoom) < 1);
 await page.locator('.wl-fit').click();
 assert.ok(await viewer(()=>document.querySelector('weightlang-viewer').zoom) >= 0.75 - 1e-9);

 // Left-to-right layout and arrow navigation along the horizontal axis.
 await page.selectOption('.wl-direction','LR');
 assert.equal(await page.locator('.wl-flow > svg').getAttribute('data-direction'),'LR');
 const xs = await page.locator('.wl-node').evaluateAll(ns=>ns.map(n=>n.transform.baseVal[0].matrix.e));
 assert.ok(new Set(xs).size === xs.length);
 await page.locator('.wl-node[data-op="input"]').focus();
 await page.keyboard.press('ArrowRight');
 assert.equal(await page.locator('.wl-selected').getAttribute('data-path'),'embed');
 await page.selectOption('.wl-direction','TB');

 // selectNode() expands the overview when the target is an inner node.
 const inner = await viewer(()=>document.querySelector('weightlang-viewer').data.nodes.find(n=>n.path==='body.attention.attention').id);
 await viewer(id=>document.querySelector('weightlang-viewer').selectNode(id), inner);
 assert.equal(await viewer(()=>document.querySelector('weightlang-viewer').expanded), true);
 assert.equal(await page.locator('.wl-selected').getAttribute('data-id'), inner);

 // Export is standalone: inline colors, no CSS variables or classes required.
 const exported = await viewer(()=>document.querySelector('weightlang-viewer').exportSVG());
 assert.match(exported, /^<\?xml/);
 assert.ok(!exported.includes('var(--'));
 assert.match(exported, /marker-end="url\(#wl\d+-arrow-/);
 const png = await viewer(async()=>(await document.querySelector('weightlang-viewer').exportPNG(1)).type);
 assert.equal(png,'image/png');

 // Custom layout hook and custom category colors.
 await viewer(()=>{
  const v=document.querySelector('weightlang-viewer');
  v.configure({layout:({nodes,width,height})=>({width:nodes.length*(width+20)+40,height:height+60,
    positions:new Map(nodes.map((n,i)=>[n.id,{x:20+i*(width+20),y:30,width,height}]))})});
 });
 const ys = await page.locator('.wl-node').evaluateAll(ns=>ns.map(n=>n.transform.baseVal[0].matrix.f));
 assert.ok(ys.every(y=>y===30));
 await viewer(()=>document.querySelector('weightlang-viewer').configure({layout:null}));

 await page.setViewportSize({width:390,height:844});
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 assert.equal(await page.locator('.wl-error').textContent(),'');
 assert.deepEqual(errors,[]);
 console.log('PASS: categories, search, edge emphasis, neighbour links, arrow navigation, zoom, LR layout, selectNode expansion, SVG/PNG export, custom layout, mobile');
} finally {await browser.close();}
