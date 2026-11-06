import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
import {withBrowser} from './support/browser.mjs';
await withBrowser(async page => {
 await page.goto(pathToFileURL(resolve(process.argv[2],'tiny_transformer.html')).href);
 await page.waitForSelector('.wl-tensor-cell');
 await page.selectOption('.wl-language','zh-CN');
 const grid=page.locator('.wl-node[data-path="embed"] [data-side="output"]');
 assert.equal(await grid.locator('.wl-tensor-cell').count(),32);
 assert.equal(await page.locator('.wl-node[data-path="input"] .wl-tensor-cell').count(),4);
 await page.selectOption('.wl-theme','dark');
 assert.equal(await page.locator('weightlang-viewer').getAttribute('data-theme'),'dark');
 await page.locator('.wl-node[data-path="embed"]').click();
 await page.screenshot({path:resolve(process.argv[2],'dark.png'),fullPage:true});
 await page.reload();
 await page.waitForSelector('.wl-tensor-cell');
 assert.equal(await page.locator('.wl-theme').inputValue(),'dark');
 await page.selectOption('.wl-theme','auto');
 await page.emulateMedia({colorScheme:'light'});
 await page.waitForFunction(()=>document.querySelector('weightlang-viewer').dataset.theme==='light');
 await page.emulateMedia({colorScheme:'dark'});
 await page.waitForFunction(()=>document.querySelector('weightlang-viewer').dataset.theme==='dark');
 await page.selectOption('.wl-theme','light');
 await page.locator('.wl-node[data-path="embed"]').click();
 await page.screenshot({path:resolve(process.argv[2],'light.png'),fullPage:true});
 await page.selectOption('.wl-presentation','compact');
 assert.equal(await page.locator('.wl-tensor-cell').count(),0);
 await page.selectOption('.wl-presentation','tensors');
 assert.equal(await grid.locator('.wl-tensor-cell').count(),32);
 // Library extensions work on arbitrary graphs and never depend on model names.
 await page.evaluate(()=>{
  const viewer=document.querySelector('weightlang-viewer');
  viewer.configure({tensorRenderer:({svg,width,height})=>svg('circle',{cx:width/2,cy:height/2,r:12,class:'custom-tensor'})});
 });
 assert.ok(await page.locator('.custom-tensor').count());
 await page.evaluate(()=>{
  document.querySelector('weightlang-viewer').configure({tensorRenderer:null,nodeRenderer:({node,defaultRender,svg})=>{
   const content=defaultRender();content.append(svg('text',{x:300,y:25,class:'custom-node'},node.op));return content;
  }});
 });
 assert.equal(await page.locator('.custom-node').count(),6);
 await page.evaluate(()=>{
  const viewer=document.querySelector('weightlang-viewer');
  viewer.configure({nodeRenderer:null,tensorLimits:{rows:2,columns:3}});
  viewer.style.setProperty('--tensor-cell','rgb(255, 0, 100)');
 });
 assert.equal(await grid.locator('.wl-tensor-cell').count(),6);
 assert.equal(await grid.locator('.wl-tensor-cell').first().evaluate(e=>getComputedStyle(e).fill),'rgb(255, 0, 100)');
 for (const width of [390,320]) {
  await page.setViewportSize({width,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 }
 await page.goto(pathToFileURL(resolve(process.argv[2],'tiny_cnn.html')).href);
 await page.waitForSelector('.wl-tensor-glyph[data-rank="3"]');
 assert.equal(await page.locator('.wl-node[data-path="features"] [data-side="output"] [data-plane]').count(),3);
 assert.ok((await page.locator('.wl-node[data-path="features"] [data-side="output"]').textContent()).includes('4 × 3 × 3'));
 assert.equal(await page.locator('.wl-error').textContent(),'');
}, {locale: 'zh-CN'});
console.log('PASS: exact small grids, bounded channel stacks, light/dark/system and reload, compact mode, custom renderers and CSS, mobile');
