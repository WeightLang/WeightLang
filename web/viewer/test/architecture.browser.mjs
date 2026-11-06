import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
import {withBrowser} from './support/browser.mjs';
await withBrowser(async page => {
 for (const model of ['xor_mlp','tiny_cnn','tiny_transformer','sequence_classifier']) {
  await page.goto(pathToFileURL(resolve(process.argv[2],model+'.html')).href);
  await page.waitForSelector('.wl-node');
  await page.selectOption('.wl-language','zh-CN');
  const count = await page.locator('.wl-node').count();
  assert.ok(count <= 6);
  await page.locator('.wl-node').nth(1).click();
  assert.match(await page.locator('.wl-node-info').textContent(),/输入.*输出/);
  if (model === 'tiny_transformer') {
   await page.locator('.wl-node[data-path="body"]').click();
   assert.match(await page.locator('.wl-node-info').textContent(),/Transformer 解码器/);
  }
  await page.locator('.wl-expand').click();
  assert.equal(await page.locator('.wl-node').count(),await page.evaluate(()=>document.querySelector('weightlang-viewer').data.nodes.length));
  if(model === 'sequence_classifier') assert.ok(await page.locator('.wl-edge[data-kind="recurrent"]').count());
  await page.locator('.wl-expand').click();
  assert.equal(await page.locator('.wl-node').count(),count);
  await page.locator('.wl-node[data-op="input"]').click();
  assert.equal(await page.locator('.wl-plot').isVisible(),false);
  await page.locator('.wl-inspect-toggle').click();
  await page.selectOption('.wl-tensor',{index:0});
  assert.equal(await page.locator('.wl-plot').isVisible(),true);
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.setViewportSize({width:1440,height:1000});
  assert.equal(await page.locator('.wl-error').textContent(),'');
  if(model==='tiny_transformer') {
   await page.locator('.wl-node[data-path="body"]').click();
   await page.locator('.wl-inspect-toggle').click();
   await page.screenshot({path:resolve(process.argv[2],'transformer.png'),fullPage:true});
  }
 }
});
console.log('PASS: four architectures, overview/detail, module explanation, recurrent edges, parameter selection, mobile width, no JS errors');
