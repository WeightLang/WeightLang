import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
const {chromium} = await import(pathToFileURL(process.env.WEIGHTLANG_PLAYWRIGHT));
const browser = await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
const page=await browser.newPage({viewport:{width:1440,height:1000},locale:'zh-CN'});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
try {
 await page.goto(pathToFileURL(resolve(process.argv[2])).href);
 await page.waitForSelector('.wl-node');
 await page.selectOption('.wl-language','zh-CN');
 await page.locator('.wl-expand').click();
 await page.locator('.wl-node[data-path="body.attention.attention"]').click();
 await page.locator('.wl-inspect-toggle').click();
 await page.selectOption('.wl-tensor','body.attention.attention.qkv.weight');
 await page.selectOption('.wl-presets select','0');
 await page.selectOption('.wl-mode','delta');
 const snapshot=()=>page.evaluate(()=>{
  const v=document.querySelector('weightlang-viewer');
  return {node:v.q('.wl-selected').dataset.id,tensor:v.selected.name,mode:v.q('.wl-mode').value,
   expanded:v.expanded,fixed:v.fixed,range:v.range,values:v.plane.values};
 });
 const before=await snapshot();
 await page.selectOption('.wl-language','en');
 assert.deepEqual(await snapshot(),before);
 assert.equal(await page.locator('weightlang-viewer').getAttribute('lang'),'en');
 assert.match(await page.locator('.wl-node-info').textContent(),/Exchange information/);
 assert.equal(await page.locator('.wl-expand').textContent(),'Module overview');
 assert.equal(await page.locator('.wl-inspect-toggle').textContent(),'Hide weights');
 assert.ok(!(await page.locator('.wl-workspace').textContent()).match(/[\u4e00-\u9fff]/));
 if (process.argv[3]) await page.screenshot({path:resolve(process.argv[3]),fullPage:true});
 await page.reload();
 await page.waitForSelector('.wl-node');
 assert.equal(await page.locator('.wl-language').inputValue(),'en');
 await page.selectOption('.wl-language','zh-CN');
 await page.locator('.wl-node[data-path="body"]').click();
 assert.match(await page.locator('.wl-node-info').textContent(),/因果自注意力/);
 await page.setViewportSize({width:390,height:844});
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth));
 // Storage may be disabled in embedded/offline environments; switching must still work.
 await page.evaluate(()=>Object.defineProperty(window,'localStorage',{get(){throw Error('blocked')}}));
 await page.selectOption('.wl-language','en');
 assert.equal(await page.locator('.wl-language').inputValue(),'en');
 assert.equal(await page.locator('.wl-error').textContent(),'');
 assert.deepEqual(errors,[]);
 console.log('PASS: bilingual labels/descriptions, preserved graph/tensor/slice state, reload preference, mobile, blocked storage');
} finally {await browser.close();}
