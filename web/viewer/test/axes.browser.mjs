import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
const {chromium}=await import(pathToFileURL(process.env.WEIGHTLANG_PLAYWRIGHT));
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
try {
 await page.goto(pathToFileURL(resolve(process.argv[2])).href);
 await page.waitForSelector('.wl-node');
 await page.evaluate(()=>document.querySelector('weightlang-viewer').load({model:'5-D axes',nodes:[{
  id:'tensor',path:'tensor',op:'input',input_shape:'f32[2,3,4,5,6]',output_shape:'f32[2,3,4,5,6]'
 }],edges:[],tensor_data:[]}));
 const glyph=page.locator('.wl-tensor-glyph');
 assert.equal(await glyph.getAttribute('data-visible-axes'),'0,1,2');
 assert.equal(await glyph.locator('.wl-tensor-cell').count(),24);
 assert.equal(await page.locator('[data-axis="4"]').isDisabled(),true);
 await page.locator('[data-axis="1"]').uncheck();
 assert.equal(await page.locator('[data-axis="4"]').isEnabled(),true);
 await page.locator('[data-axis="4"]').check();
 assert.equal(await glyph.getAttribute('data-visible-axes'),'0,2,4');
 assert.equal(await glyph.locator('.wl-tensor-cell').count(),48);
 assert.equal(await glyph.locator('[data-coordinate="1,0,3,0,5"]').count(),1);
 assert.equal(await page.locator('[data-axis="3"]').isDisabled(),true);
 await page.selectOption('.wl-language','en');
 await page.selectOption('.wl-theme','dark');
 assert.equal(await glyph.getAttribute('data-visible-axes'),'0,2,4');
 for (const axis of [0,2,4]) await page.locator(`[data-axis="${axis}"]`).uncheck();
 assert.equal(await glyph.locator('.wl-tensor-cell').count(),1);
 assert.equal(await glyph.locator('[data-coordinate="0,0,0,0,0"]').count(),1);
 assert.equal(await page.locator('section').first().evaluate(e=>getComputedStyle(e).borderTopWidth),'0px');
 await page.setViewportSize({width:320,height:844});
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 assert.deepEqual(errors,[]);
 console.log('PASS: default first 3 axes, swap axes, max-3 enforcement, hidden indices zero, scalar slice, language/theme persistence, open layout');
} finally {await browser.close();}
