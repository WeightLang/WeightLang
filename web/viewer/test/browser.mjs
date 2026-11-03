import assert from "node:assert/strict";
import {pathToFileURL} from "node:url";
import {resolve} from "node:path";
const modulePath=process.env.WEIGHTLANG_PLAYWRIGHT;
if(!modulePath)throw Error("Set WEIGHTLANG_PLAYWRIGHT to an installed playwright module");
const playwright=await import(pathToFileURL(modulePath));
const browser=await (playwright.chromium || playwright.default.chromium).launch({executablePath:process.env.CHROMIUM||"/usr/bin/chromium",headless:true,args:["--no-sandbox","--disable-dev-shm-usage"]});
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[];
page.on("pageerror",e=>errors.push(e.message));
try {
  await page.goto(pathToFileURL(resolve(process.argv[2]||"weights/transformer-edit/preview.html")).href);
  await page.waitForSelector(".wl-node");
  await page.selectOption(".wl-language","en");
  await page.locator(".wl-inspect-toggle").click();
  assert.equal(await page.locator("details").count(),0);
  assert.equal(await page.getByText("Agent-readable").count(),0);
  await page.selectOption(".wl-tensor","body.attention.attention.qkv.weight");
  await page.selectOption(".wl-presets select",{label:"Query · head 1"});
  assert.equal(await page.evaluate(()=>document.querySelector("weightlang-viewer").plane.rows),4);
  await page.selectOption(".wl-tensor","head.logits.weight");
  await page.selectOption(".wl-mode","delta");
  assert.ok(await page.evaluate(()=>document.querySelector("weightlang-viewer").plane.values.some(v=>v!==0)));
  await page.screenshot({path:process.argv[3]||"weights/transformer-edit/preview.png",fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.setViewportSize({width:1440,height:1000});
  await page.evaluate(()=>{
    const viewer=document.querySelector("weightlang-viewer");
    viewer.load({title:"Arbitrary 5-D tensor",nodes:[{id:"custom",path:"custom",op:"unseen_operator"}],edges:[],tensor_data:[{name:"custom.weight",owner:"custom",shape:[2,3,4,5,2],values:Array.from({length:240},(_,i)=>i),min:0,max:239,mean:119.5}]});
    const theme=document.createElement("style");
    theme.textContent='weightlang-viewer [data-op="unseen_operator"]{--node-fill:rgb(12, 34, 56)}';
    document.head.append(theme);
  });
  assert.equal(await page.locator(".wl-node rect").evaluate(e=>getComputedStyle(e).fill),"rgb(12, 34, 56)");
  await page.selectOption(".wl-row","1");await page.selectOption(".wl-column","3");
  for(const [axis,value] of [[0,1],[2,2],[4,1]]){const input=page.getByLabel("Axis "+axis+" slice",{exact:true});await input.fill(String(value));await input.dispatchEvent("change");}
  assert.equal(await page.evaluate(()=>document.querySelector("weightlang-viewer").plane.values[0]),141);
  assert.equal(await page.locator(".wl-error").textContent(),"");
  assert.deepEqual(errors,[]);
  console.log("Browser checks passed: no collapsed data modes, QKV metadata, delta heatmap, responsive layout, arbitrary 5-D axes and CSS-only styling.");
} finally {await browser.close();}
