import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';

export async function withBrowser(check, options = {}) {
  const modulePath = process.env.WEIGHTLANG_PLAYWRIGHT;
  if (!modulePath) throw Error('Set WEIGHTLANG_PLAYWRIGHT to an installed playwright module');
  const playwright = await import(pathToFileURL(modulePath));
  const chromium = playwright.chromium || playwright.default.chromium;
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM || '/usr/bin/chromium',
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}, ...options});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await check(page);
    assert.deepEqual(errors, [], 'unexpected browser page errors');
  } finally {
    await browser.close();
  }
}
