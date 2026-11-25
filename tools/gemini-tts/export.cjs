// Load the built reader page with the export hook set and dump the narration units.
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const [page_path, out] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.addInitScript(() => { window.__PELEARN_EXPORT__ = {}; });
  await page.route('**/*', r => r.request().url().startsWith('file:') ? r.continue() : r.abort());
  await page.goto('file://' + path.resolve(page_path), { waitUntil: 'load', timeout: 120000 });
  const units = await page.evaluate(() => window.__PELEARN_EXPORT__.units);
  require('fs').writeFileSync(out, JSON.stringify(units));
  console.log('units', units.length, 'blocks', units.reduce((a, u) => a + u.blocks.length, 0));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
