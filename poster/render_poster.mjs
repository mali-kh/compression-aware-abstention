// Render poster.html to an A0 PDF and a PNG preview.
//   node poster/render_poster.mjs
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
// require() resolves a globally installed playwright, which ESM import does not
const { chromium } = createRequire(import.meta.url)("playwright");
const html = pathToFileURL(path.join(here, "poster.html")).href;

// A0 portrait at 96 CSS px per inch
const W = Math.round((841 / 25.4) * 96);
const H = Math.round((1189 / 25.4) * 96);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
await page.goto(html, { waitUntil: "networkidle" });
await page.evaluate(() => document.fonts.ready);

// Layout check: report any column or card whose content overflows.
const problems = await page.evaluate(() => {
  const out = [];
  for (const el of document.querySelectorAll(".col, .card, main, .poster")) {
    if (el.scrollHeight > el.clientHeight + 1 || el.scrollWidth > el.clientWidth + 1) {
      out.push(`${el.className}: scroll ${el.scrollWidth}x${el.scrollHeight} vs client ${el.clientWidth}x${el.clientHeight}`);
    }
  }
  const main = document.querySelector("main").getBoundingClientRect();
  for (const [i, col] of [...document.querySelectorAll(".col")].entries()) {
    const last = col.lastElementChild.getBoundingClientRect();
    out.push(`col ${i + 1}: last card bottom ${Math.round(last.bottom)} / main bottom ${Math.round(main.bottom)}`);
  }
  return out;
});
console.log(problems.join("\n"));

await page.pdf({
  path: path.join(here, "poster.pdf"),
  width: "841mm",
  height: "1189mm",
  printBackground: true,
  preferCSSPageSize: true,
});
await page.setViewportSize({ width: W, height: H });
const preview = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 0.5 });
await preview.goto(html, { waitUntil: "networkidle" });
await preview.evaluate(() => document.fonts.ready);
await preview.screenshot({ path: path.join(here, "poster_preview.png"), fullPage: false });
await browser.close();
console.log("wrote poster.pdf and poster_preview.png");
