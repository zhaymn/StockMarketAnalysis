/**
 * Capture dashboard screenshots for the README.
 *
 * Run with the backend and frontend dev servers already running:
 *   node scripts/screenshot.mjs
 *
 * Waits on real content rather than a fixed timeout: the first analysis
 * request runs walk-forward validation and the news request runs FinBERT plus
 * Gemini, so the page is not meaningfully renderable for ~20-30s.
 */

import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(HERE, "../../../docs/screenshots");
const URL = "http://localhost:3000";

/**
 * Capture one section by its heading text.
 *
 * Uses an element screenshot rather than a page screenshot with `clip`:
 * `clip` is interpreted in page coordinates but a non-fullPage screenshot only
 * covers the viewport, so any section below the fold fails with "clipped area
 * is outside the resulting image". Element screenshots scroll into view first.
 */
async function captureSection(page, headingText, filename) {
  const handle = await page.evaluateHandle((text) => {
    const heading = [...document.querySelectorAll("h2")].find((h) =>
      h.textContent.toLowerCase().includes(text.toLowerCase()),
    );
    if (!heading) return null;
    return heading.closest("section") ?? heading.parentElement;
  }, headingText);

  const element = handle.asElement();
  if (!element) {
    console.log(`  SKIP ${filename} — no heading matching "${headingText}"`);
    return false;
  }

  await element.scrollIntoViewIfNeeded();
  await page.waitForTimeout(400); // let any lazy paint settle
  await element.screenshot({ path: join(OUT, filename) });

  const height = await element.evaluate((el) => Math.round(el.getBoundingClientRect().height));
  console.log(`  OK   ${filename}  (${height}px tall)`);
  return true;
}

const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width: 1440, height: 960 },
  deviceScaleFactor: 2, // retina, so text stays crisp on GitHub
});

await mkdir(OUT, { recursive: true });

// Hide the Next.js dev-tools indicator, which otherwise overlaps the
// bottom-left of the page and appears in captures as a stray badge.
await page.addStyleTag({
  content: `
    nextjs-portal,
    [data-nextjs-dev-tools-button],
    [data-nextjs-toast],
    #__next-build-watcher { display: none !important; }
  `,
});

console.log("loading dashboard…");
await page.goto(URL, { waitUntil: "networkidle", timeout: 120_000 });

// addStyleTag before navigation does not survive it, so re-apply.
await page.addStyleTag({
  content: `
    nextjs-portal,
    [data-nextjs-dev-tools-button],
    [data-nextjs-toast],
    #__next-build-watcher { display: none !important; }
  `,
});

// The prediction verdict only appears once walk-forward validation completes.
await page.waitForFunction(
  () => /NO DIRECTIONAL CALL|BULLISH|BEARISH|NO CALL TODAY|INSUFFICIENT/.test(document.body.innerText),
  { timeout: 120_000 },
);
console.log("prediction rendered");

// News arrives later still — FinBERT load plus Gemini classification.
await page
  .waitForFunction(
    () => /Weighted sentiment|NEWS API NOT CONFIGURED/i.test(document.body.innerText),
    { timeout: 120_000 },
  )
  .catch(() => console.log("  (news did not resolve in time)"));
console.log("news rendered");

// Let chart canvases finish painting.
await page.waitForTimeout(3000);

console.log("capturing…");

// Full page, for the README hero.
await page.screenshot({ path: join(OUT, "dashboard-full.png"), fullPage: true });
console.log("  OK   dashboard-full.png");

// Header + prediction: scroll to top and grab the first screenful.
await page.evaluate(() => window.scrollTo(0, 0));
await page.waitForTimeout(500);
await page.screenshot({
  path: join(OUT, "prediction.png"),
  clip: { x: 0, y: 0, width: 1440, height: 900 },
});
console.log("  OK   prediction.png");

await captureSection(page, "Price & technical charts", "charts.png");
await captureSection(page, "Model evidence", "model-evidence.png");
await captureSection(page, "Why this prediction", "why-this-prediction.png");
await captureSection(page, "News & current affairs", "news.png");

await browser.close();
console.log(`\nwritten to ${OUT}`);
