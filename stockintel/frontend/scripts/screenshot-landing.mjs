/**
 * Capture landing page screenshots for the README.
 *
 * Separate from screenshot.mjs because the landing page is static: it needs no
 * waiting on model validation or news, so it captures in seconds.
 */

import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(HERE, "../../../docs/screenshots");
const URL = "http://localhost:3000";

const HIDE_DEV_OVERLAY = `
  nextjs-portal,
  [data-nextjs-dev-tools-button],
  [data-nextjs-toast],
  #__next-build-watcher { display: none !important; }
`;

async function captureByHeading(page, text, filename) {
  const handle = await page.evaluateHandle((needle) => {
    const heading = [...document.querySelectorAll("h2")].find((h) =>
      h.textContent.toLowerCase().includes(needle.toLowerCase()),
    );
    return heading ? heading.closest("section") : null;
  }, text);

  const element = handle.asElement();
  if (!element) {
    console.log(`  SKIP ${filename}`);
    return;
  }
  await element.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  await element.screenshot({ path: join(OUT, filename) });
  console.log(`  OK   ${filename}`);
}

const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 2,
});

await mkdir(OUT, { recursive: true });
await page.goto(URL, { waitUntil: "networkidle", timeout: 60_000 });
await page.addStyleTag({ content: HIDE_DEV_OVERLAY });
await page.waitForTimeout(600);

console.log("capturing…");

// Hero: nav plus the opening statement.
await page.screenshot({
  path: join(OUT, "landing-hero.png"),
  clip: { x: 0, y: 0, width: 1440, height: 860 },
});
console.log("  OK   landing-hero.png");

await captureByHeading(page, "Nothing beat the baseline", "landing-finding.png");
await captureByHeading(page, "What it will not do", "landing-limits.png");

await page.screenshot({ path: join(OUT, "landing-full.png"), fullPage: true });
console.log("  OK   landing-full.png");

await browser.close();
console.log(`\nwritten to ${OUT}`);
