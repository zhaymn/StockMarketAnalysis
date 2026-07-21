/**
 * Record a demo GIF of switching markets: United States -> India.
 *
 * Run with both dev servers up:
 *   node scripts/demo-gif.mjs
 *
 * Encoded in pure JS (pngjs decode -> gifenc encode) rather than via ffmpeg,
 * which is not assumed to be installed.
 *
 * Frames are captured in explicit phases rather than continuously. Switching
 * markets triggers a fresh walk-forward validation that takes 20-30s, and a
 * real-time recording of that would be a mostly-static 30-second GIF. The
 * phases keep the loading state visible and honest while compressing the dead
 * time.
 */

import { chromium } from "playwright";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { PNG } from "pngjs";
// gifenc ships CommonJS, so its named exports are not directly importable
// from an ES module; destructure from the default export instead.
import gifenc from "gifenc";

const { GIFEncoder, quantize, applyPalette } = gifenc;

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(HERE, "../../../docs/screenshots");
const URL = "http://localhost:3000";

// Region captured: control bar + stock header + prediction verdict. Everything
// that changes when the market switches is inside this band.
const CLIP = { x: 0, y: 0, width: 1280, height: 640 };
const FRAME_DELAY_MS = 220;

const HIDE_DEV_OVERLAY = `
  nextjs-portal,
  [data-nextjs-dev-tools-button],
  [data-nextjs-toast],
  #__next-build-watcher { display: none !important; }
`;

const frames = [];

async function grab(page, count = 1, gapMs = 120) {
  for (let i = 0; i < count; i += 1) {
    frames.push(await page.screenshot({ clip: CLIP }));
    if (i < count - 1) await page.waitForTimeout(gapMs);
  }
}

/**
 * Wait for the analysis to finish, sampling a few frames so the loading state
 * stays visible in the GIF.
 *
 * Polling and frame capture are deliberately decoupled: switching markets
 * kicks off a fresh walk-forward validation that can take 30s+, so a loop that
 * captured one frame per poll would either give up early or produce a GIF made
 * almost entirely of loading skeletons.
 */
async function grabWhileLoading(page, expectSymbol, { timeoutMs = 90_000, maxFrames = 5 } = {}) {
  const started = Date.now();
  const pollMs = 400;
  let captured = 0;
  let sincePoll = 0;

  while (Date.now() - started < timeoutMs) {
    const done = await page.evaluate(
      (sym) => document.body.innerText.includes(sym),
      expectSymbol,
    );
    if (done) return true;

    // Capture only occasionally, regardless of how long the wait runs.
    sincePoll += pollMs;
    if (captured < maxFrames && sincePoll >= 900) {
      frames.push(await page.screenshot({ clip: CLIP }));
      captured += 1;
      sincePoll = 0;
    }

    await page.waitForTimeout(pollMs);
  }
  return false;
}

const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width: CLIP.width, height: 800 },
  deviceScaleFactor: 1, // keeps the GIF small; retina would quadruple it
});

await mkdir(OUT, { recursive: true });

console.log("loading dashboard…");
await page.goto(URL, { waitUntil: "networkidle", timeout: 120_000 });
await page.addStyleTag({ content: HIDE_DEV_OVERLAY });

await page.waitForFunction(
  () => /NO DIRECTIONAL CALL|BULLISH|BEARISH|NO CALL TODAY/.test(document.body.innerText),
  { timeout: 120_000 },
);
await page.waitForTimeout(1500);
console.log("US state ready");

// --- Phase 1: hold on the United States ---------------------------------
await grab(page, 8, 150);

// --- Phase 2: click India -----------------------------------------------
const indiaTab = page.getByRole("tab", { name: "India" });
await indiaTab.hover();
await grab(page, 3, 120);
await indiaTab.click();
console.log("clicked India");
await grab(page, 3, 200);

// --- Phase 3: loading ----------------------------------------------------
const loaded = await grabWhileLoading(page, "RELIANCE");
console.log(loaded ? "India state loaded" : "  (timed out waiting for India)");

// --- Phase 4: hold on India ---------------------------------------------
await page.waitForTimeout(1200);
await grab(page, 12, 150);

await browser.close();

// --- Encode --------------------------------------------------------------
console.log(`encoding ${frames.length} frames…`);
const encoder = GIFEncoder();

for (const [index, buffer] of frames.entries()) {
  const png = PNG.sync.read(buffer);
  const rgba = new Uint8ClampedArray(png.data);

  // 192 colours: the palette is mostly near-black plus lime and semantic
  // accents, so this is ample and keeps the file small.
  const palette = quantize(rgba, 192);
  const indexed = applyPalette(rgba, palette);

  encoder.writeFrame(indexed, png.width, png.height, {
    palette,
    delay: index === frames.length - 1 ? 1400 : FRAME_DELAY_MS, // pause on the last frame
  });
}

encoder.finish();

const path = join(OUT, "market-switch.gif");
await writeFile(path, encoder.bytes());

const { size } = await import("node:fs").then((fs) => fs.promises.stat(path));
console.log(`\nwrote ${path}`);
console.log(`${frames.length} frames, ${(size / 1024 / 1024).toFixed(2)} MB`);

// Dump the first and last frames so the start and end states can be checked
// without decoding the GIF. Written to a temp-ish name, not committed.
if (process.env.DUMP_FRAMES) {
  await writeFile(join(OUT, "_frame-first.png"), frames[0]);
  await writeFile(join(OUT, "_frame-last.png"), frames[frames.length - 1]);
  console.log("dumped _frame-first.png and _frame-last.png for inspection");
}
