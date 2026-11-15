// Node wasm micro-benchmark for the opend2d sampler/fill perf work.
// Usage: node wasm-bench.mjs <wasm-nodejs-pkg-dir> <scene.json> [reps] [scene2.json ...]
// Times drive_bytes(scene) (same core as native), min wall time over `reps` runs (least noise),
// after a warm-up call. Prints one line per scene: "<name> wasm-min <ms> bytes <n>".
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { basename } from "node:path";

const require = createRequire(import.meta.url);
const [pkgDir, ...rest] = process.argv.slice(2);

// args: pkgDir scene1 [reps] scene2 scene3 ...  — a bare integer anywhere sets reps.
let REPS = 9;
const scenes = [];
for (const a of rest) {
  if (/^\d+$/.test(a)) { REPS = +a; continue; }
  scenes.push(a);
}

const mod = require(`${pkgDir}/d2d_web.js`);
if (typeof mod.drive_bytes !== "function") {
  console.error("drive_bytes export missing from", pkgDir);
  process.exit(2);
}

for (const sceneFile of scenes) {
  const scene = readFileSync(sceneFile, "utf8");
  const out = mod.drive_bytes(scene); // warm
  if (!out || out.length === 0) { console.log(`${basename(sceneFile)} wasm-min SKIP (empty output)`); continue; }
  let best = Infinity;
  for (let r = 0; r < REPS; r++) {
    const t = process.hrtime.bigint();
    const px = mod.drive_bytes(scene);
    const dt = Number(process.hrtime.bigint() - t) / 1e6;
    if (px.length) best = Math.min(best, dt);
  }
  console.log(`${basename(sceneFile)} wasm-min ${best.toFixed(1)} ms bytes ${out.length}`);
}
