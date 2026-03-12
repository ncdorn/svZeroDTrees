#!/usr/bin/env node
"use strict";

const path = require("path");

let PptxGenJS;
try {
  PptxGenJS = require("pptxgenjs");
} catch (_err) {
  PptxGenJS = require(
    "/Users/ndorn/Documents/Stanford/PhD/Marsden_Lab/SimVascular/svz/repos/svZeroDTrees/docs/slides/impedance_bc_impl_comparison/node_modules/pptxgenjs"
  );
}

const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");

const W = 12.8;
const H = 8.0;
const deckDir = __dirname;
const img = (name) => path.join(deckDir, "assets", "generated", name);

const pptx = new PptxGenJS();
pptx.defineLayout({ name: "LAYOUT_16x10", width: W, height: H });
pptx.layout = "LAYOUT_16x10";
pptx.author = "Codex";
pptx.subject = "Impedance BC implementation review";
pptx.title = "Impedance BC Implementation";
pptx.company = "SimVascular";
pptx.lang = "en-US";

function header(slide, title, subtitle) {
  slide.background = { color: "FFFFFF" };
  slide.addText(title, {
    x: 0.6,
    y: 0.22,
    w: 11.6,
    h: 0.58,
    fontSize: 34,
    bold: true,
  });
  slide.addText(subtitle, {
    x: 0.6,
    y: 0.86,
    w: 11.6,
    h: 0.42,
    fontSize: 19,
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 0.6,
    y: 1.28,
    w: 11.6,
    h: 0,
    line: { pt: 1, color: "BFBFBF" },
  });
}

function footer(slide) {
  slide.addShape(pptx.ShapeType.line, {
    x: 0.6,
    y: 7.55,
    w: 11.6,
    h: 0,
    line: { pt: 1, color: "D9D9D9" },
  });
}

function bullets(slide, items, x, y, w, fontSize = 21, step = 0.84, h = 0.72) {
  let yy = y;
  for (const text of items) {
    slide.addText(text, {
      x,
      y: yy,
      w,
      h,
      fontSize,
      bullet: { indent: 20 },
      valign: "mid",
    });
    yy += step;
  }
}

function frame(slide, x, y, w, h) {
  slide.addShape(pptx.ShapeType.rect, {
    x,
    y,
    w,
    h,
    line: { pt: 1, color: "D9D9D9" },
    fill: { color: "FFFFFF", transparency: 100 },
  });
}

function finalize(slide, skipBoundsCheck = false) {
  footer(slide);
  warnIfSlideHasOverlaps(slide, pptx);
  if (!skipBoundsCheck) {
    warnIfSlideElementsOutOfBounds(slide, pptx);
  }
}

// 1) Goal + top-line result
{
  const s = pptx.addSlide();
  header(
    s,
    "Impedance BC Implementation Review",
    "MASTER-focused check: localized solver impact, BC behavior, and workflow implications"
  );

  frame(s, 0.7, 1.58, 6.2, 5.45);
  s.addImage({ path: img("iteration_pipeline.png"), x: 0.95, y: 1.86, w: 5.7, h: 4.95 });

  s.addText("This revision focuses on:", {
    x: 7.25,
    y: 1.82,
    w: 4.7,
    h: 0.44,
    fontSize: 22,
    bold: true,
  });
  bullets(
    s,
    [
      "Structural impact against master",
      "What changed inside IMPEDANCE BC",
      "Why minimal support hooks were added",
      "What this means for 3D-0D and full 0D runs",
    ],
    7.2,
    2.32,
    4.8,
    21,
    0.86
  );

  s.addText("Top line: no broad solver rewrite was introduced.", {
    x: 7.2,
    y: 6.52,
    w: 4.8,
    h: 0.55,
    fontSize: 18,
    bold: true,
  });

  finalize(s);
}

// 2) Architecture context
{
  const s = pptx.addSlide();
  header(
    s,
    "Architecture Context",
    "The same IMPEDANCE logic is used in both full 0D and coupled 3D-0D workflows"
  );

  frame(s, 0.7, 1.58, 11.4, 4.3);
  s.addImage({ path: img("architecture_context_simple.png"), x: 0.95, y: 1.85, w: 10.9, h: 3.85 });

  bullets(
    s,
    [
      "Full 0D: standard model execution path",
      "3D-0D: retry loop needs rollback-safe internal state",
      "Shared IMPEDANCE runtime rules across both modes",
    ],
    0.95,
    6.05,
    11.0,
    20,
    0.66,
    0.56
  );

  finalize(s);
}

// 3) MAIN diff footprint
{
  const s = pptx.addSlide();
  header(
    s,
    "Structural Footprint vs master",
    "Changes are concentrated in IMPEDANCE logic plus a small set of support hooks"
  );

  frame(s, 0.7, 1.58, 11.4, 4.25);
  s.addImage({ path: img("main_diff_touchpoints.png"), x: 0.95, y: 1.86, w: 10.9, h: 3.8 });

  bullets(
    s,
    [
      "Core additions stay inside IMPEDANCE BC and its config wiring",
      "Support hooks are limited to state snapshot/restore and timestep acceptance",
      "No solver loop redesign or global architecture change",
    ],
    0.95,
    6.08,
    11.0,
    20,
    0.66,
    0.56
  );

  s.addNotes(
    "Comparison method note: master history in this local clone has missing objects, so this footprint is anchored to the impedance-bc commit set around the master merge (bb926fa9, 82ba416d, 161870a5)."
  );

  finalize(s);
}

// 4) IMPEDANCE BC behavior
{
  const s = pptx.addSlide();
  header(
    s,
    "How IMPEDANCE BC Works",
    "Pressure is computed from current flow plus accepted one-cycle flow history"
  );

  frame(s, 0.7, 1.58, 7.8, 5.45);
  s.addImage({ path: img("convolution_ringbuffer_schematic.png"), x: 0.95, y: 1.86, w: 7.3, h: 5.0 });

  s.addText("Key runtime behaviors", {
    x: 8.78,
    y: 1.9,
    w: 3.25,
    h: 0.38,
    fontSize: 21,
    bold: true,
  });
  bullets(
    s,
    [
      "Implicit current-flow term enters matrix",
      "Lagged terms use accepted history only",
      "Startup follows one-cycle warm-up semantics",
      "Supports exact or truncated kernel mode",
    ],
    8.72,
    2.38,
    3.35,
    19,
    0.88
  );

  finalize(s);
}

// 5) Non-IMPEDANCE support changes + rationale
{
  const s = pptx.addSlide();
  header(
    s,
    "Support Hooks Outside IMPEDANCE",
    "Small cross-cutting additions were made to keep coupling retries safe and deterministic"
  );

  frame(s, 0.7, 1.58, 7.8, 5.45);
  s.addImage({ path: img("coupling_state_flow.png"), x: 0.95, y: 1.84, w: 7.3, h: 5.02 });

  bullets(
    s,
    [
      "Interface snapshots accepted persistent state",
      "update_state restores snapshot before each trial",
      "Integrator calls accept_timestep only on accepted steps",
      "Steady-init path clears dt-dependent persistent memory",
    ],
    8.72,
    2.16,
    3.35,
    19,
    0.86
  );

  s.addText("Why: preserve deterministic retries and avoid stale-memory carryover.", {
    x: 8.72,
    y: 6.32,
    w: 3.35,
    h: 0.62,
    fontSize: 17,
    bold: true,
  });

  finalize(s);
}

// 6) Impedance vs RCR
{
  const s = pptx.addSlide();
  header(
    s,
    "IMPEDANCE vs RCR",
    "Both remain supported; choice depends on fidelity needs and runtime budget"
  );

  frame(s, 0.7, 1.58, 8.0, 5.1);
  s.addImage({ path: img("impedance_vs_rcr_matrix.png"), x: 0.95, y: 1.85, w: 7.5, h: 4.65 });

  bullets(
    s,
    [
      "IMPEDANCE: richer waveform behavior with memory",
      "RCR: faster baseline model with simpler dynamics",
      "Select model type based on study objective",
    ],
    8.92,
    2.34,
    3.05,
    20,
    0.94
  );

  s.addText("Comparison is behavioral and use-case driven.", {
    x: 8.92,
    y: 6.18,
    w: 3.05,
    h: 0.74,
    fontSize: 17,
    bold: true,
  });

  finalize(s);
}

// 7) Workflow implications
{
  const s = pptx.addSlide();
  header(
    s,
    "Practical Implications",
    "Impact summary for coupled 3D-0D tuning loops and standard full 0D execution"
  );

  frame(s, 0.7, 1.58, 6.35, 5.45);
  s.addImage({ path: img("iteration_pipeline.png"), x: 0.95, y: 1.84, w: 5.85, h: 5.0 });

  s.addText("3D-0D", {
    x: 7.35,
    y: 1.86,
    w: 4.6,
    h: 0.42,
    fontSize: 22,
    bold: true,
  });
  bullets(
    s,
    [
      "Retry loops remain deterministic",
      "State rollback is explicit and controlled",
      "Accept/commit boundary is clearer",
    ],
    7.35,
    2.34,
    4.55,
    20,
    0.84
  );

  s.addText("Full 0D", {
    x: 7.35,
    y: 4.9,
    w: 4.6,
    h: 0.42,
    fontSize: 22,
    bold: true,
  });
  bullets(
    s,
    [
      "Configuration remains straightforward",
      "Validation checks catch kernel and timestep inconsistencies",
      "Reproducibility improves through explicit persistent-state handling",
    ],
    7.35,
    5.42,
    4.55,
    19,
    0.72,
    0.62
  );

  finalize(s);
}

// 8) Evidence + takeaway
{
  const s = pptx.addSlide();
  header(
    s,
    "Evidence and Takeaway",
    "Tests and fixtures support the implementation claims and runtime behavior"
  );

  frame(s, 0.7, 1.58, 5.6, 3.75);
  frame(s, 6.5, 1.58, 5.6, 3.75);
  s.addImage({ path: img("kernel_plot.png"), x: 0.95, y: 1.85, w: 5.1, h: 3.25 });
  s.addImage({ path: img("flow_pressure_plot.png"), x: 6.75, y: 1.85, w: 5.1, h: 3.25 });

  bullets(
    s,
    [
      "Impedance fixture validates kernel and waveform behavior",
      "Interface test_04 validates retry determinism and commit effect",
      "Overall result: targeted implementation with stable coupling behavior",
    ],
    0.95,
    5.5,
    11.0,
    19,
    0.62,
    0.52
  );

  finalize(s);
}

const outPath = path.join(deckDir, "impedance_bc_impl_comparison.pptx");
pptx
  .writeFile({ fileName: outPath })
  .then(() => {
    console.log(`Wrote ${outPath}`);
  })
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
