#!/usr/bin/env node
"use strict";

const path = require("path");

let PptxGenJS;
try {
  PptxGenJS = require("pptxgenjs");
} catch (_err) {
  PptxGenJS = require("/Users/ndorn/Documents/Stanford/PhD/Marsden_Lab/SimVascular/svz/repos/svZeroDTrees/docs/slides/impedance_bc_impl_comparison/node_modules/pptxgenjs");
}

const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");

const W = 12.8;
const H = 8.0;
const deckDir = __dirname;
const img = (name) => path.join(deckDir, "assets", "generated", name);

const C = {
  bg: "F8FAFC",
  ink: "0F172A",
  mut: "334155",
  pri: "0A9396",
  sec: "005F73",
  acc: "BB3E03",
  panel: "E2E8F0",
  pale: "ECFEFF",
  pale2: "FEF3C7",
};

const FONT_HEAD = "Helvetica Neue";
const FONT_BODY = "Helvetica Neue";
const FONT_MONO = "Courier New";

const pptx = new PptxGenJS();
pptx.defineLayout({ name: "LAYOUT_16x10", width: W, height: H });
pptx.layout = "LAYOUT_16x10";
pptx.author = "Codex";
pptx.subject = "Impedance BC implementation comparison";
pptx.company = "SimVascular";
pptx.title = "Impedance BC Implementation Comparison";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: FONT_HEAD,
  bodyFontFace: FONT_BODY,
};

function addSlideTitle(slide, title, subtitle = "") {
  slide.background = { color: C.bg };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: W,
    h: 0.82,
    fill: { color: "EAF2F8" },
    line: { color: "EAF2F8" },
  });
  slide.addText(title, {
    x: 0.45,
    y: 0.18,
    w: 8.8,
    h: 0.36,
    fontFace: FONT_HEAD,
    fontSize: 20,
    color: C.ink,
    bold: true,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.45,
      y: 0.54,
      w: 10.6,
      h: 0.18,
      fontFace: FONT_BODY,
      fontSize: 11,
      color: C.mut,
    });
  }
}

function addFooter(slide, note = "") {
  slide.addShape(pptx.ShapeType.line, {
    x: 0.45,
    y: 7.55,
    w: 11.9,
    h: 0,
    line: { color: "CBD5E1", pt: 1 },
  });
  slide.addText(
    note || "Impedance BC implementation comparison: solver + workflow implications",
    {
      x: 0.45,
      y: 7.6,
      w: 11.9,
      h: 0.2,
      fontFace: FONT_BODY,
      fontSize: 9,
      color: "64748B",
      align: "right",
    }
  );
}

function bullets(slide, items, x, y, w, opts = {}) {
  const lineH = opts.lineH || 0.36;
  const fs = opts.fs || 13;
  let yy = y;
  for (const item of items) {
    slide.addText(item, {
      x,
      y: yy,
      w,
      h: lineH,
      fontFace: FONT_BODY,
      fontSize: fs,
      color: opts.color || C.ink,
      bullet: { indent: 16 },
    });
    yy += lineH;
  }
}

function panel(slide, x, y, w, h, color = "FFFFFF") {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    fill: { color },
    line: { color: C.panel, pt: 1.2 },
    shadow: { type: "outer", color: "D1D5DB", blur: 1, angle: 45, distance: 1, opacity: 0.2 },
  });
}

function finalize(slide, options = {}) {
  addFooter(slide);
  warnIfSlideHasOverlaps(slide, pptx);
  if (!options.skipBoundsCheck) {
    warnIfSlideElementsOutOfBounds(slide, pptx);
  }
}

// Slide 1
{
  const s = pptx.addSlide();
  addSlideTitle(
    s,
    "Impedance BC Implementation Comparison",
    "Prior vs new IMPEDANCE implementation and IMPEDANCE vs RCR behavior"
  );

  panel(s, 0.55, 1.15, 5.85, 5.95, "FFFFFF");
  s.addText("Scope", {
    x: 0.8,
    y: 1.4,
    w: 2.5,
    h: 0.3,
    fontFace: FONT_HEAD,
    fontSize: 16,
    bold: true,
    color: C.sec,
  });
  bullets(
    s,
    [
      "Solver code deltas in IMPEDANCE block setup and runtime behavior",
      "Functionality: convolution controls, memory persistence, and validation",
      "Rationale for major design decisions and tradeoffs",
      "Implications for 3D-0D coupling and full 0D simulation modes",
      "Evidence from fixtures and tests in svZeroDSolver/svZeroDTrees/svzt-agent",
    ],
    0.85,
    1.85,
    5.3,
    { fs: 11.5, lineH: 0.72 }
  );

  panel(s, 6.7, 1.15, 5.55, 2.75, C.pale);
  s.addText("Comparison Axis A: Prior -> New IMPEDANCE", {
    x: 6.95,
    y: 1.45,
    w: 5.1,
    h: 0.28,
    fontFace: FONT_HEAD,
    fontSize: 14,
    bold: true,
    color: C.sec,
  });
  bullets(
    s,
    [
      "Period ownership moved to simulation_parameters.cardiac_period",
      "Configure signature simplified; parser-level period requirement removed",
      "Error handling now tied to model dt/period consistency at initialization",
    ],
    7.0,
    1.9,
    5.0,
    { fs: 11, lineH: 0.48 }
  );

  panel(s, 6.7, 4.35, 5.55, 2.75, "FFF7ED");
  s.addText("Comparison Axis B: IMPEDANCE vs RCR", {
    x: 6.95,
    y: 4.65,
    w: 5.1,
    h: 0.28,
    fontFace: FONT_HEAD,
    fontSize: 14,
    bold: true,
    color: C.acc,
  });
  bullets(
    s,
    [
      "History-based periodic convolution vs low-order lumped approximation",
      "Persistent memory semantics under coupling retries and commits",
      "Runtime cost/accuracy tradeoff via exact vs truncated kernel usage",
    ],
    7.0,
    5.1,
    5.0,
    { fs: 11, lineH: 0.48 }
  );
  finalize(s);
}

// Slide 2
{
  const s = pptx.addSlide();
  addSlideTitle(s, "Architecture Context", "Where IMPEDANCE behavior differs in full 0D and 3D-0D workflows");

  panel(s, 0.6, 1.2, 5.85, 5.95, "FFFFFF");
  panel(s, 6.55, 1.2, 5.65, 5.95, "FFFFFF");
  s.addText("Full 0D Simulation Path", {
    x: 0.9, y: 1.45, w: 5.2, h: 0.26, fontFace: FONT_HEAD, fontSize: 14, bold: true, color: C.sec,
  });
  s.addShape(pptx.ShapeType.roundRect, {
    x: 0.95, y: 2.0, w: 5.1, h: 0.95, rectRadius: 0.06,
    fill: { color: "F0FDFA" }, line: { color: "99F6E4", pt: 1.0 },
  });
  s.addText("JSON boundary_conditions + simulation_parameters", {
    x: 1.15, y: 2.25, w: 4.7, h: 0.3, fontFace: FONT_BODY, fontSize: 12, color: C.ink,
  });
  s.addShape(pptx.ShapeType.chevron, {
    x: 2.9, y: 3.02, w: 1.2, h: 0.45, fill: { color: "BAE6FD" }, line: { color: "7DD3FC", pt: 1.0 },
  });
  s.addShape(pptx.ShapeType.roundRect, {
    x: 0.95, y: 3.65, w: 5.1, h: 1.3, rectRadius: 0.06,
    fill: { color: "F8FAFC" }, line: { color: "CBD5E1", pt: 1.0 },
  });
  s.addText("Model::update_time() -> ImpedanceBC::update_time()\nConvolution uses accepted one-cycle history", {
    x: 1.15, y: 3.96, w: 4.7, h: 0.72, fontFace: FONT_BODY, fontSize: 12, color: C.ink,
  });
  bullets(
    s,
    [
      "cardiac_period must align with dt and kernel length",
      "Kernel history starts after one accepted period",
      "Exact/truncated modes control per-step cost",
    ],
    1.0, 5.25, 5.2, { fs: 11.5, lineH: 0.32 }
  );

  s.addText("3D-0D Coupled Iteration Path", {
    x: 6.65, y: 1.45, w: 5.2, h: 0.26, fontFace: FONT_HEAD, fontSize: 14, bold: true, color: C.sec,
  });
  s.addShape(pptx.ShapeType.roundRect, {
    x: 6.7, y: 2.0, w: 5.1, h: 1.15, rectRadius: 0.06,
    fill: { color: "EFF6FF" }, line: { color: "BFDBFE", pt: 1.0 },
  });
  s.addText("Interface retry loop:\nupdate_state() -> run_simulation() -> return_y()/return_ydot()", {
    x: 6.95, y: 2.3, w: 4.7, h: 0.62, fontFace: FONT_BODY, fontSize: 12, color: C.ink,
  });
  s.addShape(pptx.ShapeType.chevron, {
    x: 8.65, y: 3.2, w: 1.2, h: 0.45, fill: { color: "FDE68A" }, line: { color: "FCD34D", pt: 1.0 },
  });
  s.addShape(pptx.ShapeType.roundRect, {
    x: 6.7, y: 3.85, w: 5.1, h: 1.15, rectRadius: 0.06,
    fill: { color: "FEFCE8" }, line: { color: "FDE68A", pt: 1.0 },
  });
  s.addText("Committed persistent-state snapshot controls rollback-safe retries", {
    x: 6.95, y: 4.22, w: 4.7, h: 0.42, fontFace: FONT_BODY, fontSize: 12, color: C.ink,
  });
  bullets(
    s,
    [
      "Deterministic trial reruns from identical committed state",
      "Commit occurs when caller accepts state via return_y/return_ydot",
      "Behavior explicitly verified in interface test_04",
    ],
    6.8, 5.25, 5.2, { fs: 11.5, lineH: 0.32 }
  );
  finalize(s);
}

// Slide 3
{
  const s = pptx.addSlide();
  addSlideTitle(s, "Solver Code Changes", "Period ownership refactor and parser/configuration wiring updates");

  panel(s, 0.55, 1.2, 5.9, 2.2, "ECFEFF");
  panel(s, 6.55, 1.2, 5.7, 2.2, "FEF3C7");
  s.addText("Prior implementation", {
    x: 0.8, y: 1.48, w: 2.8, h: 0.3, fontFace: FONT_HEAD, fontSize: 14, bold: true, color: C.sec,
  });
  bullets(
    s,
    [
      "Required period inside bc_values for each IMPEDANCE BC",
      "Parser checked or set model.cardiac_cycle_period from BC field",
      "configure(z, period, pd, mode, num_kernel_terms)",
    ],
    0.82, 1.92, 5.5, { fs: 12, lineH: 0.33 }
  );
  s.addText("New implementation", {
    x: 6.6, y: 1.48, w: 2.8, h: 0.3, fontFace: FONT_HEAD, fontSize: 14, bold: true, color: C.acc,
  });
  bullets(
    s,
    [
      "period removed from IMPEDANCE bc_values contract",
      "Single source of truth: simulation_parameters.cardiac_period",
      "configure(z, pd, mode, num_kernel_terms)",
    ],
    6.62, 1.92, 5.5, { fs: 12, lineH: 0.33 }
  );

  panel(s, 0.55, 3.65, 11.7, 3.35, "FFFFFF");
  s.addText("Code-level evidence (HEAD commit 161870a5)", {
    x: 0.85, y: 3.9, w: 5.1, h: 0.24, fontFace: FONT_HEAD, fontSize: 13, bold: true, color: C.ink,
  });
  s.addText(
    [
      { text: "ImpedanceBC::configure signature\n", options: { bold: true } },
      { text: "- old: configure(const std::vector<double>& z, double period, double pd, ...)\n" },
      { text: "+ new: configure(const std::vector<double>& z, double pd, ...)\n\n" },
      { text: "Initialization guard in ensure_initialized()\n", options: { bold: true } },
      { text: "const double period = model->cardiac_cycle_period;\n" },
      { text: "if (period <= 0.0) throw runtime_error(...cardiac_period > 0...);\n\n" },
      { text: "SimulationParameters::create_boundary_conditions\n", options: { bold: true } },
      { text: "period parsing and BC-level period consistency checks removed.\n" },
      { text: "IMPEDANCE now configures from z, Pd, convolution_mode, num_kernel_terms." },
    ],
    {
      x: 0.9,
      y: 4.2,
      w: 11.0,
      h: 2.55,
      fontFace: FONT_MONO,
      fontSize: 10,
      color: "1F2937",
      breakLine: false,
    }
  );
  finalize(s);
}

// Slide 4
{
  const s = pptx.addSlide();
  addSlideTitle(s, "Runtime Behavior of New IMPEDANCE BC", "Discrete periodic convolution + one-cycle ring buffer memory");

  panel(s, 0.55, 1.2, 7.35, 5.95, "FFFFFF");
  s.addImage({ path: img("convolution_ringbuffer_schematic.png"), x: 0.85, y: 1.55, w: 6.8, h: 3.3 });
  bullets(
    s,
    [
      "Implicit term z0*Q(n+1) is assembled in F; lagged terms accumulate into C",
      "Lagged flows are read from accepted-state ring buffer (not transient trial state)",
      "Startup rule: lagged convolution disabled until one full accepted period elapsed",
      "Runtime checks enforce integer period/dt and z.size() == period steps",
    ],
    0.85, 5.0, 6.9, { fs: 11.8, lineH: 0.31 }
  );

  panel(s, 8.2, 1.2, 4.05, 5.95, "F8FAFC");
  s.addText("Execution modes", {
    x: 8.45, y: 1.5, w: 3.6, h: 0.24, fontFace: FONT_HEAD, fontSize: 13.5, bold: true, color: C.sec,
  });
  bullets(
    s,
    [
      "exact: num_kernel_terms = Np (full period)",
      "truncated: user sets num_kernel_terms in [1, Np]",
      "Per-step convolution cost scales with num_kernel_terms",
    ],
    8.45, 1.9, 3.55, { fs: 11, lineH: 0.31 }
  );
  s.addText("Key invariants", {
    x: 8.45, y: 3.1, w: 3.6, h: 0.24, fontFace: FONT_HEAD, fontSize: 13.5, bold: true, color: C.acc,
  });
  bullets(
    s,
    [
      "dt must remain fixed after initialization",
      "Persistent state compatibility checks include kernel size",
      "Corrupted state payloads fail fast with explicit runtime errors",
    ],
    8.45, 3.5, 3.55, { fs: 11, lineH: 0.31 }
  );
  finalize(s);
}

// Slide 5
{
  const s = pptx.addSlide();
  addSlideTitle(s, "Coupling Trial/Accept Semantics", "Rollback safety for persistent memory in repeated external-coupling trials");

  panel(s, 0.55, 1.2, 7.25, 5.95, "FFFFFF");
  s.addImage({ path: img("coupling_state_flow.png"), x: 0.85, y: 1.55, w: 6.7, h: 3.35 });
  bullets(
    s,
    [
      "return_y()/return_ydot() snapshots committed block persistent state",
      "update_state() restores committed state before each new trial",
      "run_simulation() advances from rollback-safe memory baseline",
    ],
    0.85, 5.05, 6.8, { fs: 11.8, lineH: 0.31 }
  );

  panel(s, 8.0, 1.2, 4.25, 5.95, "FFF7ED");
  s.addText("Evidence from tests", {
    x: 8.25, y: 1.5, w: 3.8, h: 0.24, fontFace: FONT_HEAD, fontSize: 13.5, bold: true, color: C.acc,
  });
  bullets(
    s,
    [
      "test_interface/test_04 repeats trial runs from same committed state",
      "Asserts deterministic equivalence across repeated trials",
      "Asserts observable state change after explicit commit",
      "Confirms IMPEDANCE persistent memory participates in coupling protocol",
    ],
    8.25, 1.9, 3.85, { fs: 11, lineH: 0.34 }
  );
  s.addText("Implication", {
    x: 8.25, y: 4.1, w: 3.7, h: 0.24, fontFace: FONT_HEAD, fontSize: 13.5, bold: true, color: C.sec,
  });
  s.addText(
    "3D solvers can probe multiple sub-iterations per external step without contaminating accepted impedance history.",
    {
      x: 8.25,
      y: 4.45,
      w: 3.8,
      h: 1.4,
      fontFace: FONT_BODY,
      fontSize: 11.2,
      color: C.ink,
    }
  );
  finalize(s);
}

// Slide 6
{
  const s = pptx.addSlide();
  addSlideTitle(s, "Functionality and Validation Surface", "What the new implementation enables and how it is guarded");

  panel(s, 0.55, 1.2, 11.7, 5.95, "FFFFFF");
  s.addTable(
    [
      [
        { text: "Capability / Guard", options: { bold: true, color: "FFFFFF" } },
        { text: "Behavior", options: { bold: true, color: "FFFFFF" } },
        { text: "Failure Mode / Message", options: { bold: true, color: "FFFFFF" } },
      ],
      ["Kernel modes", "exact (full Np) and truncated (k <= Np)", "Invalid mode or k out of range throws runtime_error"],
      ["Period source", "simulation_parameters.cardiac_period only", "cardiac_period <= 0 rejected at initialization"],
      ["Temporal consistency", "Requires integer period/dt and fixed dt post-init", "Non-integer ratio or dt change rejected"],
      ["Kernel shape", "z must be finite array with z.size() = Np", "Missing/invalid/non-finite kernel rejected"],
      ["Persistent state integrity", "Supports get/set with compatibility checks", "Corrupted payload or size mismatch rejected"],
    ],
    {
      x: 0.85,
      y: 1.55,
      w: 11.1,
      h: 3.8,
      border: { pt: 1, color: "CBD5E1" },
      fill: "FFFFFF",
      color: C.ink,
      fontFace: FONT_BODY,
      fontSize: 10.5,
      valign: "mid",
      rowH: [0.42, 0.55, 0.55, 0.55, 0.55, 0.55],
    }
  );
  s.addShape(pptx.ShapeType.rect, {
    x: 0.85,
    y: 1.55,
    w: 11.1,
    h: 0.42,
    fill: { color: C.sec },
    line: { color: C.sec, pt: 0 },
  });
  s.addText("Capability / Guard", {
    x: 0.95, y: 1.66, w: 2.8, h: 0.2, fontFace: FONT_BODY, fontSize: 11, bold: true, color: "FFFFFF",
  });
  s.addText("Behavior", {
    x: 3.95, y: 1.66, w: 3.8, h: 0.2, fontFace: FONT_BODY, fontSize: 11, bold: true, color: "FFFFFF",
  });
  s.addText("Failure Mode / Message", {
    x: 7.95, y: 1.66, w: 3.8, h: 0.2, fontFace: FONT_BODY, fontSize: 11, bold: true, color: "FFFFFF",
  });

  bullets(
    s,
    [
      "Compared with the prior contract, validation is centralized around model-level temporal consistency.",
      "This removes duplicated period declarations and reduces mismatch states between BCs and global simulation settings.",
    ],
    0.9,
    5.7,
    11.0,
    { fs: 11.3, lineH: 0.33 }
  );
  // Table objects are serialized with internal units that trigger false
  // positives in helper bound checks.
  finalize(s, { skipBoundsCheck: true });
}

// Slide 7
{
  const s = pptx.addSlide();
  addSlideTitle(s, "Implementation Decision Rationale", "Why these design choices were made and what they trade off");

  panel(s, 0.55, 1.2, 11.7, 5.95, "FFFFFF");
  s.addTable(
    [
      [
        { text: "Decision", options: { bold: true, color: "FFFFFF" } },
        { text: "Justification", options: { bold: true, color: "FFFFFF" } },
        { text: "Tradeoff", options: { bold: true, color: "FFFFFF" } },
      ],
      [
        "Remove BC-level period field",
        "Single source of truth in simulation_parameters improves consistency and parser simplicity.",
        "Requires users to set global cardiac_period explicitly for IMPEDANCE.",
      ],
      [
        "Initialize from model dt and cardiac_period",
        "Runtime checks align kernel discretization with actual integration time-step.",
        "Less tolerant to loosely specified inputs (fails fast by design).",
      ],
      [
        "Persistent state snapshot/restore in interface",
        "Guarantees deterministic trial reruns for 3D coupling sub-iterations.",
        "Adds interface bookkeeping for committed block states.",
      ],
      [
        "Exact + truncated convolution modes",
        "Exposes accuracy/performance knob for large kernels and long periods.",
        "Users must choose num_kernel_terms carefully for truncated mode.",
      ],
    ],
    {
      x: 0.85,
      y: 1.55,
      w: 11.1,
      h: 4.5,
      border: { pt: 1, color: "CBD5E1" },
      fill: "FFFFFF",
      color: C.ink,
      fontFace: FONT_BODY,
      fontSize: 10.4,
      valign: "mid",
      rowH: [0.42, 1.0, 1.0, 1.0, 1.0],
    }
  );
  s.addShape(pptx.ShapeType.rect, {
    x: 0.85,
    y: 1.55,
    w: 11.1,
    h: 0.42,
    fill: { color: C.sec },
    line: { color: C.sec, pt: 0 },
  });
  s.addText("Decision", { x: 0.95, y: 1.66, w: 2.0, h: 0.2, fontFace: FONT_BODY, fontSize: 11, bold: true, color: "FFFFFF" });
  s.addText("Justification", { x: 3.45, y: 1.66, w: 3.8, h: 0.2, fontFace: FONT_BODY, fontSize: 11, bold: true, color: "FFFFFF" });
  s.addText("Tradeoff", { x: 8.05, y: 1.66, w: 3.5, h: 0.2, fontFace: FONT_BODY, fontSize: 11, bold: true, color: "FFFFFF" });
  // Table objects are serialized with internal units that trigger false
  // positives in helper bound checks.
  finalize(s, { skipBoundsCheck: true });
}

// Slide 8
{
  const s = pptx.addSlide();
  addSlideTitle(s, "IMPEDANCE vs RCR", "Behavioral and practical comparison for model selection");

  panel(s, 0.55, 1.2, 11.7, 5.95, "FFFFFF");
  s.addTable(
    [
      [
        { text: "Dimension", options: { bold: true, color: "FFFFFF" } },
        { text: "IMPEDANCE BC", options: { bold: true, color: "FFFFFF" } },
        { text: "RCR BC", options: { bold: true, color: "FFFFFF" } },
      ],
      ["Memory", "One-cycle flow history ring buffer; explicit persistent state", "Low-order internal states only; no long kernel memory"],
      ["Frequency content", "Directly encodes structured-tree kernel waveform", "Approximates with 3-parameter lumped dynamics"],
      ["Coupling retries", "Designed for trial/accept rollback-safe semantics", "Also retry-safe, but without long convolution history"],
      ["Runtime cost", "O(num_kernel_terms) per accepted step", "Constant-time low-order update"],
      ["Input contract", "z kernel + global cardiac_period + mode", "Rp, C, Rd parameters (+ optional variants)"],
      ["Use cases", "High-fidelity distal bed effects / wave reflections", "Fast reduced-order approximation and robust baseline"],
    ],
    {
      x: 0.85,
      y: 1.55,
      w: 11.1,
      h: 4.8,
      border: { pt: 1, color: "CBD5E1" },
      fill: "FFFFFF",
      color: C.ink,
      fontFace: FONT_BODY,
      fontSize: 10.6,
      valign: "mid",
      rowH: [0.42, 0.7, 0.7, 0.7, 0.65, 0.65, 0.7],
    }
  );
  s.addShape(pptx.ShapeType.rect, {
    x: 0.85,
    y: 1.55,
    w: 11.1,
    h: 0.42,
    fill: { color: C.sec },
    line: { color: C.sec, pt: 0 },
  });
  s.addText("Dimension", { x: 0.95, y: 1.66, w: 2.2, h: 0.2, fontFace: FONT_BODY, fontSize: 11, bold: true, color: "FFFFFF" });
  s.addText("IMPEDANCE BC", { x: 4.15, y: 1.66, w: 2.8, h: 0.2, fontFace: FONT_BODY, fontSize: 11, bold: true, color: "FFFFFF" });
  s.addText("RCR BC", { x: 8.25, y: 1.66, w: 2.2, h: 0.2, fontFace: FONT_BODY, fontSize: 11, bold: true, color: "FFFFFF" });

  s.addText("Practical interpretation: IMPEDANCE provides richer distal dynamics at higher runtime/config complexity; RCR remains the lightweight baseline.", {
    x: 0.9, y: 6.55, w: 11.0, h: 0.5, fontFace: FONT_BODY, fontSize: 11.2, color: C.mut,
  });
  // Table objects are serialized with internal units that trigger false
  // positives in helper bound checks.
  finalize(s, { skipBoundsCheck: true });
}

// Slide 9
{
  const s = pptx.addSlide();
  addSlideTitle(s, "3D-0D Workflow Implications", "How the new impedance path behaves inside iterative tuning/submission loops");

  panel(s, 0.55, 1.2, 7.4, 5.95, "FFFFFF");
  s.addImage({ path: img("iteration_pipeline.png"), x: 0.85, y: 1.55, w: 6.9, h: 4.1 });
  bullets(
    s,
    [
      "Tuning artifacts: optimized_params.csv, PA snapshot, tuned 0D coupling config",
      "Gate outputs drive branch: converged -> postop submit, else regenerate reduced PA seed",
    ],
    0.85, 5.85, 6.95, { fs: 11.2, lineH: 0.31 }
  );

  panel(s, 8.1, 1.2, 4.15, 5.95, "F8FAFC");
  s.addText("Operational implications", {
    x: 8.35, y: 1.5, w: 3.75, h: 0.25, fontFace: FONT_HEAD, fontSize: 13.5, bold: true, color: C.sec,
  });
  bullets(
    s,
    [
      "Impedance tune_space is now validated end-to-end across defaults and patient overrides",
      "Decision payloads explicitly include convergence state and regeneration path",
      "3D submission logic can pause to needs_review on missing artifacts or failed extraction",
      "Pipeline naturally separates physics tuning from scheduler lifecycle handling",
    ],
    8.35, 1.92, 3.75, { fs: 11, lineH: 0.34 }
  );
  s.addText("Relevant modules", {
    x: 8.35, y: 4.2, w: 3.75, h: 0.24, fontFace: FONT_HEAD, fontSize: 13.5, bold: true, color: C.acc,
  });
  s.addText(
    "svzerodtrees.tuning.iteration\nsvzt-agent templates/slurm/job_template.sh\nsvzt config model/load merge rules",
    {
      x: 8.35,
      y: 4.55,
      w: 3.75,
      h: 1.35,
      fontFace: FONT_MONO,
      fontSize: 9.8,
      color: "1F2937",
    }
  );
  finalize(s);
}

// Slide 10
{
  const s = pptx.addSlide();
  addSlideTitle(s, "Full 0D Implications", "Configuration contract and reproducibility behavior after the refactor");

  panel(s, 0.55, 1.2, 5.8, 5.95, "FFFFFF");
  s.addText("Input contract update", {
    x: 0.82, y: 1.48, w: 5.2, h: 0.25, fontFace: FONT_HEAD, fontSize: 14, bold: true, color: C.sec,
  });
  s.addText(
    [
      { text: "// old (removed)\n", options: { bold: true } },
      { text: "\"bc_values\": { \"period\": 1.0, \"Pd\": 80000.0, \"z\": [...] }\n\n" },
      { text: "// new\n", options: { bold: true } },
      { text: "\"simulation_parameters\": { \"cardiac_period\": 1.0, ... }\n" },
      { text: "\"bc_values\": { \"Pd\": 80000.0, \"z\": [...], \"convolution_mode\": \"exact\" }\n" },
    ],
    {
      x: 0.85,
      y: 1.9,
      w: 5.35,
      h: 2.2,
      fontFace: FONT_MONO,
      fontSize: 9.9,
      color: "1F2937",
    }
  );
  bullets(
    s,
    [
      "Cardiac period is globally defined and consumed by IMPEDANCE at initialization",
      "Kernel length and period/dt mismatch now fail fast with explicit diagnostics",
      "Fixture updates moved period declarations from BC block to simulation parameters",
    ],
    0.82, 4.25, 5.35, { fs: 11.2, lineH: 0.32 }
  );

  panel(s, 6.45, 1.2, 5.8, 5.95, "F8FAFC");
  s.addText("Reproducibility and debugging", {
    x: 6.72, y: 1.48, w: 5.2, h: 0.25, fontFace: FONT_HEAD, fontSize: 14, bold: true, color: C.acc,
  });
  bullets(
    s,
    [
      "Deterministic behavior depends on fixed dt and explicit committed-state transitions",
      "Persistent state serialization includes dt, period steps, kernel terms, and flow history",
      "Corrupted or incompatible persistent payloads trigger guarded runtime failures",
      "Error messages now reference missing/invalid global cardiac_period rather than BC period",
    ],
    6.72, 1.9, 5.25, { fs: 11.2, lineH: 0.34 }
  );
  s.addText("Recommended full-0D checklist", {
    x: 6.72, y: 4.65, w: 5.2, h: 0.24, fontFace: FONT_HEAD, fontSize: 13.3, bold: true, color: C.sec,
  });
  bullets(
    s,
    [
      "Set simulation_parameters.cardiac_period",
      "Ensure period/dt is an integer count",
      "Set z length to one period in time steps",
      "Use truncated mode only with validated num_kernel_terms",
    ],
    6.72, 4.95, 5.2, { fs: 10.8, lineH: 0.3 }
  );
  finalize(s);
}

// Slide 11
{
  const s = pptx.addSlide();
  addSlideTitle(s, "Evidence from Fixtures and Tests", "Kernel and waveform evidence plus explicit test coverage");

  panel(s, 0.55, 1.2, 5.7, 2.95, "FFFFFF");
  s.addText("Kernel fixture evidence", {
    x: 0.82, y: 1.45, w: 5.2, h: 0.24, fontFace: FONT_HEAD, fontSize: 13.5, bold: true, color: C.sec,
  });
  s.addImage({ path: img("kernel_plot.png"), x: 0.85, y: 1.75, w: 5.15, h: 2.2 });

  panel(s, 6.55, 1.2, 5.7, 2.95, "FFFFFF");
  s.addText("Flow/pressure fixture evidence", {
    x: 6.82, y: 1.45, w: 5.2, h: 0.24, fontFace: FONT_HEAD, fontSize: 13.5, bold: true, color: C.sec,
  });
  s.addImage({ path: img("flow_pressure_plot.png"), x: 6.85, y: 1.75, w: 5.15, h: 2.2 });

  panel(s, 0.55, 4.35, 11.7, 2.8, "F8FAFC");
  s.addText("Key tests and fixtures referenced", {
    x: 0.82, y: 4.6, w: 4.8, h: 0.24, fontFace: FONT_HEAD, fontSize: 13.5, bold: true, color: C.acc,
  });
  bullets(
    s,
    [
      "tests/cases/pulsatileFlow_R_impedance.json and result_pulsatileFlow_R_impedance.json",
      "tests/test_interface/test_04 (coupling trial/accept deterministic semantics)",
      "src/model/ImpedanceBC.{h,cpp} and src/solve/SimulationParameters.cpp",
      "svzerodtrees/tuning/iteration.py and svzt-agent templates/slurm/job_template.sh",
    ],
    0.82, 4.95, 11.1, { fs: 11.1, lineH: 0.33 }
  );
  s.addText("Takeaway: the refactor improves contract consistency and coupling safety while preserving a clear runtime cost/accuracy control surface.", {
    x: 0.82, y: 6.43, w: 11.0, h: 0.45, fontFace: FONT_BODY, fontSize: 11.4, color: C.mut,
  });
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
