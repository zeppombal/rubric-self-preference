#!/usr/bin/env python3
"""Build interactive HTML dashboard for HealthBench judge analysis.

Loads all experimental data, packs it into compact JSON, and generates
a self-contained HTML file with embedded data, CSS, and JS.

Key differences from IFEval dashboard:
- Only SR method (no AR, DA, PWC)
- Scoring mode toggle (weighted vs uniform)
- Committee = reference (not committee = judge)
- Rubric points stored alongside rubric booleans
- Committee-based reference
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from math import comb
from hb_data_loading import (
    load_all_data, load_instance_tags,
    GENERATORS, JUDGES, ALL_JUDGES, FAMILIES, MODEL_TO_FAMILY,
    N_INSTANCES, SHORT_NAMES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def pack_data(all_data):
    """Pack all loaded data into compact JSON-serializable format."""
    gen_data = all_data["gen"]
    sr_data = all_data["sr"]

    packed = {
        "generators": GENERATORS,
        "allJudges": ALL_JUDGES,
        "defaultJudges": list(JUDGES),
        "families": FAMILIES,
        "modelToFamily": MODEL_TO_FAMILY,
        "shortNames": SHORT_NAMES,
        "nInstances": N_INSTANCES,
    }

    # Rubric counts per instance (shared across generators — validated identical)
    ref_gen = GENERATORS[0]
    packed["rubricCounts"] = [len(gen_data[ref_gen][i]) for i in range(N_INSTANCES)]

    # Rubric points flattened (shared across generators — validated identical)
    points_flat = []
    for i in range(N_INSTANCES):
        for r in gen_data[ref_gen][i]:
            points_flat.append(r["points"])
    packed["rubricPointsFlat"] = points_flat

    # SR rubric data: flattened binary strings per judge|gen
    packed["srRubricFlat"] = {}
    for (judge, gen), inst_list in sr_data.items():
        flat = []
        for rubrics in inst_list:
            flat.extend("1" if r["criteria_met"] else "0" for r in rubrics)
        packed["srRubricFlat"][f"{judge}|{gen}"] = "".join(flat)

    # --- Interactive Analysis: additional data for filtering ---

    # 1a. Rubric lengths (character count of criterion text)
    lengths_flat = []
    for i in range(N_INSTANCES):
        for r in gen_data[ref_gen][i]:
            lengths_flat.append(len(r["criterion"]))
    packed["rubricLengthsFlat"] = lengths_flat

    # 1b. Rubric axes bitmask (only axis:* tags)
    all_axis_tags = set()
    for i in range(N_INSTANCES):
        for r in gen_data[ref_gen][i]:
            for t in r.get("tags", []):
                if t.startswith("axis:"):
                    all_axis_tags.add(t)
    axis_names = sorted(all_axis_tags)
    axis_to_idx = {t: idx for idx, t in enumerate(axis_names)}
    packed["axisNames"] = axis_names

    axes_flat = []
    for i in range(N_INSTANCES):
        for r in gen_data[ref_gen][i]:
            bits = 0
            for t in r.get("tags", []):
                if t in axis_to_idx:
                    bits |= (1 << axis_to_idx[t])
            axes_flat.append(bits)
    packed["rubricAxesFlat"] = axes_flat

    # 1c. Instance themes bitmask (only theme:* example_tags)
    instance_tags = load_instance_tags()
    all_themes = set()
    for tags in instance_tags:
        for t in tags:
            if t.startswith("theme:"):
                all_themes.add(t)
    theme_names = sorted(all_themes)
    theme_to_idx = {t: idx for idx, t in enumerate(theme_names)}
    packed["themeNames"] = theme_names

    themes_flat = []
    for tags in instance_tags:
        bits = 0
        for t in tags:
            if t in theme_to_idx:
                bits |= (1 << theme_to_idx[t])
        themes_flat.append(bits)
    packed["instanceThemesFlat"] = themes_flat

    # 1d. Rubric agreement (pairwise agreement across 12 judges, averaged across generators)
    n_judges = len(ALL_JUDGES)
    total_pairs = comb(n_judges, 2)
    rubric_counts = packed["rubricCounts"]
    agreement_flat = []
    for i in range(N_INSTANCES):
        n_rubrics = rubric_counts[i]
        for r_idx in range(n_rubrics):
            gen_agreements = []
            for gen in GENERATORS:
                labels = []
                for judge in ALL_JUDGES:
                    key = (judge, gen)
                    if key not in sr_data:
                        continue
                    labels.append(sr_data[key][i][r_idx]["criteria_met"])
                if len(labels) == n_judges:
                    n_true = sum(labels)
                    n_false = n_judges - n_true
                    agr = (comb(n_true, 2) + comb(n_false, 2)) / total_pairs
                    gen_agreements.append(agr)
            if gen_agreements:
                agreement_flat.append(round(sum(gen_agreements) / len(gen_agreements), 4))
            else:
                agreement_flat.append(1.0)
    packed["rubricAgreementFlat"] = agreement_flat

    # 1e. Default committee
    packed["defaultCommittee"] = [
        "gemma_3_27b_it", "qwen3_235b_instruct",
        "llama_4_maverick_17b_128e_instruct", "gpt_5", "claude_4_5_sonnet",
    ]

    return packed


def generate_html(packed_data_json):
    """Generate the complete HTML dashboard."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HealthBench Judge Analysis Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400;1,700&family=Fraunces:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
{CSS_BLOCK}
</head>
<body>

<header class="page-header">
  <h1>HealthBench Judge Analysis</h1>
  <p class="subtitle">Interactive analysis of accuracy and self-preference bias in HealthBench SR evaluations (weighted &amp; uniform scoring)</p>
</header>

<div class="layout">
  <aside class="panel" id="panel">
    <div class="panel-inner">
      <div class="panel-section">
        <h3>Scoring Mode</h3>
        <label class="cb-label"><input type="radio" name="scoring-mode" value="weighted" checked onchange="onSelectionChange()"><span>Weighted (actual points)</span></label>
        <label class="cb-label"><input type="radio" name="scoring-mode" value="uniform" onchange="onSelectionChange()"><span>Uniform (+1/-1)</span></label>
      </div>
      <div class="panel-section">
        <h3>Reference Committee</h3>
        <p class="note" style="margin-bottom:0.25rem;">Select reference committee members:</p>
        <div id="committee-member-checkboxes"></div>
        <div class="panel-actions">
          <button onclick="selectAllCommittee()" class="btn-sm">All</button>
          <button onclick="selectNoneCommittee()" class="btn-sm">None</button>
        </div>
        <p class="note" id="committee-status" style="margin-top:0.5rem;"></p>
      </div>
      <div class="panel-section">
        <h3>Judges</h3>
        <div class="panel-actions">
          <button onclick="selectAll('judge')" class="btn-sm">All</button>
          <button onclick="selectNone('judge')" class="btn-sm">None</button>
          <button onclick="selectDefault('judge')" class="btn-sm">Default 9</button>
        </div>
        <div id="judge-checkboxes"></div>
      </div>
      <div class="panel-section">
        <h3>Generators</h3>
        <div class="panel-actions">
          <button onclick="selectAll('gen')" class="btn-sm">All</button>
          <button onclick="selectNone('gen')" class="btn-sm">None</button>
        </div>
        <div id="gen-checkboxes"></div>
      </div>
      <div class="panel-section info-box">
        <p><strong>Selected:</strong> <span id="sel-judges-count">0</span> judges, <span id="sel-gens-count">0</span> generators</p>
        <p class="note">* SR-only judges shown with asterisk</p>
      </div>
    </div>
  </aside>

  <main class="dashboard" id="dashboard">
    <div class="tabs" id="tabs">
      <button class="tab active" data-tab="aggregate">Aggregate</button>
      <button class="tab" data-tab="per-judge">Per-Judge</button>
      <button class="tab" data-tab="per-generator">Per-Generator</button>
      <button class="tab" data-tab="charts">Charts</button>
      <button class="tab" data-tab="interactive">Interactive Analysis</button>
    </div>

    <div class="tab-content" id="tab-aggregate">
      <div class="card-grid">
        <div class="card">
          <h3>System-Level Accuracy</h3>
          <p class="card-desc">How well judges rank generators (higher MPA = better; lower MRD = better)</p>
          <div id="tbl-sys-accuracy"></div>
        </div>
        <div class="card">
          <h3>Instance-Level Accuracy</h3>
          <p class="card-desc">Agreement with reference on individual instance-pair comparisons</p>
          <div id="tbl-inst-accuracy"></div>
        </div>
        <div class="card">
          <h3>Rubric-Level Accuracy</h3>
          <p class="card-desc">Per-rubric agreement with reference</p>
          <div id="tbl-rubric-accuracy"></div>
        </div>
      </div>
      <div class="card-grid">
        <div class="card">
          <h3>System-Level Bias</h3>
          <p class="card-desc">Self-preference in ranking and scoring (negative MRD-SP = self-inflation; positive MSD-SP = score inflation)</p>
          <div id="tbl-sys-bias"></div>
        </div>
        <div class="card">
          <h3>Instance-Level Bias</h3>
          <p class="card-desc">Overestimation of own outputs relative to others</p>
          <div id="tbl-inst-bias"></div>
        </div>
        <div class="card">
          <h3>Rubric-Level Bias</h3>
          <p class="card-desc">Self-preference at rubric level</p>
          <div id="tbl-rubric-bias"></div>
        </div>
      </div>
    </div>

    <div class="tab-content hidden" id="tab-per-judge">
      <h2 class="section-heading">Accuracy</h2>
      <div class="card">
        <h3>Pairwise Accuracy (MPA) by Judge</h3>
        <div id="tbl-mpa-judge"></div>
      </div>
      <div class="card">
        <h3>Instance Pairwise Accuracy (MIPA) by Judge</h3>
        <div id="tbl-mipa-judge"></div>
      </div>
      <div class="card">
        <h3>Rubric Accuracy (MRA) by Judge</h3>
        <div id="tbl-mra-judge"></div>
      </div>
      <h2 class="section-heading">System-Level Bias</h2>
      <div class="card">
        <h3>MRD-SP / MRD-FSP by Judge</h3>
        <p class="card-desc">Signed rank difference: self vs others (SP) and family vs non-family (FSP). Negative = judge ranks itself/family better.</p>
        <div id="tbl-mrdsp-judge"></div>
      </div>
      <div class="card">
        <h3>MSD-SP / MSD-FSP by Judge</h3>
        <p class="card-desc">Score delta: self vs others (SP) and family vs non-family (FSP). Positive = judge inflates itself/family more.</p>
        <div id="tbl-msdsp-judge"></div>
      </div>
      <h2 class="section-heading">Self-Preference Bias</h2>
      <div class="card">
        <h3>Instance Self-Preference Bias (MISPB) by Judge</h3>
        <p class="card-desc">Corrected overestimation rate: raw - other. Positive = self-preference.</p>
        <div id="tbl-mispb-judge"></div>
      </div>
      <div class="card">
        <h3>Family MISPB (MISPB-F) by Judge</h3>
        <div id="tbl-mispbf-judge"></div>
      </div>
      <div class="card">
        <h3>Family-Only MISPB (MISPB-FO) by Judge</h3>
        <p class="card-desc">Family excluding self: isolates sibling preference</p>
        <div id="tbl-mispbfo-judge"></div>
      </div>
      <div class="card">
        <h3>Harmful Self-Preference Propensity (HSPP) by Judge</h3>
        <p class="card-desc">Restricted to instances where reference says other is better</p>
        <div id="tbl-hspp-judge"></div>
      </div>
      <div class="card">
        <h3>Family HSPP (HSPP-F) by Judge</h3>
        <div id="tbl-hsppf-judge"></div>
      </div>
      <div class="card">
        <h3>Family-Only HSPP (HSPP-FO) by Judge</h3>
        <div id="tbl-hsppfo-judge"></div>
      </div>
      <div class="card">
        <h3>Rubric Self-Preference Bias (MRSPB) by Judge</h3>
        <p class="card-desc">Corrected rubric-level false positive rate.</p>
        <div id="tbl-mrspb-judge"></div>
      </div>
      <div class="card">
        <h3>Family MRSPB (MRSPB-F) by Judge</h3>
        <div id="tbl-mrspbf-judge"></div>
      </div>
      <div class="card">
        <h3>Family-Only MRSPB (MRSPB-FO) by Judge</h3>
        <p class="card-desc">Family excluding self.</p>
        <div id="tbl-mrspbfo-judge"></div>
      </div>
      <div class="card">
        <h3>MRSPB Error-Denominator (MRSPB-err) by Judge</h3>
        <p class="card-desc">Denominator restricted to rubrics where reference says "not met"</p>
        <div id="tbl-mrspberr-judge"></div>
      </div>
      <div class="card">
        <h3>Family MRSPB-err (MRSPB-err-F) by Judge</h3>
        <div id="tbl-mrspberrf-judge"></div>
      </div>
      <div class="card">
        <h3>Family-Only MRSPB-err (MRSPB-err-FO) by Judge</h3>
        <div id="tbl-mrspberrfo-judge"></div>
      </div>
    </div>

    <div class="tab-content hidden" id="tab-per-generator">
      <div class="card">
        <h3>Reference Scores</h3>
        <p class="card-desc">Committee reference system-level score per generator</p>
        <div id="tbl-ref-scores"></div>
      </div>
      <div class="card">
        <h3>System Scores (Judge Mean)</h3>
        <p class="card-desc">Mean judge-assigned score per generator, averaged across selected judges</p>
        <div id="tbl-gen-scores"></div>
      </div>
      <div class="card">
        <h3>Score Deltas (Judge - Reference)</h3>
        <p class="card-desc">Positive = overestimation by judges</p>
        <div id="tbl-gen-deltas"></div>
      </div>
      <div class="card">
        <h3>Normalized Score Deltas</h3>
        <p class="card-desc">Score delta divided by reference score</p>
        <div id="tbl-gen-deltas-norm"></div>
      </div>
      <div class="card">
        <h3>Rank Deltas (Judge - Reference)</h3>
        <p class="card-desc">Negative = judges rank this generator better than reference</p>
        <div id="tbl-gen-rank-deltas"></div>
      </div>
    </div>

    <div class="tab-content hidden" id="tab-charts">
      <div class="card">
        <h3>Accuracy Metrics</h3>
        <div id="chart-accuracy" class="chart-container"></div>
      </div>
      <div class="card">
        <h3>MISPB by Judge</h3>
        <div id="chart-mispb" class="chart-container"></div>
      </div>
      <div class="card">
        <h3>Per-Judge MPA</h3>
        <div id="chart-mpa-judge" class="chart-container"></div>
      </div>
      <div class="card">
        <h3>System Scores: Judge vs. Reference</h3>
        <div id="chart-scores" class="chart-container"></div>
      </div>
      <div class="card">
        <h3>MRSPB by Judge</h3>
        <div id="chart-mrspb" class="chart-container"></div>
      </div>
    </div>

    <div class="tab-content hidden" id="tab-interactive">
      <div class="ia-layout">
        <div class="ia-filters">
          <h3>Filters</h3>

          <div class="filter-group">
            <label>Points Range</label>
            <div class="range-inputs">
              <input type="number" id="ia-points-min" value="-10" min="-10" max="10" onchange="onIAFilterChange()">
              <span>to</span>
              <input type="number" id="ia-points-max" value="10" min="-10" max="10" onchange="onIAFilterChange()">
            </div>
          </div>

          <div class="filter-group">
            <label>Rubric Length (chars)</label>
            <div class="range-inputs">
              <input type="number" id="ia-len-min" value="0" min="0" onchange="onIAFilterChange()">
              <span>to</span>
              <input type="number" id="ia-len-max" value="9999" onchange="onIAFilterChange()">
            </div>
            <p class="note" id="ia-len-percentiles"></p>
          </div>

          <div class="filter-group">
            <label>Axis Filter</label>
            <div class="panel-actions">
              <button onclick="iaAxisAll()" class="btn-sm">All</button>
              <button onclick="iaAxisNone()" class="btn-sm">None</button>
            </div>
            <div id="ia-axis-checkboxes"></div>
          </div>

          <div class="filter-group">
            <label>Theme Filter</label>
            <div class="panel-actions">
              <button onclick="iaThemeAll()" class="btn-sm">All</button>
              <button onclick="iaThemeNone()" class="btn-sm">None</button>
            </div>
            <div id="ia-theme-checkboxes"></div>
          </div>

          <div class="filter-group">
            <label>LLM Agreement Range</label>
            <div class="range-inputs">
              <span id="ia-agr-min-val">0%</span>
              <span>to</span>
              <span id="ia-agr-max-val">100%</span>
            </div>
            <label class="range-label-sm">Min</label>
            <input type="range" id="ia-agr-min" min="0" max="100" value="0" step="5" oninput="onIAAgrSlider()">
            <label class="range-label-sm">Max</label>
            <input type="range" id="ia-agr-max" min="0" max="100" value="100" step="5" oninput="onIAAgrSlider()">
          </div>

          <div class="info-box" id="ia-filter-status">
            <p>Rubrics: <strong><span id="ia-rubric-count">0</span></strong> / <span id="ia-rubric-total">0</span></p>
            <p>Instances: <strong><span id="ia-instance-count">0</span></strong> / <span id="ia-instance-total">0</span></p>
          </div>

          <button onclick="iaResetFilters()" class="btn-sm" style="margin-top:0.5rem;width:100%">Reset All Filters</button>
        </div>

        <div class="ia-results">
          <div class="card">
            <h3>MSD Matrix (Judge x Generator)</h3>
            <p class="card-desc">Entry = dGm - mean(dGm): per-generator score delta controlled for judge's average overestimation. Diagonal = self-preference.</p>
            <div id="ia-msd-matrix" style="overflow-x:auto;"></div>
          </div>
          <div class="card">
            <h3>Delta MSD Matrix (Filtered - Unfiltered)</h3>
            <p class="card-desc">Difference between filtered and unfiltered MSD values. Zeros = no change from filtering.</p>
            <div id="ia-delta-msd-matrix" style="overflow-x:auto;"></div>
          </div>
          <div class="card">
            <h3>Summary Table</h3>
            <p class="card-desc">Self-preference and accuracy metrics per judge. "vs ref" compares to committee reference; "vs unfilt" compares to unfiltered rankings.</p>
            <div class="ia-bias-toggle">
              <span class="note">Bias metric display:</span>
              <label class="cb-label"><input type="radio" name="ia-bias-mode" value="ratio" checked onchange="iaOnBiasModeChange()"><span>Ratio (raw/other)</span></label>
              <label class="cb-label"><input type="radio" name="ia-bias-mode" value="diff" onchange="iaOnBiasModeChange()"><span>Difference (raw-other)</span></label>
            </div>
            <div id="ia-summary-table" style="overflow-x:auto;"></div>
          </div>
        </div>
      </div>
    </div>
  </main>
</div>

<script>
// ============================================================
// EMBEDDED DATA
// ============================================================
const DATA = {packed_data_json};

// ============================================================
// DATA ACCESS HELPERS
// ============================================================
const SN = DATA.shortNames;
function sn(model) {{ return SN[model] || model; }}

function getFamily(model) {{ return DATA.modelToFamily[model]; }}
function isSameFamily(a, b) {{ return getFamily(a) === getFamily(b); }}
function getOtherGens(judge, gens) {{
  const fam = getFamily(judge);
  return gens.filter(g => getFamily(g) !== fam);
}}
function getFamilyGens(judge, gens, includeSelf) {{
  const fam = getFamily(judge);
  return gens.filter(g => getFamily(g) === fam && (includeSelf || g !== judge));
}}

// Decode rubric counts and points for an instance
function getRubricSlice(instIdx) {{
  let pos = 0;
  for (let i = 0; i < instIdx; i++) pos += DATA.rubricCounts[i];
  const n = DATA.rubricCounts[instIdx];
  return {{ offset: pos, count: n }};
}}

function getInstancePoints(instIdx) {{
  const s = getRubricSlice(instIdx);
  return DATA.rubricPointsFlat.slice(s.offset, s.offset + s.count);
}}

// Build per-instance rubric offset table once
let _rubricOffsets = null;
function buildRubricOffsets() {{
  if (_rubricOffsets) return;
  _rubricOffsets = new Int32Array(DATA.nInstances);
  let pos = 0;
  for (let i = 0; i < DATA.nInstances; i++) {{
    _rubricOffsets[i] = pos;
    pos += DATA.rubricCounts[i];
  }}
}}

const _rubricCache = {{}};
function getJudgeRubrics(judge, gen) {{
  const cacheKey = judge + '|' + gen;
  if (_rubricCache[cacheKey]) return _rubricCache[cacheKey];
  const flat = DATA.srRubricFlat[cacheKey];
  if (!flat) return null;
  buildRubricOffsets();
  const result = new Array(DATA.nInstances);
  let pos = 0;
  for (let i = 0; i < DATA.nInstances; i++) {{
    const n = DATA.rubricCounts[i];
    const arr = new Array(n);
    for (let j = 0; j < n; j++) arr[j] = flat[pos + j] === '1';
    result[i] = arr;
    pos += n;
  }}
  _rubricCache[cacheKey] = result;
  return result;
}}

function judgeHasData(judge) {{
  return !!DATA.srRubricFlat[judge + '|' + DATA.generators[0]];
}}

function getActiveJudges(judges) {{
  return judges.filter(j => judgeHasData(j));
}}

// ============================================================
// SCORING FUNCTIONS
// ============================================================

function getScoringMode() {{
  const el = document.querySelector('input[name="scoring-mode"]:checked');
  return el ? el.value : 'weighted';
}}

function instanceScoreWeighted(rubricBools, instIdx) {{
  buildRubricOffsets();
  const off = _rubricOffsets[instIdx];
  const n = DATA.rubricCounts[instIdx];
  let totalPossible = 0, achieved = 0;
  for (let j = 0; j < n; j++) {{
    const pts = DATA.rubricPointsFlat[off + j];
    if (pts > 0) totalPossible += pts;
    if (rubricBools[j]) achieved += pts;
  }}
  if (totalPossible <= 0) return 0;
  return Math.max(0, Math.min(1, achieved / totalPossible));
}}

function instanceScoreUniform(rubricBools, instIdx) {{
  buildRubricOffsets();
  const off = _rubricOffsets[instIdx];
  const n = DATA.rubricCounts[instIdx];
  let nPositive = 0, achieved = 0;
  for (let j = 0; j < n; j++) {{
    const pts = DATA.rubricPointsFlat[off + j];
    if (pts > 0) nPositive++;
    if (rubricBools[j]) achieved += (pts > 0 ? 1 : -1);
  }}
  if (nPositive <= 0) return 0;
  return Math.max(0, Math.min(1, achieved / nPositive));
}}

function instanceScore(rubricBools, instIdx, mode) {{
  return mode === 'weighted'
    ? instanceScoreWeighted(rubricBools, instIdx)
    : instanceScoreUniform(rubricBools, instIdx);
}}

// Pre-compute instance scores for a judge+gen
const _instScoreCache = {{}};
function getInstScores(judge, gen, mode) {{
  const ck = mode + '|' + judge + '|' + gen;
  if (_instScoreCache[ck]) return _instScoreCache[ck];
  const rubrics = getJudgeRubrics(judge, gen);
  if (!rubrics) return null;
  const arr = new Float64Array(DATA.nInstances);
  for (let i = 0; i < DATA.nInstances; i++) {{
    arr[i] = instanceScore(rubrics[i], i, mode);
  }}
  _instScoreCache[ck] = arr;
  return arr;
}}

// ============================================================
// REFERENCE CONSTRUCTION
// ============================================================

function buildCommitteeReference(members, gens) {{
  // Majority vote on rubrics across members
  // Returns refRubrics[gen][i] = array of bools
  const ref = {{}};
  for (const gen of gens) {{
    const memberRubrics = [];
    for (const m of members) {{
      const r = getJudgeRubrics(m, gen);
      if (r) memberRubrics.push(r);
    }}
    if (memberRubrics.length === 0) {{
      // All-false reference
      ref[gen] = new Array(DATA.nInstances);
      for (let i = 0; i < DATA.nInstances; i++) {{
        ref[gen][i] = new Array(DATA.rubricCounts[i]).fill(false);
      }}
      continue;
    }}
    const nm = memberRubrics.length;
    const threshold = nm / 2;
    ref[gen] = new Array(DATA.nInstances);
    for (let i = 0; i < DATA.nInstances; i++) {{
      const n = DATA.rubricCounts[i];
      const arr = new Array(n);
      for (let ri = 0; ri < n; ri++) {{
        let votes = 0;
        for (let mi = 0; mi < nm; mi++) votes += memberRubrics[mi][i][ri] ? 1 : 0;
        arr[ri] = votes > threshold;
      }}
      ref[gen][i] = arr;
    }}
  }}
  return ref;
}}


// Compute reference instance scores
function refInstanceScore(refRubrics, instIdx, mode) {{
  return instanceScore(refRubrics, instIdx, mode);
}}

function computeRefInstScores(refRubrics, gens, mode) {{
  const scores = {{}};
  for (const gen of gens) {{
    if (!refRubrics[gen]) continue;
    const arr = new Float64Array(DATA.nInstances);
    for (let i = 0; i < DATA.nInstances; i++) {{
      arr[i] = refInstanceScore(refRubrics[gen][i], i, mode);
    }}
    scores[gen] = arr;
  }}
  return scores;
}}

function computeRefSystemScores(refRubrics, gens, mode) {{
  const instScores = computeRefInstScores(refRubrics, gens, mode);
  const scores = {{}};
  for (const gen of gens) {{
    if (!instScores[gen]) continue;
    let sum = 0;
    for (let i = 0; i < DATA.nInstances; i++) sum += instScores[gen][i];
    scores[gen] = sum / DATA.nInstances;
  }}
  return scores;
}}

// ============================================================
// RANKING HELPER
// ============================================================
function rankdata(arr) {{
  const n = arr.length;
  const indexed = arr.map((v, i) => [v, i]);
  indexed.sort((a, b) => b[0] - a[0]);
  const ranks = new Array(n);
  let i = 0;
  while (i < n) {{
    let j = i;
    while (j < n - 1 && indexed[j+1][0] === indexed[i][0]) j++;
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[indexed[k][1]] = avgRank;
    i = j + 1;
  }}
  return ranks;
}}

// ============================================================
// SYSTEM-LEVEL SCORES
// ============================================================

function computeSystemScoresSR(judges, gens, mode) {{
  const scores = {{}};
  for (const j of judges) {{
    for (const g of gens) {{
      const instScores = getInstScores(j, g, mode);
      if (!instScores) continue;
      let sum = 0;
      for (let i = 0; i < DATA.nInstances; i++) sum += instScores[i];
      scores[j + '|' + g] = sum / DATA.nInstances;
    }}
  }}
  return scores;
}}

// ============================================================
// SYSTEM-LEVEL ACCURACY
// ============================================================

function computeMPA(sysScores, refScoresByJudge, judges, gens) {{
  const perJudge = {{}}, perJudgeNConc = {{}}, perJudgeNTotal = {{}};
  for (const judge of judges) {{
    const ref = refScoresByJudge[judge];
    if (!ref) continue;
    const available = gens.filter(g => sysScores[judge+'|'+g] !== undefined && ref[g] !== undefined);
    if (available.length < 2) continue;
    let concordant = 0, total = 0;
    for (let i = 0; i < available.length; i++) {{
      for (let j = i+1; j < available.length; j++) {{
        const g1 = available[i], g2 = available[j];
        const jd = sysScores[judge+'|'+g1] - sysScores[judge+'|'+g2];
        const rd = ref[g1] - ref[g2];
        if ((jd > 0 && rd > 0) || (jd < 0 && rd < 0) || (jd === 0 && rd === 0)) concordant++;
        total++;
      }}
    }}
    perJudge[judge] = concordant / total;
    perJudgeNConc[judge] = concordant;
    perJudgeNTotal[judge] = total;
  }}
  const vals = Object.values(perJudge);
  return {{ mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null, perJudge, perJudgeNConc, perJudgeNTotal }};
}}

function computeMRD(sysScores, refScoresByJudge, judges, gens) {{
  const perJudge = {{}};
  for (const judge of judges) {{
    const ref = refScoresByJudge[judge];
    if (!ref) continue;
    const available = gens.filter(g => sysScores[judge+'|'+g] !== undefined && ref[g] !== undefined);
    if (available.length < 2) continue;
    const jArr = available.map(g => sysScores[judge+'|'+g]);
    const rArr = available.map(g => ref[g]);
    const jRanks = rankdata(jArr);
    const rRanks = rankdata(rArr);
    let sum = 0;
    for (let i = 0; i < available.length; i++) sum += Math.abs(jRanks[i] - rRanks[i]);
    perJudge[judge] = sum / available.length;
  }}
  const vals = Object.values(perJudge);
  return {{ mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null, perJudge }};
}}

function computeMSD(sysScores, refScoresByJudge, judges, gens) {{
  const perJudge = {{}};
  for (const judge of judges) {{
    const ref = refScoresByJudge[judge];
    if (!ref) continue;
    const available = gens.filter(g => sysScores[judge+'|'+g] !== undefined && ref[g] !== undefined);
    if (!available.length) continue;
    const deltas = available.map(g => sysScores[judge+'|'+g] - ref[g]);
    perJudge[judge] = deltas.reduce((a,b)=>a+b,0) / deltas.length;
  }}
  const vals = Object.values(perJudge);
  return {{ mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null, perJudge }};
}}

function computeMSDnorm(sysScores, refScoresByJudge, judges, gens) {{
  const perJudge = {{}};
  for (const judge of judges) {{
    const ref = refScoresByJudge[judge];
    if (!ref) continue;
    const available = gens.filter(g => sysScores[judge+'|'+g] !== undefined && ref[g] !== undefined && ref[g] > 0);
    if (!available.length) continue;
    const deltas = available.map(g => (sysScores[judge+'|'+g] - ref[g]) / ref[g]);
    perJudge[judge] = deltas.reduce((a,b)=>a+b,0) / deltas.length;
  }}
  const vals = Object.values(perJudge);
  return {{ mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null, perJudge }};
}}

// ============================================================
// SYSTEM-LEVEL BIAS
// ============================================================

function computeMRDSP(sysScores, refScoresByJudge, judges, gens) {{
  const perJudge = {{}}, perJudgeDSelf = {{}}, perJudgeDOther = {{}};
  for (const judge of judges) {{
    if (!gens.includes(judge)) continue;
    const ref = refScoresByJudge[judge];
    if (!ref) continue;
    const available = gens.filter(g => sysScores[judge+'|'+g] !== undefined && ref[g] !== undefined);
    if (available.length < 2 || !available.includes(judge)) continue;
    const jArr = available.map(g => sysScores[judge+'|'+g]);
    const rArr = available.map(g => ref[g]);
    const jRanks = rankdata(jArr);
    const rRanks = rankdata(rArr);
    const signedDiffs = {{}};
    for (let i = 0; i < available.length; i++) signedDiffs[available[i]] = jRanks[i] - rRanks[i];
    const dSelf = signedDiffs[judge];
    const others = getOtherGens(judge, available);
    if (!others.length) continue;
    const dOther = others.map(g => signedDiffs[g]).reduce((a,b)=>a+b,0) / others.length;
    perJudge[judge] = dSelf - dOther;
    perJudgeDSelf[judge] = dSelf;
    perJudgeDOther[judge] = dOther;
  }}
  const vals = Object.values(perJudge);
  const dSelfVals = Object.values(perJudgeDSelf);
  const dOtherVals = Object.values(perJudgeDOther);
  return {{
    mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null,
    meanDSelf: dSelfVals.length ? dSelfVals.reduce((a,b)=>a+b,0)/dSelfVals.length : null,
    meanDOther: dOtherVals.length ? dOtherVals.reduce((a,b)=>a+b,0)/dOtherVals.length : null,
    perJudge, perJudgeDSelf, perJudgeDOther
  }};
}}

function computeMRDFSP(sysScores, refScoresByJudge, judges, gens, includeSelf) {{
  if (includeSelf === undefined) includeSelf = true;
  const perJudge = {{}}, perJudgeDFam = {{}}, perJudgeDOther = {{}};
  for (const judge of judges) {{
    const ref = refScoresByJudge[judge];
    if (!ref) continue;
    const available = gens.filter(g => sysScores[judge+'|'+g] !== undefined && ref[g] !== undefined);
    if (available.length < 2) continue;
    const jArr = available.map(g => sysScores[judge+'|'+g]);
    const rArr = available.map(g => ref[g]);
    const jRanks = rankdata(jArr);
    const rRanks = rankdata(rArr);
    const signedDiffs = {{}};
    for (let i = 0; i < available.length; i++) signedDiffs[available[i]] = jRanks[i] - rRanks[i];
    const famGens = getFamilyGens(judge, available, includeSelf);
    const otherGens = getOtherGens(judge, available);
    if (!famGens.length || !otherGens.length) continue;
    const dFam = famGens.map(g => signedDiffs[g]).reduce((a,b)=>a+b,0) / famGens.length;
    const dOther = otherGens.map(g => signedDiffs[g]).reduce((a,b)=>a+b,0) / otherGens.length;
    perJudge[judge] = dFam - dOther;
    perJudgeDFam[judge] = dFam;
    perJudgeDOther[judge] = dOther;
  }}
  const vals = Object.values(perJudge);
  const dFamVals = Object.values(perJudgeDFam);
  const dOtherVals = Object.values(perJudgeDOther);
  return {{
    mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null,
    meanDFam: dFamVals.length ? dFamVals.reduce((a,b)=>a+b,0)/dFamVals.length : null,
    meanDOther: dOtherVals.length ? dOtherVals.reduce((a,b)=>a+b,0)/dOtherVals.length : null,
    perJudge, perJudgeDFam, perJudgeDOther
  }};
}}

function computeMSDSP(sysScores, refScoresByJudge, judges, gens, normalize) {{
  const perJudge = {{}}, perJudgeDSelf = {{}}, perJudgeDOther = {{}};
  for (const judge of judges) {{
    if (!gens.includes(judge)) continue;
    const ref = refScoresByJudge[judge];
    if (!ref) continue;
    const available = gens.filter(g => sysScores[judge+'|'+g] !== undefined && ref[g] !== undefined);
    if (!available.includes(judge)) continue;
    const delta = (g) => {{
      let d = sysScores[judge+'|'+g] - ref[g];
      if (normalize && ref[g] > 0) d /= ref[g];
      else if (normalize) return null;
      return d;
    }};
    const dSelf = delta(judge);
    if (dSelf === null) continue;
    const others = getOtherGens(judge, available);
    if (!others.length) continue;
    const dOthers = others.map(g => delta(g)).filter(d => d !== null);
    if (!dOthers.length) continue;
    const dOther = dOthers.reduce((a,b)=>a+b,0) / dOthers.length;
    perJudge[judge] = dSelf - dOther;
    perJudgeDSelf[judge] = dSelf;
    perJudgeDOther[judge] = dOther;
  }}
  const vals = Object.values(perJudge);
  const dSelfVals = Object.values(perJudgeDSelf);
  const dOtherVals = Object.values(perJudgeDOther);
  return {{
    mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null,
    meanDSelf: dSelfVals.length ? dSelfVals.reduce((a,b)=>a+b,0)/dSelfVals.length : null,
    meanDOther: dOtherVals.length ? dOtherVals.reduce((a,b)=>a+b,0)/dOtherVals.length : null,
    perJudge, perJudgeDSelf, perJudgeDOther
  }};
}}

function computeMSDFSP(sysScores, refScoresByJudge, judges, gens, normalize, includeSelf) {{
  if (includeSelf === undefined) includeSelf = true;
  const perJudge = {{}}, perJudgeDFam = {{}}, perJudgeDOther = {{}};
  for (const judge of judges) {{
    const ref = refScoresByJudge[judge];
    if (!ref) continue;
    const available = gens.filter(g => sysScores[judge+'|'+g] !== undefined && ref[g] !== undefined);
    if (!available.length) continue;
    const delta = (g) => {{
      let d = sysScores[judge+'|'+g] - ref[g];
      if (normalize && ref[g] > 0) d /= ref[g];
      else if (normalize) return null;
      return d;
    }};
    const famGens = getFamilyGens(judge, available, includeSelf);
    const otherGens = getOtherGens(judge, available);
    if (!famGens.length || !otherGens.length) continue;
    const famDeltas = famGens.map(g => delta(g)).filter(d => d !== null);
    const otherDeltas = otherGens.map(g => delta(g)).filter(d => d !== null);
    if (!famDeltas.length || !otherDeltas.length) continue;
    const dFam = famDeltas.reduce((a,b)=>a+b,0) / famDeltas.length;
    const dOther = otherDeltas.reduce((a,b)=>a+b,0) / otherDeltas.length;
    perJudge[judge] = dFam - dOther;
    perJudgeDFam[judge] = dFam;
    perJudgeDOther[judge] = dOther;
  }}
  const vals = Object.values(perJudge);
  const dFamVals = Object.values(perJudgeDFam);
  const dOtherVals = Object.values(perJudgeDOther);
  return {{
    mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null,
    meanDFam: dFamVals.length ? dFamVals.reduce((a,b)=>a+b,0)/dFamVals.length : null,
    meanDOther: dOtherVals.length ? dOtherVals.reduce((a,b)=>a+b,0)/dOtherVals.length : null,
    perJudge, perJudgeDFam, perJudgeDOther
  }};
}}

// ============================================================
// INSTANCE-LEVEL ACCURACY (MIPA)
// ============================================================

function computeMIPA(judges, gens, mode, refInstByJudge) {{
  const perJudge = {{}}, perJudgeNAgree = {{}}, perJudgeNTotal = {{}};
  for (const judge of judges) {{
    const refInst = refInstByJudge[judge];
    if (!refInst) continue;
    const jGens = gens.filter(g => getInstScores(judge, g, mode) && refInst[g]);
    if (jGens.length < 2) continue;
    const instScores = {{}};
    for (const g of jGens) instScores[g] = getInstScores(judge, g, mode);
    let agree = 0, total = 0;
    for (let i = 0; i < DATA.nInstances; i++) {{
      for (let gi = 0; gi < jGens.length; gi++) {{
        for (let gj = gi+1; gj < jGens.length; gj++) {{
          const jd = instScores[jGens[gi]][i] - instScores[jGens[gj]][i];
          const rd = refInst[jGens[gi]][i] - refInst[jGens[gj]][i];
          if ((jd > 0 && rd > 0) || (jd < 0 && rd < 0) || (jd === 0 && rd === 0)) agree++;
          total++;
        }}
      }}
    }}
    if (total > 0) {{
      perJudge[judge] = agree / total;
      perJudgeNAgree[judge] = agree;
      perJudgeNTotal[judge] = total;
    }}
  }}
  const vals = Object.values(perJudge);
  return {{ mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null, perJudge, perJudgeNAgree, perJudgeNTotal }};
}}

// ============================================================
// INSTANCE-LEVEL SELF-PREFERENCE BIAS (MISPB)
// ============================================================

function overestRateNonPWC(judge, targetGen, gens, mode, errorDenom, refInstByJudge) {{
  const refInst = refInstByJudge[judge];
  if (!refInst) return null;
  const tgtScores = getInstScores(judge, targetGen, mode);
  if (!tgtScores) return null;
  let nOver = 0, nTotal = 0, nT2W = 0, nL2W = 0, nL2T = 0;
  for (const opp of gens) {{
    if (opp === targetGen) continue;
    const oppScores = getInstScores(judge, opp, mode);
    if (!oppScores) continue;
    if (!refInst[targetGen] || !refInst[opp]) continue;
    for (let i = 0; i < DATA.nInstances; i++) {{
      const jSign = Math.sign(tgtScores[i] - oppScores[i]);
      const rSign = Math.sign(refInst[targetGen][i] - refInst[opp][i]);
      if (errorDenom && rSign >= 0) continue;
      nTotal++;
      if (jSign > rSign) {{
        nOver++;
        if (rSign === 0 && jSign === 1) nT2W++;
        else if (rSign === -1 && jSign === 1) nL2W++;
        else if (rSign === -1 && jSign === 0) nL2T++;
      }}
    }}
  }}
  return nTotal > 0 ? {{rate: nOver / nTotal, nOver, nTotal, nT2W, nL2W, nL2T}} : null;
}}

function computeMISPB(judges, gens, mode, errorDenom, familyMode, includeSelfInFamily, refInstByJudge) {{
  if (includeSelfInFamily === undefined) includeSelfInFamily = true;
  const perJudge = {{}}, perJudgeRaw = {{}}, perJudgeOther = {{}}, perJudgeRatio = {{}};
  const perJudgeNOverSelf = {{}}, perJudgeNTotalSelf = {{}}, perJudgeNOverOther = {{}}, perJudgeNTotalOther = {{}};
  const perJudgeNT2WSelf = {{}}, perJudgeNL2WSelf = {{}}, perJudgeNL2TSelf = {{}};
  const perJudgeNT2WOther = {{}}, perJudgeNL2WOther = {{}}, perJudgeNL2TOther = {{}};

  for (const judge of judges) {{
    const targets = familyMode ? getFamilyGens(judge, gens, includeSelfInFamily) : (gens.includes(judge) ? [judge] : []);
    if (!targets.length) continue;
    const rawResults = targets.map(t => overestRateNonPWC(judge, t, gens, mode, errorDenom, refInstByJudge)).filter(r => r !== null);
    if (!rawResults.length) continue;
    const raw = rawResults.reduce((a,b)=>a+b.rate,0) / rawResults.length;
    let totalNOverSelf = 0, totalNTotalSelf = 0;
    let totalNT2WSelf = 0, totalNL2WSelf = 0, totalNL2TSelf = 0;
    for (const res of rawResults) {{
      totalNOverSelf += res.nOver; totalNTotalSelf += res.nTotal;
      totalNT2WSelf += res.nT2W; totalNL2WSelf += res.nL2W; totalNL2TSelf += res.nL2T;
    }}
    const others = getOtherGens(judge, gens);
    const otherResults = others.map(g => overestRateNonPWC(judge, g, gens, mode, errorDenom, refInstByJudge)).filter(r => r !== null);
    if (!otherResults.length) continue;
    const other = otherResults.reduce((a,b)=>a+b.rate,0) / otherResults.length;
    let totalNOverOther = 0, totalNTotalOther = 0;
    let totalNT2WOther = 0, totalNL2WOther = 0, totalNL2TOther = 0;
    for (const res of otherResults) {{
      totalNOverOther += res.nOver; totalNTotalOther += res.nTotal;
      totalNT2WOther += res.nT2W; totalNL2WOther += res.nL2W; totalNL2TOther += res.nL2T;
    }}
    perJudge[judge] = raw - other;
    perJudgeRaw[judge] = raw;
    perJudgeOther[judge] = other;
    perJudgeRatio[judge] = other > 0 ? raw / other : Infinity;
    perJudgeNOverSelf[judge] = totalNOverSelf; perJudgeNTotalSelf[judge] = totalNTotalSelf;
    perJudgeNOverOther[judge] = totalNOverOther; perJudgeNTotalOther[judge] = totalNTotalOther;
    perJudgeNT2WSelf[judge] = totalNT2WSelf; perJudgeNL2WSelf[judge] = totalNL2WSelf; perJudgeNL2TSelf[judge] = totalNL2TSelf;
    perJudgeNT2WOther[judge] = totalNT2WOther; perJudgeNL2WOther[judge] = totalNL2WOther; perJudgeNL2TOther[judge] = totalNL2TOther;
  }}
  const vals = Object.values(perJudge);
  return {{
    mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null,
    perJudge, perJudgeRaw, perJudgeOther, perJudgeRatio,
    perJudgeNOverSelf, perJudgeNTotalSelf, perJudgeNOverOther, perJudgeNTotalOther,
    perJudgeNT2WSelf, perJudgeNL2WSelf, perJudgeNL2TSelf,
    perJudgeNT2WOther, perJudgeNL2WOther, perJudgeNL2TOther
  }};
}}

// ============================================================
// RUBRIC-LEVEL METRICS
// ============================================================

function computeMRA(judges, gens, refRubricsByJudge) {{
  const perJudge = {{}}, perJudgeNCorrect = {{}}, perJudgeNTotal = {{}};
  for (const judge of judges) {{
    const refRubrics = refRubricsByJudge[judge];
    if (!refRubrics) continue;
    let correct = 0, total = 0;
    for (const gen of gens) {{
      const jRubrics = getJudgeRubrics(judge, gen);
      if (!jRubrics || !refRubrics[gen]) continue;
      for (let i = 0; i < DATA.nInstances; i++) {{
        for (let r = 0; r < jRubrics[i].length; r++) {{
          if (jRubrics[i][r] === refRubrics[gen][i][r]) correct++;
          total++;
        }}
      }}
    }}
    if (total > 0) {{
      perJudge[judge] = correct / total;
      perJudgeNCorrect[judge] = correct;
      perJudgeNTotal[judge] = total;
    }}
  }}
  const vals = Object.values(perJudge);
  return {{ mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null, perJudge, perJudgeNCorrect, perJudgeNTotal }};
}}

function rubricOverestRate(judge, targetGen, errorDenom, refRubricsByJudge) {{
  const refRubrics = refRubricsByJudge[judge];
  if (!refRubrics || !refRubrics[targetGen]) return null;
  const jRubrics = getJudgeRubrics(judge, targetGen);
  if (!jRubrics) return null;
  const rRubrics = refRubrics[targetGen];
  buildRubricOffsets();
  let nOver = 0, nTotal = 0;
  for (let i = 0; i < DATA.nInstances; i++) {{
    const off = _rubricOffsets[i];
    for (let r = 0; r < jRubrics[i].length; r++) {{
      const pts = DATA.rubricPointsFlat[off + r];
      const isPositive = pts > 0;
      if (errorDenom) {{
        // Only count rubrics where overestimation is possible
        if (isPositive && rRubrics[i][r]) continue;
        if (!isPositive && !rRubrics[i][r]) continue;
      }}
      nTotal++;
      if (isPositive) {{
        // Positive rubric: overest = judge says met, ref says not met
        if (jRubrics[i][r] && !rRubrics[i][r]) nOver++;
      }} else {{
        // Negative rubric: overest = judge says not met, ref says met
        if (!jRubrics[i][r] && rRubrics[i][r]) nOver++;
      }}
    }}
  }}
  return nTotal > 0 ? {{rate: nOver / nTotal, nOver, nTotal}} : null;
}}

function computeMRSPB(judges, gens, errorDenom, familyMode, includeSelfInFamily, refRubricsByJudge) {{
  if (includeSelfInFamily === undefined) includeSelfInFamily = true;
  const perJudge = {{}}, perJudgeRaw = {{}}, perJudgeOther = {{}}, perJudgeRatio = {{}};
  const perJudgeNOverSelf = {{}}, perJudgeNTotalSelf = {{}}, perJudgeNOverOther = {{}}, perJudgeNTotalOther = {{}};
  for (const judge of judges) {{
    const targets = familyMode ? getFamilyGens(judge, gens, includeSelfInFamily) : (gens.includes(judge) ? [judge] : []);
    if (!targets.length) continue;
    const rawResults = targets.map(t => rubricOverestRate(judge, t, errorDenom, refRubricsByJudge)).filter(r => r !== null);
    if (!rawResults.length) continue;
    const raw = rawResults.reduce((a,b)=>a+b.rate,0) / rawResults.length;
    let totalNOverSelf = 0, totalNTotalSelf = 0;
    for (const res of rawResults) {{ totalNOverSelf += res.nOver; totalNTotalSelf += res.nTotal; }}
    const others = getOtherGens(judge, gens);
    const otherResults = others.map(g => rubricOverestRate(judge, g, errorDenom, refRubricsByJudge)).filter(r => r !== null);
    if (!otherResults.length) continue;
    const other = otherResults.reduce((a,b)=>a+b.rate,0) / otherResults.length;
    let totalNOverOther = 0, totalNTotalOther = 0;
    for (const res of otherResults) {{ totalNOverOther += res.nOver; totalNTotalOther += res.nTotal; }}
    perJudge[judge] = raw - other;
    perJudgeRaw[judge] = raw;
    perJudgeOther[judge] = other;
    perJudgeRatio[judge] = other > 0 ? raw / other : Infinity;
    perJudgeNOverSelf[judge] = totalNOverSelf; perJudgeNTotalSelf[judge] = totalNTotalSelf;
    perJudgeNOverOther[judge] = totalNOverOther; perJudgeNTotalOther[judge] = totalNTotalOther;
  }}
  const vals = Object.values(perJudge);
  return {{
    mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null,
    perJudge, perJudgeRaw, perJudgeOther, perJudgeRatio,
    perJudgeNOverSelf, perJudgeNTotalSelf, perJudgeNOverOther, perJudgeNTotalOther
  }};
}}

// ============================================================
// MAIN COMPUTATION
// ============================================================

let cachedResults = null;
let effectiveJudgesList = [];


function computeAll(judges, gens) {{
  if (judges.length === 0 || gens.length < 2) return null;
  // Clear per-computation caches
  for (const k in _instScoreCache) delete _instScoreCache[k];

  const mode = getScoringMode();

  // Build committee reference
  const members = getCommitteeMembers();
  if (members.length < 2) return null;
  const ref = buildCommitteeReference(members, gens);
  const refInst = computeRefInstScores(ref, gens, mode);
  const refSys = computeRefSystemScores(ref, gens, mode);

  let refRubricsByJudge = {{}};
  let refInstByJudge = {{}};
  let refSysByJudge = {{}};
  for (const judge of judges) {{
    refRubricsByJudge[judge] = ref;
    refInstByJudge[judge] = refInst;
    refSysByJudge[judge] = refSys;
  }}
  const displayRefSys = refSys;

  const sysScores = computeSystemScoresSR(judges, gens, mode);

  const r = {{ sysScores, refSysByJudge, displayRefSys, refInstByJudge, refRubricsByJudge }};

  // System-level accuracy
  r.mpa = computeMPA(sysScores, refSysByJudge, judges, gens);
  r.mrd = computeMRD(sysScores, refSysByJudge, judges, gens);
  r.msd = computeMSD(sysScores, refSysByJudge, judges, gens);
  r.msdNorm = computeMSDnorm(sysScores, refSysByJudge, judges, gens);

  // System-level bias
  r.mrdSP = computeMRDSP(sysScores, refSysByJudge, judges, gens);
  r.mrdFSP = computeMRDFSP(sysScores, refSysByJudge, judges, gens, true);
  r.mrdFOSP = computeMRDFSP(sysScores, refSysByJudge, judges, gens, false);
  r.msdSP = computeMSDSP(sysScores, refSysByJudge, judges, gens, false);
  r.msdSPnorm = computeMSDSP(sysScores, refSysByJudge, judges, gens, true);
  r.msdFSP = computeMSDFSP(sysScores, refSysByJudge, judges, gens, false, true);
  r.msdFSPnorm = computeMSDFSP(sysScores, refSysByJudge, judges, gens, true, true);
  r.msdFOSP = computeMSDFSP(sysScores, refSysByJudge, judges, gens, false, false);
  r.msdFOSPnorm = computeMSDFSP(sysScores, refSysByJudge, judges, gens, true, false);

  // Instance-level accuracy
  r.mipa = computeMIPA(judges, gens, mode, refInstByJudge);

  // Instance-level bias
  r.mispb = computeMISPB(judges, gens, mode, false, false, true, refInstByJudge);
  r.mispbF = computeMISPB(judges, gens, mode, false, true, true, refInstByJudge);
  r.mispbFO = computeMISPB(judges, gens, mode, false, true, false, refInstByJudge);
  r.hspp = computeMISPB(judges, gens, mode, true, false, true, refInstByJudge);
  r.hsppF = computeMISPB(judges, gens, mode, true, true, true, refInstByJudge);
  r.hsppFO = computeMISPB(judges, gens, mode, true, true, false, refInstByJudge);

  // Rubric-level metrics
  r.mra = computeMRA(judges, gens, refRubricsByJudge);
  r.mrspb = computeMRSPB(judges, gens, false, false, true, refRubricsByJudge);
  r.mrspbF = computeMRSPB(judges, gens, false, true, true, refRubricsByJudge);
  r.mrspbFO = computeMRSPB(judges, gens, false, true, false, refRubricsByJudge);
  r.mrspbErr = computeMRSPB(judges, gens, true, false, true, refRubricsByJudge);
  r.mrspbErrF = computeMRSPB(judges, gens, true, true, true, refRubricsByJudge);
  r.mrspbErrFO = computeMRSPB(judges, gens, true, true, false, refRubricsByJudge);

  cachedResults = r;
  return r;
}}

// ============================================================
// UI RENDERING
// ============================================================

function fmt(v, d=4) {{
  if (v === null || v === undefined || isNaN(v)) return '\u2014';
  return v.toFixed(d);
}}

function makeTable(headers, rows, opts={{}}) {{
  let html = '<table><thead><tr>';
  for (const h of headers) html += '<th>' + h + '</th>';
  html += '</tr></thead><tbody>';
  for (let ri = 0; ri < rows.length; ri++) {{
    const row = rows[ri];
    const rowCls = opts.rowClasses ? (opts.rowClasses[ri] || '') : '';
    html += '<tr' + (rowCls ? ' class="' + rowCls + '"' : '') + '>';
    for (let i = 0; i < row.length; i++) {{
      const cls = i === 0 ? ' class="row-label"' : '';
      html += '<td' + cls + '>' + row[i] + '</td>';
    }}
    html += '</tr>';
  }}
  html += '</tbody></table>';
  return html;
}}

function makeDetailSection(title, headersFn, rowsFn, judges) {{
  const fJudges = judges.filter(j => rowsFn(j) !== null);
  if (!fJudges.length) return '';
  const headers = headersFn();
  let html = '<details class="metric-detail"><summary>' + title + '</summary><div class="detail-body">';
  html += '<table><thead><tr>';
  for (const h of headers) html += '<th>' + h + '</th>';
  html += '</tr></thead><tbody>';
  for (const j of fJudges) {{
    const row = rowsFn(j);
    if (!row) continue;
    html += '<tr>';
    for (let i = 0; i < row.length; i++) {{
      const cls = i === 0 ? ' class="row-label"' : '';
      html += '<td' + cls + '>' + row[i] + '</td>';
    }}
    html += '</tr>';
  }}
  html += '</tbody></table></div></details>';
  return html;
}}

const meanPJ = (obj) => {{
  if (!obj) return null;
  const v = Object.values(obj);
  return v.length ? v.reduce((a,b)=>a+b,0)/v.length : null;
}};

function renderAggregate(r) {{
  const noData = '<p class="no-data">Select at least 2 generators and 1 judge</p>';
  const allIds = ['tbl-sys-accuracy','tbl-inst-accuracy','tbl-rubric-accuracy',
                  'tbl-sys-bias','tbl-inst-bias','tbl-rubric-bias'];
  if (!r) {{
    for (const id of allIds) document.getElementById(id).innerHTML = noData;
    return;
  }}
  const judges = effectiveJudgesList;
  const mode = getScoringMode();

  // System-level accuracy
  let sysAccHtml = makeTable(
    ['Metric', 'Value'],
    [['MPA \u2191', fmt(r.mpa?.mean)],
     ['MRD \u2193', fmt(r.mrd?.mean, 2)],
     ['MSD', fmt(r.msd?.mean)],
     ['MSD-norm', fmt(r.msdNorm?.mean)]]
  );
  sysAccHtml += makeDetailSection(
    'Per-judge MPA (n_concordant / n_total)',
    () => ['Judge', 'MPA', 'n_conc/n_total'],
    (j) => {{
      if (r.mpa?.perJudge[j] === undefined) return null;
      return [sn(j), fmt(r.mpa.perJudge[j]), r.mpa.perJudgeNConc[j]+'/'+r.mpa.perJudgeNTotal[j]];
    }},
    judges
  );
  document.getElementById('tbl-sys-accuracy').innerHTML = sysAccHtml;

  // Instance-level accuracy
  let instAccHtml = makeTable(
    ['Metric', 'Value'],
    [['MIPA \u2191', fmt(r.mipa?.mean)]]
  );
  instAccHtml += makeDetailSection(
    'Per-judge MIPA (n_agree / n_total)',
    () => ['Judge', 'MIPA', 'n_agree/n_total'],
    (j) => {{
      if (r.mipa?.perJudge[j] === undefined) return null;
      return [sn(j), fmt(r.mipa.perJudge[j]), r.mipa.perJudgeNAgree[j]+'/'+r.mipa.perJudgeNTotal[j]];
    }},
    judges
  );
  document.getElementById('tbl-inst-accuracy').innerHTML = instAccHtml;

  // Rubric-level accuracy
  let rubAccHtml = makeTable(
    ['Metric', 'Value'],
    [['MRA \u2191', fmt(r.mra?.mean)]]
  );
  rubAccHtml += makeDetailSection(
    'Per-judge MRA (n_correct / n_total)',
    () => ['Judge', 'MRA', 'n_correct/n_total'],
    (j) => {{
      if (r.mra?.perJudge[j] === undefined) return null;
      return [sn(j), fmt(r.mra.perJudge[j]), r.mra.perJudgeNCorrect[j]+'/'+r.mra.perJudgeNTotal[j]];
    }},
    judges
  );
  document.getElementById('tbl-rubric-accuracy').innerHTML = rubAccHtml;

  // System-level bias
  let sysBiasHtml = makeTable(
    ['Metric', 'Value', 'd_self/d_fam', 'd_other/d_nonfam'],
    [
      ['MRD-SP', fmt(r.mrdSP?.mean, 3), fmt(r.mrdSP?.meanDSelf, 2), fmt(r.mrdSP?.meanDOther, 2)],
      ['MRD-FSP', fmt(r.mrdFSP?.mean, 3), fmt(r.mrdFSP?.meanDFam, 2), fmt(r.mrdFSP?.meanDOther, 2)],
      ['MRD-FOSP', fmt(r.mrdFOSP?.mean, 3), fmt(r.mrdFOSP?.meanDFam, 2), fmt(r.mrdFOSP?.meanDOther, 2)],
      ['MSD-SP', fmt(r.msdSP?.mean), fmt(r.msdSP?.meanDSelf), fmt(r.msdSP?.meanDOther)],
      ['MSD-SP-norm', fmt(r.msdSPnorm?.mean), fmt(r.msdSPnorm?.meanDSelf), fmt(r.msdSPnorm?.meanDOther)],
      ['MSD-FSP', fmt(r.msdFSP?.mean), fmt(r.msdFSP?.meanDFam), fmt(r.msdFSP?.meanDOther)],
      ['MSD-FSP-norm', fmt(r.msdFSPnorm?.mean), fmt(r.msdFSPnorm?.meanDFam), fmt(r.msdFSPnorm?.meanDOther)],
      ['MSD-FOSP', fmt(r.msdFOSP?.mean), fmt(r.msdFOSP?.meanDFam), fmt(r.msdFOSP?.meanDOther)],
      ['MSD-FOSP-norm', fmt(r.msdFOSPnorm?.mean), fmt(r.msdFOSPnorm?.meanDFam), fmt(r.msdFOSPnorm?.meanDOther)],
    ]
  );
  document.getElementById('tbl-sys-bias').innerHTML = sysBiasHtml;

  // Instance-level bias
  let instBiasHtml = makeTable(
    ['Metric', 'Value', 'raw', 'other', 'ratio'],
    [
      ['MISPB', fmt(r.mispb?.mean), fmt(meanPJ(r.mispb?.perJudgeRaw)), fmt(meanPJ(r.mispb?.perJudgeOther)), fmt(meanPJ(r.mispb?.perJudgeRatio), 2)],
      ['MISPB-F', fmt(r.mispbF?.mean), fmt(meanPJ(r.mispbF?.perJudgeRaw)), fmt(meanPJ(r.mispbF?.perJudgeOther)), fmt(meanPJ(r.mispbF?.perJudgeRatio), 2)],
      ['MISPB-FO', fmt(r.mispbFO?.mean), fmt(meanPJ(r.mispbFO?.perJudgeRaw)), fmt(meanPJ(r.mispbFO?.perJudgeOther)), fmt(meanPJ(r.mispbFO?.perJudgeRatio), 2)],
      ['HSPP', fmt(r.hspp?.mean), fmt(meanPJ(r.hspp?.perJudgeRaw)), fmt(meanPJ(r.hspp?.perJudgeOther)), fmt(meanPJ(r.hspp?.perJudgeRatio), 2)],
      ['HSPP-F', fmt(r.hsppF?.mean), fmt(meanPJ(r.hsppF?.perJudgeRaw)), fmt(meanPJ(r.hsppF?.perJudgeOther)), fmt(meanPJ(r.hsppF?.perJudgeRatio), 2)],
      ['HSPP-FO', fmt(r.hsppFO?.mean), fmt(meanPJ(r.hsppFO?.perJudgeRaw)), fmt(meanPJ(r.hsppFO?.perJudgeOther)), fmt(meanPJ(r.hsppFO?.perJudgeRatio), 2)],
    ]
  );
  document.getElementById('tbl-inst-bias').innerHTML = instBiasHtml;

  // Rubric-level bias
  let rubBiasHtml = makeTable(
    ['Metric', 'Value', 'raw', 'other', 'ratio'],
    [
      ['MRSPB', fmt(r.mrspb?.mean), fmt(meanPJ(r.mrspb?.perJudgeRaw)), fmt(meanPJ(r.mrspb?.perJudgeOther)), fmt(meanPJ(r.mrspb?.perJudgeRatio), 2)],
      ['MRSPB-F', fmt(r.mrspbF?.mean), fmt(meanPJ(r.mrspbF?.perJudgeRaw)), fmt(meanPJ(r.mrspbF?.perJudgeOther)), fmt(meanPJ(r.mrspbF?.perJudgeRatio), 2)],
      ['MRSPB-FO', fmt(r.mrspbFO?.mean), fmt(meanPJ(r.mrspbFO?.perJudgeRaw)), fmt(meanPJ(r.mrspbFO?.perJudgeOther)), fmt(meanPJ(r.mrspbFO?.perJudgeRatio), 2)],
      ['MRSPB-err', fmt(r.mrspbErr?.mean), fmt(meanPJ(r.mrspbErr?.perJudgeRaw)), fmt(meanPJ(r.mrspbErr?.perJudgeOther)), fmt(meanPJ(r.mrspbErr?.perJudgeRatio), 2)],
      ['MRSPB-err-F', fmt(r.mrspbErrF?.mean), fmt(meanPJ(r.mrspbErrF?.perJudgeRaw)), fmt(meanPJ(r.mrspbErrF?.perJudgeOther)), fmt(meanPJ(r.mrspbErrF?.perJudgeRatio), 2)],
      ['MRSPB-err-FO', fmt(r.mrspbErrFO?.mean), fmt(meanPJ(r.mrspbErrFO?.perJudgeRaw)), fmt(meanPJ(r.mrspbErrFO?.perJudgeOther)), fmt(meanPJ(r.mrspbErrFO?.perJudgeRatio), 2)],
    ]
  );
  document.getElementById('tbl-rubric-bias').innerHTML = rubBiasHtml;
}}

function renderPerJudge(r) {{
  const allIds = ['tbl-mpa-judge','tbl-mipa-judge','tbl-mra-judge',
    'tbl-mrdsp-judge','tbl-msdsp-judge',
    'tbl-mispb-judge','tbl-mispbf-judge','tbl-mispbfo-judge',
    'tbl-hspp-judge','tbl-hsppf-judge','tbl-hsppfo-judge',
    'tbl-mrspb-judge','tbl-mrspbf-judge','tbl-mrspbfo-judge',
    'tbl-mrspberr-judge','tbl-mrspberrf-judge','tbl-mrspberrfo-judge'];
  const judges = effectiveJudgesList;
  if (!r || !judges.length) {{
    for (const id of allIds) document.getElementById(id).innerHTML = '<p class="no-data">No data</p>';
    return;
  }}

  // MPA per judge
  document.getElementById('tbl-mpa-judge').innerHTML = makeTable(
    ['Judge', 'MPA'],
    judges.map(j => [sn(j), fmt(r.mpa?.perJudge[j])])
  );

  // MIPA per judge
  document.getElementById('tbl-mipa-judge').innerHTML = makeTable(
    ['Judge', 'MIPA'],
    judges.map(j => [sn(j), fmt(r.mipa?.perJudge[j])])
  );

  // MRA per judge
  document.getElementById('tbl-mra-judge').innerHTML = makeTable(
    ['Judge', 'MRA'],
    judges.map(j => [sn(j), fmt(r.mra?.perJudge[j])])
  );

  // MRD-SP per judge
  let mrdspHtml = makeTable(
    ['Judge', 'MRD-SP', 'd_self', 'd_other'],
    judges.map(j => [sn(j), fmt(r.mrdSP?.perJudge[j], 3), fmt(r.mrdSP?.perJudgeDSelf?.[j], 2), fmt(r.mrdSP?.perJudgeDOther?.[j], 2)])
  );
  mrdspHtml += makeDetailSection(
    'MRD-FSP per judge',
    () => ['Judge', 'MRD-FSP', 'd_fam', 'd_nonfam'],
    (j) => {{
      if (r.mrdFSP?.perJudge[j] === undefined) return null;
      return [sn(j), fmt(r.mrdFSP.perJudge[j], 3), fmt(r.mrdFSP.perJudgeDFam?.[j], 2), fmt(r.mrdFSP.perJudgeDOther?.[j], 2)];
    }},
    judges
  );
  mrdspHtml += makeDetailSection(
    'MRD-FOSP per judge',
    () => ['Judge', 'MRD-FOSP', 'd_fam', 'd_nonfam'],
    (j) => {{
      if (r.mrdFOSP?.perJudge[j] === undefined) return null;
      return [sn(j), fmt(r.mrdFOSP.perJudge[j], 3), fmt(r.mrdFOSP.perJudgeDFam?.[j], 2), fmt(r.mrdFOSP.perJudgeDOther?.[j], 2)];
    }},
    judges
  );
  document.getElementById('tbl-mrdsp-judge').innerHTML = mrdspHtml;

  // MSD-SP per judge
  let msdspHtml = makeTable(
    ['Judge', 'MSD-SP', 'd_self', 'd_other'],
    judges.map(j => [sn(j), fmt(r.msdSP?.perJudge[j]), fmt(r.msdSP?.perJudgeDSelf?.[j]), fmt(r.msdSP?.perJudgeDOther?.[j])])
  );
  msdspHtml += makeDetailSection(
    'MSD-SP-norm per judge',
    () => ['Judge', 'MSD-SP-n', 'd_self', 'd_other'],
    (j) => {{
      if (r.msdSPnorm?.perJudge[j] === undefined) return null;
      return [sn(j), fmt(r.msdSPnorm.perJudge[j]), fmt(r.msdSPnorm.perJudgeDSelf?.[j]), fmt(r.msdSPnorm.perJudgeDOther?.[j])];
    }},
    judges
  );
  msdspHtml += makeDetailSection(
    'MSD-FSP per judge',
    () => ['Judge', 'MSD-FSP', 'd_fam', 'd_nonfam'],
    (j) => {{
      if (r.msdFSP?.perJudge[j] === undefined) return null;
      return [sn(j), fmt(r.msdFSP.perJudge[j]), fmt(r.msdFSP.perJudgeDFam?.[j]), fmt(r.msdFSP.perJudgeDOther?.[j])];
    }},
    judges
  );
  msdspHtml += makeDetailSection(
    'MSD-FOSP per judge',
    () => ['Judge', 'MSD-FOSP', 'd_fam', 'd_nonfam'],
    (j) => {{
      if (r.msdFOSP?.perJudge[j] === undefined) return null;
      return [sn(j), fmt(r.msdFOSP.perJudge[j]), fmt(r.msdFOSP.perJudgeDFam?.[j]), fmt(r.msdFOSP.perJudgeDOther?.[j])];
    }},
    judges
  );
  document.getElementById('tbl-msdsp-judge').innerHTML = msdspHtml;

  // Helper: build a bias table with value + raw + other + ratio
  function biasTable(metric, judgeList) {{
    const jl = judgeList || judges;
    return makeTable(
      ['Judge', 'Value', 'raw', 'other', 'ratio'],
      jl.map(j => [sn(j), fmt(metric?.perJudge[j]), fmt(metric?.perJudgeRaw[j]), fmt(metric?.perJudgeOther[j]), fmt(metric?.perJudgeRatio[j], 2)])
    );
  }}

  function biasDetail(metric, label, judgeList) {{
    return makeDetailSection(
      label + ' counts (n_overest/n_total for self | other)',
      () => ['Judge', 'Self', 'Other'],
      (j) => {{
        if (metric?.perJudge[j] === undefined) return null;
        const nos = metric.perJudgeNOverSelf?.[j];
        const nts = metric.perJudgeNTotalSelf?.[j];
        const noo = metric.perJudgeNOverOther?.[j];
        const nto = metric.perJudgeNTotalOther?.[j];
        return [sn(j),
          nos !== undefined ? nos+'/'+nts : '\u2014',
          noo !== undefined ? noo+'/'+nto : '\u2014'];
      }},
      judgeList || judges
    );
  }}

  // MISPB per judge
  document.getElementById('tbl-mispb-judge').innerHTML = biasTable(r.mispb) + biasDetail(r.mispb, 'MISPB');
  document.getElementById('tbl-mispbf-judge').innerHTML = biasTable(r.mispbF) + biasDetail(r.mispbF, 'MISPB-F');
  document.getElementById('tbl-mispbfo-judge').innerHTML = biasTable(r.mispbFO) + biasDetail(r.mispbFO, 'MISPB-FO');
  document.getElementById('tbl-hspp-judge').innerHTML = biasTable(r.hspp) + biasDetail(r.hspp, 'HSPP');
  document.getElementById('tbl-hsppf-judge').innerHTML = biasTable(r.hsppF) + biasDetail(r.hsppF, 'HSPP-F');
  document.getElementById('tbl-hsppfo-judge').innerHTML = biasTable(r.hsppFO) + biasDetail(r.hsppFO, 'HSPP-FO');

  // MRSPB per judge
  document.getElementById('tbl-mrspb-judge').innerHTML = biasTable(r.mrspb) + biasDetail(r.mrspb, 'MRSPB');
  document.getElementById('tbl-mrspbf-judge').innerHTML = biasTable(r.mrspbF) + biasDetail(r.mrspbF, 'MRSPB-F');
  document.getElementById('tbl-mrspbfo-judge').innerHTML = biasTable(r.mrspbFO) + biasDetail(r.mrspbFO, 'MRSPB-FO');
  document.getElementById('tbl-mrspberr-judge').innerHTML = biasTable(r.mrspbErr) + biasDetail(r.mrspbErr, 'MRSPB-err');
  document.getElementById('tbl-mrspberrf-judge').innerHTML = biasTable(r.mrspbErrF) + biasDetail(r.mrspbErrF, 'MRSPB-err-F');
  document.getElementById('tbl-mrspberrfo-judge').innerHTML = biasTable(r.mrspbErrFO) + biasDetail(r.mrspbErrFO, 'MRSPB-err-FO');
}}

function renderPerGenerator(r) {{
  const gens = getSelectedGens();
  const allIds = ['tbl-ref-scores','tbl-gen-scores','tbl-gen-deltas','tbl-gen-deltas-norm','tbl-gen-rank-deltas'];
  if (!r || !gens.length) {{
    for (const id of allIds) document.getElementById(id).innerHTML = '<p class="no-data">No data</p>';
    return;
  }}

  const judges = effectiveJudgesList;
  const refSys = r.displayRefSys;

  // Reference scores
  document.getElementById('tbl-ref-scores').innerHTML = makeTable(
    ['Generator', 'Ref Score'],
    gens.sort((a,b) => (refSys[b]||0) - (refSys[a]||0)).map(g => [sn(g), fmt(refSys[g])])
  );

  // Mean judge scores per generator
  document.getElementById('tbl-gen-scores').innerHTML = makeTable(
    ['Generator', 'SR Score'],
    gens.map(g => {{
      const vals = judges.map(j => r.sysScores[j+'|'+g]).filter(v => v !== undefined);
      return [sn(g), fmt(vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null)];
    }})
  );

  // Score deltas
  document.getElementById('tbl-gen-deltas').innerHTML = makeTable(
    ['Generator', 'Score Delta'],
    gens.map(g => {{
      const vals = judges.map(j => r.sysScores[j+'|'+g]).filter(v => v !== undefined);
      const meanJ = vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null;
      return [sn(g), fmt(meanJ !== null && refSys[g] !== undefined ? meanJ - refSys[g] : null)];
    }})
  );

  // Normalized score deltas
  document.getElementById('tbl-gen-deltas-norm').innerHTML = makeTable(
    ['Generator', 'Norm Delta'],
    gens.map(g => {{
      const vals = judges.map(j => r.sysScores[j+'|'+g]).filter(v => v !== undefined);
      const meanJ = vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null;
      if (meanJ !== null && refSys[g] !== undefined && refSys[g] > 0) return [sn(g), fmt((meanJ - refSys[g]) / refSys[g])];
      return [sn(g), fmt(null)];
    }})
  );

  // Rank deltas
  const refVals = gens.map(g => ({{ gen: g, val: refSys[g] || 0 }}));
  refVals.sort((a, b) => b.val - a.val);
  const refRank = {{}};
  refVals.forEach((item, i) => {{ refRank[item.gen] = i + 1; }});
  const judgeRanks = {{}};
  for (const j of judges) {{
    const jVals = gens.map(g => ({{ gen: g, val: r.sysScores[j + '|' + g] || 0 }}));
    jVals.sort((a, b) => b.val - a.val);
    const jRank = {{}};
    jVals.forEach((item, i) => {{ jRank[item.gen] = i + 1; }});
    judgeRanks[j] = jRank;
  }}
  document.getElementById('tbl-gen-rank-deltas').innerHTML = makeTable(
    ['Generator', 'Rank Delta'],
    gens.map(g => {{
      const deltas = judges.map(j => (judgeRanks[j]?.[g] || 0) - (refRank[g] || 0));
      return [sn(g), fmt(deltas.length ? deltas.reduce((a,b)=>a+b,0)/deltas.length : null, 2)];
    }})
  );
}}

// ============================================================
// CHARTS
// ============================================================

const PLOTLY_LAYOUT = {{
  font: {{ family: "'Atkinson Hyperlegible', sans-serif", color: '#3d3a35' }},
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  margin: {{ t: 30, r: 20, b: 40, l: 50 }},
}};
const ACCENT = '#c17f59';
const SAND200 = '#e8e4de';
const SAND700 = '#3d3a35';
const SUCCESS = '#5b8a72';

function renderCharts(r) {{
  const chartIds = ['chart-accuracy','chart-mispb','chart-mpa-judge','chart-scores','chart-mrspb'];
  if (!r) {{
    for (const id of chartIds) document.getElementById(id).innerHTML = '<p class="no-data">No data</p>';
    return;
  }}

  const judges = effectiveJudgesList;
  const gens = getSelectedGens();

  // 1. Accuracy bar chart
  Plotly.newPlot('chart-accuracy', [
    {{
      x: ['MPA', 'MIPA', 'MRA'],
      y: [r.mpa?.mean, r.mipa?.mean, r.mra?.mean],
      type: 'bar',
      marker: {{ color: [ACCENT, SUCCESS, '#6b8cae'] }},
    }}
  ], {{
    ...PLOTLY_LAYOUT,
    yaxis: {{ title: 'Score', gridcolor: SAND200 }},
    xaxis: {{ gridcolor: SAND200 }},
  }}, {{responsive: true}});

  // 2. MISPB by judge
  const mispbJudges = judges.filter(j => r.mispb?.perJudge[j] !== undefined);
  if (mispbJudges.length > 0) {{
    Plotly.newPlot('chart-mispb', [{{
      x: mispbJudges.map(j => sn(j)),
      y: mispbJudges.map(j => r.mispb.perJudge[j]),
      type: 'bar',
      marker: {{ color: mispbJudges.map(j => r.mispb.perJudge[j] >= 0 ? '#b2182b' : '#2166ac') }},
    }}], {{
      ...PLOTLY_LAYOUT,
      xaxis: {{ tickangle: -45, gridcolor: SAND200 }},
      yaxis: {{ title: 'MISPB', gridcolor: SAND200 }},
      margin: {{ t: 30, r: 20, b: 80, l: 50 }},
    }}, {{responsive: true}});
  }} else {{
    document.getElementById('chart-mispb').innerHTML = '<p class="no-data">No bias data</p>';
  }}

  // 3. MPA by judge
  const mpaJudges = judges.filter(j => r.mpa?.perJudge[j] !== undefined);
  if (mpaJudges.length > 0) {{
    Plotly.newPlot('chart-mpa-judge', [{{
      x: mpaJudges.map(j => sn(j)),
      y: mpaJudges.map(j => r.mpa.perJudge[j]),
      type: 'bar',
      marker: {{ color: ACCENT }},
    }}], {{
      ...PLOTLY_LAYOUT,
      xaxis: {{ tickangle: -45, gridcolor: SAND200 }},
      yaxis: {{ title: 'MPA', gridcolor: SAND200 }},
      margin: {{ t: 30, r: 20, b: 80, l: 50 }},
    }}, {{responsive: true}});
  }} else {{
    document.getElementById('chart-mpa-judge').innerHTML = '<p class="no-data">No data</p>';
  }}

  // 4. System scores: Judge vs Reference
  const sortedGens = [...gens].sort((a,b) => (r.displayRefSys[b]||0) - (r.displayRefSys[a]||0));
  Plotly.newPlot('chart-scores', [
    {{
      x: sortedGens.map(g => sn(g)),
      y: sortedGens.map(g => r.displayRefSys[g]),
      name: 'Reference',
      type: 'bar',
      marker: {{ color: SAND700 }},
    }},
    {{
      x: sortedGens.map(g => sn(g)),
      y: sortedGens.map(g => {{
        const vals = judges.map(j => r.sysScores[j+'|'+g]).filter(v => v !== undefined);
        return vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null;
      }}),
      name: 'SR (Judge Mean)',
      type: 'bar',
      marker: {{ color: ACCENT }},
    }}
  ], {{
    ...PLOTLY_LAYOUT,
    barmode: 'group',
    xaxis: {{ tickangle: -45, gridcolor: SAND200 }},
    yaxis: {{ title: 'System Score', gridcolor: SAND200 }},
    legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.8)' }},
    margin: {{ t: 30, r: 20, b: 80, l: 50 }},
  }}, {{responsive: true}});

  // 5. MRSPB by judge
  const mrspbJudges = judges.filter(j => r.mrspb?.perJudge[j] !== undefined);
  if (mrspbJudges.length > 0) {{
    Plotly.newPlot('chart-mrspb', [{{
      x: mrspbJudges.map(j => sn(j)),
      y: mrspbJudges.map(j => r.mrspb.perJudge[j]),
      type: 'bar',
      marker: {{ color: mrspbJudges.map(j => r.mrspb.perJudge[j] >= 0 ? '#b2182b' : '#2166ac') }},
    }}], {{
      ...PLOTLY_LAYOUT,
      xaxis: {{ tickangle: -45, gridcolor: SAND200 }},
      yaxis: {{ title: 'MRSPB', gridcolor: SAND200 }},
      margin: {{ t: 30, r: 20, b: 80, l: 50 }},
    }}, {{responsive: true}});
  }} else {{
    document.getElementById('chart-mrspb').innerHTML = '<p class="no-data">No data</p>';
  }}
}}

// ============================================================
// SELECTION & UI CONTROLLER
// ============================================================

function getSelectedJudges() {{
  return [...document.querySelectorAll('#judge-checkboxes input:checked')].map(cb => cb.value);
}}

function getSelectedGens() {{
  return [...document.querySelectorAll('#gen-checkboxes input:checked')].map(cb => cb.value);
}}

function getCommitteeMembers() {{
  return [...document.querySelectorAll('#committee-member-checkboxes input:checked')].map(cb => cb.value);
}}

function selectAll(type) {{
  document.querySelectorAll('#' + type + '-checkboxes input').forEach(cb => cb.checked = true);
  onSelectionChange();
}}

function selectNone(type) {{
  document.querySelectorAll('#' + type + '-checkboxes input').forEach(cb => cb.checked = false);
  onSelectionChange();
}}

function selectDefault(type) {{
  if (type === 'judge') {{
    document.querySelectorAll('#judge-checkboxes input').forEach(cb => {{
      cb.checked = DATA.defaultJudges.includes(cb.value);
    }});
  }}
  onSelectionChange();
}}

function selectAllCommittee() {{
  document.querySelectorAll('#committee-member-checkboxes input').forEach(cb => cb.checked = true);
  onCommitteeChange();
}}

function selectNoneCommittee() {{
  document.querySelectorAll('#committee-member-checkboxes input').forEach(cb => cb.checked = false);
  onCommitteeChange();
}}

let debounceTimer = null;
function onSelectionChange() {{
  document.getElementById('sel-judges-count').textContent = getSelectedJudges().length;
  document.getElementById('sel-gens-count').textContent = getSelectedGens().length;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(recompute, 150);
}}

function onCommitteeChange() {{
  const members = getCommitteeMembers();
  const statusEl = document.getElementById('committee-status');
  statusEl.textContent = members.length >= 2
    ? 'Ref: ' + members.map(m => sn(m)).join(' + ')
    : 'Select at least 2 members for committee';
  onSelectionChange();
}}

function recompute() {{
  const judges = getActiveJudges(getSelectedJudges());
  const gens = getSelectedGens();
  effectiveJudgesList = judges;
  const r = computeAll(judges, gens);
  renderAggregate(r);
  renderPerJudge(r);
  renderPerGenerator(r);
  renderCharts(r);
  // Also update IA tab if it's active
  if (iaInitialized && document.querySelector('.tab.active')?.dataset.tab === 'interactive') {{
    iaBaselineResults = null;
    iaBaselineKey = '';
    setTimeout(iaRecompute, 50);
  }}
}}

function buildCheckboxes() {{
  const judgeDiv = document.getElementById('judge-checkboxes');
  const genDiv = document.getElementById('gen-checkboxes');
  const committeeDiv = document.getElementById('committee-member-checkboxes');
  const familyOrder = ['Gemma', 'Llama', 'Qwen', 'GPT', 'Claude'];

  for (const fam of familyOrder) {{
    const members = DATA.families[fam];

    // Judges
    const jFamDiv = document.createElement('div');
    jFamDiv.className = 'family-group';
    jFamDiv.innerHTML = '<span class="family-label">' + fam + '</span>';
    for (const m of members) {{
      if (!DATA.allJudges.includes(m)) continue;
      const isDefault = DATA.defaultJudges.includes(m);
      const label = sn(m) + (isDefault ? '' : ' *');
      jFamDiv.innerHTML += '<label class="cb-label"><input type="checkbox" value="' + m + '" checked onchange="onSelectionChange()"><span>' + label + '</span></label>';
    }}
    judgeDiv.appendChild(jFamDiv);

    // Generators
    const gFamDiv = document.createElement('div');
    gFamDiv.className = 'family-group';
    gFamDiv.innerHTML = '<span class="family-label">' + fam + '</span>';
    for (const m of members) {{
      if (!DATA.generators.includes(m)) continue;
      gFamDiv.innerHTML += '<label class="cb-label"><input type="checkbox" value="' + m + '" checked onchange="onSelectionChange()"><span>' + sn(m) + '</span></label>';
    }}
    genDiv.appendChild(gFamDiv);

    // Committee members
    const cFamDiv = document.createElement('div');
    cFamDiv.className = 'family-group';
    cFamDiv.innerHTML = '<span class="family-label">' + fam + '</span>';
    for (const m of members) {{
      if (!DATA.allJudges.includes(m)) continue;
      const isDefComm = DATA.defaultCommittee && DATA.defaultCommittee.includes(m);
      cFamDiv.innerHTML += '<label class="cb-label"><input type="checkbox" value="' + m + '" ' + (isDefComm ? 'checked' : '') + ' onchange="onCommitteeChange()"><span>' + sn(m) + '</span></label>';
    }}
    committeeDiv.appendChild(cFamDiv);
  }}
}}

// ============================================================
// INTERACTIVE ANALYSIS: FILTER STATE
// ============================================================

const IA_TOTAL_RUBRICS = DATA.rubricPointsFlat.length;
let iaRubricMask = new Uint8Array(IA_TOTAL_RUBRICS).fill(1);
let iaInstanceMask = new Uint8Array(DATA.nInstances).fill(1);
let iaBaselineResults = null;
let iaBaselineKey = '';
let iaInitialized = false;
let iaCachedSummary = null;
let iaCachedSummaryJudges = null;

const iaLenPercentiles = (() => {{
  const sorted = [...DATA.rubricLengthsFlat].sort((a,b) => a - b);
  const n = sorted.length;
  return {{
    p10: sorted[Math.floor(n * 0.1)],
    p25: sorted[Math.floor(n * 0.25)],
    p50: sorted[Math.floor(n * 0.5)],
    p75: sorted[Math.floor(n * 0.75)],
    p90: sorted[Math.floor(n * 0.9)],
    min: sorted[0],
    max: sorted[n - 1]
  }};
}})();

// ============================================================
// INTERACTIVE ANALYSIS: FILTER APPLICATION
// ============================================================

function iaApplyFilters() {{
  buildRubricOffsets();
  const ptsMin = +document.getElementById('ia-points-min').value;
  const ptsMax = +document.getElementById('ia-points-max').value;
  const lenMin = +document.getElementById('ia-len-min').value;
  const lenMax = +document.getElementById('ia-len-max').value;
  const agrMin = +document.getElementById('ia-agr-min').value / 100;
  const agrMax = +document.getElementById('ia-agr-max').value / 100;

  // Gather selected axis tags as bitmask
  let axisMask = 0;
  document.querySelectorAll('#ia-axis-checkboxes input:checked').forEach(cb => {{
    axisMask |= (1 << +cb.dataset.idx);
  }});
  const anyAxisSelected = axisMask !== 0;

  // Gather selected theme tags as bitmask
  let themeMask = 0;
  document.querySelectorAll('#ia-theme-checkboxes input:checked').forEach(cb => {{
    themeMask |= (1 << +cb.dataset.idx);
  }});
  const anyThemeSelected = themeMask !== 0;

  // Reset masks
  iaRubricMask.fill(0);
  iaInstanceMask.fill(1);

  // Pass 1: Apply instance-level theme filter
  for (let i = 0; i < DATA.nInstances; i++) {{
    if (anyThemeSelected) {{
      if ((DATA.instanceThemesFlat[i] & themeMask) === 0) {{
        iaInstanceMask[i] = 0;
      }}
    }}
  }}

  // Pass 2: Apply rubric-level filters
  let rubricIdx = 0;
  let filteredRubrics = 0;
  let filteredInstances = 0;
  for (let i = 0; i < DATA.nInstances; i++) {{
    const n = DATA.rubricCounts[i];
    let instHasRubric = false;
    for (let r = 0; r < n; r++) {{
      const pos = rubricIdx + r;
      if (!iaInstanceMask[i]) continue;

      const pts = DATA.rubricPointsFlat[pos];
      if (pts < ptsMin || pts > ptsMax) continue;

      const len = DATA.rubricLengthsFlat[pos];
      if (len < lenMin || len > lenMax) continue;

      if (anyAxisSelected) {{
        if ((DATA.rubricAxesFlat[pos] & axisMask) === 0) continue;
      }}

      const agr = DATA.rubricAgreementFlat[pos];
      if (agr < agrMin || agr > agrMax) continue;

      iaRubricMask[pos] = 1;
      instHasRubric = true;
      filteredRubrics++;
    }}
    rubricIdx += n;
    if (iaInstanceMask[i] && !instHasRubric) {{
      iaInstanceMask[i] = 0;
    }}
    if (iaInstanceMask[i]) filteredInstances++;
  }}

  document.getElementById('ia-rubric-count').textContent = filteredRubrics;
  document.getElementById('ia-instance-count').textContent = filteredInstances;
}}

// ============================================================
// INTERACTIVE ANALYSIS: FILTERED SCORING
// ============================================================

function iaInstanceScoreFiltered(rubricBools, instIdx, mode) {{
  buildRubricOffsets();
  const off = _rubricOffsets[instIdx];
  const n = DATA.rubricCounts[instIdx];
  if (mode === 'weighted') {{
    let totalPossible = 0, achieved = 0;
    for (let j = 0; j < n; j++) {{
      if (!iaRubricMask[off + j]) continue;
      const pts = DATA.rubricPointsFlat[off + j];
      if (pts > 0) totalPossible += pts;
      if (rubricBools[j]) achieved += pts;
    }}
    if (totalPossible <= 0) return null;
    return Math.max(0, Math.min(1, achieved / totalPossible));
  }} else {{
    let nPositive = 0, achieved = 0;
    for (let j = 0; j < n; j++) {{
      if (!iaRubricMask[off + j]) continue;
      const pts = DATA.rubricPointsFlat[off + j];
      if (pts > 0) nPositive++;
      if (rubricBools[j]) achieved += (pts > 0 ? 1 : -1);
    }}
    if (nPositive <= 0) return null;
    return Math.max(0, Math.min(1, achieved / nPositive));
  }}
}}

const _iaInstScoreCache = {{}};
let _iaInstScoreCacheKey = '';

function iaGetFilteredInstScores(judge, gen, mode) {{
  const ck = mode + '|' + judge + '|' + gen;
  if (_iaInstScoreCache[ck]) return _iaInstScoreCache[ck];
  const rubrics = getJudgeRubrics(judge, gen);
  if (!rubrics) return null;
  const scores = new Float64Array(DATA.nInstances);
  const valid = new Uint8Array(DATA.nInstances);
  for (let i = 0; i < DATA.nInstances; i++) {{
    if (!iaInstanceMask[i]) {{ scores[i] = NaN; continue; }}
    const s = iaInstanceScoreFiltered(rubrics[i], i, mode);
    if (s === null) {{ scores[i] = NaN; continue; }}
    scores[i] = s;
    valid[i] = 1;
  }}
  const result = {{ scores, valid }};
  _iaInstScoreCache[ck] = result;
  return result;
}}

function iaClearInstScoreCache() {{
  for (const k in _iaInstScoreCache) delete _iaInstScoreCache[k];
}}

function iaFilteredSystemScore(judge, gen, mode) {{
  const data = iaGetFilteredInstScores(judge, gen, mode);
  if (!data) return null;
  let sum = 0, count = 0;
  for (let i = 0; i < DATA.nInstances; i++) {{
    if (data.valid[i]) {{ sum += data.scores[i]; count++; }}
  }}
  return count > 0 ? sum / count : null;
}}

function iaFilteredRefScore(refRubrics, gen, mode) {{
  if (!refRubrics[gen]) return null;
  let sum = 0, count = 0;
  for (let i = 0; i < DATA.nInstances; i++) {{
    if (!iaInstanceMask[i]) continue;
    const s = iaInstanceScoreFiltered(refRubrics[gen][i], i, mode);
    if (s === null) continue;
    sum += s; count++;
  }}
  return count > 0 ? sum / count : null;
}}

// ============================================================
// INTERACTIVE ANALYSIS: MSD MATRIX
// ============================================================

function iaComputeMSDMatrix(judges, gens, mode, refRubrics) {{
  const judgeScores = {{}};
  const refScores = {{}};

  for (const g of gens) {{
    refScores[g] = iaFilteredRefScore(refRubrics, g, mode);
    for (const j of judges) {{
      judgeScores[j + '|' + g] = iaFilteredSystemScore(j, g, mode);
    }}
  }}

  // rawDeltas[j][g] = judgeScore - refScore
  const rawDeltas = {{}};
  const matrix = {{}};
  for (const j of judges) {{
    rawDeltas[j] = {{}};
    matrix[j] = {{}};
    const deltas = [];
    for (const g of gens) {{
      const js = judgeScores[j + '|' + g];
      const rs = refScores[g];
      if (js !== null && rs !== null) {{
        rawDeltas[j][g] = js - rs;
        deltas.push(js - rs);
      }}
    }}
    const meanD = deltas.length > 0 ? deltas.reduce((a,b) => a+b, 0) / deltas.length : 0;
    for (const g of gens) {{
      if (rawDeltas[j][g] !== undefined) {{
        matrix[j][g] = rawDeltas[j][g] - meanD;
      }}
    }}
  }}

  return {{ matrix, judgeScores, refScores, rawDeltas }};
}}

function iaRenderMSDMatrix(result, judges, gens, containerId) {{
  const m = result.matrix;
  let html = '<div class="msd-matrix"><table><thead><tr><th></th>';
  for (const g of gens) html += '<th>' + sn(g) + '</th>';
  html += '<th class="avg-col">Avg</th></tr></thead><tbody>';

  const colSums = {{}};
  const colCounts = {{}};
  let diagSum = 0, diagCount = 0;

  for (const j of judges) {{
    html += '<tr><td class="row-label">' + sn(j) + '</td>';
    let rowSum = 0, rowCount = 0;
    for (const g of gens) {{
      const val = m[j]?.[g];
      const isDiag = (j === g);
      let cls = isDiag ? 'diag' : '';
      if (val !== undefined) {{
        if (val > 0.0001) cls += ' pos';
        else if (val < -0.0001) cls += ' neg';
      }}
      html += '<td class="' + cls + '">' + (val !== undefined ? val.toFixed(4) : '\u2014') + '</td>';
      if (val !== undefined) {{
        rowSum += val; rowCount++;
        colSums[g] = (colSums[g] || 0) + val;
        colCounts[g] = (colCounts[g] || 0) + 1;
        if (isDiag) {{ diagSum += val; diagCount++; }}
      }}
    }}
    html += '<td class="avg-col">' + (rowCount > 0 ? (rowSum/rowCount).toFixed(4) : '\u2014') + '</td>';
    html += '</tr>';
  }}

  // Average row
  html += '<tr class="avg-row"><td class="row-label"><strong>Avg</strong></td>';
  for (const g of gens) {{
    const avg = colCounts[g] ? colSums[g] / colCounts[g] : null;
    html += '<td class="avg-col">' + (avg !== null ? avg.toFixed(4) : '\u2014') + '</td>';
  }}
  html += '<td class="avg-col diag">' + (diagCount > 0 ? (diagSum/diagCount).toFixed(4) : '\u2014') + '</td>';
  html += '</tr></tbody></table></div>';

  document.getElementById(containerId).innerHTML = html;
}}

function iaRenderDeltaMSDMatrix(filtered, baseline, judges, gens) {{
  const fm = filtered.matrix;
  const bm = baseline.matrix;
  const deltaMatrix = {{}};
  for (const j of judges) {{
    deltaMatrix[j] = {{}};
    for (const g of gens) {{
      if (fm[j]?.[g] !== undefined && bm[j]?.[g] !== undefined) {{
        deltaMatrix[j][g] = fm[j][g] - bm[j][g];
      }}
    }}
  }}
  iaRenderMSDMatrix({{ matrix: deltaMatrix }}, judges, gens, 'ia-delta-msd-matrix');
}}

// ============================================================
// INTERACTIVE ANALYSIS: SPEARMAN CORRELATION
// ============================================================

function spearmanCorr(x, y) {{
  const n = x.length;
  if (n < 3 || n !== y.length) return null;
  const rx = rankdata(x);
  const ry = rankdata(y);
  let sumRx = 0, sumRy = 0;
  for (let i = 0; i < n; i++) {{ sumRx += rx[i]; sumRy += ry[i]; }}
  const meanRx = sumRx / n, meanRy = sumRy / n;
  let num = 0, denX = 0, denY = 0;
  for (let i = 0; i < n; i++) {{
    const dx = rx[i] - meanRx, dy = ry[i] - meanRy;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }}
  const den = Math.sqrt(denX * denY);
  return den > 0 ? num / den : 0;
}}

// ============================================================
// INTERACTIVE ANALYSIS: FILTERED HSPP / MRSPB-err / MIPA
// ============================================================

function iaOverestRateFiltered(judge, targetGen, gens, mode, errorDenom, refRubrics) {{
  const tgtData = iaGetFilteredInstScores(judge, targetGen, mode);
  if (!tgtData) return null;
  const refTgt = iaGetFilteredRefInstScores(refRubrics, targetGen, mode);
  if (!refTgt) return null;
  let nOver = 0, nTotal = 0;
  for (const opp of gens) {{
    if (opp === targetGen) continue;
    const oppData = iaGetFilteredInstScores(judge, opp, mode);
    if (!oppData) continue;
    const refOpp = iaGetFilteredRefInstScores(refRubrics, opp, mode);
    if (!refOpp) continue;
    for (let i = 0; i < DATA.nInstances; i++) {{
      if (!iaInstanceMask[i] || !tgtData.valid[i] || !oppData.valid[i] || !refTgt.valid[i] || !refOpp.valid[i]) continue;
      const jSign = Math.sign(tgtData.scores[i] - oppData.scores[i]);
      const rSign = Math.sign(refTgt.scores[i] - refOpp.scores[i]);
      if (errorDenom && rSign >= 0) continue;
      nTotal++;
      if (jSign > rSign) nOver++;
    }}
  }}
  return nTotal > 0 ? {{ rate: nOver / nTotal, nOver, nTotal }} : null;
}}

function iaGetFilteredRefInstScores(refRubrics, gen, mode) {{
  if (!refRubrics[gen]) return null;
  const scores = new Float64Array(DATA.nInstances);
  const valid = new Uint8Array(DATA.nInstances);
  for (let i = 0; i < DATA.nInstances; i++) {{
    if (!iaInstanceMask[i]) {{ scores[i] = NaN; continue; }}
    const s = iaInstanceScoreFiltered(refRubrics[gen][i], i, mode);
    if (s === null) {{ scores[i] = NaN; continue; }}
    scores[i] = s;
    valid[i] = 1;
  }}
  return {{ scores, valid }};
}}

function iaComputeMISPBFiltered(judges, gens, mode, errorDenom, familyMode, includeSelf, refRubrics) {{
  if (includeSelf === undefined) includeSelf = true;
  const perJudge = {{}}, perJudgeRaw = {{}}, perJudgeOther = {{}}, perJudgeRatio = {{}};
  for (const judge of judges) {{
    const targets = familyMode ? getFamilyGens(judge, gens, includeSelf) : (gens.includes(judge) ? [judge] : []);
    if (!targets.length) continue;
    const rawResults = targets.map(t => iaOverestRateFiltered(judge, t, gens, mode, errorDenom, refRubrics)).filter(r => r !== null);
    if (!rawResults.length) continue;
    const raw = rawResults.reduce((a,b) => a + b.rate, 0) / rawResults.length;
    const others = getOtherGens(judge, gens);
    const otherResults = others.map(g => iaOverestRateFiltered(judge, g, gens, mode, errorDenom, refRubrics)).filter(r => r !== null);
    if (!otherResults.length) continue;
    const other = otherResults.reduce((a,b) => a + b.rate, 0) / otherResults.length;
    perJudge[judge] = raw - other;
    perJudgeRaw[judge] = raw;
    perJudgeOther[judge] = other;
    perJudgeRatio[judge] = other > 0 ? raw / other : (raw > 0 ? Infinity : 1);
  }}
  const vals = Object.values(perJudge);
  return {{ mean: vals.length ? vals.reduce((a,b) => a+b, 0) / vals.length : null, perJudge, perJudgeRaw, perJudgeOther, perJudgeRatio }};
}}

function iaRubricOverestRateFiltered(judge, targetGen, errorDenom, refRubrics) {{
  if (!refRubrics[targetGen]) return null;
  const jRubrics = getJudgeRubrics(judge, targetGen);
  if (!jRubrics) return null;
  const rRubrics = refRubrics[targetGen];
  buildRubricOffsets();
  let nOver = 0, nTotal = 0;
  for (let i = 0; i < DATA.nInstances; i++) {{
    if (!iaInstanceMask[i]) continue;
    const off = _rubricOffsets[i];
    for (let r = 0; r < jRubrics[i].length; r++) {{
      if (!iaRubricMask[off + r]) continue;
      const pts = DATA.rubricPointsFlat[off + r];
      const isPositive = pts > 0;
      if (errorDenom) {{
        if (isPositive && rRubrics[i][r]) continue;
        if (!isPositive && !rRubrics[i][r]) continue;
      }}
      nTotal++;
      if (isPositive) {{
        if (jRubrics[i][r] && !rRubrics[i][r]) nOver++;
      }} else {{
        if (!jRubrics[i][r] && rRubrics[i][r]) nOver++;
      }}
    }}
  }}
  return nTotal > 0 ? {{ rate: nOver / nTotal, nOver, nTotal }} : null;
}}

function iaComputeMRSPBFiltered(judges, gens, errorDenom, familyMode, includeSelf, refRubrics) {{
  if (includeSelf === undefined) includeSelf = true;
  const perJudge = {{}}, perJudgeRaw = {{}}, perJudgeOther = {{}}, perJudgeRatio = {{}};
  for (const judge of judges) {{
    const targets = familyMode ? getFamilyGens(judge, gens, includeSelf) : (gens.includes(judge) ? [judge] : []);
    if (!targets.length) continue;
    const rawResults = targets.map(t => iaRubricOverestRateFiltered(judge, t, errorDenom, refRubrics)).filter(r => r !== null);
    if (!rawResults.length) continue;
    const raw = rawResults.reduce((a,b) => a + b.rate, 0) / rawResults.length;
    const others = getOtherGens(judge, gens);
    const otherResults = others.map(g => iaRubricOverestRateFiltered(judge, g, errorDenom, refRubrics)).filter(r => r !== null);
    if (!otherResults.length) continue;
    const other = otherResults.reduce((a,b) => a + b.rate, 0) / otherResults.length;
    perJudge[judge] = raw - other;
    perJudgeRaw[judge] = raw;
    perJudgeOther[judge] = other;
    perJudgeRatio[judge] = other > 0 ? raw / other : (raw > 0 ? Infinity : 1);
  }}
  const vals = Object.values(perJudge);
  return {{ mean: vals.length ? vals.reduce((a,b) => a+b, 0) / vals.length : null, perJudge, perJudgeRaw, perJudgeOther, perJudgeRatio }};
}}

function iaComputeMIPAFiltered(judges, gens, mode, refRubrics) {{
  const perJudge = {{}};
  for (const judge of judges) {{
    const jGens = gens.filter(g => iaGetFilteredInstScores(judge, g, mode) && refRubrics[g]);
    if (jGens.length < 2) continue;
    const instScores = {{}};
    const refInst = {{}};
    for (const g of jGens) {{
      instScores[g] = iaGetFilteredInstScores(judge, g, mode);
      refInst[g] = iaGetFilteredRefInstScores(refRubrics, g, mode);
    }}
    let agree = 0, total = 0;
    for (let i = 0; i < DATA.nInstances; i++) {{
      if (!iaInstanceMask[i]) continue;
      for (let gi = 0; gi < jGens.length; gi++) {{
        for (let gj = gi + 1; gj < jGens.length; gj++) {{
          const g1 = jGens[gi], g2 = jGens[gj];
          if (!instScores[g1].valid[i] || !instScores[g2].valid[i] || !refInst[g1].valid[i] || !refInst[g2].valid[i]) continue;
          const jd = instScores[g1].scores[i] - instScores[g2].scores[i];
          const rd = refInst[g1].scores[i] - refInst[g2].scores[i];
          if ((jd > 0 && rd > 0) || (jd < 0 && rd < 0) || (jd === 0 && rd === 0)) agree++;
          total++;
        }}
      }}
    }}
    if (total > 0) perJudge[judge] = agree / total;
  }}
  const vals = Object.values(perJudge);
  return {{ mean: vals.length ? vals.reduce((a,b) => a+b, 0) / vals.length : null, perJudge }};
}}

function iaComputeMIPAFilteredVsBaseline(judges, gens, mode) {{
  // MIPA comparing filtered instance rankings to unfiltered (baseline) rankings
  // Uses getInstScores() for baseline (regular unfiltered cache)
  const perJudge = {{}};
  for (const judge of judges) {{
    const jGens = gens.filter(g => iaGetFilteredInstScores(judge, g, mode) && getInstScores(judge, g, mode));
    if (jGens.length < 2) continue;
    const filteredScores = {{}};
    const baselineScores = {{}};
    for (const g of jGens) {{
      filteredScores[g] = iaGetFilteredInstScores(judge, g, mode);
      baselineScores[g] = getInstScores(judge, g, mode);
    }}

    let agree = 0, total = 0;
    for (let i = 0; i < DATA.nInstances; i++) {{
      if (!filteredScores[jGens[0]]?.valid[i]) continue;
      for (let gi = 0; gi < jGens.length; gi++) {{
        for (let gj = gi + 1; gj < jGens.length; gj++) {{
          const g1 = jGens[gi], g2 = jGens[gj];
          if (!filteredScores[g1].valid[i] || !filteredScores[g2].valid[i]) continue;
          const fd = filteredScores[g1].scores[i] - filteredScores[g2].scores[i];
          const bd = baselineScores[g1][i] - baselineScores[g2][i];
          if ((fd > 0 && bd > 0) || (fd < 0 && bd < 0) || (fd === 0 && bd === 0)) agree++;
          total++;
        }}
      }}
    }}
    if (total > 0) perJudge[judge] = agree / total;
  }}
  const vals = Object.values(perJudge);
  return {{ mean: vals.length ? vals.reduce((a,b) => a+b, 0) / vals.length : null, perJudge }};
}}

// ============================================================
// INTERACTIVE ANALYSIS: SUMMARY TABLE
// ============================================================

function iaComputeSummaryTable(judges, gens, mode, refRubrics, msdResult, baselineMsdResult) {{
  // HSPP variants
  const hspp = iaComputeMISPBFiltered(judges, gens, mode, true, false, true, refRubrics);
  const hsppF = iaComputeMISPBFiltered(judges, gens, mode, true, true, true, refRubrics);
  const hsppFO = iaComputeMISPBFiltered(judges, gens, mode, true, true, false, refRubrics);

  // MRSPB-err variants
  const mrspbErr = iaComputeMRSPBFiltered(judges, gens, true, false, true, refRubrics);
  const mrspbErrF = iaComputeMRSPBFiltered(judges, gens, true, true, true, refRubrics);
  const mrspbErrFO = iaComputeMRSPBFiltered(judges, gens, true, true, false, refRubrics);

  // MIPA vs ref
  const mipa = iaComputeMIPAFiltered(judges, gens, mode, refRubrics);

  // SPA + MIPA vs baseline
  const rows = {{}};
  for (const judge of judges) {{
    const row = {{}};
    // Difference variants (raw - other)
    row.hspp = hspp.perJudge[judge];
    row.mrspbErr = mrspbErr.perJudge[judge];
    row.hsppF = hsppF.perJudge[judge];
    row.hsppFO = hsppFO.perJudge[judge];
    row.mrspbErrF = mrspbErrF.perJudge[judge];
    row.mrspbErrFO = mrspbErrFO.perJudge[judge];
    // Ratio variants (raw / other)
    row.hsppR = hspp.perJudgeRatio[judge];
    row.mrspbErrR = mrspbErr.perJudgeRatio[judge];
    row.hsppFR = hsppF.perJudgeRatio[judge];
    row.hsppFOR = hsppFO.perJudgeRatio[judge];
    row.mrspbErrFR = mrspbErrF.perJudgeRatio[judge];
    row.mrspbErrFOR = mrspbErrFO.perJudgeRatio[judge];
    row.mipa = mipa.perJudge[judge];

    // SPA vs reference: Spearman of filtered gen rankings vs ref gen rankings
    const filteredArr = [], refArr = [];
    for (const g of gens) {{
      const js = msdResult.judgeScores[judge + '|' + g];
      const rs = msdResult.refScores[g];
      if (js !== null && js !== undefined && rs !== null && rs !== undefined) {{
        filteredArr.push(js);
        refArr.push(rs);
      }}
    }}
    row.spa = filteredArr.length >= 3 ? spearmanCorr(filteredArr, refArr) : null;

    // SPA vs unfiltered: Spearman of filtered rankings vs baseline rankings (same judge)
    if (baselineMsdResult) {{
      const filtArr2 = [], baseArr = [];
      for (const g of gens) {{
        const js = msdResult.judgeScores[judge + '|' + g];
        const bs = baselineMsdResult.judgeScores[judge + '|' + g];
        if (js !== null && js !== undefined && bs !== null && bs !== undefined) {{
          filtArr2.push(js);
          baseArr.push(bs);
        }}
      }}
      row.spaBaseline = filtArr2.length >= 3 ? spearmanCorr(filtArr2, baseArr) : null;
    }}

    rows[judge] = row;
  }}

  // MIPA vs baseline
  const mipaBaseline = iaComputeMIPAFilteredVsBaseline(judges, gens, mode);
  for (const judge of judges) {{
    if (rows[judge]) rows[judge].mipaBaseline = mipaBaseline.perJudge[judge];
  }}

  return rows;
}}

function iaGetBiasMode() {{
  const el = document.querySelector('input[name="ia-bias-mode"]:checked');
  return el ? el.value : 'ratio';
}}

function iaRenderSummaryTable(summary, judges) {{
  const isRatio = iaGetBiasMode() === 'ratio';
  const suffix = isRatio ? 'R' : '';
  const biasLabel = isRatio ? ' (ratio)' : ' (diff)';
  const biasFmt = isRatio ? 2 : 4;

  const headers = ['Judge',
    'HSPP' + biasLabel, 'MRSPB-err' + biasLabel,
    'SPA (ref)', 'MIPA (ref)',
    'HSPP-F' + biasLabel, 'HSPP-FO' + biasLabel,
    'MRSPB-err-F' + biasLabel, 'MRSPB-err-FO' + biasLabel,
    'SPA (unfilt)', 'MIPA (unfilt)'];

  const keys = [
    isRatio ? 'hsppR' : 'hspp',
    isRatio ? 'mrspbErrR' : 'mrspbErr',
    'spa', 'mipa',
    isRatio ? 'hsppFR' : 'hsppF',
    isRatio ? 'hsppFOR' : 'hsppFO',
    isRatio ? 'mrspbErrFR' : 'mrspbErrF',
    isRatio ? 'mrspbErrFOR' : 'mrspbErrFO',
    'spaBaseline', 'mipaBaseline'
  ];

  // Per-key format: ratio uses 2 decimals, difference and others use 4
  const isBiasKey = (k) => k.startsWith('hspp') || k.startsWith('mrspb');
  const fmtVal = (v, k) => {{
    if (v === null || v === undefined || isNaN(v)) return '\u2014';
    if (v === Infinity) return '\u221e';
    return isBiasKey(k) ? v.toFixed(biasFmt) : v.toFixed(4);
  }};

  let html = '<table><thead><tr>';
  for (const h of headers) html += '<th>' + h + '</th>';
  html += '</tr></thead><tbody>';

  for (const j of judges) {{
    const r = summary[j] || {{}};
    html += '<tr><td class="row-label">' + sn(j) + '</td>';
    for (const k of keys) html += '<td>' + fmtVal(r[k], k) + '</td>';
    html += '</tr>';
  }}

  // Average row
  html += '<tr><td class="row-label"><strong>Average</strong></td>';
  for (const k of keys) {{
    const vals = judges.map(j => summary[j]?.[k]).filter(v => v !== null && v !== undefined && !isNaN(v) && isFinite(v));
    const avg = vals.length ? vals.reduce((a,b) => a+b, 0) / vals.length : null;
    html += '<td><strong>' + fmtVal(avg, k) + '</strong></td>';
  }}
  html += '</tr></tbody></table>';

  document.getElementById('ia-summary-table').innerHTML = html;
}}

// ============================================================
// INTERACTIVE ANALYSIS: MASTER RECOMPUTE
// ============================================================

function iaRecompute() {{
  const mode = getScoringMode();
  const gens = getSelectedGens();
  const members = getCommitteeMembers();
  const judges = getActiveJudges(getSelectedJudges());

  if (members.length < 2 || gens.length < 2 || judges.length < 1) {{
    const noData = '<p class="no-data">Need >= 2 committee members, >= 2 generators, and >= 1 judge</p>';
    document.getElementById('ia-msd-matrix').innerHTML = noData;
    document.getElementById('ia-delta-msd-matrix').innerHTML = noData;
    document.getElementById('ia-summary-table').innerHTML = noData;
    return;
  }}

  const refRubrics = buildCommitteeReference(members, gens);

  // Apply filters and clear filtered score cache
  iaApplyFilters();
  iaClearInstScoreCache();

  // Compute filtered MSD matrix
  const msdResult = iaComputeMSDMatrix(judges, gens, mode, refRubrics);

  // Compute or use cached baseline
  const currentKey = members.slice().sort().join('+') + '|' + mode + '|' + gens.slice().sort().join('+') + '|' + judges.slice().sort().join('+');
  if (!iaBaselineResults || iaBaselineKey !== currentKey) {{
    const savedRM = iaRubricMask.slice();
    const savedIM = iaInstanceMask.slice();
    iaRubricMask.fill(1);
    iaInstanceMask.fill(1);
    iaClearInstScoreCache();
    iaBaselineResults = iaComputeMSDMatrix(judges, gens, mode, refRubrics);
    iaBaselineKey = currentKey;
    // Restore masks
    iaRubricMask = savedRM;
    iaInstanceMask = savedIM;
    iaClearInstScoreCache();
  }}

  // Render MSD matrices
  iaRenderMSDMatrix(msdResult, judges, gens, 'ia-msd-matrix');
  iaRenderDeltaMSDMatrix(msdResult, iaBaselineResults, judges, gens);

  // Compute and render summary table
  const summary = iaComputeSummaryTable(judges, gens, mode, refRubrics, msdResult, iaBaselineResults);
  iaCachedSummary = summary;
  iaCachedSummaryJudges = judges;
  iaRenderSummaryTable(summary, judges);
}}

// ============================================================
// INTERACTIVE ANALYSIS: FILTER UI
// ============================================================

function iaInitFilters() {{
  // Axis checkboxes
  const axisDiv = document.getElementById('ia-axis-checkboxes');
  for (let idx = 0; idx < DATA.axisNames.length; idx++) {{
    const tag = DATA.axisNames[idx];
    const label = tag.replace('axis:', '');
    axisDiv.innerHTML += '<label class="cb-label"><input type="checkbox" data-idx="' + idx + '" checked onchange="onIAFilterChange()"><span>' + label + '</span></label>';
  }}

  // Theme checkboxes
  const themeDiv = document.getElementById('ia-theme-checkboxes');
  for (let idx = 0; idx < DATA.themeNames.length; idx++) {{
    const tag = DATA.themeNames[idx];
    const label = tag.replace('theme:', '');
    themeDiv.innerHTML += '<label class="cb-label"><input type="checkbox" data-idx="' + idx + '" checked onchange="onIAFilterChange()"><span>' + label + '</span></label>';
  }}

  // Set length percentile text and max
  document.getElementById('ia-len-percentiles').textContent =
    'p10=' + iaLenPercentiles.p10 + ' | p25=' + iaLenPercentiles.p25 +
    ' | med=' + iaLenPercentiles.p50 + ' | p75=' + iaLenPercentiles.p75 +
    ' | p90=' + iaLenPercentiles.p90;
  document.getElementById('ia-len-max').value = iaLenPercentiles.max;

  // Set totals
  document.getElementById('ia-rubric-total').textContent = IA_TOTAL_RUBRICS;
  document.getElementById('ia-instance-total').textContent = DATA.nInstances;
  document.getElementById('ia-rubric-count').textContent = IA_TOTAL_RUBRICS;
  document.getElementById('ia-instance-count').textContent = DATA.nInstances;

  iaInitialized = true;
}}

let iaDebounceTimer = null;
function onIAFilterChange() {{
  clearTimeout(iaDebounceTimer);
  iaDebounceTimer = setTimeout(iaRecompute, 300);
}}

function onIAAgrSlider() {{
  const minVal = +document.getElementById('ia-agr-min').value;
  const maxVal = +document.getElementById('ia-agr-max').value;
  document.getElementById('ia-agr-min-val').textContent = minVal + '%';
  document.getElementById('ia-agr-max-val').textContent = maxVal + '%';
  // Enforce min <= max
  if (minVal > maxVal) {{
    document.getElementById('ia-agr-max').value = minVal;
    document.getElementById('ia-agr-max-val').textContent = minVal + '%';
  }}
  onIAFilterChange();
}}

function iaOnBiasModeChange() {{
  // Re-render summary table with cached data (no recompute needed)
  if (iaCachedSummary && iaCachedSummaryJudges) {{
    iaRenderSummaryTable(iaCachedSummary, iaCachedSummaryJudges);
  }}
}}

function iaAxisAll() {{
  document.querySelectorAll('#ia-axis-checkboxes input').forEach(cb => cb.checked = true);
  onIAFilterChange();
}}
function iaAxisNone() {{
  document.querySelectorAll('#ia-axis-checkboxes input').forEach(cb => cb.checked = false);
  onIAFilterChange();
}}
function iaThemeAll() {{
  document.querySelectorAll('#ia-theme-checkboxes input').forEach(cb => cb.checked = true);
  onIAFilterChange();
}}
function iaThemeNone() {{
  document.querySelectorAll('#ia-theme-checkboxes input').forEach(cb => cb.checked = false);
  onIAFilterChange();
}}

function iaResetFilters() {{
  document.getElementById('ia-points-min').value = -10;
  document.getElementById('ia-points-max').value = 10;
  document.getElementById('ia-len-min').value = 0;
  document.getElementById('ia-len-max').value = iaLenPercentiles.max;
  document.getElementById('ia-agr-min').value = 0;
  document.getElementById('ia-agr-max').value = 100;
  document.getElementById('ia-agr-min-val').textContent = '0%';
  document.getElementById('ia-agr-max-val').textContent = '100%';
  iaAxisAll();
  iaThemeAll();
}}

// Tab switching
document.querySelectorAll('.tab').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.remove('hidden');
    if (btn.dataset.tab === 'charts' && cachedResults) {{
      setTimeout(() => renderCharts(cachedResults), 50);
    }}
    if (btn.dataset.tab === 'interactive') {{
      setTimeout(() => iaRecompute(), 50);
    }}
  }});
}});

// Initialize
buildCheckboxes();
iaInitFilters();
onCommitteeChange();
</script>

</body>
</html>"""


# CSS block (same as IFEval with minor color adjustments)
CSS_BLOCK = """<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --sand-50: #faf9f7;
    --sand-100: #f5f3f0;
    --sand-200: #e8e4de;
    --sand-300: #d4cec4;
    --sand-400: #a69f94;
    --sand-500: #7a7267;
    --sand-600: #5c564d;
    --sand-700: #3d3a35;
    --sand-800: #252320;
    --accent: #c17f59;
    --accent-light: #daa88a;
    --accent-soft: #f7ebe4;
    --text-primary: var(--sand-800);
    --text-secondary: var(--sand-500);
    --text-muted: var(--sand-400);
    --bg-page: var(--sand-50);
    --bg-elevated: #ffffff;
    --border-subtle: var(--sand-200);
    --success: #5b8a72;
    --success-bg: #f0f5f2;
    --error: #c75a5a;
    --error-bg: #fdf2f2;
}

html { font-size: 14px; }
body {
    font-family: 'Atkinson Hyperlegible', sans-serif;
    background: var(--bg-page);
    color: var(--text-primary);
    line-height: 1.5;
    scrollbar-width: thin;
}

h1, h2, h3 { font-family: 'Fraunces', serif; font-weight: 500; }

.page-header {
    padding: 2rem 2rem 1.5rem;
    border-bottom: 1px solid var(--border-subtle);
    background: var(--bg-elevated);
}
.page-header h1 { font-size: 1.75rem; color: var(--sand-800); }
.page-header .subtitle { color: var(--text-secondary); margin-top: 0.25rem; font-size: 0.9rem; }

.layout {
    display: flex;
    min-height: calc(100vh - 100px);
}

.panel {
    width: 260px;
    min-width: 260px;
    background: var(--bg-elevated);
    border-right: 1px solid var(--border-subtle);
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
    scrollbar-width: thin;
}
.panel-inner { padding: 1rem; }
.panel-section { margin-bottom: 1.5rem; }
.panel-section h3 { font-size: 1rem; margin-bottom: 0.5rem; color: var(--sand-700); }

.panel-actions {
    display: flex;
    gap: 0.25rem;
    margin-bottom: 0.5rem;
}
.btn-sm {
    font-family: 'Atkinson Hyperlegible', sans-serif;
    font-size: 0.75rem;
    padding: 0.2rem 0.5rem;
    border: 1px solid var(--accent);
    background: transparent;
    color: var(--accent);
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s;
}
.btn-sm:hover { background: var(--accent-soft); }

.family-group { margin-bottom: 0.5rem; }
.family-label {
    display: block;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.15rem;
    font-weight: 700;
}
.cb-label {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.15rem 0;
    font-size: 0.85rem;
    cursor: pointer;
}
.cb-label input[type="checkbox"], .cb-label input[type="radio"] {
    accent-color: var(--accent);
    width: 14px;
    height: 14px;
}

.info-box {
    background: var(--sand-100);
    border-radius: 8px;
    padding: 0.75rem;
    font-size: 0.8rem;
}
.info-box .note { color: var(--text-muted); font-size: 0.75rem; margin-top: 0.25rem; }
.note { color: var(--text-muted); font-size: 0.75rem; }

.dashboard {
    flex: 1;
    padding: 1.5rem;
    max-width: calc(100% - 260px);
    min-width: 0;
}

.tabs {
    display: flex;
    gap: 0.25rem;
    margin-bottom: 1.5rem;
    border-bottom: 2px solid var(--border-subtle);
    padding-bottom: 0;
}
.tab {
    font-family: 'Atkinson Hyperlegible', sans-serif;
    font-size: 0.9rem;
    padding: 0.5rem 1rem;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: all 0.15s;
}
.tab:hover { color: var(--text-primary); }
.tab.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
    font-weight: 700;
}

.tab-content.hidden { display: none; }

.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(360px, 100%), 1fr));
    gap: 1rem;
    margin-bottom: 1rem;
}
.card {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    overflow-x: auto;
    min-width: 0;
}
.card h3 { font-size: 1rem; margin-bottom: 0.25rem; }
.card-desc { font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.75rem; }

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
thead th {
    text-align: left;
    padding: 0.5rem 0.6rem;
    border-bottom: 2px solid var(--border-subtle);
    font-weight: 700;
    color: var(--sand-600);
    font-size: 0.78rem;
    white-space: nowrap;
}
tbody td {
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid var(--sand-100);
}
tbody tr:hover { background: var(--sand-100); }
td.row-label { font-weight: 700; color: var(--sand-700); }

.no-data {
    color: var(--text-muted);
    font-style: italic;
    padding: 1rem;
    text-align: center;
}

.section-heading {
    font-size: 1.15rem;
    color: var(--sand-600);
    margin: 1.5rem 0 0.75rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--border-subtle);
}
.section-heading:first-child { margin-top: 0; }

.chart-container {
    min-height: 350px;
    width: 100%;
}

details.metric-detail {
    margin-top: 0.75rem;
    border: 1px solid var(--sand-200);
    border-radius: 8px;
    overflow: hidden;
}
details.metric-detail summary {
    padding: 0.5rem 0.75rem;
    background: var(--sand-100);
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--sand-600);
    cursor: pointer;
    user-select: none;
}
details.metric-detail summary:hover { background: var(--sand-200); }
details.metric-detail[open] summary { border-bottom: 1px solid var(--sand-200); }
details.metric-detail .detail-body {
    padding: 0.5rem;
    font-size: 0.78rem;
    overflow-x: auto;
}
details.metric-detail .detail-body table { font-size: 0.75rem; }
details.metric-detail .detail-body td, details.metric-detail .detail-body th { padding: 0.25rem 0.4rem; }

/* Interactive Analysis tab */
.ia-layout {
    display: flex;
    gap: 1.5rem;
}
.ia-filters {
    width: 260px;
    min-width: 260px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1rem;
    position: sticky;
    top: 1rem;
    max-height: calc(100vh - 8rem);
    overflow-y: auto;
    scrollbar-width: thin;
}
.ia-filters h3 { font-size: 1rem; margin-bottom: 0.75rem; color: var(--sand-700); }
.ia-results {
    flex: 1;
    min-width: 0;
}
.filter-group {
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--sand-100);
}
.filter-group label {
    font-weight: 700;
    font-size: 0.85rem;
    color: var(--sand-700);
    display: block;
    margin-bottom: 0.35rem;
}
.range-inputs {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.25rem;
}
.range-inputs input[type="number"] {
    width: 60px;
    padding: 0.2rem 0.4rem;
    font-size: 0.8rem;
    border: 1px solid var(--sand-300);
    border-radius: 4px;
    font-family: inherit;
}
.range-inputs span {
    font-size: 0.8rem;
    color: var(--text-muted);
}
.range-label-sm {
    font-size: 0.72rem !important;
    font-weight: 400 !important;
    color: var(--text-muted) !important;
    margin-bottom: 0.1rem !important;
}
input[type="range"] {
    width: 100%;
    accent-color: var(--accent);
}

.ia-bias-toggle {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
    font-size: 0.85rem;
}
.ia-bias-toggle .cb-label { padding: 0; }

.msd-matrix table {
    font-size: 0.72rem;
    border-collapse: collapse;
}
.msd-matrix th, .msd-matrix td {
    padding: 0.25rem 0.35rem;
    text-align: right;
    white-space: nowrap;
    min-width: 55px;
}
.msd-matrix th {
    font-size: 0.68rem;
}
.msd-matrix td.diag {
    background: var(--accent-soft);
    font-weight: 700;
}
.msd-matrix tr.avg-row td {
    border-top: 2px solid var(--sand-300);
    font-weight: 600;
}
.msd-matrix td.avg-col {
    border-left: 2px solid var(--sand-300);
    font-weight: 600;
}
.msd-matrix td.pos { color: #b2182b; }
.msd-matrix td.neg { color: #2166ac; }

@media (max-width: 1100px) {
    .ia-layout { flex-direction: column; }
    .ia-filters { width: 100%; min-width: 100%; position: static; max-height: none; }
}
@media (max-width: 900px) {
    .layout { flex-direction: column; }
    .panel { width: 100%; min-width: 100%; height: auto; position: static; border-right: none; border-bottom: 1px solid var(--border-subtle); }
    .panel-inner { display: flex; flex-wrap: wrap; gap: 1rem; }
    .dashboard { max-width: 100%; }
    .card-grid { grid-template-columns: 1fr; }
}
</style>"""


def main():
    logger.info("Loading all HealthBench data...")
    all_data = load_all_data()

    logger.info("Packing data...")
    packed = pack_data(all_data)
    packed_json = json.dumps(packed, separators=(',', ':'))
    logger.info(f"Packed data size: {len(packed_json):,} bytes")

    logger.info("Generating HTML...")
    html = generate_html(packed_json)

    out_path = Path(__file__).parent / "hb_dashboard.html"
    with open(out_path, "w") as f:
        f.write(html)
    logger.info(f"Dashboard written to {out_path} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
