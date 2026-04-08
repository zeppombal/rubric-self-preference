#!/usr/bin/env python3
"""Build interactive HTML dashboard for IFEval judge analysis.

Loads all experimental data, packs it into compact JSON, and generates
a self-contained HTML file with embedded data, CSS, and JS.
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from math import comb
from data_loading import (
    load_all_data, GENERATORS, JUDGES, FAMILIES, MODEL_TO_FAMILY, N_INSTANCES,
    DATA_ROOT, _load_rubric_eval_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ALL_JUDGES = list(GENERATORS)  # All 12 models can be judges
EXTRA_JUDGES = [j for j in ALL_JUDGES if j not in JUDGES]

SHORT_NAMES = {
    "gemma_3_27b_it": "Gemma-27B",
    "gemma_3_12b_it": "Gemma-12B",
    "gemma_3_4b_it": "Gemma-4B",
    "llama_4_maverick_17b_128e_instruct": "Llama-Mav",
    "llama_4_scout_17b_16e_instruct": "Llama-Scout",
    "qwen3_235b_instruct": "Qwen-235B",
    "qwen3_30b_instruct": "Qwen-30B",
    "qwen3_4b_instruct": "Qwen-4B",
    "gpt_120b_oss": "GPT-120B",
    "gpt_5": "GPT-5",
    "claude_4_5_haiku": "Claude-Haiku",
    "claude_4_5_sonnet": "Claude-Sonnet",
}


def load_extra_rubric_data(eval_dir, ref_data, method_name):
    """Load SR/AR data for the 3 extra judges (GPT-5, Claude-Haiku, Claude-Sonnet)."""
    data = {}
    for judge in EXTRA_JUDGES:
        for gen in GENERATORS:
            dir_name = f"Evaluator.{judge}+Generator.{gen}"
            path = DATA_ROOT / eval_dir / dir_name / "evaluation_allresults.json"
            if not path.exists():
                continue
            with open(path) as f:
                raw = json.load(f)
            metadata = raw["metadata"]["example_level_metadata"]
            if len(metadata) != N_INSTANCES:
                continue
            instance_rubrics = []
            for i, item in enumerate(metadata):
                rubric_items = item["rubric_items"]
                criteria = [r["criteria_met"] for r in rubric_items]
                ref_len = len(ref_data[gen]["follow_list"][i])
                if len(criteria) != ref_len:
                    logger.warning(f"{method_name} rubric mismatch for extra judge {judge}, gen {gen}, inst {i}")
                    break
                instance_rubrics.append(criteria)
            else:
                data[(judge, gen)] = instance_rubrics
    logger.info(f"Loaded extra {method_name} data: {len(data)} pairs for judges {EXTRA_JUDGES}")
    return data


def pack_data(all_data):
    """Pack all loaded data into compact JSON-serializable format."""
    ref = all_data["ref"]
    gen_data = all_data["gen"]
    sr = all_data["sr"]
    ar = all_data["ar"]
    da = all_data["da"]
    pwc = all_data["pwc"]

    packed = {
        "generators": GENERATORS,
        "allJudges": ALL_JUDGES,
        "defaultJudges": list(JUDGES),
        "families": FAMILIES,
        "modelToFamily": MODEL_TO_FAMILY,
        "shortNames": SHORT_NAMES,
        "nInstances": N_INSTANCES,
    }

    # Reference: follow_all as binary strings
    packed["refFollowAll"] = {}
    for gen in GENERATORS:
        packed["refFollowAll"][gen] = "".join("1" if v else "0" for v in ref[gen]["follow_all"])

    # Reference: rubric counts per instance
    packed["refRubricCounts"] = {}
    for gen in GENERATORS:
        packed["refRubricCounts"][gen] = gen_data[gen]

    # Reference: flattened rubric list as binary strings
    packed["refRubricFlat"] = {}
    for gen in GENERATORS:
        flat = []
        for inst_rubrics in ref[gen]["follow_list"]:
            flat.extend("1" if v else "0" for v in inst_rubrics)
        packed["refRubricFlat"][gen] = "".join(flat)

    # Reference: per-instance fraction scores (fraction of rubrics met)
    packed["refInstanceFracs"] = {}
    for gen in GENERATORS:
        packed["refInstanceFracs"][gen] = [
            sum(fl) / len(fl) for fl in ref[gen]["follow_list"]
        ]

    # SR rubric data: flattened binary strings per judge|gen
    packed["srRubricFlat"] = {}
    for (judge, gen), inst_list in sr.items():
        flat = []
        for rubrics in inst_list:
            flat.extend("1" if v else "0" for v in rubrics)
        packed["srRubricFlat"][f"{judge}|{gen}"] = "".join(flat)

    # AR rubric data: same format
    packed["arRubricFlat"] = {}
    for (judge, gen), inst_list in ar.items():
        flat = []
        for rubrics in inst_list:
            flat.extend("1" if v else "0" for v in rubrics)
        packed["arRubricFlat"][f"{judge}|{gen}"] = "".join(flat)

    # DA: instance-level fraction scores (min(n_met, n_total) / n_total)
    packed["daScores"] = {}
    for (judge, gen), scores in da.items():
        packed["daScores"][f"{judge}|{gen}"] = [min(n, m) / m for n, m in scores]

    # PWC resolved: ternary strings (X/Y/T)
    packed["pwcOutcomes"] = {}
    for (judge, gx, gy), outcomes in pwc.items():
        key = f"{judge}|{gx}|{gy}"
        packed["pwcOutcomes"][key] = "".join(
            "X" if o == "X wins" else ("Y" if o == "Y wins" else "T")
            for o in outcomes
        )

    # ----- Interactive Analysis: additional data for filtering -----

    # Load raw IFEval data for instruction_id_list mapping
    raw_path = DATA_ROOT / "CloneSimpleEvals" / "Baseline.baseline" / "simple_evals_repo" / "ifeval" / "raw.jsonl"
    raw_iids = []
    with open(raw_path) as f:
        for line in f:
            row = json.loads(line)
            raw_iids.append(row["instruction_id_list"])
    assert len(raw_iids) == N_INSTANCES, f"raw.jsonl has {len(raw_iids)} entries, expected {N_INSTANCES}"

    # Load generation data for rubric criterion text (use first generator)
    ref_gen = GENERATORS[0]
    gen_path = DATA_ROOT / "GenerateIFEval" / f"Generator.{ref_gen}" / "generation.jsonl"
    gen_rubrics = []
    with open(gen_path) as f:
        for line in f:
            row = json.loads(line)
            gen_rubrics.append(row["rubrics"])
    assert len(gen_rubrics) == N_INSTANCES

    # IA1. rubricCounts: flat array (shared across generators)
    rubric_counts = [len(gen_rubrics[i]) for i in range(N_INSTANCES)]
    packed["rubricCounts"] = rubric_counts

    # IA2. rubricPointsFlat: all 1s
    total_rubrics = sum(rubric_counts)
    packed["rubricPointsFlat"] = [1] * total_rubrics

    # IA3. rubricLengthsFlat: character count of criterion text
    lengths_flat = []
    for i in range(N_INSTANCES):
        for r in gen_rubrics[i]:
            lengths_flat.append(len(r["criterion"]))
    packed["rubricLengthsFlat"] = lengths_flat

    # IA4. categoryNames + rubricCategoriesFlat (replaces axes)
    all_categories = set()
    for iid_list in raw_iids:
        for iid in iid_list:
            all_categories.add(iid.split(":")[0])
    category_names = sorted(all_categories)
    cat_to_idx = {c: idx for idx, c in enumerate(category_names)}
    packed["categoryNames"] = category_names

    categories_flat = []
    for i in range(N_INSTANCES):
        for iid in raw_iids[i]:
            cat = iid.split(":")[0]
            categories_flat.append(1 << cat_to_idx[cat])
    packed["rubricCategoriesFlat"] = categories_flat

    # IA5. instructionIdNames + rubricInstructionIdsFlat (replaces themes, but rubric-level)
    all_iids = set()
    for iid_list in raw_iids:
        for iid in iid_list:
            all_iids.add(iid)
    iid_names = sorted(all_iids)
    iid_to_idx = {iid: idx for idx, iid in enumerate(iid_names)}
    packed["instructionIdNames"] = iid_names

    iids_flat = []
    for i in range(N_INSTANCES):
        for iid in raw_iids[i]:
            iids_flat.append(1 << iid_to_idx[iid])
    packed["rubricInstructionIdsFlat"] = iids_flat

    # IA6. rubricAgreementFlat for SR and AR
    n_judges = len(ALL_JUDGES)
    total_pairs = comb(n_judges, 2)

    for method_key, method_data in [("rubricAgreementFlatSR", sr), ("rubricAgreementFlatAR", ar)]:
        agreement_flat = []
        for i in range(N_INSTANCES):
            n_rubrics = rubric_counts[i]
            for r_idx in range(n_rubrics):
                gen_agreements = []
                for gen in GENERATORS:
                    labels = []
                    for judge in ALL_JUDGES:
                        key = (judge, gen)
                        if key not in method_data:
                            continue
                        labels.append(method_data[key][i][r_idx])
                    if len(labels) == n_judges:
                        n_true = sum(labels)
                        n_false = n_judges - n_true
                        agr = (comb(n_true, 2) + comb(n_false, 2)) / total_pairs
                        gen_agreements.append(agr)
                if gen_agreements:
                    agreement_flat.append(round(sum(gen_agreements) / len(gen_agreements), 4))
                else:
                    agreement_flat.append(1.0)
        packed[method_key] = agreement_flat

    return packed


def generate_html(packed_data_json):
    """Generate the complete HTML dashboard."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IFEval Judge Analysis Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400;1,700&family=Fraunces:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
{CSS_BLOCK}
</head>
<body>

<header class="page-header">
  <h1>IFEval Judge Prompting Methods</h1>
  <p class="subtitle">Interactive analysis of accuracy and self-preference bias across SR, AR, DA, and PWC evaluation methods</p>
</header>

<div class="layout">
  <aside class="panel" id="panel">
    <div class="panel-inner">
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
      <div class="panel-section">
        <h3>Committee Builder</h3>
        <label class="cb-label"><input type="checkbox" id="committee-enable" onchange="onCommitteeChange()"><span>Add committee as judge</span></label>
        <div id="committee-builder" style="display:none;">
          <p class="note" style="margin-top:0.25rem;">Select members:</p>
          <div id="committee-member-checkboxes"></div>
          <div class="panel-actions">
            <button onclick="selectAllCommittee()" class="btn-sm">All</button>
            <button onclick="selectNoneCommittee()" class="btn-sm">None</button>
          </div>
          <label class="cb-label" style="margin-top:0.25rem;"><input type="checkbox" id="committee-members-sp" onchange="onCommitteeChange()"><span>Per-member SP</span></label>
          <p class="note" id="committee-status" style="margin-top:0.5rem;"></p>
        </div>
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
          <p class="card-desc">How well each method ranks generators (higher MPA = better; lower MRD = better)</p>
          <div id="tbl-sys-accuracy"></div>
        </div>
        <div class="card">
          <h3>Instance-Level Accuracy</h3>
          <p class="card-desc">Agreement with reference on individual instance-pair comparisons</p>
          <div id="tbl-inst-accuracy"></div>
        </div>
        <div class="card">
          <h3>Rubric-Level Accuracy</h3>
          <p class="card-desc">Per-rubric agreement with reference (SR and AR only)</p>
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
          <p class="card-desc">Self-preference at rubric level (SR and AR only)</p>
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
        <p class="card-desc">SR and AR only</p>
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
        <h3>Family Instance Self-Preference Bias (MISPB-F) by Judge</h3>
        <p class="card-desc">Same metric but for same-family generators (including self)</p>
        <div id="tbl-mispbf-judge"></div>
      </div>
      <div class="card">
        <h3>Family-Only Instance Self-Preference Bias (MISPB-FO) by Judge</h3>
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
        <p class="card-desc">Family excluding self: isolates sibling harmful preference</p>
        <div id="tbl-hsppfo-judge"></div>
      </div>
      <div class="card">
        <h3>Rubric Self-Preference Bias (MRSPB) by Judge</h3>
        <p class="card-desc">SR and AR only. Corrected rubric-level false positive rate.</p>
        <div id="tbl-mrspb-judge"></div>
      </div>
      <div class="card">
        <h3>Family MRSPB (MRSPB-F) by Judge</h3>
        <p class="card-desc">SR and AR only</p>
        <div id="tbl-mrspbf-judge"></div>
      </div>
      <div class="card">
        <h3>Family-Only MRSPB (MRSPB-FO) by Judge</h3>
        <p class="card-desc">SR and AR only. Family excluding self.</p>
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
        <p class="card-desc">Family excluding self, error denominator</p>
        <div id="tbl-mrspberrfo-judge"></div>
      </div>
    </div>

    <div class="tab-content hidden" id="tab-per-generator">
      <div class="card">
        <h3>Reference Scores</h3>
        <p class="card-desc">Ground-truth system-level score per generator</p>
        <div id="tbl-ref-scores"></div>
      </div>
      <div class="card">
        <h3>System Scores by Method</h3>
        <p class="card-desc">Mean judge-assigned score per generator, averaged across selected judges</p>
        <div id="tbl-gen-scores"></div>
      </div>
      <div class="card">
        <h3>Score Deltas (Judge - Reference)</h3>
        <p class="card-desc">Positive = overestimation by judges</p>
        <div id="tbl-gen-deltas"></div>
      </div>
      <div class="card">
        <h3>Normalized Score Deltas (Judge - Reference) / Reference</h3>
        <p class="card-desc">Score delta divided by reference score. Controls for baseline score magnitude.</p>
        <div id="tbl-gen-deltas-norm"></div>
      </div>
      <div class="card">
        <h3>Rank Deltas (Judge - Reference)</h3>
        <p class="card-desc">Negative = judges rank this generator better than reference. Averaged across selected judges.</p>
        <div id="tbl-gen-rank-deltas"></div>
      </div>
    </div>

    <div class="tab-content hidden" id="tab-charts">
      <div class="card">
        <h3>Method Accuracy Comparison</h3>
        <div id="chart-accuracy" class="chart-container"></div>
      </div>
      <div class="card">
        <h3>MISPB by Judge and Method</h3>
        <div id="chart-mispb" class="chart-container"></div>
      </div>
      <div class="card">
        <h3>Per-Judge MPA by Method</h3>
        <div id="chart-mpa-judge" class="chart-container"></div>
      </div>
      <div class="card">
        <h3>Accuracy vs. Bias</h3>
        <div id="chart-acc-bias" class="chart-container"></div>
      </div>
      <div class="card">
        <h3>System Scores: Judge vs. Reference</h3>
        <div id="chart-scores" class="chart-container"></div>
      </div>
      <div class="card">
        <h3>MRSPB by Judge (SR vs AR)</h3>
        <div id="chart-mrspb" class="chart-container"></div>
      </div>
    </div>

    <div class="tab-content hidden" id="tab-interactive">
      <div class="ia-layout">
        <div class="ia-filters">
          <h3>Filters</h3>

          <div class="filter-group">
            <label>Method</label>
            <label class="cb-label"><input type="radio" name="ia-method" value="sr" checked onchange="onIAFilterChange()"><span>SR (Single Rubric)</span></label>
            <label class="cb-label"><input type="radio" name="ia-method" value="ar" onchange="onIAFilterChange()"><span>AR (All Rubrics)</span></label>
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
            <label>Category Filter</label>
            <div class="panel-actions">
              <button onclick="iaCategoryAll()" class="btn-sm">All</button>
              <button onclick="iaCategoryNone()" class="btn-sm">None</button>
            </div>
            <div id="ia-category-checkboxes"></div>
          </div>

          <div class="filter-group">
            <label>Instruction ID Filter</label>
            <div class="panel-actions">
              <button onclick="iaInstrIdAll()" class="btn-sm">All</button>
              <button onclick="iaInstrIdNone()" class="btn-sm">None</button>
            </div>
            <div id="ia-instrid-checkboxes"></div>
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
            <p class="card-desc">Self-preference and accuracy metrics per judge. Uses known IFEval reference.</p>
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
function sn(model) {{ if (model === COMMITTEE_KEY) return 'Committee'; if (isCmemberKey(model)) return 'Cmt\u2192' + sn(cmemberModel(model)); return SN[model] || model; }}

function getFamily(model) {{ if (model === COMMITTEE_KEY || model === '_committee_') return '_committee_family_'; if (isCmemberKey(model)) return '_committee_family_'; return DATA.modelToFamily[model]; }}
function isSameFamily(a, b) {{ return getFamily(a) === getFamily(b); }}
function getOtherGens(judge, gens) {{
  const fam = getFamily(judge);
  return gens.filter(g => getFamily(g) !== fam);
}}
function getFamilyGens(judge, gens, includeSelf) {{
  const fam = getFamily(judge);
  return gens.filter(g => getFamily(g) === fam && (includeSelf || g !== judge));
}}

function getRubricArrays(flatData, gen, rubricCounts) {{
  // Decode flattened rubric string into array of arrays
  const counts = rubricCounts[gen];
  const result = [];
  let pos = 0;
  for (let i = 0; i < counts.length; i++) {{
    const n = counts[i];
    const arr = [];
    for (let j = 0; j < n; j++) {{
      arr.push(flatData[pos + j] === '1');
    }}
    result.push(arr);
    pos += n;
  }}
  return result;
}}

const _rubricCache = {{}};
function getJudgeRubrics(method, judge, gen) {{
  const cacheKey = method + '|' + judge + '|' + gen;
  if (_rubricCache[cacheKey]) return _rubricCache[cacheKey];
  const src = method === 'sr' ? DATA.srRubricFlat : DATA.arRubricFlat;
  const key = judge + '|' + gen;
  const flat = src[key];
  if (!flat) return null;
  const result = getRubricArrays(flat, gen, DATA.refRubricCounts);
  _rubricCache[cacheKey] = result;
  return result;
}}

const _refRubricCache = {{}};
function getRefRubrics(gen) {{
  if (_refRubricCache[gen]) return _refRubricCache[gen];
  const result = getRubricArrays(DATA.refRubricFlat[gen], gen, DATA.refRubricCounts);
  _refRubricCache[gen] = result;
  return result;
}}

function getDAScore(judge, gen, i) {{
  const key = judge + '|' + gen;
  const s = DATA.daScores[key];
  if (!s) return null;
  return s[i];  // Already a float (fraction score)
}}

function getRefScore(gen, i) {{
  return DATA.refInstanceFracs[gen][i];
}}

function getPWCOutcome(judge, gx, gy, i) {{
  const key = judge + '|' + gx + '|' + gy;
  const s = DATA.pwcOutcomes[key];
  if (!s) return null;
  return s[i]; // 'X', 'Y', or 'T'
}}

// Check if judge has data for a method
function judgeHasData(method, judge) {{
  if (isCmemberKey(judge)) return false;
  // For committee, check actual injected data presence
  if (judge === COMMITTEE_KEY || judge === '_committee_') {{
    const g0 = DATA.generators[0];
    if (method === 'sr') return !!DATA.srRubricFlat[judge + '|' + g0];
    if (method === 'ar') return !!DATA.arRubricFlat[judge + '|' + g0];
    if (method === 'da') return !!DATA.daScores[judge + '|' + g0];
    if (method === 'pwc') {{
      const g1 = DATA.generators[1];
      return !!DATA.pwcOutcomes[judge + '|' + g0 + '|' + g1];
    }}
    return false;
  }}
  if (method === 'sr') {{
    return !!DATA.srRubricFlat[judge + '|' + DATA.generators[0]];
  }}
  if (!DATA.defaultJudges.includes(judge)) return false;
  if (method === 'ar') return !!DATA.arRubricFlat[judge + '|' + DATA.generators[0]];
  if (method === 'da') return !!DATA.daScores[judge + '|' + DATA.generators[0]];
  if (method === 'pwc') return true;
  return false;
}}

function getActiveJudges(method, selectedJudges) {{
  return selectedJudges.filter(j => judgeHasData(method, j));
}}

// ============================================================
// RANKING HELPER (scipy.stats.rankdata equivalent)
// ============================================================
function rankdata(arr) {{
  // Average ranking: rank 1 = largest value
  const n = arr.length;
  const indexed = arr.map((v, i) => [v, i]);
  indexed.sort((a, b) => b[0] - a[0]); // descending
  const ranks = new Array(n);
  let i = 0;
  while (i < n) {{
    let j = i;
    while (j < n - 1 && indexed[j+1][0] === indexed[i][0]) j++;
    const avgRank = (i + j) / 2 + 1; // 1-based
    for (let k = i; k <= j; k++) ranks[indexed[k][1]] = avgRank;
    i = j + 1;
  }}
  return ranks;
}}

// ============================================================
// SYSTEM-LEVEL SCORES
// ============================================================

function computeRefSystemScores(gens) {{
  const scores = {{}};
  for (const gen of gens) {{
    let sum = 0;
    for (let i = 0; i < DATA.nInstances; i++) sum += getRefScore(gen, i);
    scores[gen] = sum / DATA.nInstances;
  }}
  return scores;
}}

function computeRefPWCSystemScores(gens) {{
  const scores = {{}};
  for (const gx of gens) {{
    const pts = [];
    for (const gy of gens) {{
      if (gx === gy) continue;
      for (let i = 0; i < DATA.nInstances; i++) {{
        const rx = getRefScore(gx, i), ry = getRefScore(gy, i);
        pts.push(rx > ry ? 1 : (rx < ry ? 0 : 0.5));
      }}
    }}
    scores[gx] = pts.length > 0 ? pts.reduce((a,b)=>a+b,0)/pts.length : 0.5;
  }}
  return scores;
}}

function computeSystemScoresRubric(method, judges, gens) {{
  const scores = {{}};
  const src = method === 'sr' ? DATA.srRubricFlat : DATA.arRubricFlat;
  for (const j of judges) {{
    for (const g of gens) {{
      const key = j + '|' + g;
      if (!src[key]) continue;
      const rubrics = getJudgeRubrics(method, j, g);
      let sum = 0;
      for (let i = 0; i < DATA.nInstances; i++) {{
        const r = rubrics[i];
        sum += r.filter(v => v).length / r.length;
      }}
      scores[key] = sum / DATA.nInstances;
    }}
  }}
  return scores;
}}

function computeSystemScoresDA(judges, gens) {{
  const scores = {{}};
  for (const j of judges) {{
    for (const g of gens) {{
      const key = j + '|' + g;
      if (!DATA.daScores[key]) continue;
      let sum = 0;
      for (let i = 0; i < DATA.nInstances; i++) sum += getDAScore(j, g, i);
      scores[key] = sum / DATA.nInstances;
    }}
  }}
  return scores;
}}

function computeSystemScoresPWC(judges, gens) {{
  const points = {{}};
  for (const j of judges) {{
    for (let gi = 0; gi < gens.length; gi++) {{
      for (let gj = gi+1; gj < gens.length; gj++) {{
        // Find canonical pair in DATA (original order from GENERATORS)
        const gx = gens[gi], gy = gens[gj];
        const origIdxX = DATA.generators.indexOf(gx);
        const origIdxY = DATA.generators.indexOf(gy);
        let cX, cY;
        if (origIdxX < origIdxY) {{ cX = gx; cY = gy; }}
        else {{ cX = gy; cY = gx; }}

        const pkey = j + '|' + cX + '|' + cY;
        const outcomes = DATA.pwcOutcomes[pkey];
        if (!outcomes) continue;

        const kx = j + '|' + gx, ky = j + '|' + gy;
        if (!points[kx]) points[kx] = [];
        if (!points[ky]) points[ky] = [];

        for (let i = 0; i < DATA.nInstances; i++) {{
          const o = outcomes[i];
          // cX is the canonical "X". Determine from gx/gy perspective.
          if (gx === cX) {{
            // gx = canonical X
            points[kx].push(o === 'X' ? 1 : (o === 'Y' ? 0 : 0.5));
            points[ky].push(o === 'X' ? 0 : (o === 'Y' ? 1 : 0.5));
          }} else {{
            // gx = canonical Y
            points[kx].push(o === 'Y' ? 1 : (o === 'X' ? 0 : 0.5));
            points[ky].push(o === 'X' ? 1 : (o === 'Y' ? 0 : 0.5));
          }}
        }}
      }}
    }}
  }}
  const scores = {{}};
  for (const [k, pts] of Object.entries(points)) {{
    scores[k] = pts.reduce((a,b)=>a+b,0) / pts.length;
  }}
  return scores;
}}

// ============================================================
// SYSTEM-LEVEL ACCURACY
// ============================================================

function computeMPA(sysScores, refScores, judges, gens) {{
  const perJudge = {{}}, perJudgeNConc = {{}}, perJudgeNTotal = {{}};
  for (const judge of judges) {{
    const jScores = {{}};
    for (const g of gens) {{
      const k = judge + '|' + g;
      if (sysScores[k] !== undefined) jScores[g] = sysScores[k];
    }}
    const available = gens.filter(g => jScores[g] !== undefined && refScores[g] !== undefined);
    if (available.length < 2) continue;
    let concordant = 0, total = 0;
    for (let i = 0; i < available.length; i++) {{
      for (let j = i+1; j < available.length; j++) {{
        const g1 = available[i], g2 = available[j];
        const jd = jScores[g1] - jScores[g2];
        const rd = refScores[g1] - refScores[g2];
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

function computeMRD(sysScores, refScores, judges, gens) {{
  const perJudge = {{}};
  for (const judge of judges) {{
    const available = gens.filter(g => sysScores[judge+'|'+g] !== undefined && refScores[g] !== undefined);
    if (available.length < 2) continue;
    const jArr = available.map(g => sysScores[judge+'|'+g]);
    const rArr = available.map(g => refScores[g]);
    const jRanks = rankdata(jArr);
    const rRanks = rankdata(rArr);
    let sum = 0;
    for (let i = 0; i < available.length; i++) sum += Math.abs(jRanks[i] - rRanks[i]);
    perJudge[judge] = sum / available.length;
  }}
  const vals = Object.values(perJudge);
  return {{ mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null, perJudge }};
}}

function computeMSD(sysScores, refScores, judges, gens) {{
  const perJudge = {{}};
  for (const judge of judges) {{
    const available = gens.filter(g => sysScores[judge+'|'+g] !== undefined && refScores[g] !== undefined);
    if (!available.length) continue;
    const deltas = available.map(g => sysScores[judge+'|'+g] - refScores[g]);
    perJudge[judge] = deltas.reduce((a,b)=>a+b,0) / deltas.length;
  }}
  const vals = Object.values(perJudge);
  return {{ mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null, perJudge }};
}}

function computeMSDnorm(sysScores, refScores, judges, gens) {{
  const perJudge = {{}};
  for (const judge of judges) {{
    const available = gens.filter(g => sysScores[judge+'|'+g] !== undefined && refScores[g] !== undefined && refScores[g] > 0);
    if (!available.length) continue;
    const deltas = available.map(g => (sysScores[judge+'|'+g] - refScores[g]) / refScores[g]);
    perJudge[judge] = deltas.reduce((a,b)=>a+b,0) / deltas.length;
  }}
  const vals = Object.values(perJudge);
  return {{ mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null, perJudge }};
}}

// ============================================================
// SYSTEM-LEVEL BIAS
// ============================================================

function computeMRDSP(sysScores, refScores, judges, gens) {{
  const perJudge = {{}}, perJudgeDSelf = {{}}, perJudgeDOther = {{}};
  for (const judge of judges) {{
    if (!gens.includes(judge)) continue;
    const available = gens.filter(g => sysScores[judge+'|'+g] !== undefined && refScores[g] !== undefined);
    if (available.length < 2 || !available.includes(judge)) continue;
    const jArr = available.map(g => sysScores[judge+'|'+g]);
    const rArr = available.map(g => refScores[g]);
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

function computeMRDFSP(sysScores, refScores, judges, gens, includeSelf) {{
  if (includeSelf === undefined) includeSelf = true;
  const perJudge = {{}}, perJudgeDFam = {{}}, perJudgeDOther = {{}};
  for (const judge of judges) {{
    const available = gens.filter(g => sysScores[judge+'|'+g] !== undefined && refScores[g] !== undefined);
    if (available.length < 2) continue;
    const jArr = available.map(g => sysScores[judge+'|'+g]);
    const rArr = available.map(g => refScores[g]);
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

function computeMSDSP(sysScores, refScores, judges, gens, normalize) {{
  const perJudge = {{}}, perJudgeDSelf = {{}}, perJudgeDOther = {{}};
  for (const judge of judges) {{
    if (!gens.includes(judge)) continue;
    const available = gens.filter(g => sysScores[judge+'|'+g] !== undefined && refScores[g] !== undefined);
    if (!available.includes(judge)) continue;
    const delta = (g) => {{
      let d = sysScores[judge+'|'+g] - refScores[g];
      if (normalize && refScores[g] > 0) d /= refScores[g];
      return d;
    }};
    const dSelf = delta(judge);
    const others = getOtherGens(judge, available);
    if (!others.length) continue;
    const dOther = others.map(g => delta(g)).reduce((a,b)=>a+b,0) / others.length;
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

function computeMSDFSP(sysScores, refScores, judges, gens, normalize, includeSelf) {{
  if (includeSelf === undefined) includeSelf = true;
  const perJudge = {{}}, perJudgeDFam = {{}}, perJudgeDOther = {{}};
  for (const judge of judges) {{
    const available = gens.filter(g => sysScores[judge+'|'+g] !== undefined && refScores[g] !== undefined);
    if (!available.length) continue;
    const delta = (g) => {{
      let d = sysScores[judge+'|'+g] - refScores[g];
      if (normalize && refScores[g] > 0) d /= refScores[g];
      return d;
    }};
    const famGens = getFamilyGens(judge, available, includeSelf);
    const otherGens = getOtherGens(judge, available);
    if (!famGens.length || !otherGens.length) continue;
    const dFam = famGens.map(g => delta(g)).reduce((a,b)=>a+b,0) / famGens.length;
    const dOther = otherGens.map(g => delta(g)).reduce((a,b)=>a+b,0) / otherGens.length;
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

function getInstanceScore(method, judge, gen, i) {{
  if (method === 'sr' || method === 'ar') {{
    const rubrics = getJudgeRubrics(method, judge, gen);
    if (!rubrics) return null;
    const r = rubrics[i];
    return r.filter(v => v).length / r.length;
  }} else if (method === 'da') {{
    return getDAScore(judge, gen, i);
  }}
  return null;
}}

function computeMIPA(method, judges, gens) {{
  if (method === 'pwc') return computeMIPAPWC(judges, gens);
  const perJudge = {{}}, perJudgeNAgree = {{}}, perJudgeNTotal = {{}};
  for (const judge of judges) {{
    const jGens = gens.filter(g => {{
      if (method === 'sr') return !!DATA.srRubricFlat[judge+'|'+g];
      if (method === 'ar') return !!DATA.arRubricFlat[judge+'|'+g];
      if (method === 'da') return !!DATA.daScores[judge+'|'+g];
      return false;
    }});
    if (jGens.length < 2) continue;
    // Pre-compute instance scores for all generators (fraction of rubrics met)
    const instScores = {{}};
    for (const g of jGens) {{
      instScores[g] = new Float64Array(DATA.nInstances);
      if (method === 'sr' || method === 'ar') {{
        const rubrics = getJudgeRubrics(method, judge, g);
        for (let i = 0; i < DATA.nInstances; i++) {{
          const r = rubrics[i];
          instScores[g][i] = r.filter(v => v).length / r.length;
        }}
      }} else {{
        for (let i = 0; i < DATA.nInstances; i++) instScores[g][i] = getDAScore(judge, g, i);
      }}
    }}
    const refScores = {{}};
    for (const g of jGens) {{
      refScores[g] = new Float64Array(DATA.nInstances);
      for (let i = 0; i < DATA.nInstances; i++) refScores[g][i] = getRefScore(g, i);
    }}
    let agree = 0, total = 0;
    for (let i = 0; i < DATA.nInstances; i++) {{
      for (let gi = 0; gi < jGens.length; gi++) {{
        for (let gj = gi+1; gj < jGens.length; gj++) {{
          const jd = instScores[jGens[gi]][i] - instScores[jGens[gj]][i];
          const rd = refScores[jGens[gi]][i] - refScores[jGens[gj]][i];
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

function computeMIPAPWC(judges, gens) {{
  const perJudge = {{}}, perJudgeNAgree = {{}}, perJudgeNTotal = {{}};
  for (const judge of judges) {{
    let agree = 0, total = 0;
    for (let gi = 0; gi < gens.length; gi++) {{
      for (let gj = gi+1; gj < gens.length; gj++) {{
        const gx = gens[gi], gy = gens[gj];
        const oIdxX = DATA.generators.indexOf(gx);
        const oIdxY = DATA.generators.indexOf(gy);
        let cX, cY;
        if (oIdxX < oIdxY) {{ cX = gx; cY = gy; }} else {{ cX = gy; cY = gx; }}
        const pkey = judge + '|' + cX + '|' + cY;
        const outcomes = DATA.pwcOutcomes[pkey];
        if (!outcomes) continue;
        for (let i = 0; i < DATA.nInstances; i++) {{
          const o = outcomes[i];
          // Determine judge sign from gx's perspective
          let jSign;
          if (gx === cX) jSign = o === 'X' ? 1 : (o === 'Y' ? -1 : 0);
          else jSign = o === 'Y' ? 1 : (o === 'X' ? -1 : 0);
          const rd = getRefScore(gx, i) - getRefScore(gy, i);
          const rSign = rd > 0 ? 1 : (rd < 0 ? -1 : 0);
          if (jSign === rSign) agree++;
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

// Pre-compute instance scores for a judge+method, cached for the computation batch
const _instScoreCache = {{}};
function _getInstScores(method, judge, gen) {{
  const ck = method + '|' + judge + '|' + gen;
  if (_instScoreCache[ck]) return _instScoreCache[ck];
  const src = method === 'sr' ? DATA.srRubricFlat : (method === 'ar' ? DATA.arRubricFlat : null);
  if (method === 'da') {{
    if (!DATA.daScores[judge+'|'+gen]) return null;
    const arr = new Float64Array(DATA.nInstances);
    for (let i = 0; i < DATA.nInstances; i++) arr[i] = getDAScore(judge, gen, i);
    _instScoreCache[ck] = arr;
    return arr;
  }}
  if (!src || !src[judge+'|'+gen]) return null;
  const rubrics = getJudgeRubrics(method, judge, gen);
  const arr = new Float64Array(DATA.nInstances);
  for (let i = 0; i < DATA.nInstances; i++) {{
    const r = rubrics[i];
    arr[i] = r.filter(v => v).length / r.length;
  }}
  _instScoreCache[ck] = arr;
  return arr;
}}

function _overestRateNonPWC(method, judge, targetGen, gens, errorDenom) {{
  const tgtScores = _getInstScores(method, judge, targetGen);
  if (!tgtScores) return null;
  let nOver = 0, nTotal = 0, nT2W = 0, nL2W = 0, nL2T = 0;
  for (const opp of gens) {{
    if (opp === targetGen) continue;
    const oppScores = _getInstScores(method, judge, opp);
    if (!oppScores) continue;
    for (let i = 0; i < DATA.nInstances; i++) {{
      const jSign = Math.sign(tgtScores[i] - oppScores[i]);
      const rSign = Math.sign(getRefScore(targetGen, i) - getRefScore(opp, i));
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

function _overestRatePWC(judge, targetGen, gens, errorDenom) {{
  let nOver = 0, nTotal = 0, nT2W = 0, nL2W = 0, nL2T = 0;
  for (const opp of gens) {{
    if (opp === targetGen) continue;
    const oIdxT = DATA.generators.indexOf(targetGen);
    const oIdxO = DATA.generators.indexOf(opp);
    let cX, cY;
    if (oIdxT < oIdxO) {{ cX = targetGen; cY = opp; }} else {{ cX = opp; cY = targetGen; }}
    const pkey = judge + '|' + cX + '|' + cY;
    const outcomes = DATA.pwcOutcomes[pkey];
    if (!outcomes) continue;
    for (let i = 0; i < DATA.nInstances; i++) {{
      const o = outcomes[i];
      let jSign;
      if (targetGen === cX) jSign = o === 'X' ? 1 : (o === 'Y' ? -1 : 0);
      else jSign = o === 'Y' ? 1 : (o === 'X' ? -1 : 0);
      const rt = getRefScore(targetGen, i);
      const ro = getRefScore(opp, i);
      const rSign = Math.sign(rt - ro);
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

function computeMISPB(method, judges, gens, errorDenom, familyMode, includeSelfInFamily) {{
  if (includeSelfInFamily === undefined) includeSelfInFamily = true;
  const perJudge = {{}}, perJudgeRaw = {{}}, perJudgeOther = {{}}, perJudgeRatio = {{}};
  const perJudgeNOverSelf = {{}}, perJudgeNTotalSelf = {{}}, perJudgeNOverOther = {{}}, perJudgeNTotalOther = {{}};
  const perJudgeNT2WSelf = {{}}, perJudgeNL2WSelf = {{}}, perJudgeNL2TSelf = {{}};
  const perJudgeNT2WOther = {{}}, perJudgeNL2WOther = {{}}, perJudgeNL2TOther = {{}};
  const overestFn = method === 'pwc'
    ? (j, tg) => _overestRatePWC(j, tg, gens, errorDenom)
    : (j, tg) => _overestRateNonPWC(method, j, tg, gens, errorDenom);

  for (const judge of judges) {{
    const targets = familyMode ? getFamilyGens(judge, gens, includeSelfInFamily) : (gens.includes(judge) ? [judge] : []);
    if (!targets.length) continue;
    const rawResults = targets.map(t => overestFn(judge, t)).filter(r => r !== null);
    if (!rawResults.length) continue;
    const raw = rawResults.reduce((a,b)=>a+b.rate,0) / rawResults.length;
    let totalNOverSelf = 0, totalNTotalSelf = 0;
    let totalNT2WSelf = 0, totalNL2WSelf = 0, totalNL2TSelf = 0;
    for (const res of rawResults) {{
      totalNOverSelf += res.nOver; totalNTotalSelf += res.nTotal;
      totalNT2WSelf += res.nT2W; totalNL2WSelf += res.nL2W; totalNL2TSelf += res.nL2T;
    }}
    const others = getOtherGens(judge, gens);
    const otherResults = others.map(g => overestFn(judge, g)).filter(r => r !== null);
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
    perJudgeNOverSelf[judge] = totalNOverSelf;
    perJudgeNTotalSelf[judge] = totalNTotalSelf;
    perJudgeNOverOther[judge] = totalNOverOther;
    perJudgeNTotalOther[judge] = totalNTotalOther;
    perJudgeNT2WSelf[judge] = totalNT2WSelf;
    perJudgeNL2WSelf[judge] = totalNL2WSelf;
    perJudgeNL2TSelf[judge] = totalNL2TSelf;
    perJudgeNT2WOther[judge] = totalNT2WOther;
    perJudgeNL2WOther[judge] = totalNL2WOther;
    perJudgeNL2TOther[judge] = totalNL2TOther;
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
// RUBRIC-LEVEL METRICS (SR and AR only)
// ============================================================

function computeMRA(method, judges, gens) {{
  const perJudge = {{}}, perJudgeNCorrect = {{}}, perJudgeNTotal = {{}};
  for (const judge of judges) {{
    let correct = 0, total = 0;
    for (const gen of gens) {{
      const jRubrics = getJudgeRubrics(method, judge, gen);
      if (!jRubrics) continue;
      const rRubrics = getRefRubrics(gen);
      for (let i = 0; i < DATA.nInstances; i++) {{
        for (let r = 0; r < jRubrics[i].length; r++) {{
          if (jRubrics[i][r] === rRubrics[i][r]) correct++;
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

function _rubricOverestRate(method, judge, targetGen, errorDenom) {{
  const jRubrics = getJudgeRubrics(method, judge, targetGen);
  if (!jRubrics) return null;
  const rRubrics = getRefRubrics(targetGen);
  let nOver = 0, nTotal = 0;
  for (let i = 0; i < DATA.nInstances; i++) {{
    for (let r = 0; r < jRubrics[i].length; r++) {{
      if (errorDenom && rRubrics[i][r]) continue; // skip ref=met
      nTotal++;
      if (jRubrics[i][r] && !rRubrics[i][r]) nOver++;
    }}
  }}
  return nTotal > 0 ? {{rate: nOver / nTotal, nOver, nTotal}} : null;
}}

function computeMRSPB(method, judges, gens, errorDenom, familyMode, includeSelfInFamily) {{
  if (includeSelfInFamily === undefined) includeSelfInFamily = true;
  const perJudge = {{}}, perJudgeRaw = {{}}, perJudgeOther = {{}}, perJudgeRatio = {{}};
  const perJudgeNOverSelf = {{}}, perJudgeNTotalSelf = {{}}, perJudgeNOverOther = {{}}, perJudgeNTotalOther = {{}};
  for (const judge of judges) {{
    const targets = familyMode ? getFamilyGens(judge, gens, includeSelfInFamily) : (gens.includes(judge) ? [judge] : []);
    if (!targets.length) continue;
    const rawResults = targets.map(t => _rubricOverestRate(method, judge, t, errorDenom)).filter(r => r !== null);
    if (!rawResults.length) continue;
    const raw = rawResults.reduce((a,b)=>a+b.rate,0) / rawResults.length;
    let totalNOverSelf = 0, totalNTotalSelf = 0;
    for (const res of rawResults) {{ totalNOverSelf += res.nOver; totalNTotalSelf += res.nTotal; }}
    const others = getOtherGens(judge, gens);
    const otherResults = others.map(g => _rubricOverestRate(method, judge, g, errorDenom)).filter(r => r !== null);
    if (!otherResults.length) continue;
    const other = otherResults.reduce((a,b)=>a+b.rate,0) / otherResults.length;
    let totalNOverOther = 0, totalNTotalOther = 0;
    for (const res of otherResults) {{ totalNOverOther += res.nOver; totalNTotalOther += res.nTotal; }}
    perJudge[judge] = raw - other;
    perJudgeRaw[judge] = raw;
    perJudgeOther[judge] = other;
    perJudgeRatio[judge] = other > 0 ? raw / other : Infinity;
    perJudgeNOverSelf[judge] = totalNOverSelf;
    perJudgeNTotalSelf[judge] = totalNTotalSelf;
    perJudgeNOverOther[judge] = totalNOverOther;
    perJudgeNTotalOther[judge] = totalNTotalOther;
  }}
  const vals = Object.values(perJudge);
  return {{
    mean: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null,
    perJudge, perJudgeRaw, perJudgeOther, perJudgeRatio,
    perJudgeNOverSelf, perJudgeNTotalSelf, perJudgeNOverOther, perJudgeNTotalOther
  }};
}}


// ============================================================
// COMMITTEE INJECTION
// ============================================================

const COMMITTEE_KEY = '_committee_';
const CMEMBER_PREFIX = '_cmember_';
function cmemberKey(model) {{ return CMEMBER_PREFIX + model; }}
function isCmemberKey(key) {{ return typeof key === 'string' && key.startsWith(CMEMBER_PREFIX); }}
function cmemberModel(key) {{ return key.slice(CMEMBER_PREFIX.length); }}

function getCommitteeMembers() {{
  return [...document.querySelectorAll('#committee-member-checkboxes input:checked')].map(cb => cb.value);
}}

function isCommitteeEnabled() {{
  return document.getElementById('committee-enable')?.checked && getCommitteeMembers().length >= 2;
}}

function isShowCmemberSP() {{
  return isCommitteeEnabled() && (document.getElementById('committee-members-sp')?.checked || false);
}}

function clearCommitteeData() {{
  for (const store of [DATA.srRubricFlat, DATA.arRubricFlat, DATA.daScores, DATA.pwcOutcomes]) {{
    for (const k of Object.keys(store)) {{
      if (k.startsWith(COMMITTEE_KEY)) delete store[k];
    }}
  }}
  // Clear rubric cache entries for committee
  for (const k of Object.keys(_rubricCache)) {{
    if (k.startsWith('sr|' + COMMITTEE_KEY) || k.startsWith('ar|' + COMMITTEE_KEY)) delete _rubricCache[k];
  }}
}}

function injectCommitteeData(members, gens) {{
  // SR/AR: majority-vote rubrics, then pack back to flat binary string
  for (const mk of ['sr', 'ar']) {{
    const src = mk === 'sr' ? DATA.srRubricFlat : DATA.arRubricFlat;
    for (const g of gens) {{
      // Collect rubric arrays from members who have data
      const memberRubrics = [];
      for (const m of members) {{
        const r = getJudgeRubrics(mk, m, g);
        if (r) memberRubrics.push(r);
      }}
      if (memberRubrics.length < 2) continue;
      const nm = memberRubrics.length;
      // Majority vote per rubric, pack to binary string
      let flat = '';
      for (let i = 0; i < DATA.nInstances; i++) {{
        const nRub = memberRubrics[0][i].length;
        for (let ri = 0; ri < nRub; ri++) {{
          let votes = 0;
          for (let mi = 0; mi < nm; mi++) votes += memberRubrics[mi][i][ri] ? 1 : 0;
          flat += (votes > nm / 2) ? '1' : '0';
        }}
      }}
      src[COMMITTEE_KEY + '|' + g] = flat;
    }}
  }}

  // DA: average scores across members
  for (const g of gens) {{
    const arrays = [];
    for (const m of members) {{
      const key = m + '|' + g;
      if (DATA.daScores[key]) arrays.push(DATA.daScores[key]);
    }}
    if (arrays.length < 2) continue;
    const avg = new Array(DATA.nInstances);
    for (let i = 0; i < DATA.nInstances; i++) {{
      let s = 0;
      for (const a of arrays) s += a[i];
      avg[i] = s / arrays.length;
    }}
    DATA.daScores[COMMITTEE_KEY + '|' + g] = avg;
  }}

  // PWC: majority vote on resolved outcomes
  for (let gi = 0; gi < DATA.generators.length; gi++) {{
    for (let gj = gi + 1; gj < DATA.generators.length; gj++) {{
      const gx = DATA.generators[gi], gy = DATA.generators[gj];
      const pkey = gx + '|' + gy;
      const allOutcomes = [];
      for (const m of members) {{
        const o = DATA.pwcOutcomes[m + '|' + pkey];
        if (o) allOutcomes.push(o);
      }}
      if (allOutcomes.length < 2) continue;
      const nm = allOutcomes.length;
      let result = '';
      for (let i = 0; i < DATA.nInstances; i++) {{
        let sum = 0;
        for (const o of allOutcomes) {{
          const c = o[i];
          sum += c === 'X' ? 1 : (c === 'Y' ? -1 : 0);
        }}
        result += sum > 0 ? 'X' : (sum < 0 ? 'Y' : 'T');
      }}
      DATA.pwcOutcomes[COMMITTEE_KEY + '|' + pkey] = result;
    }}
  }}
}}

// Committee self-preference: compute per-member SP using committee's aggregated data
function computeCommitteeMISPB(method, gens, errorDenom, familyMode, includeSelfInFamily) {{
  if (includeSelfInFamily === undefined) includeSelfInFamily = true;
  const members = getCommitteeMembers();
  const committeeFamilies = new Set(members.map(m => getFamily(m)));
  const nonCommitteeGens = gens.filter(g => !members.includes(g));
  const otherTargets = gens.filter(g => !committeeFamilies.has(getFamily(g)));
  const perJudge = {{}}, perJudgeRaw = {{}}, perJudgeOther = {{}}, perJudgeRatio = {{}};
  const perJudgeNOverSelf = {{}}, perJudgeNTotalSelf = {{}}, perJudgeNOverOther = {{}}, perJudgeNTotalOther = {{}};
  const perJudgeNT2WSelf = {{}}, perJudgeNL2WSelf = {{}}, perJudgeNL2TSelf = {{}};
  const perJudgeNT2WOther = {{}}, perJudgeNL2WOther = {{}}, perJudgeNL2TOther = {{}};

  // Use non-committee generators as opponents (no head-to-head)
  const overestFn = method === 'pwc'
    ? (tg) => _overestRatePWC(COMMITTEE_KEY, tg, nonCommitteeGens, errorDenom)
    : (tg) => _overestRateNonPWC(method, COMMITTEE_KEY, tg, nonCommitteeGens, errorDenom);

  for (const member of members) {{
    // Determine targets
    const targets = familyMode
      ? getFamilyGens(member, gens, includeSelfInFamily)
      : (gens.includes(member) ? [member] : []);
    if (!targets.length) continue;

    const rawResults = targets.map(t => overestFn(t)).filter(r => r !== null);
    if (!rawResults.length) continue;
    const raw = rawResults.reduce((a, b) => a + b.rate, 0) / rawResults.length;
    let totalNOverSelf = 0, totalNTotalSelf = 0;
    let totalNT2WSelf = 0, totalNL2WSelf = 0, totalNL2TSelf = 0;
    for (const res of rawResults) {{ totalNOverSelf += res.nOver; totalNTotalSelf += res.nTotal; totalNT2WSelf += res.nT2W; totalNL2WSelf += res.nL2W; totalNL2TSelf += res.nL2T; }}

    const otherResults = otherTargets.map(g => overestFn(g)).filter(r => r !== null);
    if (!otherResults.length) continue;
    const other = otherResults.reduce((a, b) => a + b.rate, 0) / otherResults.length;
    let totalNOverOther = 0, totalNTotalOther = 0;
    let totalNT2WOther = 0, totalNL2WOther = 0, totalNL2TOther = 0;
    for (const res of otherResults) {{ totalNOverOther += res.nOver; totalNTotalOther += res.nTotal; totalNT2WOther += res.nT2W; totalNL2WOther += res.nL2W; totalNL2TOther += res.nL2T; }}

    perJudge[member] = raw - other;
    perJudgeRaw[member] = raw;
    perJudgeOther[member] = other;
    perJudgeRatio[member] = other > 0 ? raw / other : Infinity;
    perJudgeNOverSelf[member] = totalNOverSelf;
    perJudgeNTotalSelf[member] = totalNTotalSelf;
    perJudgeNOverOther[member] = totalNOverOther;
    perJudgeNTotalOther[member] = totalNTotalOther;
    perJudgeNT2WSelf[member] = totalNT2WSelf; perJudgeNL2WSelf[member] = totalNL2WSelf; perJudgeNL2TSelf[member] = totalNL2TSelf;
    perJudgeNT2WOther[member] = totalNT2WOther; perJudgeNL2WOther[member] = totalNL2WOther; perJudgeNL2TOther[member] = totalNL2TOther;
  }}
  const vals = Object.values(perJudge);
  const median = vals.length ? vals.slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null;
  const sumV = obj => Object.values(obj).reduce((a,b) => a+b, 0);
  // Add committee-level totals (sum across members)
  perJudgeNOverSelf[COMMITTEE_KEY] = sumV(perJudgeNOverSelf);
  perJudgeNTotalSelf[COMMITTEE_KEY] = sumV(perJudgeNTotalSelf);
  perJudgeNOverOther[COMMITTEE_KEY] = sumV(perJudgeNOverOther);
  perJudgeNTotalOther[COMMITTEE_KEY] = sumV(perJudgeNTotalOther);
  perJudgeNT2WSelf[COMMITTEE_KEY] = sumV(perJudgeNT2WSelf);
  perJudgeNL2WSelf[COMMITTEE_KEY] = sumV(perJudgeNL2WSelf);
  perJudgeNL2TSelf[COMMITTEE_KEY] = sumV(perJudgeNL2TSelf);
  perJudgeNT2WOther[COMMITTEE_KEY] = sumV(perJudgeNT2WOther);
  perJudgeNL2WOther[COMMITTEE_KEY] = sumV(perJudgeNL2WOther);
  perJudgeNL2TOther[COMMITTEE_KEY] = sumV(perJudgeNL2TOther);
  // Build result maps with committee median + per-member cmember entries
  const rPerJudge = {{ [COMMITTEE_KEY]: median }};
  const rPerJudgeRaw = {{ [COMMITTEE_KEY]: vals.length ? Object.values(perJudgeRaw).slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null }};
  const rPerJudgeOther = {{ [COMMITTEE_KEY]: vals.length ? Object.values(perJudgeOther).slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null }};
  const rPerJudgeRatio = {{ [COMMITTEE_KEY]: vals.length ? Object.values(perJudgeRatio).slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null }};
  for (const member of members) {{
    if (perJudge[member] === undefined) continue;
    const ck = cmemberKey(member);
    rPerJudge[ck] = perJudge[member];
    rPerJudgeRaw[ck] = perJudgeRaw[member];
    rPerJudgeOther[ck] = perJudgeOther[member];
    rPerJudgeRatio[ck] = perJudgeRatio[member];
    perJudgeNOverSelf[ck] = perJudgeNOverSelf[member] || 0;
    perJudgeNTotalSelf[ck] = perJudgeNTotalSelf[member] || 0;
    perJudgeNOverOther[ck] = perJudgeNOverOther[member] || 0;
    perJudgeNTotalOther[ck] = perJudgeNTotalOther[member] || 0;
    perJudgeNT2WSelf[ck] = perJudgeNT2WSelf[member] || 0;
    perJudgeNL2WSelf[ck] = perJudgeNL2WSelf[member] || 0;
    perJudgeNL2TSelf[ck] = perJudgeNL2TSelf[member] || 0;
    perJudgeNT2WOther[ck] = perJudgeNT2WOther[member] || 0;
    perJudgeNL2WOther[ck] = perJudgeNL2WOther[member] || 0;
    perJudgeNL2TOther[ck] = perJudgeNL2TOther[member] || 0;
  }}
  return {{
    mean: median,
    perJudge: rPerJudge,
    perJudgeRaw: rPerJudgeRaw,
    perJudgeOther: rPerJudgeOther,
    perJudgeRatio: rPerJudgeRatio,
    perJudgeNOverSelf, perJudgeNTotalSelf, perJudgeNOverOther, perJudgeNTotalOther,
    perJudgeNT2WSelf, perJudgeNL2WSelf, perJudgeNL2TSelf,
    perJudgeNT2WOther, perJudgeNL2WOther, perJudgeNL2TOther,
    _perMemberDetail: perJudge,
  }};
}}

function computeCommitteeMSDSP(sysScores, refScores, gens, normalize) {{
  const members = getCommitteeMembers();
  const committeeFamilies = new Set(members.map(m => getFamily(m)));
  const perMember = {{}};
  const perMemberDSelf = {{}};
  const perMemberDOther = {{}};
  for (const member of members) {{
    if (!gens.includes(member)) continue;
    const available = gens.filter(g => sysScores[COMMITTEE_KEY + '|' + g] !== undefined && refScores[g] !== undefined);
    if (!available.includes(member)) continue;
    const delta = (g) => {{
      let d = sysScores[COMMITTEE_KEY + '|' + g] - refScores[g];
      if (normalize && refScores[g] > 0) d /= refScores[g];
      return d;
    }};
    const dSelf = delta(member);
    const others = available.filter(g => !committeeFamilies.has(getFamily(g)));
    if (!others.length) continue;
    const dOther = others.map(g => delta(g)).reduce((a,b)=>a+b,0) / others.length;
    perMember[member] = dSelf - dOther;
    perMemberDSelf[member] = dSelf;
    perMemberDOther[member] = dOther;
  }}
  const vals = Object.values(perMember);
  const median = vals.length ? vals.slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null;
  const rPJ = {{ [COMMITTEE_KEY]: median }};
  const rPJDS = {{ [COMMITTEE_KEY]: vals.length ? Object.values(perMemberDSelf).slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null }};
  const rPJDO = {{ [COMMITTEE_KEY]: vals.length ? Object.values(perMemberDOther).slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null }};
  for (const member of members) {{
    if (perMember[member] === undefined) continue;
    const ck = cmemberKey(member);
    rPJ[ck] = perMember[member];
    rPJDS[ck] = perMemberDSelf[member];
    rPJDO[ck] = perMemberDOther[member];
  }}
  return {{
    mean: median,
    meanDSelf: vals.length ? Object.values(perMemberDSelf).slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null,
    meanDOther: vals.length ? Object.values(perMemberDOther).slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null,
    perJudge: rPJ,
    perJudgeDSelf: rPJDS,
    perJudgeDOther: rPJDO,
  }};
}}

function computeCommitteeMRDSP(sysScores, refScores, gens) {{
  const members = getCommitteeMembers();
  const committeeFamilies = new Set(members.map(m => getFamily(m)));
  const perMember = {{}};
  const perMemberDSelf = {{}};
  const perMemberDOther = {{}};
  for (const member of members) {{
    if (!gens.includes(member)) continue;
    const available = gens.filter(g => sysScores[COMMITTEE_KEY + '|' + g] !== undefined && refScores[g] !== undefined);
    if (available.length < 2 || !available.includes(member)) continue;
    const jArr = available.map(g => sysScores[COMMITTEE_KEY + '|' + g]);
    const rArr = available.map(g => refScores[g]);
    const jRanks = rankdata(jArr);
    const rRanks = rankdata(rArr);
    const signedDiffs = {{}};
    for (let i = 0; i < available.length; i++) signedDiffs[available[i]] = jRanks[i] - rRanks[i];
    const dSelf = signedDiffs[member];
    const others = available.filter(g => !committeeFamilies.has(getFamily(g)));
    if (!others.length) continue;
    const dOther = others.map(g => signedDiffs[g]).reduce((a,b)=>a+b,0) / others.length;
    perMember[member] = dSelf - dOther;
    perMemberDSelf[member] = dSelf;
    perMemberDOther[member] = dOther;
  }}
  const vals = Object.values(perMember);
  const median = vals.length ? vals.slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null;
  const rPJ = {{ [COMMITTEE_KEY]: median }};
  const rPJDS = {{ [COMMITTEE_KEY]: vals.length ? Object.values(perMemberDSelf).slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null }};
  const rPJDO = {{ [COMMITTEE_KEY]: vals.length ? Object.values(perMemberDOther).slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null }};
  for (const member of members) {{
    if (perMember[member] === undefined) continue;
    const ck = cmemberKey(member);
    rPJ[ck] = perMember[member];
    rPJDS[ck] = perMemberDSelf[member];
    rPJDO[ck] = perMemberDOther[member];
  }}
  return {{
    mean: median,
    meanDSelf: vals.length ? Object.values(perMemberDSelf).slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null,
    meanDOther: vals.length ? Object.values(perMemberDOther).slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null,
    perJudge: rPJ,
    perJudgeDSelf: rPJDS,
    perJudgeDOther: rPJDO,
  }};
}}

function computeCommitteeMRSPB(method, gens, errorDenom, familyMode, includeSelfInFamily) {{
  if (includeSelfInFamily === undefined) includeSelfInFamily = true;
  const members = getCommitteeMembers();
  const committeeFamilies = new Set(members.map(m => getFamily(m)));
  const otherTargets = gens.filter(g => !committeeFamilies.has(getFamily(g)));
  const perMember = {{}}, perMemberRaw = {{}}, perMemberOther = {{}}, perMemberRatio = {{}};
  const pmNOverSelf = {{}}, pmNTotalSelf = {{}}, pmNOverOther = {{}}, pmNTotalOther = {{}};
  let totNOverSelf = 0, totNTotalSelf = 0, totNOverOther = 0, totNTotalOther = 0;

  for (const member of members) {{
    const targets = familyMode
      ? getFamilyGens(member, gens, includeSelfInFamily)
      : (gens.includes(member) ? [member] : []);
    if (!targets.length) continue;

    function rubricOverestRate(targetGen) {{
      const jRubrics = getJudgeRubrics(method, COMMITTEE_KEY, targetGen);
      if (!jRubrics) return null;
      const rRubrics = getRefRubrics(targetGen);
      let nOver = 0, nTotal = 0;
      for (let i = 0; i < DATA.nInstances; i++) {{
        for (let ri = 0; ri < jRubrics[i].length; ri++) {{
          if (errorDenom && rRubrics[i][ri]) continue;
          nTotal++;
          if (jRubrics[i][ri] && !rRubrics[i][ri]) nOver++;
        }}
      }}
      return nTotal > 0 ? {{ rate: nOver / nTotal, nOver, nTotal }} : null;
    }}

    const rawResults = targets.map(t => rubricOverestRate(t)).filter(r => r !== null);
    if (!rawResults.length) continue;
    const raw = rawResults.reduce((a, b) => a + b.rate, 0) / rawResults.length;
    let mNOS = 0, mNTS = 0;
    for (const res of rawResults) {{ totNOverSelf += res.nOver; totNTotalSelf += res.nTotal; mNOS += res.nOver; mNTS += res.nTotal; }}

    const otherResults = otherTargets.map(g => rubricOverestRate(g)).filter(r => r !== null);
    if (!otherResults.length) continue;
    const other = otherResults.reduce((a, b) => a + b.rate, 0) / otherResults.length;
    let mNOO = 0, mNTO = 0;
    for (const res of otherResults) {{ totNOverOther += res.nOver; totNTotalOther += res.nTotal; mNOO += res.nOver; mNTO += res.nTotal; }}
    perMember[member] = raw - other;
    perMemberRaw[member] = raw;
    perMemberOther[member] = other;
    perMemberRatio[member] = other > 0 ? raw / other : Infinity;
    pmNOverSelf[member] = mNOS; pmNTotalSelf[member] = mNTS;
    pmNOverOther[member] = mNOO; pmNTotalOther[member] = mNTO;
  }}

  const vals = Object.values(perMember);
  const median = vals.length ? vals.slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null;
  const rPJ = {{ [COMMITTEE_KEY]: median }};
  const rPJRaw = {{ [COMMITTEE_KEY]: vals.length ? Object.values(perMemberRaw).slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null }};
  const rPJOther = {{ [COMMITTEE_KEY]: vals.length ? Object.values(perMemberOther).slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null }};
  const rPJRatio = {{ [COMMITTEE_KEY]: vals.length ? Object.values(perMemberRatio).slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)] : null }};
  const rNOS = {{ [COMMITTEE_KEY]: totNOverSelf }};
  const rNTS = {{ [COMMITTEE_KEY]: totNTotalSelf }};
  const rNOO = {{ [COMMITTEE_KEY]: totNOverOther }};
  const rNTO = {{ [COMMITTEE_KEY]: totNTotalOther }};
  for (const member of members) {{
    if (perMember[member] === undefined) continue;
    const ck = cmemberKey(member);
    rPJ[ck] = perMember[member];
    rPJRaw[ck] = perMemberRaw[member];
    rPJOther[ck] = perMemberOther[member];
    rPJRatio[ck] = perMemberRatio[member];
    rNOS[ck] = pmNOverSelf[member] || 0;
    rNTS[ck] = pmNTotalSelf[member] || 0;
    rNOO[ck] = pmNOverOther[member] || 0;
    rNTO[ck] = pmNTotalOther[member] || 0;
  }}
  return {{
    mean: median,
    perJudge: rPJ,
    perJudgeRaw: rPJRaw,
    perJudgeOther: rPJOther,
    perJudgeRatio: rPJRatio,
    perJudgeNOverSelf: rNOS,
    perJudgeNTotalSelf: rNTS,
    perJudgeNOverOther: rNOO,
    perJudgeNTotalOther: rNTO,
  }};
}}

// ============================================================
// MAIN COMPUTATION
// ============================================================

let cachedResults = null;
let effectiveJudgesList = [];

function computeAll(judges, gens) {{
  if (judges.length === 0 || gens.length < 2) {{
    return null;
  }}
  // Clear per-computation caches
  for (const k in _instScoreCache) delete _instScoreCache[k];

  const refScores = computeRefSystemScores(gens);
  const refPWCScores = computeRefPWCSystemScores(gens);

  const methods = ['sr', 'ar', 'da', 'pwc'];
  const sysScores = {{}};
  sysScores.sr = computeSystemScoresRubric('sr', judges, gens);
  sysScores.ar = computeSystemScoresRubric('ar', judges, gens);
  sysScores.da = computeSystemScoresDA(judges, gens);
  sysScores.pwc = computeSystemScoresPWC(judges, gens);

  const r = {{ refScores, refPWCScores, sysScores }};
  const hasCommittee = judges.includes(COMMITTEE_KEY);

  // System-level accuracy
  r.mpa = {{}}; r.mrd = {{}}; r.msd = {{}}; r.msdNorm = {{}};
  for (const m of methods) {{
    const ref = m === 'pwc' ? refPWCScores : refScores;
    const aj = getActiveJudges(m, judges);
    r.mpa[m] = computeMPA(sysScores[m], ref, aj, gens);
    r.mrd[m] = computeMRD(sysScores[m], ref, aj, gens);
    r.msd[m] = computeMSD(sysScores[m], ref, aj, gens);
    r.msdNorm[m] = computeMSDnorm(sysScores[m], ref, aj, gens);
  }}

  // System-level bias
  r.mrdSP = {{}}; r.mrdFSP = {{}}; r.mrdFOSP = {{}};
  r.msdSP = {{}}; r.msdSPnorm = {{}}; r.msdFSP = {{}}; r.msdFSPnorm = {{}};
  r.msdFOSP = {{}}; r.msdFOSPnorm = {{}};
  for (const m of methods) {{
    const ref = m === 'pwc' ? refPWCScores : refScores;
    const aj = getActiveJudges(m, judges);
    const realJ = aj.filter(j => j !== COMMITTEE_KEY);
    r.mrdSP[m] = realJ.length ? computeMRDSP(sysScores[m], ref, realJ, gens) : {{ mean: null, perJudge: {{}}, perJudgeDSelf: {{}}, perJudgeDOther: {{}} }};
    r.mrdFSP[m] = realJ.length ? computeMRDFSP(sysScores[m], ref, realJ, gens, true) : {{ mean: null, perJudge: {{}}, perJudgeDFam: {{}}, perJudgeDOther: {{}} }};
    r.mrdFOSP[m] = realJ.length ? computeMRDFSP(sysScores[m], ref, realJ, gens, false) : {{ mean: null, perJudge: {{}}, perJudgeDFam: {{}}, perJudgeDOther: {{}} }};
    r.msdSP[m] = realJ.length ? computeMSDSP(sysScores[m], ref, realJ, gens, false) : {{ mean: null, perJudge: {{}}, perJudgeDSelf: {{}}, perJudgeDOther: {{}} }};
    r.msdSPnorm[m] = realJ.length ? computeMSDSP(sysScores[m], ref, realJ, gens, true) : {{ mean: null, perJudge: {{}}, perJudgeDSelf: {{}}, perJudgeDOther: {{}} }};
    r.msdFSP[m] = realJ.length ? computeMSDFSP(sysScores[m], ref, realJ, gens, false, true) : {{ mean: null, perJudge: {{}}, perJudgeDFam: {{}}, perJudgeDOther: {{}} }};
    r.msdFSPnorm[m] = realJ.length ? computeMSDFSP(sysScores[m], ref, realJ, gens, true, true) : {{ mean: null, perJudge: {{}}, perJudgeDFam: {{}}, perJudgeDOther: {{}} }};
    r.msdFOSP[m] = realJ.length ? computeMSDFSP(sysScores[m], ref, realJ, gens, false, false) : {{ mean: null, perJudge: {{}}, perJudgeDFam: {{}}, perJudgeDOther: {{}} }};
    r.msdFOSPnorm[m] = realJ.length ? computeMSDFSP(sysScores[m], ref, realJ, gens, true, false) : {{ mean: null, perJudge: {{}}, perJudgeDFam: {{}}, perJudgeDOther: {{}} }};
    // Merge committee system-level bias
    if (hasCommittee && judgeHasData(m, COMMITTEE_KEY)) {{
      const cmrdSP = computeCommitteeMRDSP(sysScores[m], ref, gens);
      Object.assign(r.mrdSP[m].perJudge, cmrdSP.perJudge);
      Object.assign(r.mrdSP[m].perJudgeDSelf || {{}}, cmrdSP.perJudgeDSelf || {{}});
      Object.assign(r.mrdSP[m].perJudgeDOther || {{}}, cmrdSP.perJudgeDOther || {{}});
      const cmsdSP = computeCommitteeMSDSP(sysScores[m], ref, gens, false);
      Object.assign(r.msdSP[m].perJudge, cmsdSP.perJudge);
      Object.assign(r.msdSP[m].perJudgeDSelf || {{}}, cmsdSP.perJudgeDSelf || {{}});
      Object.assign(r.msdSP[m].perJudgeDOther || {{}}, cmsdSP.perJudgeDOther || {{}});
      const cmsdSPn = computeCommitteeMSDSP(sysScores[m], ref, gens, true);
      Object.assign(r.msdSPnorm[m].perJudge, cmsdSPn.perJudge);
      Object.assign(r.msdSPnorm[m].perJudgeDSelf || {{}}, cmsdSPn.perJudgeDSelf || {{}});
      Object.assign(r.msdSPnorm[m].perJudgeDOther || {{}}, cmsdSPn.perJudgeDOther || {{}});
    }}
  }}

  // Instance-level accuracy
  r.mipa = {{}};
  for (const m of methods) {{
    const aj = getActiveJudges(m, judges);
    r.mipa[m] = computeMIPA(m, aj, gens);
  }}

  // Instance-level bias
  r.mispb = {{}}; r.mispbF = {{}}; r.mispbFO = {{}}; r.hspp = {{}}; r.hsppF = {{}}; r.hsppFO = {{}};

  for (const m of methods) {{
    const aj = getActiveJudges(m, judges);
    const realJudges = aj.filter(j => j !== COMMITTEE_KEY);
    // Compute for real judges
    r.mispb[m] = realJudges.length ? computeMISPB(m, realJudges, gens, false, false) : {{ mean: null, perJudge: {{}} }};
    r.mispbF[m] = realJudges.length ? computeMISPB(m, realJudges, gens, false, true, true) : {{ mean: null, perJudge: {{}} }};
    r.mispbFO[m] = realJudges.length ? computeMISPB(m, realJudges, gens, false, true, false) : {{ mean: null, perJudge: {{}} }};
    r.hspp[m] = realJudges.length ? computeMISPB(m, realJudges, gens, true, false) : {{ mean: null, perJudge: {{}} }};
    r.hsppF[m] = realJudges.length ? computeMISPB(m, realJudges, gens, true, true, true) : {{ mean: null, perJudge: {{}} }};
    r.hsppFO[m] = realJudges.length ? computeMISPB(m, realJudges, gens, true, true, false) : {{ mean: null, perJudge: {{}} }};
    // Merge committee SP if committee is active and has data for this method
    const _spKeys = ['perJudge','perJudgeRaw','perJudgeOther','perJudgeRatio',
      'perJudgeNOverSelf','perJudgeNTotalSelf','perJudgeNOverOther','perJudgeNTotalOther',
      'perJudgeNT2WSelf','perJudgeNL2WSelf','perJudgeNL2TSelf',
      'perJudgeNT2WOther','perJudgeNL2WOther','perJudgeNL2TOther'];
    const _mergeC = (target, source) => {{ for (const k of _spKeys) if (source[k]) Object.assign(target[k] = target[k] || {{}}, source[k]); }};
    if (hasCommittee && judgeHasData(m, COMMITTEE_KEY)) {{
      _mergeC(r.mispb[m], computeCommitteeMISPB(m, gens, false, false));
      _mergeC(r.mispbF[m], computeCommitteeMISPB(m, gens, false, true, true));
      _mergeC(r.mispbFO[m], computeCommitteeMISPB(m, gens, false, true, false));
      _mergeC(r.hspp[m], computeCommitteeMISPB(m, gens, true, false));
      _mergeC(r.hsppF[m], computeCommitteeMISPB(m, gens, true, true, true));
      _mergeC(r.hsppFO[m], computeCommitteeMISPB(m, gens, true, true, false));
      // Recompute mean including committee (exclude cmember keys to avoid double-counting)
      const allVals = Object.entries(r.mispb[m].perJudge).filter(([k,v]) => !isCmemberKey(k) && v !== null && v !== undefined).map(([,v]) => v);
      r.mispb[m].mean = allVals.length ? allVals.reduce((a,b)=>a+b,0)/allVals.length : null;
      const allValsH = Object.entries(r.hspp[m].perJudge).filter(([k,v]) => !isCmemberKey(k) && v !== null && v !== undefined).map(([,v]) => v);
      r.hspp[m].mean = allValsH.length ? allValsH.reduce((a,b)=>a+b,0)/allValsH.length : null;
    }}
  }}

  // Rubric-level metrics (SR and AR only)
  r.mra = {{}};
  r.mrspb = {{}}; r.mrspbF = {{}}; r.mrspbFO = {{}}; r.mrspbErr = {{}}; r.mrspbErrF = {{}}; r.mrspbErrFO = {{}};
  for (const m of ['sr', 'ar']) {{
    const aj = getActiveJudges(m, judges);
    const realJ = aj.filter(j => j !== COMMITTEE_KEY);
    r.mra[m] = realJ.length ? computeMRA(m, realJ, gens) : {{ mean: null, perJudge: {{}}, perJudgeNCorrect: {{}}, perJudgeNTotal: {{}} }};
    r.mrspb[m] = realJ.length ? computeMRSPB(m, realJ, gens, false, false) : {{ mean: null, perJudge: {{}} }};
    r.mrspbF[m] = realJ.length ? computeMRSPB(m, realJ, gens, false, true, true) : {{ mean: null, perJudge: {{}} }};
    r.mrspbFO[m] = realJ.length ? computeMRSPB(m, realJ, gens, false, true, false) : {{ mean: null, perJudge: {{}} }};
    r.mrspbErr[m] = realJ.length ? computeMRSPB(m, realJ, gens, true, false) : {{ mean: null, perJudge: {{}} }};
    r.mrspbErrF[m] = realJ.length ? computeMRSPB(m, realJ, gens, true, true, true) : {{ mean: null, perJudge: {{}} }};
    r.mrspbErrFO[m] = realJ.length ? computeMRSPB(m, realJ, gens, true, true, false) : {{ mean: null, perJudge: {{}} }};
    // Merge committee MRA + MRSPB
    if (hasCommittee && judgeHasData(m, COMMITTEE_KEY)) {{
      // MRA for committee
      const cMRA = computeMRA(m, [COMMITTEE_KEY], gens);
      Object.assign(r.mra[m].perJudge, cMRA.perJudge);
      Object.assign(r.mra[m].perJudgeNCorrect || {{}}, cMRA.perJudgeNCorrect || {{}});
      Object.assign(r.mra[m].perJudgeNTotal || {{}}, cMRA.perJudgeNTotal || {{}});
      // MRSPB for committee
      const _rspKeys = ['perJudge','perJudgeRaw','perJudgeOther','perJudgeRatio',
        'perJudgeNOverSelf','perJudgeNTotalSelf','perJudgeNOverOther','perJudgeNTotalOther'];
      const _mergeR = (target, source) => {{ for (const k of _rspKeys) if (source[k]) Object.assign(target[k] = target[k] || {{}}, source[k]); }};
      _mergeR(r.mrspb[m], computeCommitteeMRSPB(m, gens, false, false));
      _mergeR(r.mrspbF[m], computeCommitteeMRSPB(m, gens, false, true, true));
      _mergeR(r.mrspbFO[m], computeCommitteeMRSPB(m, gens, false, true, false));
      _mergeR(r.mrspbErr[m], computeCommitteeMRSPB(m, gens, true, false));
      _mergeR(r.mrspbErrF[m], computeCommitteeMRSPB(m, gens, true, true, true));
      _mergeR(r.mrspbErrFO[m], computeCommitteeMRSPB(m, gens, true, true, false));
    }}
  }}

  cachedResults = r;
  return r;
}}

// ============================================================
// UI RENDERING
// ============================================================

const METHODS = ['sr', 'ar', 'da', 'pwc'];
const METHOD_LABELS = {{ sr: 'SR', ar: 'AR', da: 'DA', pwc: 'PWC' }};
const ACCENT = '#c17f59';
const ACCENT_LIGHT = '#daa88a';
const SAND50 = '#faf9f7';
const SAND200 = '#e8e4de';
const SAND700 = '#3d3a35';
const SUCCESS = '#5b8a72';
const ERROR = '#c75a5a';

function fmt(v, d=4) {{
  if (v === null || v === undefined || isNaN(v)) return '—';
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

// Helper: build an expandable detail section with per-judge counts
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

function renderAggregate(r) {{
  if (!r) {{
    for (const id of ['tbl-sys-accuracy','tbl-inst-accuracy','tbl-rubric-accuracy',
                       'tbl-sys-bias','tbl-inst-bias','tbl-rubric-bias']) {{
      document.getElementById(id).innerHTML = '<p class="no-data">Select at least 2 generators and 1 judge</p>';
    }}
    return;
  }}
  const judges = effectiveJudgesList;

  // System-level accuracy
  let sysAccHtml = makeTable(
    ['Method', 'MPA ↑', 'MRD ↓', 'MSD', 'MSD-norm'],
    METHODS.map(m => [
      METHOD_LABELS[m],
      fmt(r.mpa[m]?.mean), fmt(r.mrd[m]?.mean, 2), fmt(r.msd[m]?.mean), fmt(r.msdNorm[m]?.mean)
    ])
  );
  sysAccHtml += makeDetailSection(
    'Per-judge MPA sample sizes (n_concordant / n_total)',
    () => ['Judge', ...METHODS.map(m => METHOD_LABELS[m])],
    (j) => {{
      const hasSome = METHODS.some(m => r.mpa[m]?.perJudge[j] !== undefined);
      if (!hasSome) return null;
      return [sn(j), ...METHODS.map(m => {{
        const nc = r.mpa[m]?.perJudgeNConc?.[j];
        const nt = r.mpa[m]?.perJudgeNTotal?.[j];
        return nc !== undefined ? nc + '/' + nt : '—';
      }})];
    }},
    judges
  );
  document.getElementById('tbl-sys-accuracy').innerHTML = sysAccHtml;

  // Instance-level accuracy
  let instAccHtml = makeTable(
    ['Method', 'MIPA ↑'],
    METHODS.map(m => [METHOD_LABELS[m], fmt(r.mipa[m]?.mean)])
  );
  instAccHtml += makeDetailSection(
    'Per-judge MIPA sample sizes (n_agree / n_total)',
    () => ['Judge', ...METHODS.map(m => METHOD_LABELS[m])],
    (j) => {{
      const hasSome = METHODS.some(m => r.mipa[m]?.perJudge[j] !== undefined);
      if (!hasSome) return null;
      return [sn(j), ...METHODS.map(m => {{
        const na = r.mipa[m]?.perJudgeNAgree?.[j];
        const nt = r.mipa[m]?.perJudgeNTotal?.[j];
        return na !== undefined ? na + '/' + nt : '—';
      }})];
    }},
    judges
  );
  document.getElementById('tbl-inst-accuracy').innerHTML = instAccHtml;

  // Rubric-level accuracy
  let rubAccHtml = makeTable(
    ['Method', 'MRA ↑'],
    ['sr', 'ar'].map(m => [METHOD_LABELS[m], fmt(r.mra[m]?.mean)])
  );
  rubAccHtml += makeDetailSection(
    'Per-judge MRA sample sizes (n_correct / n_total)',
    () => ['Judge', 'SR', 'AR'],
    (j) => {{
      const hasSR = r.mra.sr?.perJudge[j] !== undefined;
      const hasAR = r.mra.ar?.perJudge[j] !== undefined;
      if (!hasSR && !hasAR) return null;
      const srDetail = hasSR ? (r.mra.sr.perJudgeNCorrect[j] + '/' + r.mra.sr.perJudgeNTotal[j]) : '—';
      const arDetail = hasAR ? (r.mra.ar.perJudgeNCorrect[j] + '/' + r.mra.ar.perJudgeNTotal[j]) : '—';
      return [sn(j), srDetail, arDetail];
    }},
    judges
  );
  document.getElementById('tbl-rubric-accuracy').innerHTML = rubAccHtml;

  // System-level bias — include d_self/d_other components, ratios alongside differences
  let sysBiasHtml = makeTable(
    ['Method', 'MRD-SP', 'd_self', 'd_other', 'MRD-FSP', 'd_fam', 'd_nonfam', 'MRD-FOSP', 'd_fam', 'd_nonfam',
     'MSD-SP', 'd_self', 'd_other', 'MSD-SP-n', 'MSD-FSP', 'd_fam', 'd_nonfam', 'MSD-FSP-n', 'MSD-FOSP', 'd_fam', 'd_nonfam', 'MSD-FOSP-n'],
    METHODS.map(m => [
      METHOD_LABELS[m],
      fmt(r.mrdSP[m]?.mean, 3), fmt(r.mrdSP[m]?.meanDSelf, 2), fmt(r.mrdSP[m]?.meanDOther, 2),
      fmt(r.mrdFSP[m]?.mean, 3), fmt(r.mrdFSP[m]?.meanDFam, 2), fmt(r.mrdFSP[m]?.meanDOther, 2),
      fmt(r.mrdFOSP[m]?.mean, 3), fmt(r.mrdFOSP[m]?.meanDFam, 2), fmt(r.mrdFOSP[m]?.meanDOther, 2),
      fmt(r.msdSP[m]?.mean), fmt(r.msdSP[m]?.meanDSelf), fmt(r.msdSP[m]?.meanDOther),
      fmt(r.msdSPnorm[m]?.mean),
      fmt(r.msdFSP[m]?.mean), fmt(r.msdFSP[m]?.meanDFam), fmt(r.msdFSP[m]?.meanDOther),
      fmt(r.msdFSPnorm[m]?.mean),
      fmt(r.msdFOSP[m]?.mean), fmt(r.msdFOSP[m]?.meanDFam), fmt(r.msdFOSP[m]?.meanDOther),
      fmt(r.msdFOSPnorm[m]?.mean)
    ])
  );
  sysBiasHtml += makeDetailSection(
    'Per-judge MRD-SP components (d_self / d_other)',
    () => ['Judge', ...METHODS.map(m => METHOD_LABELS[m] + ' d_self'), ...METHODS.map(m => METHOD_LABELS[m] + ' d_other')],
    (j) => {{
      const hasSome = METHODS.some(m => r.mrdSP[m]?.perJudge[j] !== undefined);
      if (!hasSome) return null;
      const selfCols = METHODS.map(m => fmt(r.mrdSP[m]?.perJudgeDSelf?.[j], 2));
      const otherCols = METHODS.map(m => fmt(r.mrdSP[m]?.perJudgeDOther?.[j], 2));
      return [sn(j), ...selfCols, ...otherCols];
    }},
    judges
  );
  document.getElementById('tbl-sys-bias').innerHTML = sysBiasHtml;

  // Instance-level bias
  // Helper to compute mean of per-judge values
  const meanPJ = (obj) => {{
    if (!obj) return null;
    const v = Object.entries(obj).filter(([k]) => !isCmemberKey(k)).map(([,v]) => v);
    return v.length ? v.reduce((a,b)=>a+b,0)/v.length : null;
  }};

  let instBiasHtml = makeTable(
    ['Method', 'MISPB', 'raw', 'other', 'ratio', 'MISPB-F', 'F_ratio', 'MISPB-FO', 'FO_ratio',
     'HSPP', 'H_raw', 'H_other', 'H_ratio', 'HSPP-F', 'HF_ratio', 'HSPP-FO', 'HFO_ratio'],
    METHODS.map(m => [
      METHOD_LABELS[m],
      fmt(r.mispb[m]?.mean),
      fmt(meanPJ(r.mispb[m]?.perJudgeRaw)),
      fmt(meanPJ(r.mispb[m]?.perJudgeOther)),
      fmt(meanPJ(r.mispb[m]?.perJudgeRatio), 2),
      fmt(r.mispbF[m]?.mean),
      fmt(meanPJ(r.mispbF[m]?.perJudgeRatio), 2),
      fmt(r.mispbFO[m]?.mean),
      fmt(meanPJ(r.mispbFO[m]?.perJudgeRatio), 2),
      fmt(r.hspp[m]?.mean),
      fmt(meanPJ(r.hspp[m]?.perJudgeRaw)),
      fmt(meanPJ(r.hspp[m]?.perJudgeOther)),
      fmt(meanPJ(r.hspp[m]?.perJudgeRatio), 2),
      fmt(r.hsppF[m]?.mean),
      fmt(meanPJ(r.hsppF[m]?.perJudgeRatio), 2),
      fmt(r.hsppFO[m]?.mean),
      fmt(meanPJ(r.hsppFO[m]?.perJudgeRatio), 2)
    ])
  );
  instBiasHtml += makeDetailSection(
    'Per-judge MISPB sample sizes (n_overest_self/n_total_self | n_overest_other/n_total_other)',
    () => ['Judge', ...METHODS.map(m => METHOD_LABELS[m] + ' self'), ...METHODS.map(m => METHOD_LABELS[m] + ' other')],
    (j) => {{
      const hasSome = METHODS.some(m => r.mispb[m]?.perJudge[j] !== undefined);
      if (!hasSome) return null;
      const selfCols = METHODS.map(m => {{
        const nos = r.mispb[m]?.perJudgeNOverSelf?.[j];
        const nts = r.mispb[m]?.perJudgeNTotalSelf?.[j];
        return nos !== undefined ? nos + '/' + nts : '\u2014';
      }});
      const otherCols = METHODS.map(m => {{
        const noo = r.mispb[m]?.perJudgeNOverOther?.[j];
        const nto = r.mispb[m]?.perJudgeNTotalOther?.[j];
        return noo !== undefined ? noo + '/' + nto : '\u2014';
      }});
      return [sn(j), ...selfCols, ...otherCols];
    }},
    judges
  );
  instBiasHtml += makeDetailSection(
    'Overestimation sub-types: t2w (tie\u2192win) / l2w (loss\u2192win) / l2t (loss\u2192tie) for self | other',
    () => ['Judge', ...METHODS.flatMap(m => [METHOD_LABELS[m]+' self', METHOD_LABELS[m]+' other'])],
    (j) => {{
      const hasSome = METHODS.some(m => r.mispb[m]?.perJudge[j] !== undefined);
      if (!hasSome) return null;
      const cols = METHODS.flatMap(m => {{
        const t2ws = r.mispb[m]?.perJudgeNT2WSelf?.[j];
        const l2ws = r.mispb[m]?.perJudgeNL2WSelf?.[j];
        const l2ts = r.mispb[m]?.perJudgeNL2TSelf?.[j];
        const t2wo = r.mispb[m]?.perJudgeNT2WOther?.[j];
        const l2wo = r.mispb[m]?.perJudgeNL2WOther?.[j];
        const l2to = r.mispb[m]?.perJudgeNL2TOther?.[j];
        return [
          t2ws !== undefined ? t2ws+'/'+l2ws+'/'+l2ts : '\u2014',
          t2wo !== undefined ? t2wo+'/'+l2wo+'/'+l2to : '\u2014'
        ];
      }});
      return [sn(j), ...cols];
    }},
    judges
  );
  document.getElementById('tbl-inst-bias').innerHTML = instBiasHtml;

  // Rubric-level bias
  let rubBiasHtml = makeTable(
    ['Method', 'MRSPB', 'raw', 'other', 'ratio', 'MRSPB-F', 'F_ratio', 'MRSPB-FO', 'FO_ratio',
     'MRSPB-err', 'err_ratio', 'MRSPB-err-F', 'errF_ratio', 'MRSPB-err-FO', 'errFO_ratio'],
    ['sr', 'ar'].map(m => [
      METHOD_LABELS[m],
      fmt(r.mrspb[m]?.mean),
      fmt(meanPJ(r.mrspb[m]?.perJudgeRaw)),
      fmt(meanPJ(r.mrspb[m]?.perJudgeOther)),
      fmt(meanPJ(r.mrspb[m]?.perJudgeRatio), 2),
      fmt(r.mrspbF[m]?.mean),
      fmt(meanPJ(r.mrspbF[m]?.perJudgeRatio), 2),
      fmt(r.mrspbFO[m]?.mean),
      fmt(meanPJ(r.mrspbFO[m]?.perJudgeRatio), 2),
      fmt(r.mrspbErr[m]?.mean),
      fmt(meanPJ(r.mrspbErr[m]?.perJudgeRatio), 2),
      fmt(r.mrspbErrF[m]?.mean),
      fmt(meanPJ(r.mrspbErrF[m]?.perJudgeRatio), 2),
      fmt(r.mrspbErrFO[m]?.mean),
      fmt(meanPJ(r.mrspbErrFO[m]?.perJudgeRatio), 2)
    ])
  );
  rubBiasHtml += makeDetailSection(
    'Per-judge MRSPB sample sizes (n_overest_self/n_total_self | n_overest_other/n_total_other)',
    () => ['Judge', 'SR self', 'SR other', 'AR self', 'AR other'],
    (j) => {{
      const hasSR = r.mrspb.sr?.perJudge[j] !== undefined;
      const hasAR = r.mrspb.ar?.perJudge[j] !== undefined;
      if (!hasSR && !hasAR) return null;
      const srSelf = hasSR ? (r.mrspb.sr.perJudgeNOverSelf[j] + '/' + r.mrspb.sr.perJudgeNTotalSelf[j]) : '—';
      const srOther = hasSR ? (r.mrspb.sr.perJudgeNOverOther[j] + '/' + r.mrspb.sr.perJudgeNTotalOther[j]) : '—';
      const arSelf = hasAR ? (r.mrspb.ar.perJudgeNOverSelf[j] + '/' + r.mrspb.ar.perJudgeNTotalSelf[j]) : '—';
      const arOther = hasAR ? (r.mrspb.ar.perJudgeNOverOther[j] + '/' + r.mrspb.ar.perJudgeNTotalOther[j]) : '—';
      return [sn(j), srSelf, srOther, arSelf, arOther];
    }},
    judges
  );
  document.getElementById('tbl-rubric-bias').innerHTML = rubBiasHtml;
}}

function renderPerJudge(r) {{
  const judges = effectiveJudgesList;
  // Build extended judge list with per-member committee SP rows
  let spJudges = judges;
  if (isShowCmemberSP()) {{
    spJudges = [...judges, ...getCommitteeMembers().map(cmemberKey)];
  }}
  const allIds = ['tbl-mpa-judge','tbl-mipa-judge','tbl-mra-judge',
    'tbl-mrdsp-judge','tbl-msdsp-judge',
    'tbl-mispb-judge','tbl-mispbf-judge','tbl-hspp-judge','tbl-hsppf-judge',
    'tbl-mrspb-judge','tbl-mrspbf-judge','tbl-mrspberr-judge','tbl-mrspberrf-judge',
    'tbl-mispbfo-judge','tbl-hsppfo-judge','tbl-mrspbfo-judge','tbl-mrspberrfo-judge'];
  if (!r || !judges.length) {{
    for (const id of allIds) document.getElementById(id).innerHTML = '<p class="no-data">No data</p>';
    return;
  }}

  // MPA per judge (with expandable counts)
  let mpaHtml = makeTable(
    ['Judge', ...METHODS.map(m => METHOD_LABELS[m])],
    judges.map(j => [sn(j), ...METHODS.map(m => fmt(r.mpa[m]?.perJudge[j]))])
  );
  mpaHtml += makeDetailSection(
    'Sample sizes (n_concordant / n_total pairs)',
    () => ['Judge', ...METHODS.map(m => METHOD_LABELS[m])],
    (j) => {{
      const hasSome = METHODS.some(m => r.mpa[m]?.perJudgeNConc?.[j] !== undefined);
      if (!hasSome) return null;
      return [sn(j), ...METHODS.map(m => {{
        const nc = r.mpa[m]?.perJudgeNConc?.[j];
        const nt = r.mpa[m]?.perJudgeNTotal?.[j];
        return nc !== undefined ? nc + '/' + nt : '—';
      }})];
    }},
    judges
  );
  document.getElementById('tbl-mpa-judge').innerHTML = mpaHtml;

  // MIPA per judge (with expandable counts)
  let mipaHtml = makeTable(
    ['Judge', ...METHODS.map(m => METHOD_LABELS[m])],
    judges.map(j => [sn(j), ...METHODS.map(m => fmt(r.mipa[m]?.perJudge[j]))])
  );
  mipaHtml += makeDetailSection(
    'Sample sizes (n_agree / n_total comparisons)',
    () => ['Judge', ...METHODS.map(m => METHOD_LABELS[m])],
    (j) => {{
      const hasSome = METHODS.some(m => r.mipa[m]?.perJudgeNAgree?.[j] !== undefined);
      if (!hasSome) return null;
      return [sn(j), ...METHODS.map(m => {{
        const na = r.mipa[m]?.perJudgeNAgree?.[j];
        const nt = r.mipa[m]?.perJudgeNTotal?.[j];
        return na !== undefined ? na + '/' + nt : '—';
      }})];
    }},
    judges
  );
  document.getElementById('tbl-mipa-judge').innerHTML = mipaHtml;

  // MRA per judge (with expandable counts)
  let mraHtml = makeTable(
    ['Judge', 'SR', 'AR'],
    judges.map(j => [sn(j), fmt(r.mra.sr?.perJudge[j]), fmt(r.mra.ar?.perJudge[j])])
  );
  mraHtml += makeDetailSection(
    'Sample sizes (n_correct / n_total rubrics)',
    () => ['Judge', 'SR', 'AR'],
    (j) => {{
      const hasSR = r.mra.sr?.perJudgeNCorrect?.[j] !== undefined;
      const hasAR = r.mra.ar?.perJudgeNCorrect?.[j] !== undefined;
      if (!hasSR && !hasAR) return null;
      return [sn(j),
        hasSR ? (r.mra.sr.perJudgeNCorrect[j] + '/' + r.mra.sr.perJudgeNTotal[j]) : '—',
        hasAR ? (r.mra.ar.perJudgeNCorrect[j] + '/' + r.mra.ar.perJudgeNTotal[j]) : '—',
      ];
    }},
    judges
  );
  document.getElementById('tbl-mra-judge').innerHTML = mraHtml;

  // Helper: build a bias table with value + raw + other + ratio per method
  function biasTable4M(metric, judgeList) {{
    const jl = judgeList || judges;
    const headers = ['Judge'];
    for (const m of METHODS) {{
      const ml = METHOD_LABELS[m];
      headers.push(ml, ml+' raw', ml+' oth', ml+' rat');
    }}
    const rowClasses = jl.map(j => isCmemberKey(j) ? 'cmember-row' : '');
    return makeTable(headers, jl.map(j => {{
      const row = [sn(j)];
      for (const m of METHODS) {{
        row.push(
          fmt(metric[m]?.perJudge[j]),
          fmt(metric[m]?.perJudgeRaw[j]),
          fmt(metric[m]?.perJudgeOther[j]),
          fmt(metric[m]?.perJudgeRatio[j], 2)
        );
      }}
      return row;
    }}), {{ rowClasses }});
  }}

  // Helper: build a bias table with value + raw + other + ratio for SR/AR
  function biasTable2M(metric, judgeList) {{
    const jl = judgeList || judges;
    const rowClasses = jl.map(j => isCmemberKey(j) ? 'cmember-row' : '');
    return makeTable(
      ['Judge', 'SR', 'SR raw', 'SR oth', 'SR rat', 'AR', 'AR raw', 'AR oth', 'AR rat'],
      jl.map(j => [
        sn(j),
        fmt(metric.sr?.perJudge[j]), fmt(metric.sr?.perJudgeRaw[j]),
        fmt(metric.sr?.perJudgeOther[j]), fmt(metric.sr?.perJudgeRatio[j], 2),
        fmt(metric.ar?.perJudge[j]), fmt(metric.ar?.perJudgeRaw[j]),
        fmt(metric.ar?.perJudgeOther[j]), fmt(metric.ar?.perJudgeRatio[j], 2),
      ]),
      {{ rowClasses }}
    );
  }}

  // Helper: build expandable count section for 4-method bias metrics
  function biasDetail4M(metric, label, judgeList) {{
    return makeDetailSection(
      label + ' counts (n_overest_self/n_total_self | n_overest_other/n_total_other)',
      () => ['Judge', ...METHODS.flatMap(m => [METHOD_LABELS[m]+' self', METHOD_LABELS[m]+' other'])],
      (j) => {{
        const hasSome = METHODS.some(m => metric[m]?.perJudge[j] !== undefined);
        if (!hasSome) return null;
        const cols = METHODS.flatMap(m => {{
          const nos = metric[m]?.perJudgeNOverSelf?.[j];
          const nts = metric[m]?.perJudgeNTotalSelf?.[j];
          const noo = metric[m]?.perJudgeNOverOther?.[j];
          const nto = metric[m]?.perJudgeNTotalOther?.[j];
          return [
            nos !== undefined ? nos+'/'+nts : '—',
            noo !== undefined ? noo+'/'+nto : '—'
          ];
        }});
        return [sn(j), ...cols];
      }},
      judgeList || judges
    );
  }}

  // Helper: build expandable count section for 2-method rubric bias metrics
  function biasDetail2M(metric, label, judgeList) {{
    return makeDetailSection(
      label + ' counts (n_overest_self/n_total_self | n_overest_other/n_total_other)',
      () => ['Judge', 'SR self', 'SR other', 'AR self', 'AR other'],
      (j) => {{
        const hasSR = metric.sr?.perJudge[j] !== undefined;
        const hasAR = metric.ar?.perJudge[j] !== undefined;
        if (!hasSR && !hasAR) return null;
        return [sn(j),
          hasSR ? (metric.sr.perJudgeNOverSelf[j]+'/'+metric.sr.perJudgeNTotalSelf[j]) : '\u2014',
          hasSR ? (metric.sr.perJudgeNOverOther[j]+'/'+metric.sr.perJudgeNTotalOther[j]) : '\u2014',
          hasAR ? (metric.ar.perJudgeNOverSelf[j]+'/'+metric.ar.perJudgeNTotalSelf[j]) : '\u2014',
          hasAR ? (metric.ar.perJudgeNOverOther[j]+'/'+metric.ar.perJudgeNTotalOther[j]) : '\u2014',
        ];
      }},
      judgeList || judges
    );
  }}

  // Helper: build expandable sub-type breakdown for 4-method bias metrics
  function biasSubtypeDetail4M(metric, label, judgeList) {{
    return makeDetailSection(
      label + ' sub-types: t2w (tie\u2192win) / l2w (loss\u2192win) / l2t (loss\u2192tie)',
      () => ['Judge', ...METHODS.flatMap(m => [METHOD_LABELS[m]+' self', METHOD_LABELS[m]+' other'])],
      (j) => {{
        const hasSome = METHODS.some(m => metric[m]?.perJudge[j] !== undefined);
        if (!hasSome) return null;
        const cols = METHODS.flatMap(m => {{
          const t2ws = metric[m]?.perJudgeNT2WSelf?.[j];
          const l2ws = metric[m]?.perJudgeNL2WSelf?.[j];
          const l2ts = metric[m]?.perJudgeNL2TSelf?.[j];
          const t2wo = metric[m]?.perJudgeNT2WOther?.[j];
          const l2wo = metric[m]?.perJudgeNL2WOther?.[j];
          const l2to = metric[m]?.perJudgeNL2TOther?.[j];
          return [
            t2ws !== undefined ? t2ws+'/'+l2ws+'/'+l2ts : '\u2014',
            t2wo !== undefined ? t2wo+'/'+l2wo+'/'+l2to : '\u2014'
          ];
        }});
        return [sn(j), ...cols];
      }},
      judgeList || judges
    );
  }}

  // MRD-SP / MRD-FSP per judge
  const mrdspRC = spJudges.map(j => isCmemberKey(j) ? 'cmember-row' : '');
  let mrdspHtml = makeTable(
    ['Judge', ...METHODS.flatMap(m => [METHOD_LABELS[m]+' MRD-SP', 'd_self', 'd_other'])],
    spJudges.map(j => {{
      const row = [sn(j)];
      for (const m of METHODS) {{
        row.push(
          fmt(r.mrdSP[m]?.perJudge[j], 3),
          fmt(r.mrdSP[m]?.perJudgeDSelf?.[j], 2),
          fmt(r.mrdSP[m]?.perJudgeDOther?.[j], 2)
        );
      }}
      return row;
    }}),
    {{ rowClasses: mrdspRC }}
  );
  mrdspHtml += makeDetailSection(
    'MRD-FSP per judge (d_family / d_nonfamily)',
    () => ['Judge', ...METHODS.flatMap(m => [METHOD_LABELS[m]+' MRD-FSP', 'd_fam', 'd_nonfam'])],
    (j) => {{
      const hasSome = METHODS.some(m => r.mrdFSP[m]?.perJudge[j] !== undefined);
      if (!hasSome) return null;
      const cols = METHODS.flatMap(m => [
        fmt(r.mrdFSP[m]?.perJudge[j], 3),
        fmt(r.mrdFSP[m]?.perJudgeDFam?.[j], 2),
        fmt(r.mrdFSP[m]?.perJudgeDOther?.[j], 2)
      ]);
      return [sn(j), ...cols];
    }},
    spJudges
  );
  mrdspHtml += makeDetailSection(
    'MRD-FOSP per judge (d_family_only / d_nonfamily)',
    () => ['Judge', ...METHODS.flatMap(m => [METHOD_LABELS[m]+' MRD-FOSP', 'd_fam', 'd_nonfam'])],
    (j) => {{
      const hasSome = METHODS.some(m => r.mrdFOSP[m]?.perJudge[j] !== undefined);
      if (!hasSome) return null;
      const cols = METHODS.flatMap(m => [
        fmt(r.mrdFOSP[m]?.perJudge[j], 3),
        fmt(r.mrdFOSP[m]?.perJudgeDFam?.[j], 2),
        fmt(r.mrdFOSP[m]?.perJudgeDOther?.[j], 2)
      ]);
      return [sn(j), ...cols];
    }},
    spJudges
  );
  document.getElementById('tbl-mrdsp-judge').innerHTML = mrdspHtml;

  // MSD-SP / MSD-FSP per judge
  const msdspRC = spJudges.map(j => isCmemberKey(j) ? 'cmember-row' : '');
  let msdspHtml = makeTable(
    ['Judge', ...METHODS.flatMap(m => [METHOD_LABELS[m]+' MSD-SP', 'd_self', 'd_other'])],
    spJudges.map(j => {{
      const row = [sn(j)];
      for (const m of METHODS) {{
        row.push(
          fmt(r.msdSP[m]?.perJudge[j]),
          fmt(r.msdSP[m]?.perJudgeDSelf?.[j]),
          fmt(r.msdSP[m]?.perJudgeDOther?.[j])
        );
      }}
      return row;
    }}),
    {{ rowClasses: msdspRC }}
  );
  msdspHtml += makeDetailSection(
    'MSD-SP-norm per judge (d_self / d_other)',
    () => ['Judge', ...METHODS.flatMap(m => [METHOD_LABELS[m]+' MSD-SP-n', 'd_self', 'd_other'])],
    (j) => {{
      const hasSome = METHODS.some(m => r.msdSPnorm[m]?.perJudge[j] !== undefined);
      if (!hasSome) return null;
      const cols = METHODS.flatMap(m => [
        fmt(r.msdSPnorm[m]?.perJudge[j]),
        fmt(r.msdSPnorm[m]?.perJudgeDSelf?.[j]),
        fmt(r.msdSPnorm[m]?.perJudgeDOther?.[j])
      ]);
      return [sn(j), ...cols];
    }},
    spJudges
  );
  msdspHtml += makeDetailSection(
    'MSD-FSP per judge (d_family / d_nonfamily)',
    () => ['Judge', ...METHODS.flatMap(m => [METHOD_LABELS[m]+' MSD-FSP', 'd_fam', 'd_nonfam'])],
    (j) => {{
      const hasSome = METHODS.some(m => r.msdFSP[m]?.perJudge[j] !== undefined);
      if (!hasSome) return null;
      const cols = METHODS.flatMap(m => [
        fmt(r.msdFSP[m]?.perJudge[j]),
        fmt(r.msdFSP[m]?.perJudgeDFam?.[j]),
        fmt(r.msdFSP[m]?.perJudgeDOther?.[j])
      ]);
      return [sn(j), ...cols];
    }},
    spJudges
  );
  msdspHtml += makeDetailSection(
    'MSD-FSP-norm per judge (d_family / d_nonfamily)',
    () => ['Judge', ...METHODS.flatMap(m => [METHOD_LABELS[m]+' MSD-FSP-n', 'd_fam', 'd_nonfam'])],
    (j) => {{
      const hasSome = METHODS.some(m => r.msdFSPnorm[m]?.perJudge[j] !== undefined);
      if (!hasSome) return null;
      const cols = METHODS.flatMap(m => [
        fmt(r.msdFSPnorm[m]?.perJudge[j]),
        fmt(r.msdFSPnorm[m]?.perJudgeDFam?.[j]),
        fmt(r.msdFSPnorm[m]?.perJudgeDOther?.[j])
      ]);
      return [sn(j), ...cols];
    }},
    spJudges
  );
  msdspHtml += makeDetailSection(
    'MSD-FOSP per judge (d_family_only / d_nonfamily)',
    () => ['Judge', ...METHODS.flatMap(m => [METHOD_LABELS[m]+' MSD-FOSP', 'd_fam', 'd_nonfam'])],
    (j) => {{
      const hasSome = METHODS.some(m => r.msdFOSP[m]?.perJudge[j] !== undefined);
      if (!hasSome) return null;
      const cols = METHODS.flatMap(m => [
        fmt(r.msdFOSP[m]?.perJudge[j]),
        fmt(r.msdFOSP[m]?.perJudgeDFam?.[j]),
        fmt(r.msdFOSP[m]?.perJudgeDOther?.[j])
      ]);
      return [sn(j), ...cols];
    }},
    spJudges
  );
  msdspHtml += makeDetailSection(
    'MSD-FOSP-norm per judge (d_family_only / d_nonfamily)',
    () => ['Judge', ...METHODS.flatMap(m => [METHOD_LABELS[m]+' MSD-FOSP-n', 'd_fam', 'd_nonfam'])],
    (j) => {{
      const hasSome = METHODS.some(m => r.msdFOSPnorm[m]?.perJudge[j] !== undefined);
      if (!hasSome) return null;
      const cols = METHODS.flatMap(m => [
        fmt(r.msdFOSPnorm[m]?.perJudge[j]),
        fmt(r.msdFOSPnorm[m]?.perJudgeDFam?.[j]),
        fmt(r.msdFOSPnorm[m]?.perJudgeDOther?.[j])
      ]);
      return [sn(j), ...cols];
    }},
    spJudges
  );
  document.getElementById('tbl-msdsp-judge').innerHTML = msdspHtml;

  // MISPB per judge (self variant)
  document.getElementById('tbl-mispb-judge').innerHTML = biasTable4M(r.mispb, spJudges) + biasDetail4M(r.mispb, 'MISPB', spJudges) + biasSubtypeDetail4M(r.mispb, 'MISPB', spJudges);

  // MISPB-F per judge (family variant)
  document.getElementById('tbl-mispbf-judge').innerHTML = biasTable4M(r.mispbF, spJudges) + biasDetail4M(r.mispbF, 'MISPB-F', spJudges) + biasSubtypeDetail4M(r.mispbF, 'MISPB-F', spJudges);

  // MISPB-FO per judge (family-only variant)
  document.getElementById('tbl-mispbfo-judge').innerHTML = biasTable4M(r.mispbFO, spJudges) + biasDetail4M(r.mispbFO, 'MISPB-FO', spJudges) + biasSubtypeDetail4M(r.mispbFO, 'MISPB-FO', spJudges);

  // HSPP per judge
  document.getElementById('tbl-hspp-judge').innerHTML = biasTable4M(r.hspp, spJudges) + biasDetail4M(r.hspp, 'HSPP', spJudges) + biasSubtypeDetail4M(r.hspp, 'HSPP', spJudges);

  // HSPP-F per judge
  document.getElementById('tbl-hsppf-judge').innerHTML = biasTable4M(r.hsppF, spJudges) + biasDetail4M(r.hsppF, 'HSPP-F', spJudges) + biasSubtypeDetail4M(r.hsppF, 'HSPP-F', spJudges);

  // HSPP-FO per judge (family-only variant)
  document.getElementById('tbl-hsppfo-judge').innerHTML = biasTable4M(r.hsppFO, spJudges) + biasDetail4M(r.hsppFO, 'HSPP-FO', spJudges) + biasSubtypeDetail4M(r.hsppFO, 'HSPP-FO', spJudges);

  // MRSPB per judge (self)
  document.getElementById('tbl-mrspb-judge').innerHTML = biasTable2M(r.mrspb, spJudges) + biasDetail2M(r.mrspb, 'MRSPB', spJudges);

  // MRSPB-F per judge (family)
  document.getElementById('tbl-mrspbf-judge').innerHTML = biasTable2M(r.mrspbF, spJudges) + biasDetail2M(r.mrspbF, 'MRSPB-F', spJudges);

  // MRSPB-FO per judge (family-only)
  document.getElementById('tbl-mrspbfo-judge').innerHTML = biasTable2M(r.mrspbFO, spJudges) + biasDetail2M(r.mrspbFO, 'MRSPB-FO', spJudges);

  // MRSPB-err per judge (error denominator)
  document.getElementById('tbl-mrspberr-judge').innerHTML = biasTable2M(r.mrspbErr, spJudges) + biasDetail2M(r.mrspbErr, 'MRSPB-err', spJudges);

  // MRSPB-err-F per judge (family + error denominator)
  document.getElementById('tbl-mrspberrf-judge').innerHTML = biasTable2M(r.mrspbErrF, spJudges) + biasDetail2M(r.mrspbErrF, 'MRSPB-err-F', spJudges);

  // MRSPB-err-FO per judge (family-only + error denominator)
  document.getElementById('tbl-mrspberrfo-judge').innerHTML = biasTable2M(r.mrspbErrFO, spJudges) + biasDetail2M(r.mrspbErrFO, 'MRSPB-err-FO', spJudges);
}}

function renderPerGenerator(r) {{
  const gens = getSelectedGens();
  if (!r || !gens.length) {{
    for (const id of ['tbl-ref-scores','tbl-gen-scores','tbl-gen-deltas','tbl-gen-deltas-norm','tbl-gen-rank-deltas']) {{
      document.getElementById(id).innerHTML = '<p class="no-data">No data</p>';
    }}
    return;
  }}

  const refScores = r.refScores;
  const judges = effectiveJudgesList;

  // Reference scores
  document.getElementById('tbl-ref-scores').innerHTML = makeTable(
    ['Generator', 'Ref Score'],
    gens.sort((a,b) => (refScores[b]||0) - (refScores[a]||0)).map(g => [sn(g), fmt(refScores[g])])
  );

  // Mean judge scores per method per generator
  const genScoreRows = gens.map(g => {{
    const row = [sn(g)];
    for (const m of METHODS) {{
      const ss = r.sysScores[m];
      const activeJ = getActiveJudges(m, judges);
      const vals = activeJ.map(j => ss[j+'|'+g]).filter(v => v !== undefined);
      row.push(fmt(vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null));
    }}
    return row;
  }});
  document.getElementById('tbl-gen-scores').innerHTML = makeTable(
    ['Generator', ...METHODS.map(m => METHOD_LABELS[m])],
    genScoreRows
  );

  // Score deltas
  const deltaRows = gens.map(g => {{
    const row = [sn(g)];
    for (const m of METHODS) {{
      const ss = r.sysScores[m];
      const ref = m === 'pwc' ? r.refPWCScores : r.refScores;
      const activeJ = getActiveJudges(m, judges);
      const vals = activeJ.map(j => ss[j+'|'+g]).filter(v => v !== undefined);
      const meanJ = vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null;
      row.push(fmt(meanJ !== null && ref[g] !== undefined ? meanJ - ref[g] : null));
    }}
    return row;
  }});
  document.getElementById('tbl-gen-deltas').innerHTML = makeTable(
    ['Generator', ...METHODS.map(m => METHOD_LABELS[m])],
    deltaRows
  );

  // Normalized score deltas (delta / ref_score)
  const deltaNormRows = gens.map(g => {{
    const row = [sn(g)];
    for (const m of METHODS) {{
      const ss = r.sysScores[m];
      const ref = m === 'pwc' ? r.refPWCScores : r.refScores;
      const activeJ = getActiveJudges(m, judges);
      const vals = activeJ.map(j => ss[j+'|'+g]).filter(v => v !== undefined);
      const meanJ = vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null;
      if (meanJ !== null && ref[g] !== undefined && ref[g] > 0) {{
        row.push(fmt((meanJ - ref[g]) / ref[g]));
      }} else {{
        row.push(fmt(null));
      }}
    }}
    return row;
  }});
  document.getElementById('tbl-gen-deltas-norm').innerHTML = makeTable(
    ['Generator', ...METHODS.map(m => METHOD_LABELS[m])],
    deltaNormRows
  );

  // Rank deltas (judge rank - reference rank)
  // For each method, compute reference ranks and per-judge ranks, then average rank delta across judges
  const rankDeltaRows = [];
  for (const m of METHODS) {{
    const ss = m === 'pwc' ? r.refPWCScores : r.refScores;
    // Reference scores → ranks (higher score → rank 1)
    const refVals = gens.map(g => ({{ gen: g, val: ss[g] || 0 }}));
    refVals.sort((a, b) => b.val - a.val);
    const refRank = {{}};
    refVals.forEach((item, i) => {{ refRank[item.gen] = i + 1; }});

    const activeJ = getActiveJudges(m, judges);
    // Per-judge ranks
    const judgeRanks = {{}};
    for (const j of activeJ) {{
      const jVals = gens.map(g => ({{ gen: g, val: r.sysScores[m][j + '|' + g] || 0 }}));
      jVals.sort((a, b) => b.val - a.val);
      const jRank = {{}};
      jVals.forEach((item, i) => {{ jRank[item.gen] = i + 1; }});
      judgeRanks[j] = jRank;
    }}

    // Mean rank delta per generator
    for (const g of gens) {{
      if (!rankDeltaRows.find(r => r[0] === sn(g))) {{
        rankDeltaRows.push([sn(g), ...METHODS.map(() => null)]);
      }}
      const row = rankDeltaRows.find(r => r[0] === sn(g));
      const mi = METHODS.indexOf(m);
      const deltas = activeJ.map(j => (judgeRanks[j][g] || 0) - (refRank[g] || 0));
      row[mi + 1] = fmt(deltas.length ? deltas.reduce((a, b) => a + b, 0) / deltas.length : null);
    }}
  }}
  document.getElementById('tbl-gen-rank-deltas').innerHTML = makeTable(
    ['Generator', ...METHODS.map(m => METHOD_LABELS[m])],
    rankDeltaRows
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

const METHOD_COLORS = {{
  sr: '#c17f59',
  ar: '#5b8a72',
  da: '#6b8cae',
  pwc: '#9b7bb8',
}};

function renderCharts(r) {{
  if (!r) {{
    for (const id of ['chart-accuracy','chart-mispb','chart-mpa-judge','chart-acc-bias','chart-scores','chart-mrspb']) {{
      document.getElementById(id).innerHTML = '<p class="no-data">No data</p>';
    }}
    return;
  }}

  const judges = effectiveJudgesList;
  const gens = getSelectedGens();

  // 1. Method Accuracy Comparison
  const accTraces = [];
  const accMetrics = [
    {{ key: 'mpa', label: 'MPA', getter: m => r.mpa[m]?.mean }},
    {{ key: 'mipa', label: 'MIPA', getter: m => r.mipa[m]?.mean }},
  ];
  for (const metric of accMetrics) {{
    accTraces.push({{
      x: METHODS.map(m => METHOD_LABELS[m]),
      y: METHODS.map(m => metric.getter(m)),
      name: metric.label,
      type: 'bar',
      marker: {{ color: metric.key === 'mpa' ? ACCENT : SUCCESS }},
    }});
  }}
  Plotly.newPlot('chart-accuracy', accTraces, {{
    ...PLOTLY_LAYOUT,
    barmode: 'group',
    yaxis: {{ title: 'Score', gridcolor: SAND200 }},
    xaxis: {{ gridcolor: SAND200 }},
    legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.8)' }},
  }}, {{responsive: true}});

  // 2. MISPB Heatmap
  const mispbJudges = judges.filter(j => {{
    for (const m of METHODS) if (r.mispb[m]?.perJudge[j] !== undefined) return true;
    return false;
  }});
  if (mispbJudges.length > 0) {{
    const z = METHODS.map(m => mispbJudges.map(j => r.mispb[m]?.perJudge[j] ?? null));
    Plotly.newPlot('chart-mispb', [{{
      z: z,
      x: mispbJudges.map(j => sn(j)),
      y: METHODS.map(m => METHOD_LABELS[m]),
      type: 'heatmap',
      colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
      zmid: 0,
      colorbar: {{ title: 'MISPB' }},
    }}], {{
      ...PLOTLY_LAYOUT,
      xaxis: {{ tickangle: -45 }},
      margin: {{ t: 30, r: 80, b: 80, l: 50 }},
    }}, {{responsive: true}});
  }} else {{
    document.getElementById('chart-mispb').innerHTML = '<p class="no-data">No bias data available</p>';
  }}

  // 3. Per-Judge MPA grouped bar
  const mpaJudges = judges.filter(j => {{
    for (const m of METHODS) if (r.mpa[m]?.perJudge[j] !== undefined) return true;
    return false;
  }});
  if (mpaJudges.length > 0) {{
    const mpaTraces = METHODS.map(m => ({{
      x: mpaJudges.map(j => sn(j)),
      y: mpaJudges.map(j => r.mpa[m]?.perJudge[j] ?? null),
      name: METHOD_LABELS[m],
      type: 'bar',
      marker: {{ color: METHOD_COLORS[m] }},
    }}));
    Plotly.newPlot('chart-mpa-judge', mpaTraces, {{
      ...PLOTLY_LAYOUT,
      barmode: 'group',
      xaxis: {{ tickangle: -45, gridcolor: SAND200 }},
      yaxis: {{ title: 'MPA', gridcolor: SAND200 }},
      legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.8)' }},
      margin: {{ t: 30, r: 20, b: 80, l: 50 }},
    }}, {{responsive: true}});
  }} else {{
    document.getElementById('chart-mpa-judge').innerHTML = '<p class="no-data">No data</p>';
  }}

  // 4. Accuracy vs Bias scatter
  const scatterData = METHODS.map(m => ({{
    x: [r.mpa[m]?.mean],
    y: [r.mispb[m]?.mean],
    text: [METHOD_LABELS[m]],
    name: METHOD_LABELS[m],
    mode: 'markers+text',
    textposition: 'top center',
    marker: {{ color: METHOD_COLORS[m], size: 14 }},
    type: 'scatter',
  }}));
  Plotly.newPlot('chart-acc-bias', scatterData, {{
    ...PLOTLY_LAYOUT,
    xaxis: {{ title: 'MPA (accuracy) ↑', gridcolor: SAND200 }},
    yaxis: {{ title: 'MISPB (bias)', gridcolor: SAND200 }},
    showlegend: false,
  }}, {{responsive: true}});

  // 5. System scores comparison
  const sortedGens = [...gens].sort((a,b) => (r.refScores[b]||0) - (r.refScores[a]||0));
  const scoreTraces = [{{
    x: sortedGens.map(g => sn(g)),
    y: sortedGens.map(g => r.refScores[g]),
    name: 'Reference',
    type: 'bar',
    marker: {{ color: SAND700 }},
  }}];
  for (const m of ['sr', 'ar', 'da']) {{
    const activeJ = getActiveJudges(m, judges);
    scoreTraces.push({{
      x: sortedGens.map(g => sn(g)),
      y: sortedGens.map(g => {{
        const vals = activeJ.map(j => r.sysScores[m][j+'|'+g]).filter(v => v !== undefined);
        return vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null;
      }}),
      name: METHOD_LABELS[m],
      type: 'bar',
      marker: {{ color: METHOD_COLORS[m] }},
    }});
  }}
  Plotly.newPlot('chart-scores', scoreTraces, {{
    ...PLOTLY_LAYOUT,
    barmode: 'group',
    xaxis: {{ tickangle: -45, gridcolor: SAND200 }},
    yaxis: {{ title: 'System Score', gridcolor: SAND200, range: [0.6, 1.0] }},
    legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.8)' }},
    margin: {{ t: 30, r: 20, b: 80, l: 50 }},
  }}, {{responsive: true}});

  // 6. MRSPB heatmap
  const mrspbJudges = judges.filter(j => {{
    return (r.mrspb.sr?.perJudge[j] !== undefined) || (r.mrspb.ar?.perJudge[j] !== undefined);
  }});
  if (mrspbJudges.length > 0) {{
    const z = ['sr', 'ar'].map(m => mrspbJudges.map(j => r.mrspb[m]?.perJudge[j] ?? null));
    Plotly.newPlot('chart-mrspb', [{{
      z: z,
      x: mrspbJudges.map(j => sn(j)),
      y: ['SR', 'AR'],
      type: 'heatmap',
      colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
      zmid: 0,
      colorbar: {{ title: 'MRSPB' }},
    }}], {{
      ...PLOTLY_LAYOUT,
      xaxis: {{ tickangle: -45 }},
      margin: {{ t: 30, r: 80, b: 80, l: 50 }},
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

let debounceTimer = null;
function onSelectionChange() {{
  document.getElementById('sel-judges-count').textContent = getSelectedJudges().length;
  document.getElementById('sel-gens-count').textContent = getSelectedGens().length;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(recompute, 150);
}}

function recompute() {{
  const judges = getSelectedJudges();
  const gens = getSelectedGens();
  clearCommitteeData();
  let effectiveJudges = [...judges];
  if (isCommitteeEnabled()) {{
    const members = getCommitteeMembers();
    injectCommitteeData(members, gens);
    effectiveJudges.push(COMMITTEE_KEY);
  }}
  effectiveJudgesList = effectiveJudges;
  const r = computeAll(effectiveJudges, gens);
  renderAggregate(r);
  renderPerJudge(r);
  renderPerGenerator(r);
  renderCharts(r);

  // Trigger IA recompute if IA tab is active
  if (iaInitialized && document.querySelector('.tab.active')?.dataset.tab === 'interactive') {{
    iaBaselineResults = null;
    iaBaselineKey = '';
    setTimeout(iaRecompute, 50);
  }}
}}

function onCommitteeChange() {{
  const enabled = document.getElementById('committee-enable')?.checked;
  document.getElementById('committee-builder').style.display = enabled ? 'block' : 'none';
  const members = getCommitteeMembers();
  const statusEl = document.getElementById('committee-status');
  if (statusEl) {{
    if (enabled && members.length >= 2) {{
      statusEl.textContent = members.map(m => sn(m)).join(' + ');
    }} else if (enabled) {{
      statusEl.textContent = 'Select at least 2 members';
    }} else {{
      statusEl.textContent = '';
    }}
  }}
  onSelectionChange();
}}

function selectAllCommittee() {{
  document.querySelectorAll('#committee-member-checkboxes input').forEach(cb => {{ cb.checked = true; }});
  onCommitteeChange();
}}

function selectNoneCommittee() {{
  document.querySelectorAll('#committee-member-checkboxes input').forEach(cb => {{ cb.checked = false; }});
  onCommitteeChange();
}}

function buildCommitteeMemberCheckboxes() {{
  const div = document.getElementById('committee-member-checkboxes');
  if (!div) return;
  const familyOrder = ['Gemma', 'Llama', 'Qwen', 'GPT', 'Claude'];
  for (const fam of familyOrder) {{
    const members = DATA.families[fam];
    const famDiv = document.createElement('div');
    famDiv.className = 'family-group';
    famDiv.innerHTML = '<span class="family-label">' + fam + '</span>';
    for (const m of members) {{
      famDiv.innerHTML += '<label class="cb-label"><input type="checkbox" value="' + m + '" checked onchange="onCommitteeChange()"><span>' + sn(m) + '</span></label>';
    }}
    div.appendChild(famDiv);
  }}
}}

function buildCheckboxes() {{
  const judgeDiv = document.getElementById('judge-checkboxes');
  const genDiv = document.getElementById('gen-checkboxes');

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
      jFamDiv.innerHTML += '<label class="cb-label"><input type="checkbox" value="' + m + '" ' + (isDefault ? 'checked' : '') + ' onchange="onSelectionChange()"><span>' + label + '</span></label>';
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

// Build rubric offset table for IA
let _iaRubricOffsets = null;
function iaBuildRubricOffsets() {{
  if (_iaRubricOffsets) return;
  _iaRubricOffsets = new Int32Array(DATA.nInstances);
  let pos = 0;
  for (let i = 0; i < DATA.nInstances; i++) {{
    _iaRubricOffsets[i] = pos;
    pos += DATA.rubricCounts[i];
  }}
}}

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

function iaGetMethod() {{
  const el = document.querySelector('input[name="ia-method"]:checked');
  return el ? el.value : 'sr';
}}

// ============================================================
// INTERACTIVE ANALYSIS: FILTER APPLICATION
// ============================================================

function iaApplyFilters() {{
  iaBuildRubricOffsets();
  const lenMin = +document.getElementById('ia-len-min').value;
  const lenMax = +document.getElementById('ia-len-max').value;
  const agrMin = +document.getElementById('ia-agr-min').value / 100;
  const agrMax = +document.getElementById('ia-agr-max').value / 100;
  const method = iaGetMethod();
  const agreementData = method === 'ar' ? DATA.rubricAgreementFlatAR : DATA.rubricAgreementFlatSR;

  // Gather selected category tags as bitmask
  let catMask = 0;
  document.querySelectorAll('#ia-category-checkboxes input:checked').forEach(cb => {{
    catMask |= (1 << +cb.dataset.idx);
  }});
  const anyCatSelected = catMask !== 0;

  // Gather selected instruction ID tags as bitmask
  let iidMask = 0;
  document.querySelectorAll('#ia-instrid-checkboxes input:checked').forEach(cb => {{
    iidMask |= (1 << +cb.dataset.idx);
  }});
  const anyIidSelected = iidMask !== 0;

  // Reset masks
  iaRubricMask.fill(0);
  iaInstanceMask.fill(1);

  // All filters are rubric-level in IFEval (no instance-level themes)
  let rubricIdx = 0;
  let filteredRubrics = 0;
  let filteredInstances = 0;
  for (let i = 0; i < DATA.nInstances; i++) {{
    const n = DATA.rubricCounts[i];
    let instHasRubric = false;
    for (let r = 0; r < n; r++) {{
      const pos = rubricIdx + r;

      const len = DATA.rubricLengthsFlat[pos];
      if (len < lenMin || len > lenMax) continue;

      if (anyCatSelected) {{
        if ((DATA.rubricCategoriesFlat[pos] & catMask) === 0) continue;
      }}

      if (anyIidSelected) {{
        if ((DATA.rubricInstructionIdsFlat[pos] & iidMask) === 0) continue;
      }}

      const agr = agreementData[pos];
      if (agr < agrMin || agr > agrMax) continue;

      iaRubricMask[pos] = 1;
      instHasRubric = true;
      filteredRubrics++;
    }}
    rubricIdx += n;
    if (!instHasRubric) {{
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

function iaInstanceScoreFiltered(rubricBools, instIdx) {{
  // IFEval: all points = 1, score = fraction of filtered rubrics met
  iaBuildRubricOffsets();
  const off = _iaRubricOffsets[instIdx];
  const n = DATA.rubricCounts[instIdx];
  let nFiltered = 0, nMet = 0;
  for (let j = 0; j < n; j++) {{
    if (!iaRubricMask[off + j]) continue;
    nFiltered++;
    if (rubricBools[j]) nMet++;
  }}
  if (nFiltered <= 0) return null;
  return nMet / nFiltered;
}}

const _iaInstScoreCache = {{}};

function iaGetFilteredInstScores(method, judge, gen) {{
  const ck = method + '|' + judge + '|' + gen;
  if (_iaInstScoreCache[ck]) return _iaInstScoreCache[ck];
  const rubrics = getJudgeRubrics(method, judge, gen);
  if (!rubrics) return null;
  const scores = new Float64Array(DATA.nInstances);
  const valid = new Uint8Array(DATA.nInstances);
  for (let i = 0; i < DATA.nInstances; i++) {{
    if (!iaInstanceMask[i]) {{ scores[i] = NaN; continue; }}
    const s = iaInstanceScoreFiltered(rubrics[i], i);
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

function iaFilteredSystemScore(method, judge, gen) {{
  const data = iaGetFilteredInstScores(method, judge, gen);
  if (!data) return null;
  let sum = 0, count = 0;
  for (let i = 0; i < DATA.nInstances; i++) {{
    if (data.valid[i]) {{ sum += data.scores[i]; count++; }}
  }}
  return count > 0 ? sum / count : null;
}}

function iaFilteredRefScore(gen) {{
  // Use known IFEval reference with rubric mask
  const refRubrics = getRefRubrics(gen);
  if (!refRubrics) return null;
  let sum = 0, count = 0;
  for (let i = 0; i < DATA.nInstances; i++) {{
    if (!iaInstanceMask[i]) continue;
    const s = iaInstanceScoreFiltered(refRubrics[i], i);
    if (s === null) continue;
    sum += s; count++;
  }}
  return count > 0 ? sum / count : null;
}}

function iaGetFilteredRefInstScores(gen) {{
  const refRubrics = getRefRubrics(gen);
  if (!refRubrics) return null;
  const scores = new Float64Array(DATA.nInstances);
  const valid = new Uint8Array(DATA.nInstances);
  for (let i = 0; i < DATA.nInstances; i++) {{
    if (!iaInstanceMask[i]) {{ scores[i] = NaN; continue; }}
    const s = iaInstanceScoreFiltered(refRubrics[i], i);
    if (s === null) {{ scores[i] = NaN; continue; }}
    scores[i] = s;
    valid[i] = 1;
  }}
  return {{ scores, valid }};
}}

// ============================================================
// INTERACTIVE ANALYSIS: MSD MATRIX
// ============================================================

function iaComputeMSDMatrix(judges, gens, method) {{
  const judgeScores = {{}};
  const refScores = {{}};

  for (const g of gens) {{
    refScores[g] = iaFilteredRefScore(g);
    for (const j of judges) {{
      judgeScores[j + '|' + g] = iaFilteredSystemScore(method, j, g);
    }}
  }}

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
// INTERACTIVE ANALYSIS: FILTERED BIAS METRICS
// ============================================================

function iaOverestRateFiltered(method, judge, targetGen, gens) {{
  const tgtData = iaGetFilteredInstScores(method, judge, targetGen);
  if (!tgtData) return null;
  const refTgt = iaGetFilteredRefInstScores(targetGen);
  if (!refTgt) return null;
  let nOver = 0, nTotal = 0;
  for (const opp of gens) {{
    if (opp === targetGen) continue;
    const oppData = iaGetFilteredInstScores(method, judge, opp);
    if (!oppData) continue;
    const refOpp = iaGetFilteredRefInstScores(opp);
    if (!refOpp) continue;
    for (let i = 0; i < DATA.nInstances; i++) {{
      if (!iaInstanceMask[i] || !tgtData.valid[i] || !oppData.valid[i] || !refTgt.valid[i] || !refOpp.valid[i]) continue;
      const jSign = Math.sign(tgtData.scores[i] - oppData.scores[i]);
      const rSign = Math.sign(refTgt.scores[i] - refOpp.scores[i]);
      if (rSign >= 0) continue; // HSPP: error denominator only
      nTotal++;
      if (jSign > rSign) nOver++;
    }}
  }}
  return nTotal > 0 ? {{ rate: nOver / nTotal, nOver, nTotal }} : null;
}}

function iaComputeHSPPFiltered(method, judges, gens, familyMode, includeSelf) {{
  if (includeSelf === undefined) includeSelf = true;
  const perJudge = {{}}, perJudgeRaw = {{}}, perJudgeOther = {{}}, perJudgeRatio = {{}};
  for (const judge of judges) {{
    const targets = familyMode ? getFamilyGens(judge, gens, includeSelf) : (gens.includes(judge) ? [judge] : []);
    if (!targets.length) continue;
    const rawResults = targets.map(t => iaOverestRateFiltered(method, judge, t, gens)).filter(r => r !== null);
    if (!rawResults.length) continue;
    const raw = rawResults.reduce((a,b) => a + b.rate, 0) / rawResults.length;
    const others = getOtherGens(judge, gens);
    const otherResults = others.map(g => iaOverestRateFiltered(method, judge, g, gens)).filter(r => r !== null);
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

function iaRubricOverestRateFiltered(method, judge, targetGen, errorDenom) {{
  const jRubrics = getJudgeRubrics(method, judge, targetGen);
  if (!jRubrics) return null;
  const rRubrics = getRefRubrics(targetGen);
  if (!rRubrics) return null;
  iaBuildRubricOffsets();
  let nOver = 0, nTotal = 0;
  for (let i = 0; i < DATA.nInstances; i++) {{
    if (!iaInstanceMask[i]) continue;
    const off = _iaRubricOffsets[i];
    for (let r = 0; r < jRubrics[i].length; r++) {{
      if (!iaRubricMask[off + r]) continue;
      // IFEval: all points positive, so overest = judge says met, ref says not met
      if (errorDenom && rRubrics[i][r]) continue; // skip ref=met
      nTotal++;
      if (jRubrics[i][r] && !rRubrics[i][r]) nOver++;
    }}
  }}
  return nTotal > 0 ? {{ rate: nOver / nTotal, nOver, nTotal }} : null;
}}

function iaComputeMRSPBFiltered(method, judges, gens, errorDenom, familyMode, includeSelf) {{
  if (includeSelf === undefined) includeSelf = true;
  const perJudge = {{}}, perJudgeRaw = {{}}, perJudgeOther = {{}}, perJudgeRatio = {{}};
  for (const judge of judges) {{
    const targets = familyMode ? getFamilyGens(judge, gens, includeSelf) : (gens.includes(judge) ? [judge] : []);
    if (!targets.length) continue;
    const rawResults = targets.map(t => iaRubricOverestRateFiltered(method, judge, t, errorDenom)).filter(r => r !== null);
    if (!rawResults.length) continue;
    const raw = rawResults.reduce((a,b) => a + b.rate, 0) / rawResults.length;
    const others = getOtherGens(judge, gens);
    const otherResults = others.map(g => iaRubricOverestRateFiltered(method, judge, g, errorDenom)).filter(r => r !== null);
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

function iaComputeMIPAFiltered(method, judges, gens) {{
  const perJudge = {{}};
  for (const judge of judges) {{
    const jGens = gens.filter(g => iaGetFilteredInstScores(method, judge, g));
    if (jGens.length < 2) continue;
    const instScores = {{}};
    const refInst = {{}};
    for (const g of jGens) {{
      instScores[g] = iaGetFilteredInstScores(method, judge, g);
      refInst[g] = iaGetFilteredRefInstScores(g);
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

function iaComputeMIPAFilteredVsBaseline(method, judges, gens) {{
  const perJudge = {{}};
  for (const judge of judges) {{
    const jGens = gens.filter(g => iaGetFilteredInstScores(method, judge, g) && _getInstScores(method, judge, g));
    if (jGens.length < 2) continue;
    const filteredScores = {{}};
    const baselineScores = {{}};
    for (const g of jGens) {{
      filteredScores[g] = iaGetFilteredInstScores(method, judge, g);
      baselineScores[g] = _getInstScores(method, judge, g);
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

function iaComputeSummaryTable(judges, gens, method, msdResult, baselineMsdResult) {{
  const hspp = iaComputeHSPPFiltered(method, judges, gens, false, true);
  const hsppF = iaComputeHSPPFiltered(method, judges, gens, true, true);
  const hsppFO = iaComputeHSPPFiltered(method, judges, gens, true, false);
  const mrspbErr = iaComputeMRSPBFiltered(method, judges, gens, true, false, true);
  const mrspbErrF = iaComputeMRSPBFiltered(method, judges, gens, true, true, true);
  const mrspbErrFO = iaComputeMRSPBFiltered(method, judges, gens, true, true, false);
  const mipa = iaComputeMIPAFiltered(method, judges, gens);

  const rows = {{}};
  for (const judge of judges) {{
    const row = {{}};
    row.hspp = hspp.perJudge[judge];
    row.mrspbErr = mrspbErr.perJudge[judge];
    row.hsppF = hsppF.perJudge[judge];
    row.hsppFO = hsppFO.perJudge[judge];
    row.mrspbErrF = mrspbErrF.perJudge[judge];
    row.mrspbErrFO = mrspbErrFO.perJudge[judge];
    row.hsppR = hspp.perJudgeRatio[judge];
    row.mrspbErrR = mrspbErr.perJudgeRatio[judge];
    row.hsppFR = hsppF.perJudgeRatio[judge];
    row.hsppFOR = hsppFO.perJudgeRatio[judge];
    row.mrspbErrFR = mrspbErrF.perJudgeRatio[judge];
    row.mrspbErrFOR = mrspbErrFO.perJudgeRatio[judge];
    row.mipa = mipa.perJudge[judge];

    // SPA vs reference
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

    // SPA vs unfiltered
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

  const mipaBaseline = iaComputeMIPAFilteredVsBaseline(method, judges, gens);
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
  const method = iaGetMethod();
  const gens = getSelectedGens();
  const judges = effectiveJudgesList.filter(j => judgeHasData(method, j));

  if (gens.length < 2 || judges.length < 1) {{
    const noData = '<p class="no-data">Need >= 2 generators and >= 1 judge with ' + method.toUpperCase() + ' data</p>';
    document.getElementById('ia-msd-matrix').innerHTML = noData;
    document.getElementById('ia-delta-msd-matrix').innerHTML = noData;
    document.getElementById('ia-summary-table').innerHTML = noData;
    return;
  }}

  iaApplyFilters();
  iaClearInstScoreCache();

  const msdResult = iaComputeMSDMatrix(judges, gens, method);

  // Compute or use cached baseline
  const currentKey = method + '|' + gens.slice().sort().join('+') + '|' + judges.slice().sort().join('+');
  if (!iaBaselineResults || iaBaselineKey !== currentKey) {{
    const savedRM = iaRubricMask.slice();
    const savedIM = iaInstanceMask.slice();
    iaRubricMask.fill(1);
    iaInstanceMask.fill(1);
    iaClearInstScoreCache();
    iaBaselineResults = iaComputeMSDMatrix(judges, gens, method);
    iaBaselineKey = currentKey;
    iaRubricMask = savedRM;
    iaInstanceMask = savedIM;
    iaClearInstScoreCache();
  }}

  iaRenderMSDMatrix(msdResult, judges, gens, 'ia-msd-matrix');
  iaRenderDeltaMSDMatrix(msdResult, iaBaselineResults, judges, gens);

  const summary = iaComputeSummaryTable(judges, gens, method, msdResult, iaBaselineResults);
  iaCachedSummary = summary;
  iaCachedSummaryJudges = judges;
  iaRenderSummaryTable(summary, judges);
}}

// ============================================================
// INTERACTIVE ANALYSIS: FILTER UI
// ============================================================

function iaInitFilters() {{
  // Category checkboxes
  const catDiv = document.getElementById('ia-category-checkboxes');
  if (DATA.categoryNames) {{
    for (let idx = 0; idx < DATA.categoryNames.length; idx++) {{
      catDiv.innerHTML += '<label class="cb-label"><input type="checkbox" data-idx="' + idx + '" checked onchange="onIAFilterChange()"><span>' + DATA.categoryNames[idx] + '</span></label>';
    }}
  }}

  // Instruction ID checkboxes
  const iidDiv = document.getElementById('ia-instrid-checkboxes');
  if (DATA.instructionIdNames) {{
    for (let idx = 0; idx < DATA.instructionIdNames.length; idx++) {{
      iidDiv.innerHTML += '<label class="cb-label"><input type="checkbox" data-idx="' + idx + '" checked onchange="onIAFilterChange()"><span>' + DATA.instructionIdNames[idx] + '</span></label>';
    }}
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
  if (minVal > maxVal) {{
    document.getElementById('ia-agr-max').value = minVal;
    document.getElementById('ia-agr-max-val').textContent = minVal + '%';
  }}
  onIAFilterChange();
}}

function iaOnBiasModeChange() {{
  if (iaCachedSummary && iaCachedSummaryJudges) {{
    iaRenderSummaryTable(iaCachedSummary, iaCachedSummaryJudges);
  }}
}}

function iaCategoryAll() {{
  document.querySelectorAll('#ia-category-checkboxes input').forEach(cb => cb.checked = true);
  onIAFilterChange();
}}
function iaCategoryNone() {{
  document.querySelectorAll('#ia-category-checkboxes input').forEach(cb => cb.checked = false);
  onIAFilterChange();
}}
function iaInstrIdAll() {{
  document.querySelectorAll('#ia-instrid-checkboxes input').forEach(cb => cb.checked = true);
  onIAFilterChange();
}}
function iaInstrIdNone() {{
  document.querySelectorAll('#ia-instrid-checkboxes input').forEach(cb => cb.checked = false);
  onIAFilterChange();
}}

function iaResetFilters() {{
  document.getElementById('ia-len-min').value = 0;
  document.getElementById('ia-len-max').value = iaLenPercentiles.max;
  document.getElementById('ia-agr-min').value = 0;
  document.getElementById('ia-agr-max').value = 100;
  document.getElementById('ia-agr-min-val').textContent = '0%';
  document.getElementById('ia-agr-max-val').textContent = '100%';
  document.querySelector('input[name="ia-method"][value="sr"]').checked = true;
  iaCategoryAll();
  iaInstrIdAll();
}}

// Tab switching
document.querySelectorAll('.tab').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.remove('hidden');
    // Re-render charts when switching to charts tab (Plotly needs visible containers)
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
buildCommitteeMemberCheckboxes();
iaInitFilters();
onSelectionChange();
</script>

</body>
</html>"""


# CSS block (separated for readability)
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
    width: 240px;
    min-width: 240px;
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
.cb-label input[type="checkbox"] {
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

.dashboard {
    flex: 1;
    padding: 1.5rem;
    max-width: calc(100% - 240px);
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
tr.cmember-row td { font-style: italic; color: var(--sand-500); font-size: 0.85em; }
tr.cmember-row td.row-label { padding-left: 1.2em; }

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
    logger.info("Loading base data (9 default judges)...")
    all_data = load_all_data()

    logger.info("Loading extra SR data (GPT-5, Claude-Haiku, Claude-Sonnet)...")
    extra_sr = load_extra_rubric_data("EvaluateIFEval", all_data["ref"], "SR-extra")
    all_data["sr"].update(extra_sr)

    logger.info("Loading extra AR data (partial)...")
    extra_ar = load_extra_rubric_data("EvaluateIFEvalAR", all_data["ref"], "AR-extra")
    all_data["ar"].update(extra_ar)

    logger.info("Packing data...")
    packed = pack_data(all_data)
    packed_json = json.dumps(packed, separators=(',', ':'))
    logger.info(f"Packed data size: {len(packed_json):,} bytes")

    logger.info("Generating HTML...")
    html = generate_html(packed_json)

    out_path = Path(__file__).parent / "dashboard.html"
    with open(out_path, "w") as f:
        f.write(html)
    logger.info(f"Dashboard written to {out_path} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
