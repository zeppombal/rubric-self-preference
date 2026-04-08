#!/usr/bin/env python3
"""Compute per-rubric inter-rater pairwise agreement from HealthBench meta-eval data.

For each unique rubric, computes pairwise agreement among physician raters,
producing both a JSONL data file and a self-contained sortable HTML table.
"""

import html as html_module
import json
import logging
import os
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

# Allow importing hb_data_loading from sandbox/hb
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sandbox" / "hb"))
import hb_data_loading  # noqa: E402

logger = logging.getLogger(__name__)

META_EVAL_PATH = Path(
    "/mnt/data/jpombal/checklist-bias/experiments/ducttape_outputs/"
    "CloneSimpleEvals/Baseline.baseline/simple_evals_repo/hb_data/meta_eval.jsonl"
)

OSS_EVAL_PATH = Path(
    "/mnt/data/jpombal/checklist-bias/experiments/ducttape_outputs/"
    "CloneSimpleEvals/Baseline.baseline/simple_evals_repo/hb_data/"
    "2025-05-07-06-14-12_oss_eval.jsonl"
)

OUTPUT_DIR = Path("/mnt/data/jpombal/checklist-bias/checklist_bias/results")


def load_meta_eval(path: Path) -> list[dict[str, Any]]:
    """Load meta-eval JSONL file."""
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    logger.info("Loaded %d rows from %s", len(rows), path)
    return rows


def load_rubric_points(path: Path) -> dict[str, int]:
    """Load rubric -> points mapping from oss_eval JSONL."""
    rubric_points: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            for r in json.loads(line)["rubrics"]:
                rubric_points[r["criterion"]] = r["points"]
    logger.info("Loaded points for %d rubrics from %s", len(rubric_points), path)
    return rubric_points


def compute_pairwise_agreement(binary_labels: list[bool]) -> float:
    """Pairwise agreement: (C(n_true,2) + C(n_false,2)) / C(n,2).

    Returns 1.0 if all raters agree, 0.0 if maximally split (2 raters disagree).
    """
    n = len(binary_labels)
    if n < 2:
        return 1.0
    n_true = sum(binary_labels)
    n_false = n - n_true
    return (comb(n_true, 2) + comb(n_false, 2)) / comb(n, 2)


def compute_llm_rubric_stats(
    rubric_texts: list[str],
    gen_data: dict[str, list[list[dict]]],
    sr_data: dict[tuple[str, str], list[list[dict]]],
) -> dict[str, dict[str, float]]:
    """Compute LLM pairwise agreement per rubric (vectorized).

    For each rubric, finds all (instance_idx, rubric_idx) positions in the
    generation data, then for each position × generator collects 12 judge
    labels and computes pairwise agreement.
    """
    generators = hb_data_loading.GENERATORS
    all_judges = hb_data_loading.ALL_JUDGES
    n_judges = len(all_judges)
    total_pairs = comb(n_judges, 2)  # C(12,2) = 66

    # Build criterion -> positions mapping from first generator
    ref_gen = generators[0]
    criterion_positions: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for inst_idx, rubrics in enumerate(gen_data[ref_gen]):
        for rub_idx, r in enumerate(rubrics):
            criterion_positions[r["criterion"]].append((inst_idx, rub_idx))

    llm_stats: dict[str, dict[str, float]] = {}
    for rubric_text in rubric_texts:
        positions = criterion_positions.get(rubric_text, [])
        if not positions:
            logger.warning("No positions found for rubric: %s", rubric_text[:60])
            continue

        # Collect all judgment arrays: (n_positions * n_generators, n_judges)
        rows_list: list[list[bool]] = []
        for inst_idx, rub_idx in positions:
            for gen in generators:
                judge_labels = []
                for judge in all_judges:
                    key = (judge, gen)
                    if key not in sr_data:
                        continue
                    judge_labels.append(sr_data[key][inst_idx][rub_idx]["criteria_met"])
                if len(judge_labels) == n_judges:
                    rows_list.append(judge_labels)

        if not rows_list:
            continue

        labels = np.array(rows_list, dtype=np.int8)  # (K, 12)
        n_true = labels.sum(axis=1)
        n_false = n_judges - n_true
        agreements = (n_true * (n_true - 1) + n_false * (n_false - 1)) / (
            2 * total_pairs
        )

        llm_stats[rubric_text] = {
            "llm_mean_agreement": round(float(agreements.mean()), 4),
            "llm_min_agreement": round(float(agreements.min()), 4),
            "llm_max_agreement": round(float(agreements.max()), 4),
            "llm_support": int(len(agreements)),
        }

    logger.info("Computed LLM agreement for %d rubrics", len(llm_stats))
    return llm_stats


def compute_rubric_stats(
    rows: list[dict[str, Any]],
    rubric_points: dict[str, int],
    llm_stats: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Compute agreement statistics for each unique rubric."""
    rubric_agreements: dict[str, list[float]] = defaultdict(list)
    rubric_labels: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        rubric_agreements[row["rubric"]].append(
            compute_pairwise_agreement(row["binary_labels"])
        )
        rubric_labels[row["rubric"]].extend(row["binary_labels"])

    stats = []
    for rubric, agreements in rubric_agreements.items():
        all_labels = rubric_labels[rubric]
        n_labels = len(all_labels)
        n_true = sum(all_labels)
        entry: dict[str, Any] = {
            "rubric": rubric,
            "mean_agreement": round(sum(agreements) / len(agreements), 4),
            "min_agreement": round(min(agreements), 4),
            "max_agreement": round(max(agreements), 4),
            "pct_true": round(n_true / n_labels, 4),
            "pct_false": round((n_labels - n_true) / n_labels, 4),
            "points": rubric_points.get(rubric, 0),
            "rubric_length": len(rubric),
            "support": len(agreements),
        }
        if rubric in llm_stats:
            entry.update(llm_stats[rubric])
        stats.append(entry)

    stats.sort(key=lambda x: x["mean_agreement"])
    return stats


def write_jsonl(stats: list[dict[str, Any]], path: Path) -> None:
    """Write rubric statistics to JSONL file."""
    with open(path, "w") as f:
        for row in stats:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d lines to %s", len(stats), path)


def compute_summary_stats(stats: list[dict[str, Any]]) -> str:
    """Compute summary statistics and return an HTML section."""
    pos_rubrics = [s for s in stats if s["pct_true"] > 0.5]
    neg_rubrics = [s for s in stats if s["pct_true"] <= 0.5]

    def avg(vals: list[float]) -> str:
        return f"{sum(vals) / len(vals):.4f}" if vals else "N/A"

    def fmt_row(label: str, rubrics: list[dict[str, Any]], n_label: str) -> str:
        return f"""<tr>
            <td><strong>{label}</strong></td>
            <td class="num">{n_label}</td>
            <td class="num">{avg([r["mean_agreement"] for r in rubrics])}</td>
            <td class="num">{avg([r["min_agreement"] for r in rubrics])}</td>
            <td class="num">{avg([r["max_agreement"] for r in rubrics])}</td>
            <td class="num">{avg([r.get("llm_mean_agreement", 0.0) for r in rubrics])}</td>
            <td class="num">{avg([r.get("llm_min_agreement", 0.0) for r in rubrics])}</td>
            <td class="num">{avg([r.get("llm_max_agreement", 0.0) for r in rubrics])}</td>
            <td class="num">{avg([r["pct_true"] for r in rubrics])}</td>
            <td class="num">{avg([float(r["support"]) for r in rubrics])}</td>
        </tr>"""

    # Pearson correlations
    n = len(stats)
    x = [s["pct_true"] for s in stats]
    y = [s["mean_agreement"] for s in stats]
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / n
    std_x = (sum((xi - x_mean) ** 2 for xi in x) / n) ** 0.5
    std_y = (sum((yi - y_mean) ** 2 for yi in y) / n) ** 0.5
    corr_pct_agr = cov / (std_x * std_y) if std_x > 0 and std_y > 0 else 0

    z = [float(s["rubric_length"]) for s in stats]
    z_mean = sum(z) / n
    cov_z = sum((zi - z_mean) * (yi - y_mean) for zi, yi in zip(z, y)) / n
    std_z = (sum((zi - z_mean) ** 2 for zi in z) / n) ** 0.5
    corr_len_agr = cov_z / (std_z * std_y) if std_z > 0 and std_y > 0 else 0

    # Human vs LLM agreement: Spearman + Kendall
    human_agr = np.array([s["mean_agreement"] for s in stats])
    llm_agr = np.array([s.get("llm_mean_agreement", np.nan) for s in stats])
    has_llm = ~np.isnan(llm_agr)
    if has_llm.sum() >= 3:
        sp_rho, sp_p = scipy_stats.spearmanr(human_agr[has_llm], llm_agr[has_llm])
        kt_tau, kt_p = scipy_stats.kendalltau(human_agr[has_llm], llm_agr[has_llm])
    else:
        sp_rho = sp_p = kt_tau = kt_p = float("nan")

    most_subj = stats[0]
    least_subj = stats[-1]

    return f"""
<section class="summary-section">
    <h2>Summary Statistics</h2>

    <div class="summary-cards">
        <div class="stat-card">
            <div class="stat-value">{avg(y)}</div>
            <div class="stat-label">Human Mean Agreement</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg(llm_agr[has_llm].tolist())}</div>
            <div class="stat-label">LLM Mean Agreement</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{corr_pct_agr:+.3f}</div>
            <div class="stat-label">Pearson(% True, Human Agr.)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{corr_len_agr:+.3f}</div>
            <div class="stat-label">Pearson(Length, Human Agr.)</div>
        </div>
    </div>

    <h3>Human vs LLM Agreement Correlation</h3>
    <div class="summary-cards">
        <div class="stat-card">
            <div class="stat-value">{sp_rho:+.3f}</div>
            <div class="stat-label">Spearman &rho; (p={sp_p:.4f})</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{kt_tau:+.3f}</div>
            <div class="stat-label">Kendall &tau; (p={kt_p:.4f})</div>
        </div>
    </div>

    <h3>Agreement by Rubric Polarity</h3>
    <p class="summary-note">Rubrics split by majority label: "Mostly True" if &gt;50% of individual judgments are True, "Mostly False" otherwise.</p>
    <div class="table-card">
    <table class="summary-table">
    <thead><tr>
        <th>Group</th>
        <th class="num">Count</th>
        <th class="num">Human Mean</th>
        <th class="num">Human Min</th>
        <th class="num">Human Max</th>
        <th class="num">LLM Mean</th>
        <th class="num">LLM Min</th>
        <th class="num">LLM Max</th>
        <th class="num">Avg % True</th>
        <th class="num">Avg Support</th>
    </tr></thead>
    <tbody>
        {fmt_row("Mostly True", pos_rubrics, str(len(pos_rubrics)))}
        {fmt_row("Mostly False", neg_rubrics, str(len(neg_rubrics)))}
        {fmt_row("All Rubrics", stats, str(len(stats)))}
    </tbody>
    </table>
    </div>

    <h3>Extremes</h3>
    <div class="extremes">
        <div class="extreme-card">
            <div class="extreme-label">Most Subjective (Human)</div>
            <div class="extreme-value">Human = {most_subj["mean_agreement"]:.4f}, LLM = {most_subj.get("llm_mean_agreement", 0):.4f}</div>
            <div class="extreme-rubric">{html_module.escape(most_subj["rubric"][:150])}{"..." if len(most_subj["rubric"]) > 150 else ""}</div>
        </div>
        <div class="extreme-card">
            <div class="extreme-label">Least Subjective (Human)</div>
            <div class="extreme-value">Human = {least_subj["mean_agreement"]:.4f}, LLM = {least_subj.get("llm_mean_agreement", 0):.4f}</div>
            <div class="extreme-rubric">{html_module.escape(least_subj["rubric"][:150])}{"..." if len(least_subj["rubric"]) > 150 else ""}</div>
        </div>
    </div>
</section>"""


def generate_html(stats: list[dict[str, Any]]) -> str:
    """Generate self-contained HTML with a sortable rubric subjectivity table."""
    summary_html = compute_summary_stats(stats)

    # Prepare data for JS embedding (escape rubric text for HTML)
    js_data = []
    for s in stats:
        js_data.append(
            {
                "rubric": html_module.escape(s["rubric"]),
                "mean_agreement": s["mean_agreement"],
                "min_agreement": s["min_agreement"],
                "max_agreement": s["max_agreement"],
                "llm_mean_agreement": s.get("llm_mean_agreement", 0),
                "llm_min_agreement": s.get("llm_min_agreement", 0),
                "llm_max_agreement": s.get("llm_max_agreement", 0),
                "llm_support": s.get("llm_support", 0),
                "pct_true": s["pct_true"],
                "pct_false": s["pct_false"],
                "points": s["points"],
                "rubric_length": s["rubric_length"],
                "support": s["support"],
            }
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HealthBench Rubric Subjectivity</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400;1,700&family=Fraunces:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
<style>
:root {{
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
}}

*, *::before, *::after {{ box-sizing: border-box; }}

body {{
    font-family: 'Atkinson Hyperlegible', system-ui, sans-serif;
    background: var(--bg-page);
    color: var(--text-primary);
    margin: 0;
    padding: 0;
    line-height: 1.5;
    scrollbar-width: thin;
}}

header {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 48px 32px 24px;
}}

header h1 {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 500;
    font-size: 2rem;
    margin: 0 0 8px;
    color: var(--text-primary);
}}

header p {{
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin: 0;
    max-width: 720px;
}}

main {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 32px 64px;
}}

.table-card {{
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}}

thead th {{
    position: sticky;
    top: 0;
    z-index: 10;
    background: var(--sand-100);
    font-weight: 700;
    text-align: left;
    padding: 14px 16px;
    border-bottom: 2px solid var(--sand-300);
    white-space: nowrap;
    user-select: none;
}}

thead th.sortable {{
    cursor: pointer;
}}

thead th.sortable:hover {{
    color: var(--accent);
}}

thead th .sort-arrow {{
    display: inline-block;
    width: 16px;
    text-align: center;
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-left: 4px;
}}

thead th.sort-active .sort-arrow {{
    color: var(--accent);
}}

tbody tr {{
    border-bottom: 1px solid var(--border-subtle);
    transition: background 0.1s;
}}

tbody tr:hover {{
    background: var(--accent-soft);
}}

tbody td {{
    padding: 12px 16px;
    vertical-align: top;
}}

td.num {{
    text-align: right;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}}

td.idx {{
    text-align: center;
    color: var(--text-muted);
    font-size: 0.85rem;
    width: 48px;
}}

.rubric-cell {{
    max-width: 560px;
    min-width: 300px;
}}

.rubric-text {{
    white-space: pre-line;
    word-break: break-word;
    font-size: 0.88rem;
    line-height: 1.55;
}}

.rubric-text.collapsed {{
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}}

.expand-btn {{
    display: inline-block;
    margin-top: 6px;
    background: none;
    border: none;
    color: var(--accent);
    font-family: inherit;
    font-size: 0.82rem;
    font-weight: 700;
    cursor: pointer;
    padding: 2px 0;
}}

.expand-btn:hover {{
    text-decoration: underline;
}}

.summary-section {{
    margin-top: 48px;
}}

.summary-section h2 {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 500;
    font-size: 1.5rem;
    margin: 0 0 20px;
}}

.summary-section h3 {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 500;
    font-size: 1.15rem;
    margin: 32px 0 8px;
}}

.summary-note {{
    color: var(--text-secondary);
    font-size: 0.88rem;
    margin: 0 0 12px;
}}

.summary-cards {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}}

.stat-card {{
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 20px 28px;
    flex: 1;
    min-width: 180px;
    text-align: center;
}}

.stat-value {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--accent);
}}

.stat-label {{
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-top: 4px;
}}

.summary-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}}

.summary-table thead th {{
    position: static;
}}

.summary-table td, .summary-table th {{
    padding: 10px 16px;
}}

.summary-table tbody tr {{
    border-bottom: 1px solid var(--border-subtle);
}}

.summary-table .num {{
    text-align: right;
    font-variant-numeric: tabular-nums;
}}

.extremes {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 12px;
}}

.extreme-card {{
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 20px 24px;
    flex: 1;
    min-width: 280px;
}}

.extreme-label {{
    font-weight: 700;
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

.extreme-value {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.2rem;
    font-weight: 500;
    color: var(--accent);
    margin: 4px 0 8px;
}}

.extreme-rubric {{
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.5;
}}

footer {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 32px 32px;
    font-size: 0.8rem;
    color: var(--text-muted);
}}
</style>
</head>
<body>
<header>
    <h1>HealthBench Rubric Subjectivity</h1>
    <p>{len(stats)} rubrics from {sum(s["support"] for s in stats):,} physician-rated meta-evaluation rows.
    Pairwise agreement = fraction of rater pairs that agree on each row, averaged across all rows per rubric.
    Lower agreement indicates more subjective rubrics.</p>
</header>
<main>
<div class="table-card">
<table id="rubric-table">
<thead>
<tr>
    <th class="idx-col">#</th>
    <th>Rubric</th>
    <th class="sortable" data-col="mean_agreement">Human Mean<span class="sort-arrow"></span></th>
    <th class="sortable" data-col="min_agreement">Human Min<span class="sort-arrow"></span></th>
    <th class="sortable" data-col="max_agreement">Human Max<span class="sort-arrow"></span></th>
    <th class="sortable" data-col="llm_mean_agreement">LLM Mean<span class="sort-arrow"></span></th>
    <th class="sortable" data-col="llm_min_agreement">LLM Min<span class="sort-arrow"></span></th>
    <th class="sortable" data-col="llm_max_agreement">LLM Max<span class="sort-arrow"></span></th>
    <th class="sortable" data-col="pct_true">% True<span class="sort-arrow"></span></th>
    <th class="sortable" data-col="pct_false">% False<span class="sort-arrow"></span></th>
    <th class="sortable" data-col="points">Points<span class="sort-arrow"></span></th>
    <th class="sortable" data-col="rubric_length">Length<span class="sort-arrow"></span></th>
    <th class="sortable" data-col="support">Human N<span class="sort-arrow"></span></th>
    <th class="sortable" data-col="llm_support">LLM N<span class="sort-arrow"></span></th>
</tr>
</thead>
<tbody id="tbody"></tbody>
</table>
</div>
{summary_html}
</main>
<footer>Generated by rubric_subjectivity.py</footer>

<script>
const DATA = {json.dumps(js_data)};

let sortState = {{ col: 'mean_agreement', asc: true }};

function agreementBg(val) {{
    // Interpolate from error-bg (low agreement) to success-bg (high agreement)
    const t = Math.max(0, Math.min(1, val));
    const r = Math.round(253 + (240 - 253) * t);
    const g = Math.round(242 + (245 - 242) * t);
    const b = Math.round(242 + (242 - 242) * t);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
}}

function render() {{
    const sorted = [...DATA];
    sorted.sort((a, b) => {{
        const av = a[sortState.col], bv = b[sortState.col];
        return sortState.asc ? av - bv : bv - av;
    }});

    const tbody = document.getElementById('tbody');
    let html = '';
    sorted.forEach((row, i) => {{
        const needsExpand = row.rubric.length > 200;
        html += '<tr>';
        html += '<td class="idx">' + (i + 1) + '</td>';
        html += '<td class="rubric-cell"><div class="rubric-text' + (needsExpand ? ' collapsed' : '') + '">' + row.rubric + '</div>';
        if (needsExpand) html += '<button class="expand-btn" onclick="toggleExpand(this)">Show more</button>';
        html += '</td>';
        html += '<td class="num" style="background:' + agreementBg(row.mean_agreement) + '">' + row.mean_agreement.toFixed(4) + '</td>';
        html += '<td class="num" style="background:' + agreementBg(row.min_agreement) + '">' + row.min_agreement.toFixed(4) + '</td>';
        html += '<td class="num" style="background:' + agreementBg(row.max_agreement) + '">' + row.max_agreement.toFixed(4) + '</td>';
        html += '<td class="num" style="background:' + agreementBg(row.llm_mean_agreement) + '">' + row.llm_mean_agreement.toFixed(4) + '</td>';
        html += '<td class="num" style="background:' + agreementBg(row.llm_min_agreement) + '">' + row.llm_min_agreement.toFixed(4) + '</td>';
        html += '<td class="num" style="background:' + agreementBg(row.llm_max_agreement) + '">' + row.llm_max_agreement.toFixed(4) + '</td>';
        html += '<td class="num">' + (row.pct_true * 100).toFixed(1) + '%</td>';
        html += '<td class="num">' + (row.pct_false * 100).toFixed(1) + '%</td>';
        html += '<td class="num">' + row.points + '</td>';
        html += '<td class="num">' + row.rubric_length.toLocaleString() + '</td>';
        html += '<td class="num">' + row.support.toLocaleString() + '</td>';
        html += '<td class="num">' + row.llm_support.toLocaleString() + '</td>';
        html += '</tr>';
    }});
    tbody.innerHTML = html;

    document.querySelectorAll('th.sortable').forEach(th => {{
        const col = th.dataset.col;
        const arrow = th.querySelector('.sort-arrow');
        if (col === sortState.col) {{
            th.classList.add('sort-active');
            arrow.textContent = sortState.asc ? ' \\u25B2' : ' \\u25BC';
        }} else {{
            th.classList.remove('sort-active');
            arrow.textContent = '';
        }}
    }});
}}

function toggleExpand(btn) {{
    const textDiv = btn.previousElementSibling;
    const collapsed = textDiv.classList.toggle('collapsed');
    btn.textContent = collapsed ? 'Show more' : 'Show less';
}}

document.querySelectorAll('th.sortable').forEach(th => {{
    th.addEventListener('click', () => {{
        const col = th.dataset.col;
        if (sortState.col === col) {{
            sortState.asc = !sortState.asc;
        }} else {{
            sortState.col = col;
            sortState.asc = true;
        }}
        render();
    }});
}});

render();
</script>
</body>
</html>"""


def write_html(html_str: str, path: Path) -> None:
    """Write HTML string to file."""
    with open(path, "w") as f:
        f.write(html_str)
    logger.info("Wrote HTML to %s", path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Loading meta-eval data...")
    rows = load_meta_eval(META_EVAL_PATH)

    logger.info("Loading rubric points...")
    rubric_points = load_rubric_points(OSS_EVAL_PATH)

    # Load LLM evaluation data
    logger.info("Loading HealthBench generation + SR evaluation data...")
    hb_data = hb_data_loading.load_all_data()
    gen_data = hb_data["gen"]
    sr_data = hb_data["sr"]

    rubric_texts = list({row["rubric"] for row in rows})
    logger.info("Computing LLM agreement for %d rubrics...", len(rubric_texts))
    llm_stats = compute_llm_rubric_stats(rubric_texts, gen_data, sr_data)

    logger.info("Computing rubric agreement statistics...")
    stats = compute_rubric_stats(rows, rubric_points, llm_stats)

    jsonl_path = OUTPUT_DIR / "rubric_subjectivity.jsonl"
    write_jsonl(stats, jsonl_path)

    html_path = OUTPUT_DIR / "rubric_subjectivity.html"
    html_str = generate_html(stats)
    write_html(html_str, html_path)

    logger.info("Done. %d rubrics processed.", len(stats))


if __name__ == "__main__":
    main()
