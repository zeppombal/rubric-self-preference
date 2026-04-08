"""
Main analysis script for HealthBench SR study.

Orchestrates data loading, reference construction, metric computation,
visualization, and reporting. Runs for both weighted and uniform scoring modes.

Adapted from the IFEval run_analysis.py but with:
- HealthBench weighted/uniform scoring
- Committee-based reference (leave-one-family-out)
- Only SR method (no AR, DA, PWC)
- Committee-as-reference exploration
"""

import sys
import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hb_data_loading import (
    load_all_data,
    GENERATORS, JUDGES, ALL_JUDGES, FAMILIES, MODEL_TO_FAMILY, N_INSTANCES,
    short,
)
from hb_scoring import instance_score, ref_instance_score
from hb_committee import (
    build_leave_one_family_out_references,
    build_committee_reference,
    enumerate_committees,
)
from hb_metrics import (
    compute_reference_system_scores,
    compute_system_scores_sr,
    _compute_instance_scores,
    _compute_ref_instance_scores,
    compute_mpa, compute_mrd, compute_msd, compute_msd_norm,
    compute_mrd_sp, compute_mrd_fsp, compute_mrd_fosp,
    compute_msd_sp, compute_msd_sp_norm,
    compute_msd_fsp, compute_msd_fsp_norm,
    compute_msd_fosp, compute_msd_fosp_norm,
    compute_per_generator_deltas, compute_per_generator_rank_deltas,
    compute_mipa,
    compute_mispb,
    compute_mra, compute_mrspb,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# Output paths
# ============================================================

SANDBOX_HB = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(SANDBOX_HB, "results", "tables")
FIGURES_DIR = os.path.join(SANDBOX_HB, "results", "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# Reference precomputation
# ============================================================


def precompute_references(ref_by_family, gen_data, scoring_mode):
    """Pre-compute per-judge reference scores (system and instance level)."""
    ref_sys_by_judge = {}
    ref_inst_by_judge = {}
    ref_data_by_judge = {}

    for judge in ALL_JUDGES:
        family = MODEL_TO_FAMILY[judge]
        ref = ref_by_family[family]
        ref_sys_by_judge[judge] = compute_reference_system_scores(ref, gen_data, scoring_mode)
        ref_inst_by_judge[judge] = _compute_ref_instance_scores(ref, gen_data, scoring_mode)
        ref_data_by_judge[judge] = ref

    return ref_sys_by_judge, ref_inst_by_judge, ref_data_by_judge


# ============================================================
# Run analysis for a single scoring mode
# ============================================================


def run_for_mode(scoring_mode, data, ref_by_family):
    """Run the full analysis pipeline for one scoring mode."""
    gen_data = data["gen"]
    sr_data = data["sr"]
    suffix = f"_{scoring_mode}"

    logger.info(f"{'='*60}")
    logger.info(f"SCORING MODE: {scoring_mode.upper()}")
    logger.info(f"{'='*60}")

    # ----------------------------------------------------------
    # 2. Pre-compute references
    # ----------------------------------------------------------
    logger.info("Pre-computing per-judge references...")
    ref_sys_by_judge, ref_inst_by_judge, ref_data_by_judge = precompute_references(
        ref_by_family, gen_data, scoring_mode
    )

    # ----------------------------------------------------------
    # 3. System-level scores
    # ----------------------------------------------------------
    logger.info("STEP 3: Computing system-level scores")

    method_system_scores = compute_system_scores_sr(sr_data, scoring_mode)

    rows = []
    for gen in GENERATORS:
        # Use mean ref score across families for display
        ref_scores_list = [ref_sys_by_judge[j][gen] for j in ALL_JUDGES if gen in ref_sys_by_judge[j]]
        ref_mean = np.mean(ref_scores_list) if ref_scores_list else float("nan")
        vals = [method_system_scores[(j, gen)] for j in JUDGES if (j, gen) in method_system_scores]
        rows.append({
            "Generator": short(gen),
            "Reference_mean": ref_mean,
            "SR_mean": np.mean(vals) if vals else float("nan"),
        })
    df_sys = pd.DataFrame(rows)
    df_sys.to_csv(os.path.join(TABLES_DIR, f"system_scores{suffix}.csv"), index=False)
    logger.info(f"System scores:\n{df_sys.to_string(index=False)}")

    # ----------------------------------------------------------
    # 4. System-level accuracy metrics (RQ1)
    # ----------------------------------------------------------
    logger.info("STEP 4: System-level accuracy metrics")

    mpa_res = compute_mpa(method_system_scores, ref_sys_by_judge)
    mrd_res = compute_mrd(method_system_scores, ref_sys_by_judge)
    msd_res = compute_msd(method_system_scores, ref_sys_by_judge)
    msd_norm_res = compute_msd_norm(method_system_scores, ref_sys_by_judge)

    accuracy_results = {
        "MPA": mpa_res["mean"],
        "MRD": mrd_res["mean"],
        "MSD": msd_res["mean"],
        "MSD-norm": msd_norm_res["mean"],
        "MPA_detail": mpa_res,
        "MRD_detail": mrd_res,
        "MSD_detail": msd_res,
        "MSD-norm_detail": msd_norm_res,
    }

    df_acc = pd.DataFrame([{
        "Method": "SR",
        "MPA": accuracy_results["MPA"],
        "MRD": accuracy_results["MRD"],
        "MSD": accuracy_results["MSD"],
        "MSD-norm": accuracy_results["MSD-norm"],
    }])
    df_acc.to_csv(os.path.join(TABLES_DIR, f"system_accuracy{suffix}.csv"), index=False)
    logger.info(f"System-level accuracy:\n{df_acc.to_string(index=False)}")

    # Per-judge MPA
    mpa_rows = []
    for judge in JUDGES:
        row = {"Judge": short(judge)}
        row["MPA"] = mpa_res["per_judge"].get(judge, float("nan"))
        row["MPA_n_concordant"] = mpa_res["per_judge_n_concordant"].get(judge, 0)
        row["MPA_n_total"] = mpa_res["per_judge_n_total"].get(judge, 0)
        mpa_rows.append(row)
    pd.DataFrame(mpa_rows).to_csv(os.path.join(TABLES_DIR, f"mpa_per_judge{suffix}.csv"), index=False)

    # ----------------------------------------------------------
    # 5. System-level bias metrics (RQ2)
    # ----------------------------------------------------------
    logger.info("STEP 5: System-level bias metrics")

    mrd_sp_res = compute_mrd_sp(method_system_scores, ref_sys_by_judge)
    mrd_fsp_res = compute_mrd_fsp(method_system_scores, ref_sys_by_judge)
    mrd_fosp_res = compute_mrd_fosp(method_system_scores, ref_sys_by_judge)
    msd_sp_res = compute_msd_sp(method_system_scores, ref_sys_by_judge)
    msd_sp_norm_res = compute_msd_sp_norm(method_system_scores, ref_sys_by_judge)
    msd_fsp_res = compute_msd_fsp(method_system_scores, ref_sys_by_judge)
    msd_fsp_norm_res = compute_msd_fsp_norm(method_system_scores, ref_sys_by_judge)
    msd_fosp_res = compute_msd_fosp(method_system_scores, ref_sys_by_judge)
    msd_fosp_norm_res = compute_msd_fosp_norm(method_system_scores, ref_sys_by_judge)

    bias_results = {
        "MRD-SP": mrd_sp_res, "MRD-FSP": mrd_fsp_res, "MRD-FOSP": mrd_fosp_res,
        "MSD-SP": msd_sp_res, "MSD-SP-norm": msd_sp_norm_res,
        "MSD-FSP": msd_fsp_res, "MSD-FSP-norm": msd_fsp_norm_res,
        "MSD-FOSP": msd_fosp_res, "MSD-FOSP-norm": msd_fosp_norm_res,
    }

    # Summary row
    bias_row = {"Method": "SR"}
    for name, res in bias_results.items():
        bias_row[name] = res["mean"]
        if "mean_d_self" in res:
            bias_row[f"{name}_d_self"] = res["mean_d_self"]
            bias_row[f"{name}_d_other"] = res["mean_d_other"]
        if "mean_d_family" in res:
            bias_row[f"{name}_d_family"] = res["mean_d_family"]
            bias_row[f"{name}_d_nonfamily"] = res["mean_d_nonfamily"]
    pd.DataFrame([bias_row]).to_csv(os.path.join(TABLES_DIR, f"system_bias{suffix}.csv"), index=False)
    logger.info("Saved system_bias")

    # Per-judge system bias
    sys_bias_pj_rows = []
    for judge in JUDGES:
        row = {"Judge": short(judge)}
        for name, res in bias_results.items():
            row[name] = res["per_judge"].get(judge, float("nan"))
            if "per_judge_d_self" in res:
                row[f"{name}_d_self"] = res["per_judge_d_self"].get(judge, float("nan"))
                row[f"{name}_d_other"] = res["per_judge_d_other"].get(judge, float("nan"))
            if "per_judge_d_family" in res:
                row[f"{name}_d_family"] = res["per_judge_d_family"].get(judge, float("nan"))
                row[f"{name}_d_nonfamily"] = res["per_judge_d_nonfamily"].get(judge, float("nan"))
        sys_bias_pj_rows.append(row)
    pd.DataFrame(sys_bias_pj_rows).to_csv(
        os.path.join(TABLES_DIR, f"system_bias_per_judge{suffix}.csv"), index=False
    )

    # Per-generator deltas
    gen_delta_rows = []
    for gen in GENERATORS:
        gen_deltas = compute_per_generator_deltas(method_system_scores, ref_sys_by_judge)
        gen_deltas_norm = compute_per_generator_deltas(method_system_scores, ref_sys_by_judge, normalize=True)
        gen_rank_deltas = compute_per_generator_rank_deltas(method_system_scores, ref_sys_by_judge)
        gen_delta_rows.append({
            "Generator": short(gen),
            "score_delta": gen_deltas.get(gen, float("nan")),
            "score_delta_norm": gen_deltas_norm.get(gen, float("nan")),
            "rank_delta": gen_rank_deltas.get(gen, float("nan")),
        })
    pd.DataFrame(gen_delta_rows).to_csv(
        os.path.join(TABLES_DIR, f"per_generator_deltas{suffix}.csv"), index=False
    )

    # ----------------------------------------------------------
    # 6. Instance-level accuracy (MIPA)
    # ----------------------------------------------------------
    logger.info("STEP 6: Instance-level accuracy (MIPA)")

    sr_inst = _compute_instance_scores(sr_data, scoring_mode)
    mipa, mipa_per, mipa_n_agree, mipa_n_total = compute_mipa(sr_inst, ref_inst_by_judge)

    pd.DataFrame([{"Method": "SR", "MIPA": mipa}]).to_csv(
        os.path.join(TABLES_DIR, f"mipa{suffix}.csv"), index=False
    )

    mipa_judge_rows = []
    for judge in JUDGES:
        mipa_judge_rows.append({
            "Judge": short(judge),
            "MIPA": mipa_per.get(judge, float("nan")),
            "n_agree": mipa_n_agree.get(judge, 0),
            "n_total": mipa_n_total.get(judge, 0),
        })
    pd.DataFrame(mipa_judge_rows).to_csv(
        os.path.join(TABLES_DIR, f"mipa_per_judge{suffix}.csv"), index=False
    )
    logger.info(f"MIPA: {mipa:.4f}")

    # ----------------------------------------------------------
    # 7. Instance-level bias (MISPB)
    # ----------------------------------------------------------
    logger.info("STEP 7: Instance-level self-preference bias (MISPB)")

    mispb = compute_mispb(sr_data, ref_inst_by_judge, scoring_mode)
    mispb_fam = compute_mispb(sr_data, ref_inst_by_judge, scoring_mode, family_mode=True)
    mispb_fo = compute_mispb(sr_data, ref_inst_by_judge, scoring_mode,
                              family_mode=True, include_self_in_family=False)
    hspp = compute_mispb(sr_data, ref_inst_by_judge, scoring_mode, error_denom=True)
    hspp_fam = compute_mispb(sr_data, ref_inst_by_judge, scoring_mode,
                              error_denom=True, family_mode=True)
    hspp_fo = compute_mispb(sr_data, ref_inst_by_judge, scoring_mode,
                             error_denom=True, family_mode=True, include_self_in_family=False)

    mispb_results = {
        "MISPB": mispb, "MISPB-F": mispb_fam, "MISPB-FO": mispb_fo,
        "HSPP": hspp, "HSPP-F": hspp_fam, "HSPP-FO": hspp_fo,
    }

    # Summary table
    mispb_row = {"Method": "SR"}
    for variant_label, v in mispb_results.items():
        mispb_row[variant_label] = v["mean"]
        mispb_row[f"{variant_label}_raw"] = v["mean_raw"]
        mispb_row[f"{variant_label}_other"] = v["mean_other"]
        mispb_row[f"{variant_label}_ratio"] = v["mean_ratio"]
        for side in ["self", "other"]:
            for sub in ["t2w", "l2w", "l2t"]:
                key = f"per_judge_n_{sub}_{side}"
                mispb_row[f"{variant_label}_n_{sub}_{side}"] = sum(v[key].values())
    pd.DataFrame([mispb_row]).to_csv(os.path.join(TABLES_DIR, f"mispb{suffix}.csv"), index=False)

    # Per-judge MISPB
    mispb_detail_rows = []
    for judge in JUDGES:
        r = mispb_results["MISPB"]
        if judge in r["per_judge"]:
            mispb_detail_rows.append({
                "Judge": short(judge),
                "MISPB": r["per_judge"][judge],
                "MISPB_raw": r["per_judge_raw"][judge],
                "MISPB_other": r["per_judge_other"][judge],
                "MISPB_ratio": r["per_judge_ratio"][judge],
                "n_overest_self": r["per_judge_n_overest_self"][judge],
                "n_total_self": r["per_judge_n_total_self"][judge],
                "n_overest_other": r["per_judge_n_overest_other"][judge],
                "n_total_other": r["per_judge_n_total_other"][judge],
                "n_t2w_self": r["per_judge_n_t2w_self"][judge],
                "n_l2w_self": r["per_judge_n_l2w_self"][judge],
                "n_l2t_self": r["per_judge_n_l2t_self"][judge],
                "n_t2w_other": r["per_judge_n_t2w_other"][judge],
                "n_l2w_other": r["per_judge_n_l2w_other"][judge],
                "n_l2t_other": r["per_judge_n_l2t_other"][judge],
            })
    pd.DataFrame(mispb_detail_rows).to_csv(
        os.path.join(TABLES_DIR, f"mispb_per_judge{suffix}.csv"), index=False
    )

    # ----------------------------------------------------------
    # 8. Rubric-level metrics
    # ----------------------------------------------------------
    logger.info("STEP 8: Rubric-level metrics (MRA, MRSPB)")

    mra_res = compute_mra(sr_data, ref_data_by_judge)
    mrspb_res = compute_mrspb(sr_data, ref_data_by_judge)
    mrspb_fam = compute_mrspb(sr_data, ref_data_by_judge, family_mode=True)
    mrspb_fo = compute_mrspb(sr_data, ref_data_by_judge, family_mode=True,
                              include_self_in_family=False)
    mrspb_err = compute_mrspb(sr_data, ref_data_by_judge, error_denom=True)
    mrspb_err_fam = compute_mrspb(sr_data, ref_data_by_judge, error_denom=True, family_mode=True)
    mrspb_err_fo = compute_mrspb(sr_data, ref_data_by_judge, error_denom=True,
                                  family_mode=True, include_self_in_family=False)

    rubric_results = {
        "MRA": mra_res,
        "MRSPB": mrspb_res, "MRSPB-F": mrspb_fam, "MRSPB-FO": mrspb_fo,
        "MRSPB-err": mrspb_err, "MRSPB-err-F": mrspb_err_fam, "MRSPB-err-FO": mrspb_err_fo,
    }

    rubric_row = {"Method": "SR", "MRA": mra_res["mean"]}
    for name in ["MRSPB", "MRSPB-F", "MRSPB-FO", "MRSPB-err", "MRSPB-err-F", "MRSPB-err-FO"]:
        r = rubric_results[name]
        rubric_row[name] = r["mean"]
        rubric_row[f"{name}_raw"] = r["mean_raw"]
        rubric_row[f"{name}_other"] = r["mean_other"]
        rubric_row[f"{name}_ratio"] = r["mean_ratio"]
    pd.DataFrame([rubric_row]).to_csv(
        os.path.join(TABLES_DIR, f"rubric_metrics{suffix}.csv"), index=False
    )

    # Per-judge MRA
    mra_rows = []
    for judge in JUDGES:
        mra_rows.append({
            "Judge": short(judge),
            "MRA": mra_res["per_judge"].get(judge, float("nan")),
            "n_correct": mra_res["per_judge_n_correct"].get(judge, 0),
            "n_total": mra_res["per_judge_n_total"].get(judge, 0),
        })
    pd.DataFrame(mra_rows).to_csv(os.path.join(TABLES_DIR, f"mra_per_judge{suffix}.csv"), index=False)

    # Per-judge MRSPB
    mrspb_detail_rows = []
    for judge in JUDGES:
        r = rubric_results["MRSPB"]
        if judge in r["per_judge"]:
            mrspb_detail_rows.append({
                "Judge": short(judge),
                "MRSPB": r["per_judge"][judge],
                "MRSPB_raw": r["per_judge_raw"][judge],
                "MRSPB_other": r["per_judge_other"][judge],
                "MRSPB_ratio": r["per_judge_ratio"][judge],
                "n_overest_self": r["per_judge_n_overest_self"][judge],
                "n_total_self": r["per_judge_n_total_self"][judge],
                "n_overest_other": r["per_judge_n_overest_other"][judge],
                "n_total_other": r["per_judge_n_total_other"][judge],
            })
    pd.DataFrame(mrspb_detail_rows).to_csv(
        os.path.join(TABLES_DIR, f"mrspb_per_judge{suffix}.csv"), index=False
    )

    # ----------------------------------------------------------
    # 9. Visualizations
    # ----------------------------------------------------------
    logger.info("STEP 9: Generating visualizations")

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # --- MISPB heatmap (per-judge) ---
    heatmap_data = []
    for judge in JUDGES:
        row = {"MISPB": mispb["per_judge"].get(judge, float("nan")),
               "MISPB-F": mispb_fam["per_judge"].get(judge, float("nan")),
               "HSPP": hspp["per_judge"].get(judge, float("nan"))}
        heatmap_data.append(row)
    df_heat = pd.DataFrame(heatmap_data, index=[short(j) for j in JUDGES])

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(df_heat, annot=True, fmt=".3f", cmap="RdYlGn_r", center=0,
                ax=ax, linewidths=0.5)
    ax.set_title(f"Self-Preference Bias per Judge ({scoring_mode})", fontsize=13, fontweight="bold")
    ax.set_ylabel("Judge")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"mispb_heatmap{suffix}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- MRSPB heatmap (per-judge) ---
    mrspb_heat_data = []
    for judge in JUDGES:
        row = {"MRSPB": mrspb_res["per_judge"].get(judge, float("nan")),
               "MRSPB-F": mrspb_fam["per_judge"].get(judge, float("nan")),
               "MRSPB-err": mrspb_err["per_judge"].get(judge, float("nan"))}
        mrspb_heat_data.append(row)
    df_mrspb_heat = pd.DataFrame(mrspb_heat_data, index=[short(j) for j in JUDGES])

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(df_mrspb_heat, annot=True, fmt=".3f", cmap="RdYlGn_r", center=0,
                ax=ax, linewidths=0.5)
    ax.set_title(f"Rubric Self-Preference Bias per Judge ({scoring_mode})", fontsize=13, fontweight="bold")
    ax.set_ylabel("Judge")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"mrspb_heatmap{suffix}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved visualizations")

    # ----------------------------------------------------------
    # 10. Committee-as-reference analysis
    # ----------------------------------------------------------
    logger.info("STEP 10: Committee-as-reference analysis")
    run_committee_as_reference(sr_data, gen_data, scoring_mode, suffix)

    return {
        "accuracy": accuracy_results,
        "bias": bias_results,
        "mipa": mipa,
        "mispb": mispb_results,
        "rubric": rubric_results,
    }


# ============================================================
# Committee-as-reference analysis
# ============================================================


def run_committee_as_reference(sr_data, gen_data, scoring_mode, suffix):
    """
    Vectorized committee-as-reference analysis.

    Pre-computes all rubric data as numpy arrays, then iterates committees
    using vectorized majority voting and metric computation.
    """
    from scipy import stats as scipy_stats

    n_judges = len(ALL_JUDGES)
    n_gens = len(GENERATORS)
    j2i = {j: i for i, j in enumerate(ALL_JUDGES)}
    g2i = {g: i for i, g in enumerate(GENERATORS)}

    # ----------------------------------------------------------
    # Phase 1: Pre-compute numpy arrays
    # ----------------------------------------------------------
    logger.info("Pre-computing numpy arrays for committee analysis...")

    # Rubric counts per instance (shared across all gens/judges)
    rubric_counts = np.array([len(gen_data[GENERATORS[0]][i]) for i in range(N_INSTANCES)])
    total_rubrics = int(rubric_counts.sum())
    cum_bounds = np.zeros(N_INSTANCES + 1, dtype=np.int64)
    cum_bounds[1:] = np.cumsum(rubric_counts)
    logger.info(f"  Total rubrics across {N_INSTANCES} instances: {total_rubrics}")

    # Points flat array (shared since rubric structure is identical across gens)
    points_flat = np.zeros(total_rubrics, dtype=np.float64)
    for i in range(N_INSTANCES):
        s = cum_bounds[i]
        for r_idx, rub in enumerate(gen_data[GENERATORS[0]][i]):
            points_flat[s + r_idx] = rub["points"]

    # Positive points total per instance (denominator, constant for weighted scoring)
    pos_total_per_inst = np.zeros(N_INSTANCES, dtype=np.float64)
    for i in range(N_INSTANCES):
        s, e = cum_bounds[i], cum_bounds[i + 1]
        pos_total_per_inst[i] = np.sum(points_flat[s:e][points_flat[s:e] > 0])

    # Uniform scoring: +1/-1 points, n_positive per instance
    uniform_points_flat = np.where(points_flat > 0, 1.0, -1.0)
    n_positive_per_inst = np.zeros(N_INSTANCES, dtype=np.float64)
    for i in range(N_INSTANCES):
        s, e = cum_bounds[i], cum_bounds[i + 1]
        n_positive_per_inst[i] = np.sum(points_flat[s:e] > 0)

    # Choose scoring arrays based on mode
    if scoring_mode == "weighted":
        score_points = points_flat
        score_denom = pos_total_per_inst
    else:
        score_points = uniform_points_flat
        score_denom = n_positive_per_inst

    # Rubric boolean arrays: (n_judges, n_gens, total_rubrics) as int8
    rubric_bools = np.zeros((n_judges, n_gens, total_rubrics), dtype=np.int8)
    for j_idx, judge in enumerate(ALL_JUDGES):
        for g_idx, gen in enumerate(GENERATORS):
            if (judge, gen) in sr_data:
                flat = []
                for inst_rubrics in sr_data[(judge, gen)]:
                    flat.extend(1 if r["criteria_met"] else 0 for r in inst_rubrics)
                rubric_bools[j_idx, g_idx] = flat

    # Pre-compute instance scores: (n_judges, n_gens, N_INSTANCES) using segment sums
    instance_scores_arr = np.zeros((n_judges, n_gens, N_INSTANCES))
    for j_idx in range(n_judges):
        for g_idx in range(n_gens):
            pw = score_points * rubric_bools[j_idx, g_idx].astype(np.float64)
            achieved = np.add.reduceat(pw, cum_bounds[:-1])
            safe_denom = np.where(score_denom > 0, score_denom, 1.0)
            instance_scores_arr[j_idx, g_idx] = np.clip(
                np.where(score_denom > 0, achieved / safe_denom, 0.0), 0, 1
            )

    # System scores: (n_judges, n_gens) = mean of instance scores
    system_scores_arr = instance_scores_arr.mean(axis=2)
    logger.info(f"  Pre-computed system scores: shape {system_scores_arr.shape}")

    # Pre-compute family info for self-preference
    judge_family_idx = np.array([list(FAMILIES.keys()).index(MODEL_TO_FAMILY[j])
                                  for j in ALL_JUDGES])
    gen_family_idx = np.array([list(FAMILIES.keys()).index(MODEL_TO_FAMILY[g])
                                for g in GENERATORS])

    # Pre-compute generator pair indices for MPA (concordance)
    gen_pairs = [(i, j) for i in range(n_gens) for j in range(i + 1, n_gens)]
    n_pairs = len(gen_pairs)
    pair_i = np.array([p[0] for p in gen_pairs])
    pair_j = np.array([p[1] for p in gen_pairs])

    # ----------------------------------------------------------
    # Phase 2: Iterate committees
    # ----------------------------------------------------------
    committees = enumerate_committees(ALL_JUDGES, min_size=2, max_size=len(ALL_JUDGES) - 2)
    logger.info(f"Processing {len(committees)} committees as references")

    accuracy_rows = []
    bias_rows = []
    sp_detail_rows = []
    rubric_rows = []

    for c_idx, (cname, members) in enumerate(committees):
        if c_idx % 500 == 0:
            logger.info(f"  Committee {c_idx + 1}/{len(committees)}: {cname}")

        member_idxs = np.array([j2i[m] for m in members])
        n_members = len(members)
        threshold = n_members / 2.0

        # Majority vote: (n_gens, total_rubrics)
        votes = rubric_bools[member_idxs].sum(axis=0)  # (n_gens, total_rubrics)
        ref_bool = (votes > threshold).astype(np.float64)

        # Reference instance scores: (n_gens, N_INSTANCES) via segment sums
        ref_inst_scores = np.zeros((n_gens, N_INSTANCES))
        for g_idx in range(n_gens):
            pw = score_points * ref_bool[g_idx]
            achieved = np.add.reduceat(pw, cum_bounds[:-1])
            safe_denom = np.where(score_denom > 0, score_denom, 1.0)
            ref_inst_scores[g_idx] = np.clip(
                np.where(score_denom > 0, achieved / safe_denom, 0.0), 0, 1
            )
        ref_sys_scores = ref_inst_scores.mean(axis=1)  # (n_gens,)

        # Eval judges = non-members
        member_set = set(member_idxs)
        eval_idxs = np.array([i for i in range(n_judges) if i not in member_set])
        if len(eval_idxs) < 1:
            continue

        # --- MPA (concordance) ---
        mpa_vals = []
        for j_idx in eval_idxs:
            j_sc = system_scores_arr[j_idx]  # (n_gens,)
            j_diffs = j_sc[pair_i] - j_sc[pair_j]
            r_diffs = ref_sys_scores[pair_i] - ref_sys_scores[pair_j]
            concordant = (
                ((j_diffs > 0) & (r_diffs > 0))
                | ((j_diffs < 0) & (r_diffs < 0))
                | ((j_diffs == 0) & (r_diffs == 0))
            )
            mpa_vals.append(concordant.sum() / n_pairs)
        mpa = np.mean(mpa_vals)

        # --- MRD (ranking difference) ---
        mrd_vals = []
        for j_idx in eval_idxs:
            j_ranks = scipy_stats.rankdata(-system_scores_arr[j_idx])
            r_ranks = scipy_stats.rankdata(-ref_sys_scores)
            mrd_vals.append(np.mean(np.abs(j_ranks - r_ranks)))
        mrd = np.mean(mrd_vals)

        # --- MSD (score delta) ---
        eval_sys = system_scores_arr[eval_idxs]  # (n_eval, n_gens)
        deltas = eval_sys - ref_sys_scores[None, :]
        msd_per_judge = deltas.mean(axis=1)
        msd = msd_per_judge.mean()

        accuracy_rows.append({
            "committee": cname,
            "n_members": n_members,
            "n_eval_judges": len(eval_idxs),
            "MPA": mpa, "MRD": mrd, "MSD": msd,
        })

        # --- MSD-SP (self-preference) ---
        sp_vals = []
        fsp_vals = []
        for j_idx in eval_idxs:
            judge_name = ALL_JUDGES[j_idx]
            g_self_idx = g2i.get(judge_name)
            if g_self_idx is None:
                continue

            j_deltas = system_scores_arr[j_idx] - ref_sys_scores
            d_self = j_deltas[g_self_idx]

            # Other gens = different family
            j_fam = judge_family_idx[j_idx]
            other_mask = gen_family_idx != j_fam
            if other_mask.sum() == 0:
                continue
            d_other = j_deltas[other_mask].mean()
            sp_vals.append(d_self - d_other)

            # Family SP
            family_mask = (gen_family_idx == j_fam)
            if family_mask.sum() > 0:
                d_family = j_deltas[family_mask].mean()
                fsp_vals.append(d_family - d_other)

            sp_detail_rows.append({
                "committee": cname,
                "eval_judge": short(judge_name),
                "MSD-SP": d_self - d_other,
            })

        msd_sp = np.mean(sp_vals) if sp_vals else float("nan")
        msd_fsp = np.mean(fsp_vals) if fsp_vals else float("nan")

        bias_rows.append({
            "committee": cname,
            "n_members": n_members,
            "MSD-SP": msd_sp,
            "MSD-FSP": msd_fsp,
        })

        # --- MRA (rubric accuracy) ---
        ref_bool_i8 = (votes > threshold).astype(np.int8)  # (n_gens, total_rubrics)
        mra_vals = []
        for j_idx in eval_idxs:
            n_correct = 0
            for g_idx in range(n_gens):
                n_correct += (rubric_bools[j_idx, g_idx] == ref_bool_i8[g_idx]).sum()
            mra_vals.append(n_correct / (n_gens * total_rubrics))
        mra = np.mean(mra_vals)

        rubric_rows.append({
            "committee": cname,
            "n_members": n_members,
            "MRA": mra,
        })

    # Save
    pd.DataFrame(accuracy_rows).to_csv(
        os.path.join(TABLES_DIR, f"committee_ref_accuracy{suffix}.csv"), index=False
    )
    pd.DataFrame(bias_rows).to_csv(
        os.path.join(TABLES_DIR, f"committee_ref_bias{suffix}.csv"), index=False
    )
    pd.DataFrame(sp_detail_rows).to_csv(
        os.path.join(TABLES_DIR, f"committee_ref_sp_detail{suffix}.csv"), index=False
    )
    pd.DataFrame(rubric_rows).to_csv(
        os.path.join(TABLES_DIR, f"committee_ref_rubric_metrics{suffix}.csv"), index=False
    )
    logger.info(f"Committee-as-reference: {len(accuracy_rows)} committees processed")


# ============================================================
# Weighted vs Uniform comparison visualization
# ============================================================


def generate_comparison_figures(results_w, results_u):
    """Generate figures comparing weighted vs uniform scoring."""
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Accuracy comparison
    metrics = ["MPA", "MRD", "MSD"]
    w_vals = [results_w["accuracy"][m] for m in metrics]
    u_vals = [results_u["accuracy"][m] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, w_vals, width, label="Weighted", color="#4c72b0")
    ax.bar(x + width/2, u_vals, width, label="Uniform", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("System Accuracy: Weighted vs Uniform Scoring", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "weighted_vs_uniform_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Bias comparison
    bias_metrics = ["MISPB", "HSPP"]
    w_bias = [results_w["mispb"][m]["mean"] for m in bias_metrics]
    u_bias = [results_u["mispb"][m]["mean"] for m in bias_metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bias_metrics))
    ax.bar(x - width/2, w_bias, width, label="Weighted", color="#4c72b0")
    ax.bar(x + width/2, u_bias, width, label="Uniform", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(bias_metrics)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_title("Self-Preference Bias: Weighted vs Uniform Scoring", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "weighted_vs_uniform_bias.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved weighted vs uniform comparison figures")


# ============================================================
# Main
# ============================================================


def main():
    # ----------------------------------------------------------
    # 1. Load all data
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Loading all data")
    logger.info("=" * 60)
    data = load_all_data()

    # ----------------------------------------------------------
    # Build leave-one-family-out references (shared across modes)
    # ----------------------------------------------------------
    logger.info("Building leave-one-family-out references...")
    ref_by_family = build_leave_one_family_out_references(data["sr"], data["gen"])

    # ----------------------------------------------------------
    # Run analysis for both scoring modes
    # ----------------------------------------------------------
    results_weighted = run_for_mode("weighted", data, ref_by_family)
    results_uniform = run_for_mode("uniform", data, ref_by_family)

    # ----------------------------------------------------------
    # Cross-mode comparison
    # ----------------------------------------------------------
    logger.info("Generating weighted vs uniform comparison figures...")
    generate_comparison_figures(results_weighted, results_uniform)

    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
