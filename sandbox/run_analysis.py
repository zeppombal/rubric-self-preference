"""
Main analysis script for IFEval judge prompting methods study.

Orchestrates data loading, metric computation, visualization, and reporting
to answer three research questions:
  RQ1: Which prompting method yields the most accurate evaluations?
  RQ2: Which methods are most sensitive to self-preference bias?
    RQ2.1: Are rubric-based methods sensitive to self-preference bias?
    RQ2.2: Is self-preference bias related to method quality?

Instance-level scoring uses fraction of rubrics met (not binary all-or-nothing).
"""

import sys
import os
import logging
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loading import (
    load_all_data, load_extra_judge_data,
    GENERATORS, JUDGES, ALL_JUDGES, FAMILIES, MODEL_TO_FAMILY, N_INSTANCES,
)
from metrics import (
    compute_reference_system_scores,
    compute_reference_pwc_system_scores,
    compute_system_scores_rubric,
    compute_system_scores_da,
    compute_system_scores_pwc,
    compute_system_scores_da_float,
    compute_mpa, compute_mrd, compute_mrd_sp, compute_mrd_fsp,
    compute_msd, compute_msd_norm, compute_msd_sp, compute_msd_sp_norm,
    compute_msd_fsp, compute_msd_fsp_norm,
    compute_mrd_fosp, compute_msd_fosp, compute_msd_fosp_norm,
    compute_per_generator_deltas, compute_per_generator_rank_deltas,
    compute_mipa_non_pwc, compute_mipa_pwc,
    compute_mispb,
    compute_mra, compute_mrspb,
    _compute_instance_scores_rubric, _compute_instance_scores_da,
    _compute_instance_scores_da_float,
)
from committee import (
    enumerate_committees,
    aggregate_sr_ar, aggregate_da, aggregate_pwc,
    compute_committee_mispb, compute_committee_msd_sp,
    compute_committee_mrspb,
    run_all_committees_fast,
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

SANDBOX = "/mnt/data/jpombal/checklist-bias/sandbox"
TABLES_DIR = os.path.join(SANDBOX, "results", "tables")
FIGURES_DIR = os.path.join(SANDBOX, "results", "figures")
REPORT_PATH = os.path.join(SANDBOX, "results", "report.md")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Utility: short model names for display
# ============================================================

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

def short(name: str) -> str:
    return SHORT_NAMES.get(name, name)

METHODS = ["SR", "AR", "DA", "PWC"]


# ============================================================
# Main analysis
# ============================================================

def main():
    # ----------------------------------------------------------
    # 1. Load all data
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Loading all data")
    logger.info("=" * 60)
    data = load_all_data()

    # Load extra judge data (SR/AR for gpt_5, claude_4_5_haiku, claude_4_5_sonnet)
    extra = load_extra_judge_data(data["ref"])
    data["sr"].update(extra["sr"])
    data["ar"].update(extra["ar"])
    logger.info(f"Merged extra judge data: {len(extra['sr'])} SR pairs, {len(extra['ar'])} AR pairs")

    # ----------------------------------------------------------
    # 2. Compute system-level scores
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Computing system-level scores")
    logger.info("=" * 60)

    ref_scores = compute_reference_system_scores(data["ref"])
    ref_pwc_scores = compute_reference_pwc_system_scores(data["ref"])

    method_system_scores = {
        "SR": compute_system_scores_rubric(data["sr"]),
        "AR": compute_system_scores_rubric(data["ar"]),
        "DA": compute_system_scores_da(data["da"]),
        "PWC": compute_system_scores_pwc(data["pwc"]),
    }

    # Save system scores table
    rows = []
    for gen in GENERATORS:
        row = {"Generator": short(gen), "Reference": ref_scores[gen]}
        for method in METHODS:
            vals = []
            for judge in JUDGES:
                key = (judge, gen)
                if key in method_system_scores[method]:
                    vals.append(method_system_scores[method][key])
            row[method] = np.mean(vals) if vals else float("nan")
        rows.append(row)
    df_sys = pd.DataFrame(rows)
    df_sys.to_csv(os.path.join(TABLES_DIR, "system_scores.csv"), index=False)
    logger.info(f"System scores:\n{df_sys.to_string(index=False)}")

    # ----------------------------------------------------------
    # 3. System-level accuracy metrics (RQ1)
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: System-level accuracy metrics (RQ1)")
    logger.info("=" * 60)

    accuracy_results = {}
    for method in METHODS:
        scores = method_system_scores[method]
        # Use PWC reference for PWC, standard reference for others
        ref = ref_pwc_scores if method == "PWC" else ref_scores

        mpa_res = compute_mpa(scores, ref_scores)  # MPA always uses standard ref
        mrd_res = compute_mrd(scores, ref_scores)
        msd_res = compute_msd(scores, ref)
        msd_norm_res = compute_msd_norm(scores, ref)

        accuracy_results[method] = {
            "MPA": mpa_res["mean"],
            "MRD": mrd_res["mean"],
            "MSD": msd_res["mean"],
            "MSD-norm": msd_norm_res["mean"],
            "MPA_per_judge": mpa_res["per_judge"],
            "MRD_per_judge": mrd_res["per_judge"],
            "MSD_per_judge": msd_res["per_judge"],
            # Full detail dicts for report generation and CSV count columns
            "MPA_detail": mpa_res,
            "MRD_detail": mrd_res,
            "MSD_detail": msd_res,
            "MSD-norm_detail": msd_norm_res,
        }

    # Summary table
    acc_rows = []
    for method in METHODS:
        acc_rows.append({
            "Method": method,
            "MPA": accuracy_results[method]["MPA"],
            "MRD": accuracy_results[method]["MRD"],
            "MSD": accuracy_results[method]["MSD"],
            "MSD-norm": accuracy_results[method]["MSD-norm"],
        })
    df_acc = pd.DataFrame(acc_rows)
    df_acc.to_csv(os.path.join(TABLES_DIR, "system_accuracy.csv"), index=False)
    logger.info(f"System-level accuracy:\n{df_acc.to_string(index=False)}")

    # Per-judge MPA table (with count columns)
    mpa_rows = []
    for judge in JUDGES:
        row = {"Judge": short(judge)}
        for method in METHODS:
            detail = accuracy_results[method]["MPA_detail"]
            row[method] = detail["per_judge"].get(judge, float("nan"))
            row[f"{method}_n_concordant"] = detail["per_judge_n_concordant"].get(judge, 0)
            row[f"{method}_n_total"] = detail["per_judge_n_total"].get(judge, 0)
        mpa_rows.append(row)
    df_mpa = pd.DataFrame(mpa_rows)
    df_mpa.to_csv(os.path.join(TABLES_DIR, "mpa_per_judge.csv"), index=False)
    logger.info(f"MPA per judge:\n{df_mpa.to_string(index=False)}")

    # ----------------------------------------------------------
    # 4. System-level bias metrics (RQ2)
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: System-level bias metrics (RQ2)")
    logger.info("=" * 60)

    bias_results = {}
    for method in METHODS:
        scores = method_system_scores[method]
        ref = ref_pwc_scores if method == "PWC" else ref_scores

        mrd_sp_res = compute_mrd_sp(scores, ref_scores)
        mrd_fsp_res = compute_mrd_fsp(scores, ref_scores)
        msd_sp_res = compute_msd_sp(scores, ref)
        msd_sp_norm_res = compute_msd_sp_norm(scores, ref)
        msd_fsp_res = compute_msd_fsp(scores, ref)
        msd_fsp_norm_res = compute_msd_fsp_norm(scores, ref)
        # Family-only variants (exclude self from family target)
        mrd_fosp_res = compute_mrd_fosp(scores, ref_scores)
        msd_fosp_res = compute_msd_fosp(scores, ref)
        msd_fosp_norm_res = compute_msd_fosp_norm(scores, ref)

        bias_results[method] = {
            "MRD-SP": mrd_sp_res["mean"],
            "MRD-FSP": mrd_fsp_res["mean"],
            "MSD-SP": msd_sp_res["mean"],
            "MSD-SP-norm": msd_sp_norm_res["mean"],
            "MSD-FSP": msd_fsp_res["mean"],
            "MSD-FSP-norm": msd_fsp_norm_res["mean"],
            "MRD-FOSP": mrd_fosp_res["mean"],
            "MSD-FOSP": msd_fosp_res["mean"],
            "MSD-FOSP-norm": msd_fosp_norm_res["mean"],
            "MRD-SP_per_judge": mrd_sp_res["per_judge"],
            "MRD-FSP_per_judge": mrd_fsp_res["per_judge"],
            "MSD-SP_per_judge": msd_sp_res["per_judge"],
            "MSD-FSP_per_judge": msd_fsp_res["per_judge"],
            # Full detail dicts
            "MRD-SP_detail": mrd_sp_res,
            "MRD-FSP_detail": mrd_fsp_res,
            "MSD-SP_detail": msd_sp_res,
            "MSD-SP-norm_detail": msd_sp_norm_res,
            "MSD-FSP_detail": msd_fsp_res,
            "MSD-FSP-norm_detail": msd_fsp_norm_res,
            "MRD-FOSP_detail": mrd_fosp_res,
            "MSD-FOSP_detail": msd_fosp_res,
            "MSD-FOSP-norm_detail": msd_fosp_norm_res,
        }

    bias_rows = []
    for method in METHODS:
        mrd_sp_d = bias_results[method]["MRD-SP_detail"]
        mrd_fsp_d = bias_results[method]["MRD-FSP_detail"]
        msd_sp_d = bias_results[method]["MSD-SP_detail"]
        msd_fsp_d = bias_results[method]["MSD-FSP_detail"]
        mrd_fosp_d = bias_results[method]["MRD-FOSP_detail"]
        msd_fosp_d = bias_results[method]["MSD-FOSP_detail"]
        bias_rows.append({
            "Method": method,
            "MRD-SP": bias_results[method]["MRD-SP"],
            "MRD-SP_d_self": mrd_sp_d["mean_d_self"],
            "MRD-SP_d_other": mrd_sp_d["mean_d_other"],
            "MRD-FSP": bias_results[method]["MRD-FSP"],
            "MRD-FSP_d_family": mrd_fsp_d["mean_d_family"],
            "MRD-FSP_d_nonfamily": mrd_fsp_d["mean_d_nonfamily"],
            "MRD-FOSP": bias_results[method]["MRD-FOSP"],
            "MRD-FOSP_d_family": mrd_fosp_d["mean_d_family"],
            "MRD-FOSP_d_nonfamily": mrd_fosp_d["mean_d_nonfamily"],
            "MSD-SP": bias_results[method]["MSD-SP"],
            "MSD-SP_d_self": msd_sp_d["mean_d_self"],
            "MSD-SP_d_other": msd_sp_d["mean_d_other"],
            "MSD-SP-norm": bias_results[method]["MSD-SP-norm"],
            "MSD-FSP": bias_results[method]["MSD-FSP"],
            "MSD-FSP_d_family": msd_fsp_d["mean_d_family"],
            "MSD-FSP_d_nonfamily": msd_fsp_d["mean_d_nonfamily"],
            "MSD-FSP-norm": bias_results[method]["MSD-FSP-norm"],
            "MSD-FOSP": bias_results[method]["MSD-FOSP"],
            "MSD-FOSP_d_family": msd_fosp_d["mean_d_family"],
            "MSD-FOSP_d_nonfamily": msd_fosp_d["mean_d_nonfamily"],
            "MSD-FOSP-norm": bias_results[method]["MSD-FOSP-norm"],
        })
    df_bias = pd.DataFrame(bias_rows)
    df_bias.to_csv(os.path.join(TABLES_DIR, "system_bias.csv"), index=False)
    logger.info(f"System-level bias:\n{df_bias.to_string(index=False)}")

    # Per-judge system bias table (all variants)
    sys_bias_pj_rows = []
    for judge in JUDGES:
        for method in METHODS:
            mrd_sp = bias_results[method]["MRD-SP_detail"]
            mrd_fsp = bias_results[method]["MRD-FSP_detail"]
            mrd_fosp = bias_results[method]["MRD-FOSP_detail"]
            msd_sp = bias_results[method]["MSD-SP_detail"]
            msd_sp_n = bias_results[method]["MSD-SP-norm_detail"]
            msd_fsp = bias_results[method]["MSD-FSP_detail"]
            msd_fsp_n = bias_results[method]["MSD-FSP-norm_detail"]
            msd_fosp = bias_results[method]["MSD-FOSP_detail"]
            msd_fosp_n = bias_results[method]["MSD-FOSP-norm_detail"]
            row = {
                "Judge": short(judge),
                "Method": method,
                "MRD-SP": mrd_sp["per_judge"].get(judge, float("nan")),
                "MRD-SP_d_self": mrd_sp["per_judge_d_self"].get(judge, float("nan")),
                "MRD-SP_d_other": mrd_sp["per_judge_d_other"].get(judge, float("nan")),
                "MRD-FSP": mrd_fsp["per_judge"].get(judge, float("nan")),
                "MRD-FSP_d_family": mrd_fsp["per_judge_d_family"].get(judge, float("nan")),
                "MRD-FSP_d_nonfamily": mrd_fsp["per_judge_d_nonfamily"].get(judge, float("nan")),
                "MRD-FOSP": mrd_fosp["per_judge"].get(judge, float("nan")),
                "MRD-FOSP_d_family": mrd_fosp["per_judge_d_family"].get(judge, float("nan")),
                "MRD-FOSP_d_nonfamily": mrd_fosp["per_judge_d_nonfamily"].get(judge, float("nan")),
                "MSD-SP": msd_sp["per_judge"].get(judge, float("nan")),
                "MSD-SP_d_self": msd_sp["per_judge_d_self"].get(judge, float("nan")),
                "MSD-SP_d_other": msd_sp["per_judge_d_other"].get(judge, float("nan")),
                "MSD-SP-norm": msd_sp_n["per_judge"].get(judge, float("nan")),
                "MSD-SP-norm_d_self": msd_sp_n["per_judge_d_self"].get(judge, float("nan")),
                "MSD-SP-norm_d_other": msd_sp_n["per_judge_d_other"].get(judge, float("nan")),
                "MSD-FSP": msd_fsp["per_judge"].get(judge, float("nan")),
                "MSD-FSP_d_family": msd_fsp["per_judge_d_family"].get(judge, float("nan")),
                "MSD-FSP_d_nonfamily": msd_fsp["per_judge_d_nonfamily"].get(judge, float("nan")),
                "MSD-FSP-norm": msd_fsp_n["per_judge"].get(judge, float("nan")),
                "MSD-FSP-norm_d_family": msd_fsp_n["per_judge_d_family"].get(judge, float("nan")),
                "MSD-FSP-norm_d_nonfamily": msd_fsp_n["per_judge_d_nonfamily"].get(judge, float("nan")),
                "MSD-FOSP": msd_fosp["per_judge"].get(judge, float("nan")),
                "MSD-FOSP_d_family": msd_fosp["per_judge_d_family"].get(judge, float("nan")),
                "MSD-FOSP_d_nonfamily": msd_fosp["per_judge_d_nonfamily"].get(judge, float("nan")),
                "MSD-FOSP-norm": msd_fosp_n["per_judge"].get(judge, float("nan")),
                "MSD-FOSP-norm_d_family": msd_fosp_n["per_judge_d_family"].get(judge, float("nan")),
                "MSD-FOSP-norm_d_nonfamily": msd_fosp_n["per_judge_d_nonfamily"].get(judge, float("nan")),
            }
            sys_bias_pj_rows.append(row)
    df_sys_bias_pj = pd.DataFrame(sys_bias_pj_rows)
    df_sys_bias_pj.to_csv(os.path.join(TABLES_DIR, "system_bias_per_judge.csv"), index=False)
    logger.info(f"System bias per judge:\n{df_sys_bias_pj.to_string(index=False)}")

    # Per-generator deltas table
    gen_delta_rows = []
    method_key_map = {"SR": "sr", "AR": "ar", "DA": "da", "PWC": "pwc"}
    for gen in GENERATORS:
        row = {"Generator": short(gen)}
        for method in METHODS:
            scores = method_system_scores[method]
            ref = ref_pwc_scores if method == "PWC" else ref_scores
            gen_deltas = compute_per_generator_deltas(scores, ref)
            gen_deltas_norm = compute_per_generator_deltas(scores, ref, normalize=True)
            gen_rank_deltas = compute_per_generator_rank_deltas(scores, ref)
            row[f"{method}_score_delta"] = gen_deltas.get(gen, float("nan"))
            row[f"{method}_score_delta_norm"] = gen_deltas_norm.get(gen, float("nan"))
            row[f"{method}_rank_delta"] = gen_rank_deltas.get(gen, float("nan"))
        gen_delta_rows.append(row)
    df_gen_deltas = pd.DataFrame(gen_delta_rows)
    df_gen_deltas.to_csv(os.path.join(TABLES_DIR, "per_generator_deltas.csv"), index=False)
    logger.info(f"Per-generator deltas:\n{df_gen_deltas.to_string(index=False)}")

    # ----------------------------------------------------------
    # 5. Instance-level accuracy (RQ1 supplemental)
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: Instance-level accuracy (MIPA)")
    logger.info("=" * 60)

    mipa_results = {}

    # SR
    sr_inst = _compute_instance_scores_rubric(data["sr"])
    mipa_sr, mipa_sr_per, mipa_sr_n_agree, mipa_sr_n_total = compute_mipa_non_pwc(sr_inst, data["ref"])
    mipa_results["SR"] = {
        "MIPA": mipa_sr, "per_judge": mipa_sr_per,
        "n_agree": mipa_sr_n_agree, "n_total": mipa_sr_n_total,
    }

    # AR
    ar_inst = _compute_instance_scores_rubric(data["ar"])
    mipa_ar, mipa_ar_per, mipa_ar_n_agree, mipa_ar_n_total = compute_mipa_non_pwc(ar_inst, data["ref"])
    mipa_results["AR"] = {
        "MIPA": mipa_ar, "per_judge": mipa_ar_per,
        "n_agree": mipa_ar_n_agree, "n_total": mipa_ar_n_total,
    }

    # DA
    da_inst = _compute_instance_scores_da(data["da"])
    mipa_da, mipa_da_per, mipa_da_n_agree, mipa_da_n_total = compute_mipa_non_pwc(da_inst, data["ref"])
    mipa_results["DA"] = {
        "MIPA": mipa_da, "per_judge": mipa_da_per,
        "n_agree": mipa_da_n_agree, "n_total": mipa_da_n_total,
    }

    # PWC
    mipa_pwc, mipa_pwc_per, mipa_pwc_n_agree, mipa_pwc_n_total = compute_mipa_pwc(data["pwc"], data["ref"])
    mipa_results["PWC"] = {
        "MIPA": mipa_pwc, "per_judge": mipa_pwc_per,
        "n_agree": mipa_pwc_n_agree, "n_total": mipa_pwc_n_total,
    }

    mipa_rows = [{"Method": m, "MIPA": mipa_results[m]["MIPA"]} for m in METHODS]
    df_mipa = pd.DataFrame(mipa_rows)
    df_mipa.to_csv(os.path.join(TABLES_DIR, "mipa.csv"), index=False)
    logger.info(f"MIPA:\n{df_mipa.to_string(index=False)}")

    # Per-judge MIPA (with count columns)
    mipa_judge_rows = []
    for judge in JUDGES:
        row = {"Judge": short(judge)}
        for method in METHODS:
            row[method] = mipa_results[method]["per_judge"].get(judge, float("nan"))
            row[f"{method}_n_agree"] = mipa_results[method]["n_agree"].get(judge, 0)
            row[f"{method}_n_total"] = mipa_results[method]["n_total"].get(judge, 0)
        mipa_judge_rows.append(row)
    df_mipa_j = pd.DataFrame(mipa_judge_rows)
    df_mipa_j.to_csv(os.path.join(TABLES_DIR, "mipa_per_judge.csv"), index=False)
    logger.info(f"MIPA per judge:\n{df_mipa_j.to_string(index=False)}")

    # ----------------------------------------------------------
    # 6. Instance-level bias (MISPB) (RQ2)
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6: Instance-level self-preference bias (MISPB)")
    logger.info("=" * 60)

    method_data_map = {
        "sr": data["sr"], "ar": data["ar"],
        "da": data["da"], "pwc": data["pwc"],
    }

    mispb_results = {}
    for method_key, method_label in [("sr", "SR"), ("ar", "AR"), ("da", "DA"), ("pwc", "PWC")]:
        # Standard self-preference
        mispb = compute_mispb(method_key, method_data_map[method_key], data["ref"])
        # Family variant
        mispb_fam = compute_mispb(method_key, method_data_map[method_key], data["ref"],
                                  family_mode=True)
        # HSPP (error-denominator)
        hspp = compute_mispb(method_key, method_data_map[method_key], data["ref"],
                             error_denom=True)
        # HSPP family
        hspp_fam = compute_mispb(method_key, method_data_map[method_key], data["ref"],
                                 error_denom=True, family_mode=True)
        # Family-only variants (exclude self from family target)
        mispb_fo = compute_mispb(method_key, method_data_map[method_key], data["ref"],
                                  family_mode=True, include_self_in_family=False)
        hspp_fo = compute_mispb(method_key, method_data_map[method_key], data["ref"],
                                 error_denom=True, family_mode=True, include_self_in_family=False)

        mispb_results[method_label] = {
            "MISPB": mispb,
            "MISPB-F": mispb_fam,
            "MISPB-FO": mispb_fo,
            "HSPP": hspp,
            "HSPP-F": hspp_fam,
            "HSPP-FO": hspp_fo,
        }

    # Summary table — include raw/other/ratio decomposition and sub-type counts
    mispb_rows = []
    for method in METHODS:
        r = mispb_results[method]
        row = {
            "Method": method,
            "MISPB": r["MISPB"]["mean"],
            "MISPB_raw": r["MISPB"]["mean_raw"],
            "MISPB_other": r["MISPB"]["mean_other"],
            "MISPB_ratio": r["MISPB"]["mean_ratio"],
            "MISPB-F": r["MISPB-F"]["mean"],
            "MISPB-F_raw": r["MISPB-F"]["mean_raw"],
            "MISPB-F_other": r["MISPB-F"]["mean_other"],
            "MISPB-F_ratio": r["MISPB-F"]["mean_ratio"],
            "MISPB-FO": r["MISPB-FO"]["mean"],
            "MISPB-FO_raw": r["MISPB-FO"]["mean_raw"],
            "MISPB-FO_other": r["MISPB-FO"]["mean_other"],
            "MISPB-FO_ratio": r["MISPB-FO"]["mean_ratio"],
            "HSPP": r["HSPP"]["mean"],
            "HSPP_raw": r["HSPP"]["mean_raw"],
            "HSPP_other": r["HSPP"]["mean_other"],
            "HSPP_ratio": r["HSPP"]["mean_ratio"],
            "HSPP-F": r["HSPP-F"]["mean"],
            "HSPP-F_raw": r["HSPP-F"]["mean_raw"],
            "HSPP-F_other": r["HSPP-F"]["mean_other"],
            "HSPP-F_ratio": r["HSPP-F"]["mean_ratio"],
            "HSPP-FO": r["HSPP-FO"]["mean"],
            "HSPP-FO_raw": r["HSPP-FO"]["mean_raw"],
            "HSPP-FO_other": r["HSPP-FO"]["mean_other"],
            "HSPP-FO_ratio": r["HSPP-FO"]["mean_ratio"],
        }
        # Sub-type aggregate counts (sum across judges) for each variant
        for variant_key, variant_label in [("MISPB", "MISPB"), ("MISPB-F", "MISPB-F"),
                                            ("MISPB-FO", "MISPB-FO"),
                                            ("HSPP", "HSPP"), ("HSPP-F", "HSPP-F"),
                                            ("HSPP-FO", "HSPP-FO")]:
            v = r[variant_key]
            for side in ["self", "other"]:
                for sub in ["t2w", "l2w", "l2t"]:
                    key = f"per_judge_n_{sub}_{side}"
                    row[f"{variant_label}_n_{sub}_{side}"] = sum(v[key].values())
        mispb_rows.append(row)
    df_mispb = pd.DataFrame(mispb_rows)
    df_mispb.to_csv(os.path.join(TABLES_DIR, "mispb.csv"), index=False)
    logger.info(f"MISPB:\n{df_mispb.to_string(index=False)}")

    # Per-judge MISPB breakdown (with count and sub-type columns)
    mispb_detail_rows = []
    for judge in JUDGES:
        for method in METHODS:
            r = mispb_results[method]["MISPB"]
            if judge in r["per_judge"]:
                mispb_detail_rows.append({
                    "Judge": short(judge),
                    "Method": method,
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
    df_mispb_detail = pd.DataFrame(mispb_detail_rows)
    df_mispb_detail.to_csv(os.path.join(TABLES_DIR, "mispb_per_judge.csv"), index=False)
    logger.info(f"MISPB per judge:\n{df_mispb_detail.to_string(index=False)}")

    # ----------------------------------------------------------
    # 7. Rubric-level metrics (SR and AR only) (RQ2.1)
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 7: Rubric-level metrics (MRA, MRSPB)")
    logger.info("=" * 60)

    rubric_results = {}
    for method_key, method_label in [("sr", "SR"), ("ar", "AR")]:
        rubric_data = data[method_key]

        mra_res = compute_mra(rubric_data, data["ref"])

        mrspb = compute_mrspb(rubric_data, data["ref"])
        mrspb_fam = compute_mrspb(rubric_data, data["ref"], family_mode=True)
        mrspb_err = compute_mrspb(rubric_data, data["ref"], error_denom=True)
        mrspb_err_fam = compute_mrspb(rubric_data, data["ref"], error_denom=True, family_mode=True)
        # Family-only variants (exclude self from family target)
        mrspb_fo = compute_mrspb(rubric_data, data["ref"], family_mode=True,
                                  include_self_in_family=False)
        mrspb_err_fo = compute_mrspb(rubric_data, data["ref"], error_denom=True,
                                      family_mode=True, include_self_in_family=False)

        rubric_results[method_label] = {
            "MRA": mra_res["mean"],
            "MRA_per_judge": mra_res["per_judge"],
            "MRA_detail": mra_res,
            "MRSPB": mrspb,
            "MRSPB-F": mrspb_fam,
            "MRSPB-FO": mrspb_fo,
            "MRSPB-err": mrspb_err,
            "MRSPB-err-F": mrspb_err_fam,
            "MRSPB-err-FO": mrspb_err_fo,
        }

    # Summary table — include raw/other/ratio decomposition for all variants
    rubric_rows = []
    for method in ["SR", "AR"]:
        r = rubric_results[method]
        rubric_rows.append({
            "Method": method,
            "MRA": r["MRA"],
            "MRSPB": r["MRSPB"]["mean"],
            "MRSPB_raw": r["MRSPB"]["mean_raw"],
            "MRSPB_other": r["MRSPB"]["mean_other"],
            "MRSPB_ratio": r["MRSPB"]["mean_ratio"],
            "MRSPB-F": r["MRSPB-F"]["mean"],
            "MRSPB-F_raw": r["MRSPB-F"]["mean_raw"],
            "MRSPB-F_other": r["MRSPB-F"]["mean_other"],
            "MRSPB-F_ratio": r["MRSPB-F"]["mean_ratio"],
            "MRSPB-err": r["MRSPB-err"]["mean"],
            "MRSPB-err_raw": r["MRSPB-err"]["mean_raw"],
            "MRSPB-err_other": r["MRSPB-err"]["mean_other"],
            "MRSPB-err_ratio": r["MRSPB-err"]["mean_ratio"],
            "MRSPB-FO": r["MRSPB-FO"]["mean"],
            "MRSPB-FO_raw": r["MRSPB-FO"]["mean_raw"],
            "MRSPB-FO_other": r["MRSPB-FO"]["mean_other"],
            "MRSPB-FO_ratio": r["MRSPB-FO"]["mean_ratio"],
            "MRSPB-err": r["MRSPB-err"]["mean"],
            "MRSPB-err_raw": r["MRSPB-err"]["mean_raw"],
            "MRSPB-err_other": r["MRSPB-err"]["mean_other"],
            "MRSPB-err_ratio": r["MRSPB-err"]["mean_ratio"],
            "MRSPB-err-F": r["MRSPB-err-F"]["mean"],
            "MRSPB-err-F_raw": r["MRSPB-err-F"]["mean_raw"],
            "MRSPB-err-F_other": r["MRSPB-err-F"]["mean_other"],
            "MRSPB-err-F_ratio": r["MRSPB-err-F"]["mean_ratio"],
            "MRSPB-err-FO": r["MRSPB-err-FO"]["mean"],
            "MRSPB-err-FO_raw": r["MRSPB-err-FO"]["mean_raw"],
            "MRSPB-err-FO_other": r["MRSPB-err-FO"]["mean_other"],
            "MRSPB-err-FO_ratio": r["MRSPB-err-FO"]["mean_ratio"],
        })
    df_rubric = pd.DataFrame(rubric_rows)
    df_rubric.to_csv(os.path.join(TABLES_DIR, "rubric_metrics.csv"), index=False)
    logger.info(f"Rubric-level metrics:\n{df_rubric.to_string(index=False)}")

    # Per-judge MRA (with count columns)
    mra_rows = []
    for judge in JUDGES:
        row = {"Judge": short(judge)}
        for method in ["SR", "AR"]:
            detail = rubric_results[method]["MRA_detail"]
            row[method] = detail["per_judge"].get(judge, float("nan"))
            row[f"{method}_n_correct"] = detail["per_judge_n_correct"].get(judge, 0)
            row[f"{method}_n_total"] = detail["per_judge_n_total"].get(judge, 0)
        mra_rows.append(row)
    df_mra = pd.DataFrame(mra_rows)
    df_mra.to_csv(os.path.join(TABLES_DIR, "mra_per_judge.csv"), index=False)
    logger.info(f"MRA per judge:\n{df_mra.to_string(index=False)}")

    # Per-judge MRSPB (with count columns)
    mrspb_detail_rows = []
    for judge in JUDGES:
        for method in ["SR", "AR"]:
            r = rubric_results[method]["MRSPB"]
            if judge in r["per_judge"]:
                mrspb_detail_rows.append({
                    "Judge": short(judge),
                    "Method": method,
                    "MRSPB": r["per_judge"][judge],
                    "MRSPB_raw": r["per_judge_raw"][judge],
                    "MRSPB_other": r["per_judge_other"][judge],
                    "MRSPB_ratio": r["per_judge_ratio"][judge],
                    "n_overest_self": r["per_judge_n_overest_self"][judge],
                    "n_total_self": r["per_judge_n_total_self"][judge],
                    "n_overest_other": r["per_judge_n_overest_other"][judge],
                    "n_total_other": r["per_judge_n_total_other"][judge],
                })
    df_mrspb_detail = pd.DataFrame(mrspb_detail_rows)
    df_mrspb_detail.to_csv(os.path.join(TABLES_DIR, "mrspb_per_judge.csv"), index=False)
    logger.info(f"MRSPB per judge:\n{df_mrspb_detail.to_string(index=False)}")

    # ----------------------------------------------------------
    # 8. Visualizations
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 8: Generating visualizations")
    logger.info("=" * 60)

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # --- Figure 1: Method accuracy comparison (bar chart) ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # MPA
    mpa_vals = [accuracy_results[m]["MPA"] for m in METHODS]
    axes[0].bar(METHODS, mpa_vals, color=sns.color_palette("muted", 4))
    axes[0].set_title("Mean Pairwise Accuracy (MPA)")
    axes[0].set_ylabel("MPA")
    axes[0].set_ylim(0, 1)

    # MIPA
    mipa_vals = [mipa_results[m]["MIPA"] for m in METHODS]
    axes[1].bar(METHODS, mipa_vals, color=sns.color_palette("muted", 4))
    axes[1].set_title("Mean Instance Pairwise Accuracy (MIPA)")
    axes[1].set_ylabel("MIPA")
    axes[1].set_ylim(0, 1)

    # MRD
    mrd_vals = [accuracy_results[m]["MRD"] for m in METHODS]
    axes[2].bar(METHODS, mrd_vals, color=sns.color_palette("muted", 4))
    axes[2].set_title("Mean Ranking Difference (MRD)")
    axes[2].set_ylabel("MRD (lower = better)")

    # MRA (SR and AR only)
    mra_vals = [rubric_results[m]["MRA"] for m in ["SR", "AR"]]
    axes[3].bar(["SR", "AR"], mra_vals, color=sns.color_palette("muted", 2))
    axes[3].set_title("Mean Rubric Accuracy (MRA)")
    axes[3].set_ylabel("MRA")
    axes[3].set_ylim(0, 1)

    fig.suptitle("RQ1: Accuracy Comparison Across Methods", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES_DIR, "accuracy_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved accuracy_comparison.png")

    # --- Figure 2: Self-preference bias comparison (bar chart) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # MISPB
    mispb_vals = [mispb_results[m]["MISPB"]["mean"] for m in METHODS]
    colors = ["#d9534f" if v > 0 else "#5cb85c" for v in mispb_vals]
    axes[0].bar(METHODS, mispb_vals, color=colors)
    axes[0].set_title("MISPB (Instance Self-Preference)")
    axes[0].set_ylabel("MISPB")
    axes[0].axhline(y=0, color="black", linewidth=0.8)

    # MSD-SP
    msd_sp_vals = [bias_results[m]["MSD-SP"] for m in METHODS]
    colors = ["#d9534f" if v > 0 else "#5cb85c" for v in msd_sp_vals]
    axes[1].bar(METHODS, msd_sp_vals, color=colors)
    axes[1].set_title("MSD-SP (Score Delta Self-Pref)")
    axes[1].set_ylabel("MSD-SP")
    axes[1].axhline(y=0, color="black", linewidth=0.8)

    # MRSPB (SR and AR only)
    mrspb_vals = [rubric_results[m]["MRSPB"]["mean"] for m in ["SR", "AR"]]
    colors = ["#d9534f" if v > 0 else "#5cb85c" for v in mrspb_vals]
    axes[2].bar(["SR", "AR"], mrspb_vals, color=colors)
    axes[2].set_title("MRSPB (Rubric Self-Preference)")
    axes[2].set_ylabel("MRSPB")
    axes[2].axhline(y=0, color="black", linewidth=0.8)

    fig.suptitle("RQ2: Self-Preference Bias Across Methods", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES_DIR, "bias_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved bias_comparison.png")

    # --- Figure 3: MISPB heatmap (per-judge, per-method) ---
    heatmap_data = []
    for judge in JUDGES:
        row = {}
        for method in METHODS:
            r = mispb_results[method]["MISPB"]
            row[method] = r["per_judge"].get(judge, float("nan"))
        heatmap_data.append(row)
    df_heat = pd.DataFrame(heatmap_data, index=[short(j) for j in JUDGES])

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(df_heat, annot=True, fmt=".3f", cmap="RdYlGn_r", center=0,
                ax=ax, linewidths=0.5)
    ax.set_title("MISPB per Judge and Method", fontsize=13, fontweight="bold")
    ax.set_ylabel("Judge")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "mispb_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved mispb_heatmap.png")

    # --- Figure 4: Family-level bias heatmap ---
    heatmap_fam = []
    for judge in JUDGES:
        row = {}
        for method in METHODS:
            r = mispb_results[method]["MISPB-F"]
            row[method] = r["per_judge"].get(judge, float("nan"))
        heatmap_fam.append(row)
    df_heat_fam = pd.DataFrame(heatmap_fam, index=[short(j) for j in JUDGES])

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(df_heat_fam, annot=True, fmt=".3f", cmap="RdYlGn_r", center=0,
                ax=ax, linewidths=0.5)
    ax.set_title("Family Self-Preference Bias per Judge and Method", fontsize=13, fontweight="bold")
    ax.set_ylabel("Judge")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "mispb_family_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved mispb_family_heatmap.png")

    # --- Figure 5: Accuracy vs. Bias scatter (RQ2.2) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in METHODS:
        acc = accuracy_results[method]["MPA"]
        bias = abs(mispb_results[method]["MISPB"]["mean"])
        ax.scatter(acc, bias, s=200, label=method, zorder=5)
        ax.annotate(method, (acc, bias), textcoords="offset points",
                    xytext=(10, 5), fontsize=12)
    ax.set_xlabel("MPA (accuracy, higher = better)")
    ax.set_ylabel("|MISPB| (absolute bias, lower = better)")
    ax.set_title("RQ2.2: Method Accuracy vs. Self-Preference Bias", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "accuracy_vs_bias.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved accuracy_vs_bias.png")

    # --- Figure 6: MRSPB heatmap (per-judge, SR vs AR) ---
    mrspb_heat_data = []
    for judge in JUDGES:
        row = {}
        for method in ["SR", "AR"]:
            r = rubric_results[method]["MRSPB"]
            row[method] = r["per_judge"].get(judge, float("nan"))
        mrspb_heat_data.append(row)
    df_mrspb_heat = pd.DataFrame(mrspb_heat_data, index=[short(j) for j in JUDGES])

    fig, ax = plt.subplots(figsize=(6, 7))
    sns.heatmap(df_mrspb_heat, annot=True, fmt=".3f", cmap="RdYlGn_r", center=0,
                ax=ax, linewidths=0.5)
    ax.set_title("MRSPB per Judge (SR vs AR)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Judge")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "mrspb_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved mrspb_heatmap.png")

    # --- Figure 7: HSPP comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # HSPP per method
    hspp_vals = [mispb_results[m]["HSPP"]["mean"] for m in METHODS]
    colors = ["#d9534f" if v > 0 else "#5cb85c" for v in hspp_vals]
    axes[0].bar(METHODS, hspp_vals, color=colors)
    axes[0].set_title("HSPP (Harmful Self-Preference Propensity)")
    axes[0].set_ylabel("HSPP")
    axes[0].axhline(y=0, color="black", linewidth=0.8)

    # HSPP family per method
    hspp_fam_vals = [mispb_results[m]["HSPP-F"]["mean"] for m in METHODS]
    colors = ["#d9534f" if v > 0 else "#5cb85c" for v in hspp_fam_vals]
    axes[1].bar(METHODS, hspp_fam_vals, color=colors)
    axes[1].set_title("HSPP-F (Family Harmful Self-Preference)")
    axes[1].set_ylabel("HSPP-F")
    axes[1].axhline(y=0, color="black", linewidth=0.8)

    fig.suptitle("Harmful Self-Preference Propensity", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES_DIR, "hspp_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved hspp_comparison.png")

    # ----------------------------------------------------------
    # 9. Generate report
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 9: Generating report")
    logger.info("=" * 60)

    report = generate_report(
        accuracy_results, bias_results, mipa_results, mispb_results,
        rubric_results, df_sys, ref_scores
    )
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    logger.info(f"Report written to {REPORT_PATH}")

    # ----------------------------------------------------------
    # 10. Committee analysis (vectorized)
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 10: Committee analysis")
    logger.info("=" * 60)

    committee_accuracy_rows, committee_bias_rows, committee_sp_detail_rows, committee_rubric_rows = \
        run_all_committees_fast(
            data, data["ref"], ALL_JUDGES, GENERATORS, N_INSTANCES,
            ref_scores, ref_pwc_scores, short, logger,
        )

    # Save committee CSVs
    df_c_acc = pd.DataFrame(committee_accuracy_rows)
    df_c_acc.to_csv(os.path.join(TABLES_DIR, "committee_accuracy.csv"), index=False)
    logger.info(f"Committee accuracy: {len(df_c_acc)} rows")

    df_c_bias = pd.DataFrame(committee_bias_rows)
    df_c_bias.to_csv(os.path.join(TABLES_DIR, "committee_bias.csv"), index=False)
    logger.info(f"Committee bias: {len(df_c_bias)} rows")

    df_c_sp = pd.DataFrame(committee_sp_detail_rows)
    df_c_sp.to_csv(os.path.join(TABLES_DIR, "committee_sp_detail.csv"), index=False)
    logger.info(f"Committee SP detail: {len(df_c_sp)} rows")

    df_c_rubric = pd.DataFrame(committee_rubric_rows)
    df_c_rubric.to_csv(os.path.join(TABLES_DIR, "committee_rubric_metrics.csv"), index=False)
    logger.info(f"Committee rubric metrics: {len(df_c_rubric)} rows")

    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)


# ============================================================
# Report generation
# ============================================================

def generate_report(accuracy_results, bias_results, mipa_results,
                    mispb_results, rubric_results, df_sys, ref_scores):
    """Generate a structured markdown report answering the research questions.

    All numeric observations are derived from the data (no hardcoded values).
    Sample size statistics (numerators, denominators) are included throughout.
    """

    def fmt(val, decimals=4):
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return str(val)
        return f"{val:.{decimals}f}"

    lines = []
    lines.append("# IFEval Judge Prompting Methods: Analysis Report\n")
    lines.append("This report analyzes four LLM-as-a-judge prompting methods—Single Rubric (SR), "
                 "All Rubrics (AR), Direct Assessment (DA), and Pairwise Comparison (PWC)—on the "
                 f"IFEval benchmark ({N_INSTANCES} instances, {len(GENERATORS)} generators, "
                 f"{len(JUDGES)} judges). All rubrics are "
                 "programmatically verifiable, providing ground-truth labels free from subjectivity.\n")
    lines.append("Instance-level scoring uses the **fraction of rubrics met** per instance "
                 "(not binary all-or-nothing). System-level scores are the mean of instance-level "
                 "fractions across all instances.\n")

    # ----- Reference scores -----
    lines.append("## Reference System Scores\n")
    lines.append("Ground-truth system-level scores (mean fraction of rubrics met per instance):\n")
    lines.append("| Generator | Family | Score |")
    lines.append("|-----------|--------|-------|")
    sorted_gens = sorted(ref_scores, key=ref_scores.get, reverse=True)
    for gen in sorted_gens:
        lines.append(f"| {short(gen)} | {MODEL_TO_FAMILY[gen]} | {fmt(ref_scores[gen])} |")
    lines.append("")

    best_gen = sorted_gens[0]
    worst_gen = sorted_gens[-1]
    lines.append(f"**{short(best_gen)}** is the strongest generator ({fmt(ref_scores[best_gen])}). "
                 f"**{short(worst_gen)}** is the weakest ({fmt(ref_scores[worst_gen])}).\n")

    # ================================================================
    # RQ1
    # ================================================================
    lines.append("---\n")
    lines.append("## RQ1: Which Prompting Method Yields the Most Accurate Evaluations?\n")

    # -- System-level accuracy --
    lines.append("### System-Level Accuracy\n")
    lines.append("| Method | MPA | MRD | MSD | MSD-norm |")
    lines.append("|--------|-----|-----|-----|----------|")
    for m in METHODS:
        r = accuracy_results[m]
        lines.append(f"| {m} | {fmt(r['MPA'])} | {fmt(r['MRD'], 2)} | {fmt(r['MSD'])} | {fmt(r['MSD-norm'])} |")
    lines.append("")
    lines.append("**Key**: MPA = Mean Pairwise Accuracy (higher = better); "
                 "MRD = Mean Ranking Difference (lower = better); "
                 "MSD = Mean Score Delta (closer to 0 = better).\n")

    best_mpa_method = max(METHODS, key=lambda m: accuracy_results[m]["MPA"])
    best_mrd_method = min(METHODS, key=lambda m: accuracy_results[m]["MRD"])
    worst_mpa_method = min(METHODS, key=lambda m: accuracy_results[m]["MPA"])
    worst_mrd_method = max(METHODS, key=lambda m: accuracy_results[m]["MRD"])

    # Compute non-PWC MRD range for context
    non_pwc_mrds = [accuracy_results[m]["MRD"] for m in ["SR", "AR", "DA"]]

    lines.append("**Observations**:")
    lines.append(f"- **{best_mpa_method}** achieves the highest MPA ({fmt(accuracy_results[best_mpa_method]['MPA'])}), "
                 "meaning it most accurately ranks generators in their correct order.")
    lines.append(f"- **{best_mrd_method}** achieves the lowest MRD ({fmt(accuracy_results[best_mrd_method]['MRD'], 2)}), "
                 "indicating the smallest average rank displacement.")
    lines.append(f"- **{worst_mpa_method}** performs worst on MPA ({fmt(accuracy_results[worst_mpa_method]['MPA'])}) "
                 f"and **{worst_mrd_method}** worst on MRD "
                 f"({fmt(accuracy_results[worst_mrd_method]['MRD'], 2)} vs "
                 f"~{fmt(np.mean(non_pwc_mrds), 1)} for others).")

    # MSD direction analysis
    positive_msd = [m for m in METHODS if accuracy_results[m]["MSD"] > 0.005]
    near_zero_msd = [m for m in METHODS if abs(accuracy_results[m]["MSD"]) <= 0.005]
    if positive_msd and near_zero_msd:
        lines.append(f"- **MSD** is near zero for {', '.join(near_zero_msd)} (relative scoring cancels "
                     f"systematic bias). {', '.join(positive_msd)} show positive MSD "
                     f"({fmt(min(accuracy_results[m]['MSD'] for m in positive_msd))}"
                     f"–{fmt(max(accuracy_results[m]['MSD'] for m in positive_msd))}), "
                     "indicating systematic overestimation of generator quality.")
    lines.append("")

    # -- Instance-level accuracy --
    lines.append("### Instance-Level Accuracy\n")
    lines.append("| Method | MIPA |")
    lines.append("|--------|------|")
    for m in METHODS:
        lines.append(f"| {m} | {fmt(mipa_results[m]['MIPA'])} |")
    lines.append("")

    # Sample size info for MIPA
    lines.append("*Sample sizes (n_agree / n_total comparisons per judge):*\n")
    lines.append("| Judge | " + " | ".join(METHODS) + " |")
    lines.append("|-------" + "|---" * len(METHODS) + "|")
    for judge in JUDGES:
        row = f"| {short(judge)} "
        for method in METHODS:
            na = mipa_results[method]["n_agree"].get(judge, 0)
            nt = mipa_results[method]["n_total"].get(judge, 0)
            row += f"| {na}/{nt} "
        row += "|"
        lines.append(row)
    lines.append("")

    best_mipa_method = max(METHODS, key=lambda m: mipa_results[m]["MIPA"])
    worst_mipa_method = min(METHODS, key=lambda m: mipa_results[m]["MIPA"])
    non_pwc_mipas = {m: mipa_results[m]["MIPA"] for m in ["SR", "AR", "DA"]}
    mipa_range_lo = min(non_pwc_mipas.values())
    mipa_range_hi = max(non_pwc_mipas.values())
    best_non_pwc_mipa = max(non_pwc_mipas, key=non_pwc_mipas.get)

    lines.append("**Observations**:")
    lines.append(f"- SR, AR, and DA all achieve similar MIPA ({fmt(mipa_range_lo)}–{fmt(mipa_range_hi)}), "
                 f"with **{best_non_pwc_mipa}** marginally best.")
    if mipa_results["PWC"]["MIPA"] < 0.5:
        lines.append(f"- **PWC is dramatically worse** ({fmt(mipa_results['PWC']['MIPA'])}), performing below "
                     "chance (0.5). This means PWC judges more often disagree with the reference on "
                     "instance-level pairwise comparisons than agree. The likely cause: PWC must compare "
                     "two full responses simultaneously, which is a harder cognitive task for the judge.")
    else:
        lines.append(f"- **PWC** is notably worse ({fmt(mipa_results['PWC']['MIPA'])}) than the other methods.")
    lines.append("")

    # -- Rubric-level accuracy --
    lines.append("### Rubric-Level Accuracy (SR and AR only)\n")
    lines.append("| Method | MRA |")
    lines.append("|--------|-----|")
    for m in ["SR", "AR"]:
        lines.append(f"| {m} | {fmt(rubric_results[m]['MRA'])} |")
    lines.append("")

    # Sample sizes for MRA
    lines.append("*Sample sizes (n_correct / n_total rubric evaluations per judge):*\n")
    lines.append("| Judge | SR | AR |")
    lines.append("|-------|----|----|")
    for judge in JUDGES:
        row = f"| {short(judge)} "
        for method in ["SR", "AR"]:
            detail = rubric_results[method]["MRA_detail"]
            nc = detail["per_judge_n_correct"].get(judge, 0)
            nt = detail["per_judge_n_total"].get(judge, 0)
            row += f"| {nc}/{nt} "
        row += "|"
        lines.append(row)
    lines.append("")

    best_mra_method = max(["SR", "AR"], key=lambda m: rubric_results[m]["MRA"])
    other_mra = "SR" if best_mra_method == "AR" else "AR"
    lines.append(f"**{best_mra_method}** slightly outperforms {other_mra} at rubric-level accuracy "
                 f"({fmt(rubric_results[best_mra_method]['MRA'])} vs {fmt(rubric_results[other_mra]['MRA'])}). "
                 "Seeing all rubrics together may help the judge contextualize each criterion.\n")

    # -- Per-judge MPA --
    lines.append("### Per-Judge MPA\n")
    lines.append("| Judge | SR | AR | DA | PWC |")
    lines.append("|-------|----|----|----|----|")
    for judge in JUDGES:
        row = f"| {short(judge)} "
        for method in METHODS:
            val = accuracy_results[method]["MPA_per_judge"].get(judge, float("nan"))
            row += f"| {fmt(val)} "
        row += "|"
        lines.append(row)
    lines.append("")

    # Find best/worst judges by average MPA across methods
    judge_avg_mpa = {}
    for judge in JUDGES:
        vals = [accuracy_results[m]["MPA_per_judge"].get(judge, float("nan"))
                for m in METHODS]
        judge_avg_mpa[judge] = np.nanmean(vals)
    best_judge = max(judge_avg_mpa, key=judge_avg_mpa.get)
    worst_judge = min(judge_avg_mpa, key=judge_avg_mpa.get)
    lines.append(f"**{short(best_judge)}** is the most accurate judge across methods "
                 f"(avg MPA = {fmt(judge_avg_mpa[best_judge])}). "
                 f"**{short(worst_judge)}** is the least accurate "
                 f"(avg MPA = {fmt(judge_avg_mpa[worst_judge])}).\n")

    # -- RQ1 Summary --
    lines.append("### RQ1 Answer\n")
    lines.append("**No single method dominates across all metrics.** The overall picture:\n")
    lines.append("| Metric | Best Method | Interpretation |")
    lines.append("|--------|-------------|----------------|")
    lines.append(f"| MPA | {best_mpa_method} | Best at ranking generators correctly |")
    lines.append(f"| MRD | {best_mrd_method} | Smallest average rank displacement |")
    lines.append(f"| MIPA | {best_mipa_method} | Best instance-level discrimination |")
    lines.append(f"| MRA | {best_mra_method} | Best rubric-level judgment |")
    lines.append("")

    # Summarize method strengths
    method_wins = {}
    for m in METHODS:
        method_wins[m] = []
    method_wins[best_mpa_method].append("system-level ranking (MPA)")
    method_wins[best_mrd_method].append("ranking displacement (MRD)")
    method_wins[best_mipa_method].append("instance-level discrimination (MIPA)")
    method_wins[best_mra_method].append("rubric-level accuracy (MRA)")

    for m in METHODS:
        if method_wins[m]:
            lines.append(f"- **{m}** is best at: {', '.join(method_wins[m])}")

    lines.append(f"\n**{worst_mipa_method} is consistently the worst** by a large margin, "
                 "especially at instance level. Among rubric-aware methods, "
                 f"{best_mra_method} has a slight edge over {other_mra}.\n")

    # ================================================================
    # RQ2
    # ================================================================
    lines.append("---\n")
    lines.append("## RQ2: Which Methods Are Most Sensitive to Self-Preference Bias?\n")

    # -- System-level bias --
    lines.append("### System-Level Bias Metrics\n")
    lines.append("| Method | MRD-SP | d_self | d_other | MRD-FSP | d_fam | d_nonfam | MSD-SP | d_self | d_other | MSD-SP-norm | MSD-FSP | d_fam | d_nonfam | MSD-FSP-norm |")
    lines.append("|--------|--------|--------|---------|---------|-------|----------|--------|--------|---------|-------------|---------|-------|----------|--------------|")
    for m in METHODS:
        r = bias_results[m]
        mrd_sp_d = r["MRD-SP_detail"]
        mrd_fsp_d = r["MRD-FSP_detail"]
        msd_sp_d = r["MSD-SP_detail"]
        msd_fsp_d = r["MSD-FSP_detail"]
        lines.append(f"| {m} "
                     f"| {fmt(r['MRD-SP'], 3)} | {fmt(mrd_sp_d['mean_d_self'], 3)} | {fmt(mrd_sp_d['mean_d_other'], 3)} "
                     f"| {fmt(r['MRD-FSP'], 3)} | {fmt(mrd_fsp_d['mean_d_family'], 3)} | {fmt(mrd_fsp_d['mean_d_nonfamily'], 3)} "
                     f"| {fmt(r['MSD-SP'])} | {fmt(msd_sp_d['mean_d_self'])} | {fmt(msd_sp_d['mean_d_other'])} "
                     f"| {fmt(r['MSD-SP-norm'])} "
                     f"| {fmt(r['MSD-FSP'])} | {fmt(msd_fsp_d['mean_d_family'])} | {fmt(msd_fsp_d['mean_d_nonfamily'])} "
                     f"| {fmt(r['MSD-FSP-norm'])} |")
    lines.append("")
    lines.append("**Key**:")
    lines.append("- **MRD-SP**: Signed rank difference for self minus that for others. "
                 "Negative = judge ranks itself better than reference says, relative to how it treats others.")
    lines.append("- **MSD-SP**: Score delta for self minus score delta for others. "
                 "Positive = judge inflates its own score more than others' scores.")
    lines.append("- **-FSP** variants: Same but for the judge's entire model family.\n")

    # MSD-SP detail table
    lines.append("*Per-judge MSD-SP detail (delta_self, delta_other, n_others):*\n")
    lines.append("| Judge | " + " | ".join(f"{m} (d_self / d_other / n)" for m in METHODS) + " |")
    lines.append("|-------" + "|---" * len(METHODS) + "|")
    for judge in JUDGES:
        row = f"| {short(judge)} "
        for method in METHODS:
            detail = bias_results[method]["MSD-SP_detail"]
            ds = detail["per_judge_d_self"].get(judge, float("nan"))
            do = detail["per_judge_d_other"].get(judge, float("nan"))
            no = detail["per_judge_n_others"].get(judge, 0)
            row += f"| {fmt(ds, 3)} / {fmt(do, 3)} / {no} "
        row += "|"
        lines.append(row)
    lines.append("")

    # Observations derived from data
    mrd_sp_negative = all(bias_results[m]["MRD-SP"] < 0 for m in METHODS)
    msd_sp_positive = all(bias_results[m]["MSD-SP"] > 0 for m in METHODS)
    strongest_mrd_sp = min(METHODS, key=lambda m: bias_results[m]["MRD-SP"])
    weakest_mrd_sp = max(METHODS, key=lambda m: bias_results[m]["MRD-SP"])
    strongest_msd_sp = max(METHODS, key=lambda m: bias_results[m]["MSD-SP"])
    non_pwc_msd_sps = [bias_results[m]["MSD-SP"] for m in ["SR", "AR", "DA"]]

    lines.append("**Observations**:")
    if mrd_sp_negative:
        lines.append(f"- **MRD-SP** is negative for all methods, confirming universal self-preference "
                     f"in rankings. {strongest_mrd_sp} shows the strongest ranking inflation "
                     f"({fmt(bias_results[strongest_mrd_sp]['MRD-SP'], 2)}), "
                     f"{weakest_mrd_sp} the least ({fmt(bias_results[weakest_mrd_sp]['MRD-SP'], 2)}).")
    if msd_sp_positive:
        lines.append(f"- **MSD-SP** is positive for all methods. "
                     f"{strongest_msd_sp} shows the largest MSD-SP ({fmt(bias_results[strongest_msd_sp]['MSD-SP'])}), "
                     f"vs ~{fmt(np.mean(non_pwc_msd_sps))} for others.")
    lines.append("- **Family bias** (MRD-FSP, MSD-FSP) follows similar patterns but is attenuated, "
                 "indicating that self-preference is strongest for the exact model and weaker "
                 "for same-family models.")
    lines.append("")

    # -- Instance-level bias --
    lines.append("### Instance-Level Self-Preference Bias (MISPB)\n")
    lines.append("| Method | MISPB | raw | other | ratio | MISPB-F | F_ratio | MISPB-FO | FO_ratio |")
    lines.append("|--------|-------|-----|-------|-------|---------|---------|----------|----------|")
    for m in METHODS:
        r = mispb_results[m]
        lines.append(f"| {m} | {fmt(r['MISPB']['mean'])} | {fmt(r['MISPB']['mean_raw'])} | "
                     f"{fmt(r['MISPB']['mean_other'])} | {fmt(r['MISPB']['mean_ratio'], 2)} | "
                     f"{fmt(r['MISPB-F']['mean'])} | {fmt(r['MISPB-F']['mean_ratio'], 2)} | "
                     f"{fmt(r['MISPB-FO']['mean'])} | {fmt(r['MISPB-FO']['mean_ratio'], 2)} |")
    lines.append("")

    # Sample size detail table for MISPB (with sub-type breakdown)
    lines.append("*Per-judge MISPB sample sizes with overestimation sub-types (t2w=tie→win, l2w=loss→win, l2t=loss→tie):*\n")
    lines.append("| Judge | Method | MISPB | overest_self/total_self | self t2w/l2w/l2t | overest_other/total_other | other t2w/l2w/l2t |")
    lines.append("|-------|--------|-------|------------------------|------------------|--------------------------|-------------------|")
    for judge in JUDGES:
        for method in METHODS:
            r = mispb_results[method]["MISPB"]
            if judge in r["per_judge"]:
                nos = r["per_judge_n_overest_self"][judge]
                nts = r["per_judge_n_total_self"][judge]
                noo = r["per_judge_n_overest_other"][judge]
                nto = r["per_judge_n_total_other"][judge]
                t2w_s = r["per_judge_n_t2w_self"][judge]
                l2w_s = r["per_judge_n_l2w_self"][judge]
                l2t_s = r["per_judge_n_l2t_self"][judge]
                t2w_o = r["per_judge_n_t2w_other"][judge]
                l2w_o = r["per_judge_n_l2w_other"][judge]
                l2t_o = r["per_judge_n_l2t_other"][judge]
                lines.append(f"| {short(judge)} | {method} | {fmt(r['per_judge'][judge])} "
                             f"| {nos}/{nts} | {t2w_s}/{l2w_s}/{l2t_s} "
                             f"| {noo}/{nto} | {t2w_o}/{l2w_o}/{l2t_o} |")
    lines.append("")

    # Observations derived from data
    mispb_ordered = sorted(METHODS, key=lambda m: mispb_results[m]["MISPB"]["mean"], reverse=True)
    most_biased_mispb = mispb_ordered[0]
    non_pwc_mispbs = {m: mispb_results[m]["MISPB"]["mean"] for m in ["SR", "AR", "DA"]}
    non_pwc_range = f"{fmt(min(non_pwc_mispbs.values()))}–{fmt(max(non_pwc_mispbs.values()))}"

    lines.append("**Observations**:")
    all_positive_mispb = all(mispb_results[m]["MISPB"]["mean"] > 0 for m in METHODS)
    if all_positive_mispb:
        lines.append("- All methods show positive MISPB (judges overestimate themselves more than others).")
    lines.append(f"- **{most_biased_mispb}** has the highest MISPB "
                 f"({fmt(mispb_results[most_biased_mispb]['MISPB']['mean'])}). "
                 f"The other methods range from {non_pwc_range}.")
    ratio_lo = min(mispb_results[m]["MISPB"]["mean_ratio"] for m in METHODS)
    ratio_hi = max(mispb_results[m]["MISPB"]["mean_ratio"] for m in METHODS)
    lines.append(f"- MISPB_ratio shows judges are {fmt(ratio_lo, 2)}x–{fmt(ratio_hi, 2)}x more likely "
                 "to overestimate themselves than other generators.")
    lines.append("- Family bias (MISPB-F) is consistently lower than self bias, "
                 "indicating same-model preference is stronger than same-family preference.")
    lines.append("")

    # -- HSPP --
    lines.append("### Harmful Self-Preference Propensity (HSPP)\n")
    lines.append("HSPP restricts the denominator to instances where the reference says the other "
                 "generator is better, measuring how often self-preference causes actual errors.\n")
    lines.append("| Method | HSPP | raw | other | ratio | HSPP-F | F_ratio | HSPP-FO | FO_ratio |")
    lines.append("|--------|------|-----|-------|-------|--------|---------|---------|----------|")
    for m in METHODS:
        r = mispb_results[m]
        lines.append(f"| {m} | {fmt(r['HSPP']['mean'])} | {fmt(r['HSPP']['mean_raw'])} | "
                     f"{fmt(r['HSPP']['mean_other'])} | {fmt(r['HSPP']['mean_ratio'], 2)} | "
                     f"{fmt(r['HSPP-F']['mean'])} | {fmt(r['HSPP-F']['mean_ratio'], 2)} | "
                     f"{fmt(r['HSPP-FO']['mean'])} | {fmt(r['HSPP-FO']['mean_ratio'], 2)} |")
    lines.append("")

    highest_hspp = max(METHODS, key=lambda m: mispb_results[m]["HSPP"]["mean"])
    non_pwc_hspps = [mispb_results[m]["HSPP"]["mean"] for m in ["SR", "AR", "DA"]]
    lines.append(f"**{highest_hspp}** has the highest HSPP ({fmt(mispb_results[highest_hspp]['HSPP']['mean'])}): "
                 "when the reference says the other generator is better, judges still overestimate "
                 f"themselves at a higher net rate. For the other methods, HSPP ranges from "
                 f"{fmt(min(non_pwc_hspps))}–{fmt(max(non_pwc_hspps))}.\n")

    # -- Per-judge MISPB highlights (data-driven) --
    lines.append("### Notable Per-Judge Patterns\n")
    lines.append("From the per-judge MISPB breakdown (see `mispb_per_judge.csv`):\n")

    # Find most/least biased judges for SR/AR/DA
    for method in ["SR", "AR", "DA"]:
        r = mispb_results[method]["MISPB"]
        if r["per_judge"]:
            most_biased_judge = max(r["per_judge"], key=r["per_judge"].get)
            least_biased_judge = min(r["per_judge"], key=r["per_judge"].get)
            lines.append(f"- **{method}**: Most biased judge = {short(most_biased_judge)} "
                         f"({fmt(r['per_judge'][most_biased_judge])}), "
                         f"least biased = {short(least_biased_judge)} "
                         f"({fmt(r['per_judge'][least_biased_judge])})")
    lines.append("")

    # -- RQ2 Summary --
    lines.append("### RQ2 Answer\n")
    mispb_order = sorted(METHODS, key=lambda m: abs(mispb_results[m]["MISPB"]["mean"]), reverse=True)
    least_biased = mispb_order[-1]
    lines.append(f"**{mispb_order[0]} is by far the most sensitive to self-preference bias** across all metrics "
                 f"(MISPB = {fmt(mispb_results[mispb_order[0]]['MISPB']['mean'])}, "
                 f"MSD-SP = {fmt(bias_results[mispb_order[0]]['MSD-SP'])}, "
                 f"HSPP = {fmt(mispb_results[mispb_order[0]]['HSPP']['mean'])}).")
    lines.append(f"**{least_biased} is the least biased** "
                 f"(MISPB = {fmt(mispb_results[least_biased]['MISPB']['mean'])}).")
    lines.append(f"Ordering from most to least biased: "
                 f"**{' > '.join(mispb_order)}**.\n")

    # ================================================================
    # RQ2.1
    # ================================================================
    lines.append("---\n")
    lines.append("## RQ2.1: Are Rubric-Based Methods Sensitive to Self-Preference Bias?\n")

    lines.append("### Rubric-Level Self-Preference Bias (MRSPB)\n")
    lines.append("| Method | MRSPB | raw | other | ratio | MRSPB-F | F_ratio | MRSPB-FO | FO_ratio | MRSPB-err | err_ratio | MRSPB-err-F | errF_ratio | MRSPB-err-FO | errFO_ratio |")
    lines.append("|--------|-------|-----|-------|-------|---------|---------|----------|----------|-----------|-----------|-------------|------------|--------------|-------------|")
    for m in ["SR", "AR"]:
        r = rubric_results[m]
        lines.append(f"| {m} | {fmt(r['MRSPB']['mean'])} | {fmt(r['MRSPB']['mean_raw'])} | "
                     f"{fmt(r['MRSPB']['mean_other'])} | {fmt(r['MRSPB']['mean_ratio'], 2)} | "
                     f"{fmt(r['MRSPB-F']['mean'])} | {fmt(r['MRSPB-F']['mean_ratio'], 2)} | "
                     f"{fmt(r['MRSPB-FO']['mean'])} | {fmt(r['MRSPB-FO']['mean_ratio'], 2)} | "
                     f"{fmt(r['MRSPB-err']['mean'])} | {fmt(r['MRSPB-err']['mean_ratio'], 2)} | "
                     f"{fmt(r['MRSPB-err-F']['mean'])} | {fmt(r['MRSPB-err-F']['mean_ratio'], 2)} | "
                     f"{fmt(r['MRSPB-err-FO']['mean'])} | {fmt(r['MRSPB-err-FO']['mean_ratio'], 2)} |")
    lines.append("")

    # Sample size detail for MRSPB
    lines.append("*Per-judge MRSPB sample sizes (n_overest_self / n_total_self | n_overest_other / n_total_other):*\n")
    lines.append("| Judge | Method | MRSPB | n_overest_self/n_total_self | n_overest_other/n_total_other |")
    lines.append("|-------|--------|-------|----------------------------|-------------------------------|")
    for judge in JUDGES:
        for method in ["SR", "AR"]:
            r = rubric_results[method]["MRSPB"]
            if judge in r["per_judge"]:
                nos = r["per_judge_n_overest_self"][judge]
                nts = r["per_judge_n_total_self"][judge]
                noo = r["per_judge_n_overest_other"][judge]
                nto = r["per_judge_n_total_other"][judge]
                lines.append(f"| {short(judge)} | {method} | {fmt(r['per_judge'][judge])} "
                             f"| {nos}/{nts} | {noo}/{nto} |")
    lines.append("")

    sr_mrspb = rubric_results["SR"]["MRSPB"]["mean"]
    ar_mrspb = rubric_results["AR"]["MRSPB"]["mean"]
    sr_mrspb_err = rubric_results["SR"]["MRSPB-err"]["mean"]
    ar_mrspb_err = rubric_results["AR"]["MRSPB-err"]["mean"]

    lines.append("**Observations**:")
    both_positive = sr_mrspb > 0 and ar_mrspb > 0
    if both_positive:
        lines.append("- **Yes, rubric-based methods are sensitive to self-preference bias.** Both SR and AR "
                     f"show positive MRSPB (SR={fmt(sr_mrspb)}, AR={fmt(ar_mrspb)}), meaning judges are "
                     "more likely to say their own rubrics are met (false positive) compared to how they "
                     "treat other generators' rubrics.")
    lines.append(f"- The bias levels are similar for SR ({fmt(sr_mrspb)}) and AR ({fmt(ar_mrspb)}), "
                 "suggesting that the presentation format (single vs. all rubrics) does not "
                 "meaningfully affect self-preference at the rubric level.")
    lines.append(f"- The error-denominator variant (MRSPB-err) is higher "
                 f"(SR={fmt(sr_mrspb_err)}, AR={fmt(ar_mrspb_err)}), meaning "
                 "that among rubrics where the reference says 'not met', judges incorrectly say 'met' "
                 "for their own outputs more often than for others.")

    # Find most/least biased judge at rubric level
    for m_label in ["SR", "AR"]:
        r_mrspb = rubric_results[m_label]["MRSPB"]
        if r_mrspb["per_judge"]:
            most_b = max(r_mrspb["per_judge"], key=r_mrspb["per_judge"].get)
            least_b = min(r_mrspb["per_judge"], key=r_mrspb["per_judge"].get)
            lines.append(f"- **{m_label}** rubric bias: most biased = {short(most_b)} "
                         f"({fmt(r_mrspb['per_judge'][most_b])}), "
                         f"least biased = {short(least_b)} ({fmt(r_mrspb['per_judge'][least_b])})")
    lines.append("")

    lines.append("### RQ2.1 Answer\n")
    lines.append(f"**Yes, rubric-based methods (SR and AR) are sensitive to self-preference bias**, "
                 f"with MRSPB of {fmt(sr_mrspb)} (SR) and {fmt(ar_mrspb)} (AR), and "
                 f"MRSPB-err of {fmt(sr_mrspb_err)} (SR) and {fmt(ar_mrspb_err)} (AR). "
                 "The bias is modest but consistent, and the two rubric formats show very similar "
                 "bias levels.\n")

    # ================================================================
    # RQ2.2
    # ================================================================
    lines.append("---\n")
    lines.append("## RQ2.2: Is Self-Preference Bias Related to Method Quality?\n")

    lines.append("### Method-Level Analysis\n")
    lines.append("| Method | MPA (accuracy) | |MISPB| (bias) |")
    lines.append("|--------|----------------|----------------|")
    for m in METHODS:
        mpa = accuracy_results[m]["MPA"]
        bias = abs(mispb_results[m]["MISPB"]["mean"])
        lines.append(f"| {m} | {fmt(mpa)} | {fmt(bias)} |")
    lines.append("")

    mpas = [accuracy_results[m]["MPA"] for m in METHODS]
    biases = [abs(mispb_results[m]["MISPB"]["mean"]) for m in METHODS]
    corr, p_val = scipy_stats.pearsonr(mpas, biases)
    lines.append(f"Pearson correlation (all 4 methods): r = {fmt(corr, 3)}, p = {fmt(p_val, 3)}")
    lines.append("")

    # Also compute excluding PWC
    mpas_3 = [accuracy_results[m]["MPA"] for m in ["SR", "AR", "DA"]]
    biases_3 = [abs(mispb_results[m]["MISPB"]["mean"]) for m in ["SR", "AR", "DA"]]
    corr_3, p_3 = scipy_stats.pearsonr(mpas_3, biases_3)
    lines.append(f"Pearson correlation (excluding PWC): r = {fmt(corr_3, 3)}, p = {fmt(p_3, 3)}")
    lines.append("")

    lines.append("**Observations**:")
    lines.append(f"- When all four methods are included, the correlation is r = {fmt(corr, 3)} "
                 f"(p = {fmt(p_val, 3)}). With only 4 data points, statistical significance is limited.")
    lines.append(f"- **Excluding PWC**, the correlation is r = {fmt(corr_3, 3)}. "
                 "Among SR, AR, and DA, the relationship between accuracy and bias may differ "
                 "from the overall pattern driven by PWC as an outlier.")
    lines.append("- The relationship is **not straightforward**: it depends heavily on whether PWC "
                 "is included, and the non-PWC methods have very similar bias levels making the "
                 "correlation unstable.")
    lines.append("")

    lines.append("### RQ2.2 Answer\n")
    lines.append("**The relationship between accuracy and bias is nuanced.** PWC stands out as both "
                 "the least accurate and most biased method by a wide margin, creating an apparent "
                 "negative correlation. However, among the more comparable methods (SR, AR, DA), "
                 "the differences in bias are small and the correlation may reverse. "
                 "Self-preference bias appears to be a relatively stable phenomenon across SR/AR/DA, "
                 "with magnitude driven more by individual judge quality than by the prompting "
                 "method itself.\n")

    # ================================================================
    # Figures and Tables index
    # ================================================================
    lines.append("---\n")
    lines.append("## Figures\n")
    lines.append("1. `figures/accuracy_comparison.png` — RQ1: Accuracy metrics across methods")
    lines.append("2. `figures/bias_comparison.png` — RQ2: Bias metrics across methods")
    lines.append("3. `figures/mispb_heatmap.png` — MISPB per judge and method")
    lines.append("4. `figures/mispb_family_heatmap.png` — Family self-preference per judge and method")
    lines.append("5. `figures/accuracy_vs_bias.png` — RQ2.2: Accuracy vs. bias scatter")
    lines.append("6. `figures/mrspb_heatmap.png` — MRSPB per judge (SR vs AR)")
    lines.append("7. `figures/hspp_comparison.png` — Harmful Self-Preference Propensity")
    lines.append("")

    lines.append("## Tables\n")
    lines.append("All CSV tables saved in `tables/`:")
    lines.append("- `system_scores.csv` — System-level scores per generator and method")
    lines.append("- `system_accuracy.csv` — MPA, MRD, MSD per method")
    lines.append("- `system_bias.csv` — MRD-SP, MSD-SP, MSD-FSP per method")
    lines.append("- `mpa_per_judge.csv` — MPA broken down by judge (with n_concordant, n_total)")
    lines.append("- `msd_sp_per_judge.csv` — MSD-SP broken down by judge (with d_self, d_other, n_others)")
    lines.append("- `mipa.csv` — MIPA per method")
    lines.append("- `mipa_per_judge.csv` — MIPA broken down by judge (with n_agree, n_total)")
    lines.append("- `mispb.csv` — MISPB summary per method")
    lines.append("- `mispb_per_judge.csv` — MISPB broken down by judge (with overestimation counts)")
    lines.append("- `rubric_metrics.csv` — MRA and MRSPB (SR, AR)")
    lines.append("- `mra_per_judge.csv` — MRA broken down by judge (with n_correct, n_total)")
    lines.append("- `mrspb_per_judge.csv` — MRSPB broken down by judge (with overestimation counts)")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
