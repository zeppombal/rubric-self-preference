"""
Metrics computation module for HealthBench SR analysis.

Computes system-level, instance-level, and rubric-level metrics for
accuracy and self-preference bias. Adapted from the IFEval metrics.py
with key differences:

1. Instance scoring uses weighted or uniform HealthBench formulas
2. Reference is per-judge (leave-one-family-out), so functions accept
   ref_scores_by_judge: Dict[str, Dict[str, float]]
3. Only SR method (no AR, DA, PWC)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats as scipy_stats

from hb_data_loading import (
    FAMILIES,
    GENERATORS,
    JUDGES,
    MODEL_TO_FAMILY,
    N_INSTANCES,
    get_family,
    get_family_generators,
    get_other_generators,
    is_same_family,
)
from hb_scoring import instance_score, ref_instance_score

logger = logging.getLogger(__name__)


# ============================================================
# Instance Scoring Helpers
# ============================================================


def _compute_instance_scores(
    sr_data: Dict[Tuple[str, str], List[List[Dict]]],
    scoring_mode: str,
) -> Dict[Tuple[str, str], List[float]]:
    """Convert rubric-level SR data to per-instance float scores."""
    return {
        (j, g): [instance_score(rubrics, scoring_mode) for rubrics in instances]
        for (j, g), instances in sr_data.items()
    }


def _compute_ref_instance_scores(
    ref_data: Dict[str, Dict],
    gen_data: Dict[str, List[List[Dict]]],
    scoring_mode: str,
) -> Dict[str, List[float]]:
    """Convert reference boolean data to per-instance float scores.

    Requires generation data to provide rubric point values.
    """
    result = {}
    for gen in GENERATORS:
        if gen not in ref_data:
            continue
        follow_list = ref_data[gen]["follow_list"]
        scores = []
        for i in range(len(follow_list)):
            scores.append(ref_instance_score(follow_list[i], gen_data[gen][i], scoring_mode))
        result[gen] = scores
    return result


# ============================================================
# System-Level Scores
# ============================================================


def compute_reference_system_scores(
    ref_data: Dict[str, Dict],
    gen_data: Dict[str, List[List[Dict]]],
    scoring_mode: str,
) -> Dict[str, float]:
    """
    Compute reference system-level scores.

    Instance score = HealthBench scoring formula (weighted or uniform).
    Generator system score = mean across all instances.
    """
    ref_inst = _compute_ref_instance_scores(ref_data, gen_data, scoring_mode)
    return {gen: np.mean(scores) for gen, scores in ref_inst.items()}


def compute_system_scores_sr(
    sr_data: Dict[Tuple[str, str], List[List[Dict]]],
    scoring_mode: str,
) -> Dict[Tuple[str, str], float]:
    """
    Compute system-level scores for SR evaluations.

    Instance score = HealthBench scoring formula.
    System score = mean across instances.
    """
    inst_scores = _compute_instance_scores(sr_data, scoring_mode)
    return {key: np.mean(scores) for key, scores in inst_scores.items()}


# ============================================================
# System-Level Accuracy Metrics
# ============================================================


def _get_judge_gen_scores(
    system_scores: Dict[Tuple[str, str], float],
    judge: str,
) -> Dict[str, float]:
    """Extract per-generator scores for a specific judge."""
    return {
        gen: system_scores[(judge, gen)]
        for gen in GENERATORS
        if (judge, gen) in system_scores
    }


def compute_mpa(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores_by_judge: Dict[str, Dict[str, float]],
    judges: Optional[List[str]] = None,
) -> Dict:
    """
    Mean Pairwise Accuracy (MPA).

    For each judge J, PA is computed against ref_scores_by_judge[J].
    MPA = mean PA across judges.
    """
    if judges is None:
        judges = JUDGES

    judge_pa = {}
    judge_n_concordant = {}
    judge_n_total = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores or judge not in ref_scores_by_judge:
            continue
        ref_scores = ref_scores_by_judge[judge]

        gens = [g for g in GENERATORS if g in j_scores and g in ref_scores]
        if len(gens) < 2:
            continue

        n_concordant = 0
        n_total = 0
        for i in range(len(gens)):
            for j_idx in range(i + 1, len(gens)):
                g1, g2 = gens[i], gens[j_idx]
                j_diff = j_scores[g1] - j_scores[g2]
                r_diff = ref_scores[g1] - ref_scores[g2]
                if (
                    (j_diff > 0 and r_diff > 0)
                    or (j_diff < 0 and r_diff < 0)
                    or (j_diff == 0 and r_diff == 0)
                ):
                    n_concordant += 1
                n_total += 1

        pa = n_concordant / n_total
        judge_pa[judge] = pa
        judge_n_concordant[judge] = n_concordant
        judge_n_total[judge] = n_total

        # Sanity check
        j_arr = np.array([j_scores[g] for g in gens])
        r_arr = np.array([ref_scores[g] for g in gens])
        tau, _ = scipy_stats.kendalltau(j_arr, r_arr)
        expected_pa = (tau + 1) / 2
        if abs(pa - expected_pa) > 0.05:
            logger.warning(
                f"PA vs tau divergence for {judge}: PA={pa:.4f}, "
                f"(tau+1)/2={expected_pa:.4f}"
            )

    mpa = np.mean(list(judge_pa.values())) if judge_pa else float("nan")
    return {
        "mean": mpa,
        "per_judge": judge_pa,
        "per_judge_n_concordant": judge_n_concordant,
        "per_judge_n_total": judge_n_total,
    }


def compute_mrd(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores_by_judge: Dict[str, Dict[str, float]],
    judges: Optional[List[str]] = None,
) -> Dict:
    """Mean Ranking Difference (MRD). Per-judge reference-aware."""
    if judges is None:
        judges = JUDGES

    judge_mrd = {}
    judge_n_gens = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores or judge not in ref_scores_by_judge:
            continue
        ref_scores = ref_scores_by_judge[judge]

        gens = [g for g in GENERATORS if g in j_scores and g in ref_scores]
        j_arr = np.array([j_scores[g] for g in gens])
        r_arr = np.array([ref_scores[g] for g in gens])
        j_ranks = scipy_stats.rankdata(-j_arr, method="average")
        r_ranks = scipy_stats.rankdata(-r_arr, method="average")

        abs_diffs = np.abs(j_ranks - r_ranks)
        judge_mrd[judge] = np.mean(abs_diffs)
        judge_n_gens[judge] = len(gens)

    mrd = np.mean(list(judge_mrd.values())) if judge_mrd else float("nan")
    return {"mean": mrd, "per_judge": judge_mrd, "per_judge_n_generators": judge_n_gens}


def compute_msd(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores_by_judge: Dict[str, Dict[str, float]],
    judges: Optional[List[str]] = None,
) -> Dict:
    """Mean Score Delta (MSD). Per-judge reference-aware."""
    if judges is None:
        judges = JUDGES

    judge_msd = {}
    judge_n_gens = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores or judge not in ref_scores_by_judge:
            continue
        ref_scores = ref_scores_by_judge[judge]

        gens = [g for g in GENERATORS if g in j_scores and g in ref_scores]
        deltas = [j_scores[g] - ref_scores[g] for g in gens]
        judge_msd[judge] = np.mean(deltas)
        judge_n_gens[judge] = len(gens)

    msd = np.mean(list(judge_msd.values())) if judge_msd else float("nan")
    return {"mean": msd, "per_judge": judge_msd, "per_judge_n_generators": judge_n_gens}


def compute_msd_norm(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores_by_judge: Dict[str, Dict[str, float]],
    judges: Optional[List[str]] = None,
) -> Dict:
    """Mean Score Delta – Normalized (MSD-norm). Per-judge reference-aware."""
    if judges is None:
        judges = JUDGES

    judge_msd = {}
    judge_n_gens = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores or judge not in ref_scores_by_judge:
            continue
        ref_scores = ref_scores_by_judge[judge]

        gens = [g for g in GENERATORS if g in j_scores and g in ref_scores]
        deltas = []
        for g in gens:
            if ref_scores[g] > 0:
                deltas.append((j_scores[g] - ref_scores[g]) / ref_scores[g])
            else:
                logger.warning(f"Ref score 0 for {g}, skipping in MSD-norm for judge {judge}")
        if deltas:
            judge_msd[judge] = np.mean(deltas)
            judge_n_gens[judge] = len(deltas)

    msd = np.mean(list(judge_msd.values())) if judge_msd else float("nan")
    return {"mean": msd, "per_judge": judge_msd, "per_judge_n_generators": judge_n_gens}


# ============================================================
# System-Level Bias Metrics
# ============================================================


def compute_mrd_sp(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores_by_judge: Dict[str, Dict[str, float]],
    judges: Optional[List[str]] = None,
) -> Dict:
    """MRD Self-Preference. Per-judge reference-aware."""
    if judges is None:
        judges = JUDGES

    judge_vals = {}
    judge_d_self = {}
    judge_d_other = {}
    judge_n_others = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores or judge not in ref_scores_by_judge:
            continue
        ref_scores = ref_scores_by_judge[judge]

        gens = [g for g in GENERATORS if g in j_scores and g in ref_scores]
        j_arr = np.array([j_scores[g] for g in gens])
        r_arr = np.array([ref_scores[g] for g in gens])
        j_ranks = scipy_stats.rankdata(-j_arr, method="average")
        r_ranks = scipy_stats.rankdata(-r_arr, method="average")
        signed_diffs = j_ranks - r_ranks
        gen_to_diff = {g: signed_diffs[idx] for idx, g in enumerate(gens)}

        if judge not in gen_to_diff:
            continue
        d_self = gen_to_diff[judge]

        others = [g for g in get_other_generators(judge) if g in gen_to_diff]
        if not others:
            continue
        d_other = np.mean([gen_to_diff[g] for g in others])

        judge_vals[judge] = d_self - d_other
        judge_d_self[judge] = d_self
        judge_d_other[judge] = d_other
        judge_n_others[judge] = len(others)

    mean_val = np.mean(list(judge_vals.values())) if judge_vals else float("nan")
    return {
        "mean": mean_val,
        "mean_d_self": np.mean(list(judge_d_self.values())) if judge_d_self else float("nan"),
        "mean_d_other": np.mean(list(judge_d_other.values())) if judge_d_other else float("nan"),
        "per_judge": judge_vals,
        "per_judge_d_self": judge_d_self,
        "per_judge_d_other": judge_d_other,
        "per_judge_n_others": judge_n_others,
    }


def compute_mrd_fsp(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores_by_judge: Dict[str, Dict[str, float]],
    include_self_in_family: bool = True,
    judges: Optional[List[str]] = None,
) -> Dict:
    """MRD Family Self-Preference. Per-judge reference-aware."""
    if judges is None:
        judges = JUDGES

    judge_vals = {}
    judge_d_family = {}
    judge_d_nonfamily = {}
    judge_n_family = {}
    judge_n_nonfamily = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores or judge not in ref_scores_by_judge:
            continue
        ref_scores = ref_scores_by_judge[judge]

        gens = [g for g in GENERATORS if g in j_scores and g in ref_scores]
        j_arr = np.array([j_scores[g] for g in gens])
        r_arr = np.array([ref_scores[g] for g in gens])
        j_ranks = scipy_stats.rankdata(-j_arr, method="average")
        r_ranks = scipy_stats.rankdata(-r_arr, method="average")
        signed_diffs = j_ranks - r_ranks
        gen_to_diff = {g: signed_diffs[idx] for idx, g in enumerate(gens)}

        family_gens = [
            g for g in get_family_generators(judge, include_self=include_self_in_family)
            if g in gen_to_diff
        ]
        other_gens = [g for g in get_other_generators(judge) if g in gen_to_diff]

        if not family_gens or not other_gens:
            continue

        d_family = np.mean([gen_to_diff[g] for g in family_gens])
        d_nonfamily = np.mean([gen_to_diff[g] for g in other_gens])

        judge_vals[judge] = d_family - d_nonfamily
        judge_d_family[judge] = d_family
        judge_d_nonfamily[judge] = d_nonfamily
        judge_n_family[judge] = len(family_gens)
        judge_n_nonfamily[judge] = len(other_gens)

    mean_val = np.mean(list(judge_vals.values())) if judge_vals else float("nan")
    return {
        "mean": mean_val,
        "mean_d_family": np.mean(list(judge_d_family.values())) if judge_d_family else float("nan"),
        "mean_d_nonfamily": np.mean(list(judge_d_nonfamily.values())) if judge_d_nonfamily else float("nan"),
        "per_judge": judge_vals,
        "per_judge_d_family": judge_d_family,
        "per_judge_d_nonfamily": judge_d_nonfamily,
        "per_judge_n_family": judge_n_family,
        "per_judge_n_nonfamily": judge_n_nonfamily,
    }


def compute_mrd_fosp(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores_by_judge: Dict[str, Dict[str, float]],
    judges: Optional[List[str]] = None,
) -> Dict:
    """MRD Family-Only Self-Preference. Family excluding self."""
    return compute_mrd_fsp(system_scores, ref_scores_by_judge,
                           include_self_in_family=False, judges=judges)


def _compute_msd_sp_generic(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores_by_judge: Dict[str, Dict[str, float]],
    normalize: bool = False,
    judges: Optional[List[str]] = None,
) -> Dict:
    """Generic MSD Self-Preference computation."""
    if judges is None:
        judges = JUDGES

    judge_vals = {}
    judge_d_self = {}
    judge_d_other = {}
    judge_n_others = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores or judge not in ref_scores_by_judge:
            continue
        ref_scores = ref_scores_by_judge[judge]

        def delta(g):
            d = j_scores[g] - ref_scores[g]
            if normalize:
                if ref_scores[g] <= 0:
                    return None
                d /= ref_scores[g]
            return d

        if judge not in j_scores:
            continue
        d_self = delta(judge)
        if d_self is None:
            continue

        others = [g for g in get_other_generators(judge) if g in j_scores]
        d_others = [delta(g) for g in others]
        d_others = [d for d in d_others if d is not None]
        if not d_others:
            continue
        d_other = np.mean(d_others)

        judge_vals[judge] = d_self - d_other
        judge_d_self[judge] = d_self
        judge_d_other[judge] = d_other
        judge_n_others[judge] = len(d_others)

    mean_val = np.mean(list(judge_vals.values())) if judge_vals else float("nan")
    return {
        "mean": mean_val,
        "mean_d_self": np.mean(list(judge_d_self.values())) if judge_d_self else float("nan"),
        "mean_d_other": np.mean(list(judge_d_other.values())) if judge_d_other else float("nan"),
        "per_judge": judge_vals,
        "per_judge_d_self": judge_d_self,
        "per_judge_d_other": judge_d_other,
        "per_judge_n_others": judge_n_others,
    }


def compute_msd_sp(system_scores, ref_scores_by_judge, judges=None):
    """MSD Self-Preference."""
    return _compute_msd_sp_generic(system_scores, ref_scores_by_judge,
                                    normalize=False, judges=judges)


def compute_msd_sp_norm(system_scores, ref_scores_by_judge, judges=None):
    """MSD Self-Preference Normalized."""
    return _compute_msd_sp_generic(system_scores, ref_scores_by_judge,
                                    normalize=True, judges=judges)


def _compute_msd_fsp_generic(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores_by_judge: Dict[str, Dict[str, float]],
    normalize: bool = False,
    include_self_in_family: bool = True,
    judges: Optional[List[str]] = None,
) -> Dict:
    """Generic MSD Family Self-Preference computation."""
    if judges is None:
        judges = JUDGES

    judge_vals = {}
    judge_d_family = {}
    judge_d_nonfamily = {}
    judge_n_family = {}
    judge_n_nonfamily = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores or judge not in ref_scores_by_judge:
            continue
        ref_scores = ref_scores_by_judge[judge]

        def delta(g):
            d = j_scores[g] - ref_scores[g]
            if normalize:
                if ref_scores[g] <= 0:
                    return None
                d /= ref_scores[g]
            return d

        family_gens = [
            g for g in get_family_generators(judge, include_self=include_self_in_family)
            if g in j_scores
        ]
        other_gens = [g for g in get_other_generators(judge) if g in j_scores]

        family_deltas = [delta(g) for g in family_gens]
        family_deltas = [d for d in family_deltas if d is not None]
        other_deltas = [delta(g) for g in other_gens]
        other_deltas = [d for d in other_deltas if d is not None]

        if not family_deltas or not other_deltas:
            continue

        d_family = np.mean(family_deltas)
        d_nonfamily = np.mean(other_deltas)

        judge_vals[judge] = d_family - d_nonfamily
        judge_d_family[judge] = d_family
        judge_d_nonfamily[judge] = d_nonfamily
        judge_n_family[judge] = len(family_deltas)
        judge_n_nonfamily[judge] = len(other_deltas)

    mean_val = np.mean(list(judge_vals.values())) if judge_vals else float("nan")
    return {
        "mean": mean_val,
        "mean_d_family": np.mean(list(judge_d_family.values())) if judge_d_family else float("nan"),
        "mean_d_nonfamily": np.mean(list(judge_d_nonfamily.values())) if judge_d_nonfamily else float("nan"),
        "per_judge": judge_vals,
        "per_judge_d_family": judge_d_family,
        "per_judge_d_nonfamily": judge_d_nonfamily,
        "per_judge_n_family": judge_n_family,
        "per_judge_n_nonfamily": judge_n_nonfamily,
    }


def compute_msd_fsp(system_scores, ref_scores_by_judge, include_self_in_family=True, judges=None):
    """MSD Family Self-Preference."""
    return _compute_msd_fsp_generic(system_scores, ref_scores_by_judge, normalize=False,
                                     include_self_in_family=include_self_in_family, judges=judges)


def compute_msd_fsp_norm(system_scores, ref_scores_by_judge, include_self_in_family=True, judges=None):
    """MSD Family Self-Preference Normalized."""
    return _compute_msd_fsp_generic(system_scores, ref_scores_by_judge, normalize=True,
                                     include_self_in_family=include_self_in_family, judges=judges)


def compute_msd_fosp(system_scores, ref_scores_by_judge, judges=None):
    """MSD Family-Only Self-Preference. Family excluding self."""
    return _compute_msd_fsp_generic(system_scores, ref_scores_by_judge, normalize=False,
                                     include_self_in_family=False, judges=judges)


def compute_msd_fosp_norm(system_scores, ref_scores_by_judge, judges=None):
    """MSD Family-Only Self-Preference Normalized. Family excluding self."""
    return _compute_msd_fsp_generic(system_scores, ref_scores_by_judge, normalize=True,
                                     include_self_in_family=False, judges=judges)


def compute_per_generator_deltas(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores_by_judge: Dict[str, Dict[str, float]],
    normalize: bool = False,
    judges: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Mean score delta per generator across all judges (per-judge reference-aware)."""
    if judges is None:
        judges = JUDGES

    gen_deltas = {}
    for gen in GENERATORS:
        deltas = []
        for judge in judges:
            if (judge, gen) not in system_scores or judge not in ref_scores_by_judge:
                continue
            ref_scores = ref_scores_by_judge[judge]
            if gen not in ref_scores:
                continue
            d = system_scores[(judge, gen)] - ref_scores[gen]
            if normalize:
                if ref_scores[gen] <= 0:
                    continue
                d /= ref_scores[gen]
            deltas.append(d)
        if deltas:
            gen_deltas[gen] = np.mean(deltas)
    return gen_deltas


def compute_per_generator_rank_deltas(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores_by_judge: Dict[str, Dict[str, float]],
    judges: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Mean signed rank delta per generator across all judges (per-judge reference-aware)."""
    if judges is None:
        judges = JUDGES

    gen_rank_diffs: Dict[str, List[float]] = {g: [] for g in GENERATORS}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores or judge not in ref_scores_by_judge:
            continue
        ref_scores = ref_scores_by_judge[judge]

        gens = [g for g in GENERATORS if g in j_scores and g in ref_scores]
        if len(gens) < 2:
            continue

        j_arr = np.array([j_scores[g] for g in gens])
        r_arr = np.array([ref_scores[g] for g in gens])
        j_ranks = scipy_stats.rankdata(-j_arr, method="average")
        r_ranks = scipy_stats.rankdata(-r_arr, method="average")
        signed_diffs = j_ranks - r_ranks

        for idx, g in enumerate(gens):
            gen_rank_diffs[g].append(signed_diffs[idx])

    return {g: np.mean(diffs) for g, diffs in gen_rank_diffs.items() if diffs}


# ============================================================
# Instance-Level Accuracy (MIPA)
# ============================================================


def compute_mipa(
    instance_scores: Dict[Tuple[str, str], List[float]],
    ref_inst_by_judge: Dict[str, Dict[str, List[float]]],
    judges: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float], Dict[str, int], Dict[str, int]]:
    """
    Mean Instance Pairwise Accuracy (MIPA).

    For each judge, instance, and pair (X, Y): does the judge's instance-level
    ordering agree with the reference?
    """
    if judges is None:
        judges = JUDGES

    judge_mipa = {}
    judge_n_agree = {}
    judge_n_total = {}

    for judge in judges:
        if judge not in ref_inst_by_judge:
            continue
        ref_inst = ref_inst_by_judge[judge]

        j_gens = [g for g in GENERATORS if (judge, g) in instance_scores and g in ref_inst]
        if len(j_gens) < 2:
            continue

        n_agree = 0
        n_total = 0

        for i in range(N_INSTANCES):
            for gi in range(len(j_gens)):
                for gj in range(gi + 1, len(j_gens)):
                    gx, gy = j_gens[gi], j_gens[gj]
                    j_diff = instance_scores[(judge, gx)][i] - instance_scores[(judge, gy)][i]
                    r_diff = ref_inst[gx][i] - ref_inst[gy][i]
                    if (
                        (j_diff > 0 and r_diff > 0)
                        or (j_diff < 0 and r_diff < 0)
                        or (j_diff == 0 and r_diff == 0)
                    ):
                        n_agree += 1
                    n_total += 1

        if n_total > 0:
            judge_mipa[judge] = n_agree / n_total
            judge_n_agree[judge] = n_agree
            judge_n_total[judge] = n_total

    mipa = np.mean(list(judge_mipa.values())) if judge_mipa else float("nan")
    return mipa, judge_mipa, judge_n_agree, judge_n_total


# ============================================================
# Instance-Level Self-Preference Bias (MISPB)
# ============================================================


def _compute_overestimation_rate(
    instance_scores: Dict[Tuple[str, str], List[float]],
    ref_inst: Dict[str, List[float]],
    judge: str,
    target_gen: str,
    error_denom: bool = False,
    generators: Optional[List[str]] = None,
) -> Optional[Dict[str, Union[float, int]]]:
    """
    Compute overestimation rate of target_gen by judge.

    Same logic as IFEval: compare judge's instance-level ordering against reference.
    """
    if generators is None:
        generators = GENERATORS

    if (judge, target_gen) not in instance_scores:
        return None

    n_overest = 0
    n_total = 0
    n_t2w = 0
    n_l2w = 0
    n_l2t = 0

    for opp in generators:
        if opp == target_gen:
            continue
        if (judge, opp) not in instance_scores:
            continue
        if opp not in ref_inst or target_gen not in ref_inst:
            continue

        for i in range(N_INSTANCES):
            j_target = instance_scores[(judge, target_gen)][i]
            j_opp = instance_scores[(judge, opp)][i]
            r_target = ref_inst[target_gen][i]
            r_opp = ref_inst[opp][i]

            j_sign = np.sign(j_target - j_opp)
            r_sign = np.sign(r_target - r_opp)

            if error_denom:
                if r_sign >= 0:
                    continue

            n_total += 1
            if j_sign > r_sign:
                n_overest += 1
                if r_sign == 0 and j_sign == 1:
                    n_t2w += 1
                elif r_sign == -1 and j_sign == 1:
                    n_l2w += 1
                elif r_sign == -1 and j_sign == 0:
                    n_l2t += 1

    if n_total == 0:
        return None
    assert n_t2w + n_l2w + n_l2t == n_overest, (
        f"Sub-type sum mismatch: {n_t2w}+{n_l2w}+{n_l2t} != {n_overest}"
    )
    return {
        "rate": n_overest / n_total,
        "n_overest": n_overest,
        "n_total": n_total,
        "n_t2w": n_t2w,
        "n_l2w": n_l2w,
        "n_l2t": n_l2t,
    }


def compute_mispb(
    sr_data: Dict[Tuple[str, str], List[List[Dict]]],
    ref_inst_by_judge: Dict[str, Dict[str, List[float]]],
    scoring_mode: str,
    error_denom: bool = False,
    family_mode: bool = False,
    include_self_in_family: bool = True,
    judges: Optional[List[str]] = None,
) -> Dict:
    """
    Mean Instance Self-Preference Bias (MISPB) or its variants.

    Uses per-judge reference instance scores.
    """
    if judges is None:
        judges = JUDGES

    instance_scores = _compute_instance_scores(sr_data, scoring_mode)

    per_judge = {}
    per_judge_raw = {}
    per_judge_other = {}
    per_judge_ratio = {}
    per_judge_n_overest_self = {}
    per_judge_n_total_self = {}
    per_judge_n_overest_other = {}
    per_judge_n_total_other = {}
    per_judge_n_t2w_self = {}
    per_judge_n_l2w_self = {}
    per_judge_n_l2t_self = {}
    per_judge_n_t2w_other = {}
    per_judge_n_l2w_other = {}
    per_judge_n_l2t_other = {}

    for judge in judges:
        if judge not in ref_inst_by_judge:
            continue
        ref_inst = ref_inst_by_judge[judge]

        if family_mode:
            target_gens = get_family_generators(judge, include_self=include_self_in_family)
        else:
            target_gens = [judge]

        if not target_gens:
            continue

        raw_rates = []
        total_n_overest_self = 0
        total_n_total_self = 0
        total_n_t2w_self = 0
        total_n_l2w_self = 0
        total_n_l2t_self = 0
        for tg in target_gens:
            result = _compute_overestimation_rate(
                instance_scores, ref_inst, judge, tg, error_denom
            )
            if result is not None:
                raw_rates.append(result["rate"])
                total_n_overest_self += result["n_overest"]
                total_n_total_self += result["n_total"]
                total_n_t2w_self += result["n_t2w"]
                total_n_l2w_self += result["n_l2w"]
                total_n_l2t_self += result["n_l2t"]

        if not raw_rates:
            continue
        mispb_raw = np.mean(raw_rates)

        other_gens = get_other_generators(judge)
        other_rates = []
        total_n_overest_other = 0
        total_n_total_other = 0
        total_n_t2w_other = 0
        total_n_l2w_other = 0
        total_n_l2t_other = 0
        for og in other_gens:
            result = _compute_overestimation_rate(
                instance_scores, ref_inst, judge, og, error_denom
            )
            if result is not None:
                other_rates.append(result["rate"])
                total_n_overest_other += result["n_overest"]
                total_n_total_other += result["n_total"]
                total_n_t2w_other += result["n_t2w"]
                total_n_l2w_other += result["n_l2w"]
                total_n_l2t_other += result["n_l2t"]

        if not other_rates:
            continue
        mispb_other = np.mean(other_rates)

        mispb = mispb_raw - mispb_other
        ratio = mispb_raw / mispb_other if mispb_other > 0 else float("inf")

        per_judge[judge] = mispb
        per_judge_raw[judge] = mispb_raw
        per_judge_other[judge] = mispb_other
        per_judge_ratio[judge] = ratio
        per_judge_n_overest_self[judge] = total_n_overest_self
        per_judge_n_total_self[judge] = total_n_total_self
        per_judge_n_overest_other[judge] = total_n_overest_other
        per_judge_n_total_other[judge] = total_n_total_other
        per_judge_n_t2w_self[judge] = total_n_t2w_self
        per_judge_n_l2w_self[judge] = total_n_l2w_self
        per_judge_n_l2t_self[judge] = total_n_l2t_self
        per_judge_n_t2w_other[judge] = total_n_t2w_other
        per_judge_n_l2w_other[judge] = total_n_l2w_other
        per_judge_n_l2t_other[judge] = total_n_l2t_other

    mean_val = np.mean(list(per_judge.values())) if per_judge else float("nan")
    mean_raw = np.mean(list(per_judge_raw.values())) if per_judge_raw else float("nan")
    mean_other = np.mean(list(per_judge_other.values())) if per_judge_other else float("nan")
    ratio_vals = [v for v in per_judge_ratio.values() if np.isfinite(v)]
    mean_ratio = np.mean(ratio_vals) if ratio_vals else float("nan")
    return {
        "mean": mean_val,
        "mean_raw": mean_raw,
        "mean_other": mean_other,
        "mean_ratio": mean_ratio,
        "per_judge": per_judge,
        "per_judge_raw": per_judge_raw,
        "per_judge_other": per_judge_other,
        "per_judge_ratio": per_judge_ratio,
        "per_judge_n_overest_self": per_judge_n_overest_self,
        "per_judge_n_total_self": per_judge_n_total_self,
        "per_judge_n_overest_other": per_judge_n_overest_other,
        "per_judge_n_total_other": per_judge_n_total_other,
        "per_judge_n_t2w_self": per_judge_n_t2w_self,
        "per_judge_n_l2w_self": per_judge_n_l2w_self,
        "per_judge_n_l2t_self": per_judge_n_l2t_self,
        "per_judge_n_t2w_other": per_judge_n_t2w_other,
        "per_judge_n_l2w_other": per_judge_n_l2w_other,
        "per_judge_n_l2t_other": per_judge_n_l2t_other,
    }


# ============================================================
# Rubric-Level Metrics
# ============================================================


def compute_mra(
    sr_data: Dict[Tuple[str, str], List[List[Dict]]],
    ref_by_judge: Dict[str, Dict[str, Dict]],
    judges: Optional[List[str]] = None,
) -> Dict:
    """
    Mean Rubric Accuracy (MRA).

    For each judge, compare rubric-level criteria_met against reference follow_list.
    Uses per-judge reference.
    """
    if judges is None:
        judges = JUDGES

    judge_mra = {}
    judge_n_correct = {}
    judge_n_total = {}

    for judge in judges:
        if judge not in ref_by_judge:
            continue
        ref_data = ref_by_judge[judge]

        n_correct = 0
        n_total = 0

        for gen in GENERATORS:
            if (judge, gen) not in sr_data or gen not in ref_data:
                continue

            judge_rubrics = sr_data[(judge, gen)]
            ref_list = ref_data[gen]["follow_list"]

            for i in range(N_INSTANCES):
                j_rubrics = judge_rubrics[i]
                r_rubrics = ref_list[i]
                assert len(j_rubrics) == len(r_rubrics), (
                    f"Rubric count mismatch at inst {i}, judge={judge}, gen={gen}"
                )

                for j_item, r_val in zip(j_rubrics, r_rubrics):
                    if j_item["criteria_met"] == r_val:
                        n_correct += 1
                    n_total += 1

        if n_total > 0:
            judge_mra[judge] = n_correct / n_total
            judge_n_correct[judge] = n_correct
            judge_n_total[judge] = n_total

    mra = np.mean(list(judge_mra.values())) if judge_mra else float("nan")
    return {
        "mean": mra,
        "per_judge": judge_mra,
        "per_judge_n_correct": judge_n_correct,
        "per_judge_n_total": judge_n_total,
    }


def compute_mrspb(
    sr_data: Dict[Tuple[str, str], List[List[Dict]]],
    ref_by_judge: Dict[str, Dict[str, Dict]],
    error_denom: bool = False,
    family_mode: bool = False,
    include_self_in_family: bool = True,
    judges: Optional[List[str]] = None,
) -> Dict:
    """
    Mean Rubric Self-Preference Bias (MRSPB) or its variants.

    MRSPB_raw = fraction of rubrics where judge says "met" but reference says "not met".
    Uses per-judge reference.
    """

    def _rubric_overest_rate(
        judge: str, target_gen: str, ref_data: Dict[str, Dict]
    ) -> Optional[Tuple[float, int, int]]:
        if (judge, target_gen) not in sr_data or target_gen not in ref_data:
            return None

        n_overest = 0
        n_total = 0

        judge_rubrics = sr_data[(judge, target_gen)]
        ref_list = ref_data[target_gen]["follow_list"]

        for i in range(N_INSTANCES):
            j_rubrics = judge_rubrics[i]
            r_rubrics = ref_list[i]

            for j_item, r_val in zip(j_rubrics, r_rubrics):
                is_positive = j_item["points"] > 0

                if error_denom:
                    # Only count rubrics where overestimation is possible:
                    # Positive: ref=False (could be falsely "met")
                    # Negative: ref=True (could be falsely "not met" to avoid penalty)
                    if is_positive and r_val:
                        continue
                    if not is_positive and not r_val:
                        continue

                n_total += 1
                if is_positive:
                    # Positive rubric: overest = judge says met, ref says not met
                    if j_item["criteria_met"] and not r_val:
                        n_overest += 1
                else:
                    # Negative rubric: overest = judge says not met, ref says met
                    if not j_item["criteria_met"] and r_val:
                        n_overest += 1

        if n_total == 0:
            return None
        return n_overest / n_total, n_overest, n_total

    if judges is None:
        judges = JUDGES

    per_judge = {}
    per_judge_raw = {}
    per_judge_other = {}
    per_judge_ratio = {}
    per_judge_n_overest_self = {}
    per_judge_n_total_self = {}
    per_judge_n_overest_other = {}
    per_judge_n_total_other = {}

    for judge in judges:
        if judge not in ref_by_judge:
            continue
        ref_data = ref_by_judge[judge]

        if family_mode:
            target_gens = get_family_generators(judge, include_self=include_self_in_family)
        else:
            target_gens = [judge]

        if not target_gens:
            continue

        raw_rates = []
        total_n_overest_self = 0
        total_n_total_self = 0
        for tg in target_gens:
            result = _rubric_overest_rate(judge, tg, ref_data)
            if result is not None:
                rate, n_over, n_tot = result
                raw_rates.append(rate)
                total_n_overest_self += n_over
                total_n_total_self += n_tot

        if not raw_rates:
            continue
        mrspb_raw = np.mean(raw_rates)

        other_gens = get_other_generators(judge)
        other_rates = []
        total_n_overest_other = 0
        total_n_total_other = 0
        for og in other_gens:
            result = _rubric_overest_rate(judge, og, ref_data)
            if result is not None:
                rate, n_over, n_tot = result
                other_rates.append(rate)
                total_n_overest_other += n_over
                total_n_total_other += n_tot

        if not other_rates:
            continue
        mrspb_other = np.mean(other_rates)

        mrspb = mrspb_raw - mrspb_other
        ratio = mrspb_raw / mrspb_other if mrspb_other > 0 else float("inf")

        per_judge[judge] = mrspb
        per_judge_raw[judge] = mrspb_raw
        per_judge_other[judge] = mrspb_other
        per_judge_ratio[judge] = ratio
        per_judge_n_overest_self[judge] = total_n_overest_self
        per_judge_n_total_self[judge] = total_n_total_self
        per_judge_n_overest_other[judge] = total_n_overest_other
        per_judge_n_total_other[judge] = total_n_total_other

    mean_val = np.mean(list(per_judge.values())) if per_judge else float("nan")
    mean_raw = np.mean(list(per_judge_raw.values())) if per_judge_raw else float("nan")
    mean_other = np.mean(list(per_judge_other.values())) if per_judge_other else float("nan")
    ratio_vals = [v for v in per_judge_ratio.values() if np.isfinite(v)]
    mean_ratio = np.mean(ratio_vals) if ratio_vals else float("nan")
    return {
        "mean": mean_val,
        "mean_raw": mean_raw,
        "mean_other": mean_other,
        "mean_ratio": mean_ratio,
        "per_judge": per_judge,
        "per_judge_raw": per_judge_raw,
        "per_judge_other": per_judge_other,
        "per_judge_ratio": per_judge_ratio,
        "per_judge_n_overest_self": per_judge_n_overest_self,
        "per_judge_n_total_self": per_judge_n_total_self,
        "per_judge_n_overest_other": per_judge_n_overest_other,
        "per_judge_n_total_other": per_judge_n_total_other,
    }
