"""
Metrics computation module for IFEval judge prompting methods analysis.

Computes system-level, instance-level, and rubric-level metrics for
accuracy and self-preference bias across SR, AR, DA, and PWC methods.

Instance-level scoring uses fraction of rubrics met (not binary all-or-nothing).

All metric definitions follow the master-plan.md specification.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from data_loading import (
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
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ============================================================
# System-Level Scores
# ============================================================


def compute_reference_system_scores(
    ref_data: Dict[str, Dict],
) -> Dict[str, float]:
    """
    Compute reference (ground truth) system-level scores.

    Instance score = fraction of rubrics met.
    Generator system score = mean across all instances.

    Returns:
        Dict mapping generator name to system-level score.
    """
    scores = {}
    for gen in GENERATORS:
        follow_list = ref_data[gen]["follow_list"]
        instance_scores = [sum(fl) / len(fl) for fl in follow_list]
        scores[gen] = np.mean(instance_scores)
    return scores


def compute_reference_pwc_system_scores(
    ref_data: Dict[str, Dict],
) -> Dict[str, float]:
    """
    Compute reference system-level scores using PWC-style scoring (for fair
    comparison with PWC judge scores in MSD metrics).

    For each generator X and each opponent Y (Y != X), for each instance:
    - Instance score = fraction of rubrics met.
    - X wins if score(X) > score(Y), loses if score(X) < score(Y), tie if equal.
    - Score: 1 for win, 0 for loss, 0.5 for tie.
    Generator score = mean across all instances and opponents.

    Returns:
        Dict mapping generator name to PWC-style reference system score.
    """
    # Pre-compute per-instance fraction scores for efficiency (hot loop below)
    ref_fracs: Dict[str, List[float]] = {}
    for gen in GENERATORS:
        ref_fracs[gen] = [sum(fl) / len(fl) for fl in ref_data[gen]["follow_list"]]

    scores = {}
    for gen_x in GENERATORS:
        points = []
        for gen_y in GENERATORS:
            if gen_x == gen_y:
                continue
            for i in range(N_INSTANCES):
                ref_x = ref_fracs[gen_x][i]
                ref_y = ref_fracs[gen_y][i]
                if ref_x > ref_y:
                    points.append(1.0)
                elif ref_x < ref_y:
                    points.append(0.0)
                else:
                    points.append(0.5)
        scores[gen_x] = np.mean(points)
    return scores


def _instance_score_rubric(rubric_evals: List[bool]) -> float:
    """Instance score for SR/AR: fraction of rubrics met."""
    return sum(rubric_evals) / len(rubric_evals)


def _instance_score_da(n_met: int, n_total: int) -> float:
    """Instance score for DA: fraction of rubrics the judge says are met, capped at 1.0."""
    return min(n_met, n_total) / n_total


def compute_system_scores_rubric(
    rubric_data: Dict[Tuple[str, str], List[List[bool]]],
) -> Dict[Tuple[str, str], float]:
    """
    Compute system-level scores for SR or AR methods.

    Instance score = fraction of rubrics the judge says are met.
    System score = mean across instances.

    Returns:
        Dict mapping (judge, generator) to system-level score.
    """
    scores = {}
    for (judge, gen), instance_rubrics in rubric_data.items():
        instance_scores = [
            _instance_score_rubric(rubrics) for rubrics in instance_rubrics
        ]
        scores[(judge, gen)] = np.mean(instance_scores)
    return scores


def compute_system_scores_da(
    da_data: Dict[Tuple[str, str], List[Tuple[int, int]]],
) -> Dict[Tuple[str, str], float]:
    """
    Compute system-level scores for DA method.

    Instance score = min(N, M) / M from score_raw (fraction, capped at 1.0).
    System score = mean across instances.
    """
    scores = {}
    for (judge, gen), instance_scores_raw in da_data.items():
        instance_scores = [_instance_score_da(n, m) for n, m in instance_scores_raw]
        scores[(judge, gen)] = np.mean(instance_scores)
    return scores


def compute_system_scores_pwc(
    resolved_data: Dict[Tuple[str, str, str], List[str]],
) -> Dict[Tuple[str, str], float]:
    """
    Compute system-level scores for PWC method.

    For each generator X, across all opponents Y and instances:
    - 1 if X wins, 0 if Y wins, 0.5 if tie.
    Score = mean across all (instance, opponent) pairs.

    The resolved_data uses canonical pairs where gen_x < gen_y (by list order).
    For gen_x: "X wins" → 1, "Y wins" → 0, "tie" → 0.5
    For gen_y: "X wins" → 0, "Y wins" → 1, "tie" → 0.5

    Returns:
        Dict mapping (judge, generator) to PWC system-level score.
    """
    # Accumulate points per (judge, generator)
    points: Dict[Tuple[str, str], List[float]] = {}

    for (judge, gen_x, gen_y), outcomes in resolved_data.items():
        # Initialize accumulators
        for gen in [gen_x, gen_y]:
            key = (judge, gen)
            if key not in points:
                points[key] = []

        for outcome in outcomes:
            if outcome == "X wins":
                points[(judge, gen_x)].append(1.0)
                points[(judge, gen_y)].append(0.0)
            elif outcome == "Y wins":
                points[(judge, gen_x)].append(0.0)
                points[(judge, gen_y)].append(1.0)
            elif outcome == "tie":
                points[(judge, gen_x)].append(0.5)
                points[(judge, gen_y)].append(0.5)
            else:
                raise ValueError(f"Unexpected resolved outcome: '{outcome}'")

    scores = {}
    for key, pts in points.items():
        scores[key] = np.mean(pts)
    return scores


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
    ref_scores: Dict[str, float],
    judges: Optional[List[str]] = None,
) -> Dict:
    """
    Mean Pairwise Accuracy (MPA).

    For each judge, check each pair of generators for concordance
    (judge and reference agree on ordering). PA = fraction concordant.
    MPA = mean PA across judges.

    Sanity check: PA should equal (Kendall tau + 1) / 2.

    Returns:
        Dict with keys: "mean", "per_judge", "per_judge_n_concordant", "per_judge_n_total"
    """
    if judges is None:
        judges = JUDGES

    judge_pa = {}
    judge_n_concordant = {}
    judge_n_total = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores:
            continue

        # Get generators present in both judge and reference scores
        gens = [g for g in GENERATORS if g in j_scores and g in ref_scores]
        assert (
            len(gens) >= 2
        ), f"Need at least 2 generators for MPA, got {len(gens)} for {judge}"

        n_concordant = 0
        n_total = 0
        for i in range(len(gens)):
            for j in range(i + 1, len(gens)):
                g1, g2 = gens[i], gens[j]
                # Judge ordering
                j_diff = j_scores[g1] - j_scores[g2]
                # Reference ordering
                r_diff = ref_scores[g1] - ref_scores[g2]
                # Concordant if same sign (or both zero)
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

        # Sanity check: PA ≈ (tau + 1) / 2.
        j_arr = np.array([j_scores[g] for g in gens])
        r_arr = np.array([ref_scores[g] for g in gens])
        tau, _ = scipy_stats.kendalltau(j_arr, r_arr)
        expected_pa = (tau + 1) / 2
        if abs(pa - expected_pa) > 0.05:
            logger.warning(
                f"PA vs tau divergence for {judge}: PA={pa:.4f}, "
                f"(tau+1)/2={expected_pa:.4f} (ties likely cause)"
            )

    mpa = np.mean(list(judge_pa.values()))
    return {
        "mean": mpa,
        "per_judge": judge_pa,
        "per_judge_n_concordant": judge_n_concordant,
        "per_judge_n_total": judge_n_total,
    }


def compute_mrd(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
    judges: Optional[List[str]] = None,
) -> Dict:
    """
    Mean Ranking Difference (MRD).

    For each judge, rank generators by judge and reference scores,
    compute mean absolute rank difference across generators.
    MRD = mean across judges.

    Returns:
        Dict with keys: "mean", "per_judge", "per_judge_n_generators"
    """
    if judges is None:
        judges = JUDGES

    judge_mrd = {}
    judge_n_gens = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores:
            continue

        gens = [g for g in GENERATORS if g in j_scores and g in ref_scores]

        # Rank by scores (higher score → lower rank number, i.e., rank 1 = best)
        j_arr = np.array([j_scores[g] for g in gens])
        r_arr = np.array([ref_scores[g] for g in gens])
        # rankdata gives rank 1 to smallest; we want rank 1 = highest, so negate
        j_ranks = scipy_stats.rankdata(-j_arr, method="average")
        r_ranks = scipy_stats.rankdata(-r_arr, method="average")

        abs_diffs = np.abs(j_ranks - r_ranks)
        judge_mrd[judge] = np.mean(abs_diffs)
        judge_n_gens[judge] = len(gens)

    mrd = np.mean(list(judge_mrd.values()))
    return {"mean": mrd, "per_judge": judge_mrd, "per_judge_n_generators": judge_n_gens}


def compute_mrd_sp(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
    judges: Optional[List[str]] = None,
) -> Dict:
    """
    Mean Ranking Difference – Self-Preference variant (MRD-SP).

    For judge J:
    - d_self = SIGNED rank difference for the generator that IS J
      (judge_rank - ref_rank; negative = judge ranks itself better than reference)
    - d_other = mean SIGNED rank diff for non-self, non-family generators
    - MRD-SP_J = d_self - d_other

    Negative MRD-SP → judge inflates its own rank relative to how it treats others.

    Returns:
        Dict with keys: "mean", "per_judge", "per_judge_d_self", "per_judge_d_other",
                         "per_judge_n_others"
    """
    if judges is None:
        judges = JUDGES

    judge_vals = {}
    judge_d_self = {}
    judge_d_other = {}
    judge_n_others = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores:
            continue

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

        others = get_other_generators(judge)
        d_others = [gen_to_diff[g] for g in others if g in gen_to_diff]
        assert len(d_others) > 0, f"No 'other' generators found for judge {judge}"
        d_other = np.mean(d_others)

        judge_vals[judge] = d_self - d_other
        judge_d_self[judge] = d_self
        judge_d_other[judge] = d_other
        judge_n_others[judge] = len(d_others)

    mean_val = np.mean(list(judge_vals.values()))
    return {
        "mean": mean_val,
        "mean_d_self": (
            np.mean(list(judge_d_self.values())) if judge_d_self else float("nan")
        ),
        "mean_d_other": (
            np.mean(list(judge_d_other.values())) if judge_d_other else float("nan")
        ),
        "per_judge": judge_vals,
        "per_judge_d_self": judge_d_self,
        "per_judge_d_other": judge_d_other,
        "per_judge_n_others": judge_n_others,
    }


def compute_mrd_fsp(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
    include_self_in_family: bool = True,
    judges: Optional[List[str]] = None,
) -> Dict:
    """
    Mean Ranking Difference – Family Self-Preference variant (MRD-FSP).

    d_family = mean SIGNED rank diff for same-family generators.
    d_nonfamily = mean SIGNED rank diff for different-family generators.
    MRD-FSP = d_family - d_nonfamily.

    Negative → judge inflates family ranks more than non-family ranks.

    Returns:
        Dict with keys: "mean", "per_judge", "per_judge_d_family", "per_judge_d_nonfamily",
                         "per_judge_n_family", "per_judge_n_nonfamily"
    """
    if judges is None:
        judges = JUDGES

    judge_vals = {}
    judge_d_family = {}
    judge_d_nonfamily = {}
    judge_n_family = {}
    judge_n_nonfamily = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores:
            continue

        gens = [g for g in GENERATORS if g in j_scores and g in ref_scores]
        j_arr = np.array([j_scores[g] for g in gens])
        r_arr = np.array([ref_scores[g] for g in gens])
        j_ranks = scipy_stats.rankdata(-j_arr, method="average")
        r_ranks = scipy_stats.rankdata(-r_arr, method="average")
        signed_diffs = j_ranks - r_ranks

        gen_to_diff = {g: signed_diffs[idx] for idx, g in enumerate(gens)}

        family_gens = [
            g
            for g in get_family_generators(judge, include_self=include_self_in_family)
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
        "mean_d_family": (
            np.mean(list(judge_d_family.values())) if judge_d_family else float("nan")
        ),
        "mean_d_nonfamily": (
            np.mean(list(judge_d_nonfamily.values()))
            if judge_d_nonfamily
            else float("nan")
        ),
        "per_judge": judge_vals,
        "per_judge_d_family": judge_d_family,
        "per_judge_d_nonfamily": judge_d_nonfamily,
        "per_judge_n_family": judge_n_family,
        "per_judge_n_nonfamily": judge_n_nonfamily,
    }


def compute_msd(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
    judges: Optional[List[str]] = None,
) -> Dict:
    """
    Mean Score Delta (MSD).

    For each judge J: delta_g = judge_score(g) - ref_score(g).
    MSD_J = mean delta across all generators.
    MSD = mean across judges.

    Returns:
        Dict with keys: "mean", "per_judge", "per_judge_n_generators"
    """
    if judges is None:
        judges = JUDGES

    judge_msd = {}
    judge_n_gens = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores:
            continue

        gens = [g for g in GENERATORS if g in j_scores and g in ref_scores]
        deltas = [j_scores[g] - ref_scores[g] for g in gens]
        judge_msd[judge] = np.mean(deltas)
        judge_n_gens[judge] = len(gens)

    msd = np.mean(list(judge_msd.values()))
    return {"mean": msd, "per_judge": judge_msd, "per_judge_n_generators": judge_n_gens}


def compute_msd_norm(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
    judges: Optional[List[str]] = None,
) -> Dict:
    """
    Mean Score Delta – Normalized (MSD-norm).

    Each delta is normalized by the reference score before averaging.
    delta_norm_g = (judge_score(g) - ref_score(g)) / ref_score(g).

    Returns:
        Dict with keys: "mean", "per_judge", "per_judge_n_generators"
    """
    if judges is None:
        judges = JUDGES

    judge_msd = {}
    judge_n_gens = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores:
            continue

        gens = [g for g in GENERATORS if g in j_scores and g in ref_scores]
        deltas = []
        for g in gens:
            assert ref_scores[g] > 0, f"Reference score is 0 for {g}, cannot normalize"
            deltas.append((j_scores[g] - ref_scores[g]) / ref_scores[g])
        judge_msd[judge] = np.mean(deltas)
        judge_n_gens[judge] = len(gens)

    msd = np.mean(list(judge_msd.values()))
    return {"mean": msd, "per_judge": judge_msd, "per_judge_n_generators": judge_n_gens}


def _compute_msd_sp_generic(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
    normalize: bool = False,
    judges: Optional[List[str]] = None,
) -> Dict:
    """
    Generic MSD Self-Preference computation (used by MSD-SP and MSD-SP-norm).

    delta_self = score delta for the judge itself as generator.
    delta_other = mean score delta for non-self, non-family generators.
    MSD-SP = delta_self - delta_other.

    If normalize=True, deltas are divided by reference scores.

    Returns:
        Dict with keys: "mean", "per_judge", "per_judge_d_self", "per_judge_d_other",
                         "per_judge_n_others"
    """
    if judges is None:
        judges = JUDGES

    judge_vals = {}
    judge_d_self = {}
    judge_d_other = {}
    judge_n_others = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores:
            continue

        gens = [g for g in GENERATORS if g in j_scores and g in ref_scores]

        def delta(g):
            d = j_scores[g] - ref_scores[g]
            if normalize:
                assert ref_scores[g] > 0, f"Ref score 0 for {g}"
                d /= ref_scores[g]
            return d

        if judge not in j_scores:
            continue
        d_self = delta(judge)

        others = [g for g in get_other_generators(judge) if g in j_scores]
        d_other = np.mean([delta(g) for g in others])

        judge_vals[judge] = d_self - d_other
        judge_d_self[judge] = d_self
        judge_d_other[judge] = d_other
        judge_n_others[judge] = len(others)

    mean_val = np.mean(list(judge_vals.values()))
    return {
        "mean": mean_val,
        "mean_d_self": (
            np.mean(list(judge_d_self.values())) if judge_d_self else float("nan")
        ),
        "mean_d_other": (
            np.mean(list(judge_d_other.values())) if judge_d_other else float("nan")
        ),
        "per_judge": judge_vals,
        "per_judge_d_self": judge_d_self,
        "per_judge_d_other": judge_d_other,
        "per_judge_n_others": judge_n_others,
    }


def compute_msd_sp(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
) -> Dict:
    """MSD Self-Preference (MSD-SP). See _compute_msd_sp_generic."""
    return _compute_msd_sp_generic(system_scores, ref_scores, normalize=False)


def compute_msd_sp_norm(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
) -> Dict:
    """MSD Self-Preference Normalized (MSD-SP-norm). See _compute_msd_sp_generic."""
    return _compute_msd_sp_generic(system_scores, ref_scores, normalize=True)


def _compute_msd_fsp_generic(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
    normalize: bool = False,
    include_self_in_family: bool = True,
    judges: Optional[List[str]] = None,
) -> Dict:
    """
    Generic MSD Family Self-Preference computation.

    delta_family = mean score delta for same-family generators.
    delta_nonfamily = mean score delta for different-family generators.
    MSD-FSP = delta_family - delta_nonfamily.

    Returns:
        Dict with keys: "mean", "per_judge", "per_judge_d_family", "per_judge_d_nonfamily",
                         "per_judge_n_family", "per_judge_n_nonfamily"
    """
    if judges is None:
        judges = JUDGES

    judge_vals = {}
    judge_d_family = {}
    judge_d_nonfamily = {}
    judge_n_family = {}
    judge_n_nonfamily = {}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores:
            continue

        def delta(g):
            d = j_scores[g] - ref_scores[g]
            if normalize:
                assert ref_scores[g] > 0, f"Ref score 0 for {g}"
                d /= ref_scores[g]
            return d

        family_gens = [
            g
            for g in get_family_generators(judge, include_self=include_self_in_family)
            if g in j_scores
        ]
        other_gens = [g for g in get_other_generators(judge) if g in j_scores]

        if not family_gens or not other_gens:
            continue

        d_family = np.mean([delta(g) for g in family_gens])
        d_nonfamily = np.mean([delta(g) for g in other_gens])

        judge_vals[judge] = d_family - d_nonfamily
        judge_d_family[judge] = d_family
        judge_d_nonfamily[judge] = d_nonfamily
        judge_n_family[judge] = len(family_gens)
        judge_n_nonfamily[judge] = len(other_gens)

    mean_val = np.mean(list(judge_vals.values())) if judge_vals else float("nan")
    return {
        "mean": mean_val,
        "mean_d_family": (
            np.mean(list(judge_d_family.values())) if judge_d_family else float("nan")
        ),
        "mean_d_nonfamily": (
            np.mean(list(judge_d_nonfamily.values()))
            if judge_d_nonfamily
            else float("nan")
        ),
        "per_judge": judge_vals,
        "per_judge_d_family": judge_d_family,
        "per_judge_d_nonfamily": judge_d_nonfamily,
        "per_judge_n_family": judge_n_family,
        "per_judge_n_nonfamily": judge_n_nonfamily,
    }


def compute_msd_fsp(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
    include_self_in_family: bool = True,
) -> Dict:
    """MSD Family Self-Preference (MSD-FSP)."""
    return _compute_msd_fsp_generic(
        system_scores,
        ref_scores,
        normalize=False,
        include_self_in_family=include_self_in_family,
    )


def compute_msd_fsp_norm(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
    include_self_in_family: bool = True,
) -> Dict:
    """MSD Family Self-Preference Normalized (MSD-FSP-norm)."""
    return _compute_msd_fsp_generic(
        system_scores,
        ref_scores,
        normalize=True,
        include_self_in_family=include_self_in_family,
    )


def compute_mrd_fosp(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
) -> Dict:
    """MRD Family-Only Self-Preference (MRD-FOSP). Family excluding self."""
    return compute_mrd_fsp(system_scores, ref_scores, include_self_in_family=False)


def compute_msd_fosp(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
) -> Dict:
    """MSD Family-Only Self-Preference (MSD-FOSP). Family excluding self."""
    return _compute_msd_fsp_generic(
        system_scores, ref_scores, normalize=False, include_self_in_family=False
    )


def compute_msd_fosp_norm(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
) -> Dict:
    """MSD Family-Only Self-Preference Normalized (MSD-FOSP-norm). Family excluding self."""
    return _compute_msd_fsp_generic(
        system_scores, ref_scores, normalize=True, include_self_in_family=False
    )


def compute_per_generator_deltas(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
    normalize: bool = False,
    judges: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Mean score delta per generator across all judges.

    For each generator G: mean_j(judge_score(G) - ref_score(G)).
    If normalize=True, each delta is divided by ref_score(G).
    Shows which generators are systematically over/under-scored.

    Returns:
        Dict mapping generator name → mean score delta.
    """
    if judges is None:
        judges = JUDGES

    gen_deltas = {}
    for gen in GENERATORS:
        if gen not in ref_scores:
            continue
        deltas = []
        for judge in judges:
            if (judge, gen) in system_scores:
                d = system_scores[(judge, gen)] - ref_scores[gen]
                if normalize:
                    assert ref_scores[gen] > 0, f"Ref score 0 for {gen}"
                    d /= ref_scores[gen]
                deltas.append(d)
        if deltas:
            gen_deltas[gen] = np.mean(deltas)
    return gen_deltas


def compute_per_generator_rank_deltas(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
    judges: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Mean signed rank delta per generator across all judges.

    For each judge, rank generators by judge and reference scores, compute
    judge_rank - ref_rank for each generator. Then average across judges.

    Negative = judges rank this generator better than reference.

    Returns:
        Dict mapping generator name → mean signed rank delta.
    """
    if judges is None:
        judges = JUDGES

    # Accumulate rank differences per generator across judges
    gen_rank_diffs: Dict[str, List[float]] = {g: [] for g in GENERATORS}

    for judge in judges:
        j_scores = _get_judge_gen_scores(system_scores, judge)
        if not j_scores:
            continue

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
# Instance-Level Metrics
# ============================================================


def _compute_instance_scores_rubric(
    rubric_data: Dict[Tuple[str, str], List[List[bool]]],
) -> Dict[Tuple[str, str], List[float]]:
    """Convert rubric-level data to per-instance fraction scores."""
    return {
        (j, g): [_instance_score_rubric(r) for r in rubrics]
        for (j, g), rubrics in rubric_data.items()
    }


def _compute_instance_scores_da(
    da_data: Dict[Tuple[str, str], List[Tuple[int, int]]],
) -> Dict[Tuple[str, str], List[float]]:
    """Convert DA data to per-instance fraction scores."""
    return {
        (j, g): [_instance_score_da(n, m) for n, m in scores]
        for (j, g), scores in da_data.items()
    }


def _compute_ref_instance_scores(
    ref_data: Dict[str, Dict],
) -> Dict[str, List[float]]:
    """Convert reference data to per-instance fraction scores (fraction of rubrics met)."""
    return {
        gen: [sum(fl) / len(fl) for fl in ref_data[gen]["follow_list"]]
        for gen in GENERATORS
    }


def compute_mipa_non_pwc(
    instance_scores: Dict[Tuple[str, str], List[float]],
    ref_data: Dict[str, Dict],
    judges: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float], Dict[str, int], Dict[str, int]]:
    """
    Mean Instance Pairwise Accuracy (MIPA) for non-PWC methods (SR, AR, DA).

    For each judge, instance, and pair (X, Y): does the judge's instance-level
    ordering agree with the reference?
    - Judge ordering: compare judge_score(X, i) vs judge_score(Y, i)
    - Reference ordering: compare ref(X, i) vs ref(Y, i)
    - Agree if same direction (both X>Y, both X<Y, or both tied)

    MIPA_J = mean agreement across all instances and pairs.
    MIPA = mean across judges.

    Returns:
        (MIPA, per_judge MIPA, per_judge n_agree, per_judge n_total)
    """
    if judges is None:
        judges = JUDGES

    ref_inst = _compute_ref_instance_scores(ref_data)
    judge_mipa = {}
    judge_n_agree = {}
    judge_n_total = {}

    for judge in judges:
        # Get all generators this judge has evaluated
        j_gens = [g for g in GENERATORS if (judge, g) in instance_scores]
        if len(j_gens) < 2:
            continue

        n_agree = 0
        n_total = 0

        for i in range(N_INSTANCES):
            for gi in range(len(j_gens)):
                for gj in range(gi + 1, len(j_gens)):
                    gx, gy = j_gens[gi], j_gens[gj]
                    # Judge ordering
                    j_diff = (
                        instance_scores[(judge, gx)][i]
                        - instance_scores[(judge, gy)][i]
                    )
                    # Reference ordering
                    r_diff = ref_inst[gx][i] - ref_inst[gy][i]
                    # Agree if same sign (including both zero)
                    if (
                        (j_diff > 0 and r_diff > 0)
                        or (j_diff < 0 and r_diff < 0)
                        or (j_diff == 0 and r_diff == 0)
                    ):
                        n_agree += 1
                    n_total += 1

        judge_mipa[judge] = n_agree / n_total
        judge_n_agree[judge] = n_agree
        judge_n_total[judge] = n_total

    mipa = np.mean(list(judge_mipa.values()))
    return mipa, judge_mipa, judge_n_agree, judge_n_total


def compute_mipa_pwc(
    resolved_data: Dict[Tuple[str, str, str], List[str]],
    ref_data: Dict[str, Dict],
    judges: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float], Dict[str, int], Dict[str, int]]:
    """
    MIPA for PWC method.

    For each judge, instance, and canonical pair (X, Y):
    - Judge outcome: "X wins" → +1, "Y wins" → -1, "tie" → 0
    - Reference outcome: compare ref(X, i) vs ref(Y, i) (fraction scores)
    - Agree if same sign.

    Returns:
        (MIPA, per_judge MIPA, per_judge n_agree, per_judge n_total)
    """
    if judges is None:
        judges = JUDGES

    ref_inst = _compute_ref_instance_scores(ref_data)
    judge_mipa = {}
    judge_n_agree = {}
    judge_n_total = {}

    for judge in judges:
        n_agree = 0
        n_total = 0

        for (j, gx, gy), outcomes in resolved_data.items():
            if j != judge:
                continue
            for i in range(N_INSTANCES):
                # Judge outcome for X
                outcome = outcomes[i]
                j_sign = (
                    1 if outcome == "X wins" else (-1 if outcome == "Y wins" else 0)
                )
                # Reference outcome for X
                r_diff = ref_inst[gx][i] - ref_inst[gy][i]
                r_sign = 1 if r_diff > 0 else (-1 if r_diff < 0 else 0)
                if j_sign == r_sign:
                    n_agree += 1
                n_total += 1

        if n_total > 0:
            judge_mipa[judge] = n_agree / n_total
            judge_n_agree[judge] = n_agree
            judge_n_total[judge] = n_total

    mipa = np.mean(list(judge_mipa.values()))
    return mipa, judge_mipa, judge_n_agree, judge_n_total


# ============================================================
# Instance-Level Self-Preference Bias (MISPB)
# ============================================================


def _compute_overestimation_rate_non_pwc(
    instance_scores: Dict[Tuple[str, str], List[float]],
    ref_data: Dict[str, Dict],
    judge: str,
    target_gen: str,
    error_denom: bool = False,
    generators: Optional[List[str]] = None,
    ref_inst: Optional[Dict[str, List[float]]] = None,
) -> Optional[Dict[str, Union[float, int]]]:
    """
    Compute the overestimation rate of a target generator by a specific judge.

    Overestimation: judge gives target_gen a more favorable outcome than reference.
    For each instance and each opponent Y (!=target_gen):
    - Judge says target wins or ties, but reference says opponent wins → overestimation
    - Specifically: overestimation if judge_outcome > ref_outcome (in {-1, 0, +1})

    Sub-types of overestimation (all satisfy j_sign > r_sign):
    - tie-to-win (t2w): ref=0, judge=+1 — ref says tie, judge says target wins
    - loss-to-win (l2w): ref=-1, judge=+1 — ref says target loses, judge says target wins
    - loss-to-tie (l2t): ref=-1, judge=0 — ref says target loses, judge says tie

    Args:
        instance_scores: Judge's per-instance fraction scores.
        ref_data: Reference data.
        judge: The judge model.
        target_gen: The generator being assessed for overestimation.
        error_denom: If True, only count instances where reference says opponent is better
                     (for HSPP variant). Under this mode, n_t2w is always 0.

    Returns:
        Dict with keys: rate, n_overest, n_total, n_t2w, n_l2w, n_l2t.
        None if no valid instances.
    """
    if generators is None:
        generators = GENERATORS

    if ref_inst is None:
        ref_inst = _compute_ref_instance_scores(ref_data)

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

        for i in range(N_INSTANCES):
            j_target = instance_scores[(judge, target_gen)][i]
            j_opp = instance_scores[(judge, opp)][i]
            r_target = ref_inst[target_gen][i]
            r_opp = ref_inst[opp][i]

            # Outcome from target_gen's perspective
            j_sign = np.sign(j_target - j_opp)  # +1, 0, -1
            r_sign = np.sign(r_target - r_opp)

            if error_denom:
                # HSPP: only count when reference says opponent is better (r_sign < 0)
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
    assert (
        n_t2w + n_l2w + n_l2t == n_overest
    ), f"Sub-type sum mismatch: {n_t2w}+{n_l2w}+{n_l2t} != {n_overest}"
    return {
        "rate": n_overest / n_total,
        "n_overest": n_overest,
        "n_total": n_total,
        "n_t2w": n_t2w,
        "n_l2w": n_l2w,
        "n_l2t": n_l2t,
    }


def _compute_overestimation_rate_pwc(
    resolved_data: Dict[Tuple[str, str, str], List[str]],
    ref_data: Dict[str, Dict],
    judge: str,
    target_gen: str,
    error_denom: bool = False,
    generators: Optional[List[str]] = None,
    ref_inst: Optional[Dict[str, List[float]]] = None,
) -> Optional[Dict[str, Union[float, int]]]:
    """
    Compute overestimation rate for PWC method.

    Uses resolved PWC outcomes. For each canonical pair involving target_gen
    and each instance, check if judge overestimates target_gen.

    Returns:
        Dict with keys: rate, n_overest, n_total, n_t2w, n_l2w, n_l2t.
        None if no valid instances.
    """
    if generators is None:
        generators = GENERATORS

    if ref_inst is None:
        ref_inst = _compute_ref_instance_scores(ref_data)

    n_overest = 0
    n_total = 0
    n_t2w = 0
    n_l2w = 0
    n_l2t = 0

    for (j, gx, gy), outcomes in resolved_data.items():
        if j != judge:
            continue

        # Determine if target_gen is involved and who is the opponent
        if gx == target_gen:
            opp = gy
            # In canonical pair (gx, gy): "X wins" favors target
            target_sign_map = {"X wins": 1, "Y wins": -1, "tie": 0}
        elif gy == target_gen:
            opp = gx
            target_sign_map = {"X wins": -1, "Y wins": 1, "tie": 0}
        else:
            continue

        # Skip opponents not in the generators list
        if opp not in generators:
            continue

        for i in range(N_INSTANCES):
            j_sign = target_sign_map[outcomes[i]]

            r_target = ref_inst[target_gen][i]
            r_opp = ref_inst[opp][i]
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
    assert (
        n_t2w + n_l2w + n_l2t == n_overest
    ), f"Sub-type sum mismatch: {n_t2w}+{n_l2w}+{n_l2t} != {n_overest}"
    return {
        "rate": n_overest / n_total,
        "n_overest": n_overest,
        "n_total": n_total,
        "n_t2w": n_t2w,
        "n_l2w": n_l2w,
        "n_l2t": n_l2t,
    }


def compute_mispb(
    method: str,
    data,
    ref_data: Dict[str, Dict],
    error_denom: bool = False,
    family_mode: bool = False,
    include_self_in_family: bool = True,
    judges: Optional[List[str]] = None,
) -> Dict:
    """
    Mean Instance Self-Preference Bias (MISPB) or its variants.

    Args:
        method: One of "sr", "ar", "da", "pwc".
        data: Method-specific data (rubric data, DA data, or resolved PWC data).
        ref_data: Reference data.
        error_denom: If True, compute HSPP variant (only errors in denominator).
        family_mode: If True, compute family variant instead of self variant.

    Returns:
        Dict with keys:
            "mean": mean MISPB across judges
            "per_judge": dict of judge → MISPB value
            "per_judge_raw": dict of judge → MISPB_raw
            "per_judge_other": dict of judge → MISPB_other
            "per_judge_ratio": dict of judge → MISPB_ratio
    """
    if judges is None:
        judges = JUDGES

    # Prepare instance scores (for non-PWC methods)
    if method in ("sr", "ar"):
        instance_scores = _compute_instance_scores_rubric(data)
        overest_fn = lambda judge, gen: _compute_overestimation_rate_non_pwc(
            instance_scores, ref_data, judge, gen, error_denom
        )
    elif method == "da":
        instance_scores = _compute_instance_scores_da(data)
        overest_fn = lambda judge, gen: _compute_overestimation_rate_non_pwc(
            instance_scores, ref_data, judge, gen, error_denom
        )
    elif method == "pwc":
        overest_fn = lambda judge, gen: _compute_overestimation_rate_pwc(
            data, ref_data, judge, gen, error_denom
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    per_judge = {}
    per_judge_raw = {}
    per_judge_other = {}
    per_judge_ratio = {}
    per_judge_n_overest_self = {}
    per_judge_n_total_self = {}
    per_judge_n_overest_other = {}
    per_judge_n_total_other = {}
    # Sub-type counts
    per_judge_n_t2w_self = {}
    per_judge_n_l2w_self = {}
    per_judge_n_l2t_self = {}
    per_judge_n_t2w_other = {}
    per_judge_n_l2w_other = {}
    per_judge_n_l2t_other = {}

    for judge in judges:
        if family_mode:
            # Family variant: target generators are same-family
            # include_self_in_family=True → family including self (MISPB-F)
            # include_self_in_family=False → family excluding self (MISPB-FO)
            target_gens = get_family_generators(
                judge, include_self=include_self_in_family
            )
        else:
            # Self variant: target generator is the judge itself
            target_gens = [judge]

        if not target_gens:
            continue

        # Compute overestimation rate for target (self or family)
        raw_rates = []
        total_n_overest_self = 0
        total_n_total_self = 0
        total_n_t2w_self = 0
        total_n_l2w_self = 0
        total_n_l2t_self = 0
        for tg in target_gens:
            result = overest_fn(judge, tg)
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

        # Compute overestimation rate for "other" generators (non-self, non-family)
        other_gens = get_other_generators(judge)
        other_rates = []
        total_n_overest_other = 0
        total_n_total_other = 0
        total_n_t2w_other = 0
        total_n_l2w_other = 0
        total_n_l2t_other = 0
        for og in other_gens:
            result = overest_fn(judge, og)
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
    mean_other = (
        np.mean(list(per_judge_other.values())) if per_judge_other else float("nan")
    )
    # Mean of per-judge ratios (not ratio of means)
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
# Rubric-Level Metrics (SR and AR only)
# ============================================================


def compute_mra(
    rubric_data: Dict[Tuple[str, str], List[List[bool]]],
    ref_data: Dict[str, Dict],
    judges: Optional[List[str]] = None,
) -> Dict:
    """
    Mean Rubric Accuracy (MRA).

    For each judge, compare rubric-level evaluations against reference
    across all instances, generators, and rubrics.
    MRA_J = fraction of correct rubric evaluations.
    MRA = mean across judges.

    Returns:
        Dict with keys: "mean", "per_judge", "per_judge_n_correct", "per_judge_n_total"
    """
    if judges is None:
        judges = JUDGES

    judge_mra = {}
    judge_n_correct = {}
    judge_n_total = {}

    for judge in judges:
        n_correct = 0
        n_total = 0

        for gen in GENERATORS:
            if (judge, gen) not in rubric_data:
                continue

            judge_rubrics = rubric_data[(judge, gen)]
            ref_list = ref_data[gen]["follow_list"]

            for i in range(N_INSTANCES):
                j_rubrics = judge_rubrics[i]
                r_rubrics = ref_list[i]
                assert len(j_rubrics) == len(
                    r_rubrics
                ), f"Rubric count mismatch at inst {i}, judge={judge}, gen={gen}"

                for j_val, r_val in zip(j_rubrics, r_rubrics):
                    if j_val == r_val:
                        n_correct += 1
                    n_total += 1

        if n_total > 0:
            judge_mra[judge] = n_correct / n_total
            judge_n_correct[judge] = n_correct
            judge_n_total[judge] = n_total

    mra = np.mean(list(judge_mra.values()))
    return {
        "mean": mra,
        "per_judge": judge_mra,
        "per_judge_n_correct": judge_n_correct,
        "per_judge_n_total": judge_n_total,
    }


def compute_mrspb(
    rubric_data: Dict[Tuple[str, str], List[List[bool]]],
    ref_data: Dict[str, Dict],
    error_denom: bool = False,
    family_mode: bool = False,
    include_self_in_family: bool = True,
    judges: Optional[List[str]] = None,
) -> Dict:
    """
    Mean Rubric Self-Preference Bias (MRSPB) or its variants.

    MRSPB_raw = fraction of rubrics where judge says "met" but reference says "not met"
    (false positive rate), restricted to instances where target is the generator.

    MRSPB_other = same for non-self, non-family generators.
    MRSPB = MRSPB_raw - MRSPB_other.

    Args:
        rubric_data: SR or AR rubric-level evaluations.
        ref_data: Reference data.
        error_denom: If True, restrict denominator to rubrics where reference says "not met"
                     (error-denominator variant = rubric-level false positive rate).
        family_mode: If True, compute family variant.

    Returns:
        Dict with keys: "mean", "per_judge", "per_judge_raw", "per_judge_other", "per_judge_ratio"
    """

    def _rubric_overest_rate(
        judge: str, target_gen: str
    ) -> Optional[Tuple[float, int, int]]:
        """Fraction of rubrics overestimated (judge says met, ref says not met).

        Returns:
            Tuple of (rate, n_overest, n_total), or None if no valid rubrics.
        """
        if (judge, target_gen) not in rubric_data:
            return None

        n_overest = 0
        n_total = 0

        judge_rubrics = rubric_data[(judge, target_gen)]
        ref_list = ref_data[target_gen]["follow_list"]

        for i in range(N_INSTANCES):
            j_rubrics = judge_rubrics[i]
            r_rubrics = ref_list[i]

            for j_val, r_val in zip(j_rubrics, r_rubrics):
                if error_denom:
                    # Only count rubrics where reference says "not met"
                    if r_val:
                        continue

                n_total += 1
                # Overestimation: judge says met, reference says not met
                if j_val and not r_val:
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
        if family_mode:
            target_gens = get_family_generators(
                judge, include_self=include_self_in_family
            )
        else:
            target_gens = [judge]

        if not target_gens:
            continue

        raw_rates = []
        total_n_overest_self = 0
        total_n_total_self = 0
        for tg in target_gens:
            result = _rubric_overest_rate(judge, tg)
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
            result = _rubric_overest_rate(judge, og)
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
    mean_other = (
        np.mean(list(per_judge_other.values())) if per_judge_other else float("nan")
    )
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


# ============================================================
# DA float variant functions (for committee-aggregated DA data)
# ============================================================


def compute_system_scores_da_float(
    da_float_data: Dict[Tuple[str, str], List[float]],
) -> Dict[Tuple[str, str], float]:
    """System scores from pre-averaged float scores (committee DA).

    Unlike compute_system_scores_da which takes (n_met, n_total) tuples,
    this accepts pre-computed float instance scores.
    """
    scores = {}
    for key, instance_scores in da_float_data.items():
        scores[key] = np.mean(instance_scores)
    return scores


def _compute_instance_scores_da_float(
    da_float_data: Dict[Tuple[str, str], List[float]],
) -> Dict[Tuple[str, str], List[float]]:
    """Identity transform — committee DA data is already in float format."""
    return da_float_data
