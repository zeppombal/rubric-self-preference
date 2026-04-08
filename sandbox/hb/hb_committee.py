"""
Committee-based reference construction for HealthBench.

In HealthBench, rubrics are not programmatically verifiable, so the reference
must be formed from a committee of judges using majority voting on rubrics.

Key functions:
- build_committee_reference: Form reference from any set of judges
- build_leave_one_family_out_references: Default references for main analysis
- enumerate_committees / run_committee_as_reference_analysis: Committee exploration
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from hb_data_loading import (
    ALL_JUDGES,
    FAMILIES,
    GENERATORS,
    MODEL_TO_FAMILY,
    N_INSTANCES,
    get_family,
    get_other_generators,
)

logger = logging.getLogger(__name__)


# ============================================================
# Committee naming and enumeration
# ============================================================


def committee_name(members: List[str]) -> str:
    """Deterministic sorted name: 'model_a+model_b+model_c'."""
    return "+".join(sorted(members))


def enumerate_committees(
    judges: List[str], min_size: int = 2, max_size: Optional[int] = None,
) -> List[Tuple[str, List[str]]]:
    """Return list of (committee_name, member_list) for all C(N,k), k=min_size..max_size."""
    if max_size is None:
        max_size = len(judges)
    committees = []
    for k in range(min_size, max_size + 1):
        for combo in combinations(sorted(judges), k):
            members = list(combo)
            committees.append((committee_name(members), members))
    return committees


# ============================================================
# Reference construction
# ============================================================


def build_committee_reference(
    sr_data: Dict[Tuple[str, str], List[List[Dict]]],
    members: List[str],
    gen_data: Dict[str, List[List[Dict]]],
) -> Dict[str, Dict]:
    """
    Build a reference from a committee of judges using majority voting on rubrics.

    For each generator, instance, and rubric: count True votes across members.
    If votes > len(available)/2, result is True. Ties → False.

    Returns:
        Dict mapping generator name to {"follow_list": List[List[bool]]}.
        Same format as IFEval reference data for compatibility with metrics.
    """
    ref_data = {}
    for gen in GENERATORS:
        available = [m for m in members if (m, gen) in sr_data]
        if len(available) == 0:
            logger.warning(f"No committee members have data for generator {gen}")
            # Build empty reference with all-False
            follow_list = []
            for i in range(N_INSTANCES):
                n_rubrics = len(gen_data[gen][i])
                follow_list.append([False] * n_rubrics)
            ref_data[gen] = {"follow_list": follow_list}
            continue

        follow_list = []
        for i in range(N_INSTANCES):
            n_rubrics = len(gen_data[gen][i])
            rubric_results = []
            for r_idx in range(n_rubrics):
                votes = sum(
                    1 for m in available
                    if sr_data[(m, gen)][i][r_idx]["criteria_met"]
                )
                # Strict majority: more than half
                rubric_results.append(votes > len(available) / 2)
            follow_list.append(rubric_results)
        ref_data[gen] = {"follow_list": follow_list}

    return ref_data


def build_leave_one_family_out_references(
    sr_data: Dict[Tuple[str, str], List[List[Dict]]],
    gen_data: Dict[str, List[List[Dict]]],
) -> Dict[str, Dict[str, Dict]]:
    """
    Build leave-one-family-out references.

    For each family F, construct a reference from ALL judges NOT in F.
    This ensures that when evaluating a judge from family F, neither the
    judge nor its family members contributed to the reference.

    Returns:
        Dict mapping family name to ref_data (same format as build_committee_reference).
    """
    ref_by_family = {}
    for family_name in FAMILIES:
        excluded_judges = [m for m in ALL_JUDGES if get_family(m) == family_name]
        included_judges = [m for m in ALL_JUDGES if get_family(m) != family_name]
        logger.info(
            f"Building reference excluding family {family_name} "
            f"({len(excluded_judges)} judges excluded, {len(included_judges)} included)"
        )
        ref_by_family[family_name] = build_committee_reference(
            sr_data, included_judges, gen_data
        )
    return ref_by_family


def build_all_judge_reference(
    sr_data: Dict[Tuple[str, str], List[List[Dict]]],
    gen_data: Dict[str, List[List[Dict]]],
) -> Dict[str, Dict]:
    """Build reference from ALL 12 judges using majority voting."""
    return build_committee_reference(sr_data, ALL_JUDGES, gen_data)


# ============================================================
# Committee self-preference helpers
# ============================================================


def get_committee_sp_generators(
    member: str,
    committee_members: List[str],
    variant: str = "standard",
) -> Tuple[List[str], List[str]]:
    """
    Get (self_gens, other_gens) for a committee member's self-preference.

    variant="standard": other = generators not in member's family
    variant="committee_aware": other = generators not in member's family
        AND not in the committee
    """
    member_family = get_family(member)

    if variant == "standard":
        other_gens = [g for g in GENERATORS if MODEL_TO_FAMILY[g] != member_family]
    elif variant == "committee_aware":
        other_gens = [
            g for g in GENERATORS
            if MODEL_TO_FAMILY[g] != member_family and g not in committee_members
        ]
    else:
        raise ValueError(f"Unknown SP variant: {variant}")

    return [member], other_gens
