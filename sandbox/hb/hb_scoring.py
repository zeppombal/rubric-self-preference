"""
HealthBench instance scoring module.

Two scoring modes:
- Weighted: uses actual rubric point values (-10 to +10)
- Uniform: treats positive rubrics as +1, negative rubrics as -1

Formula (from healthbench_eval.py):
    score = clip(sum(points for met rubrics) / sum(positive points), 0, 1)
"""

from typing import Dict, List


def instance_score_weighted(rubric_items: List[Dict]) -> float:
    """
    Weighted HealthBench instance score.

    score = clip(achieved_points / total_possible_points, 0, 1)
    where:
      achieved_points = sum of points for rubrics where criteria_met=True
      total_possible_points = sum of points for rubrics where points > 0
    """
    total_possible = sum(r["points"] for r in rubric_items if r["points"] > 0)
    if total_possible <= 0:
        return 0.0
    achieved = sum(r["points"] for r in rubric_items if r["criteria_met"])
    return max(0.0, min(1.0, achieved / total_possible))


def instance_score_uniform(rubric_items: List[Dict]) -> float:
    """
    Uniform HealthBench instance score.

    Same formula but positive rubrics count as +1, negative as -1.
    """
    n_positive = sum(1 for r in rubric_items if r["points"] > 0)
    if n_positive <= 0:
        return 0.0
    achieved = sum(
        (1 if r["points"] > 0 else -1)
        for r in rubric_items
        if r["criteria_met"]
    )
    return max(0.0, min(1.0, achieved / n_positive))


def instance_score(rubric_items: List[Dict], scoring_mode: str) -> float:
    """Dispatch to weighted or uniform scoring."""
    if scoring_mode == "weighted":
        return instance_score_weighted(rubric_items)
    elif scoring_mode == "uniform":
        return instance_score_uniform(rubric_items)
    else:
        raise ValueError(f"Unknown scoring mode: {scoring_mode}")


def ref_instance_score(
    follow_list_bools: List[bool],
    gen_rubrics: List[Dict],
    scoring_mode: str,
) -> float:
    """
    Compute instance score from reference booleans + generation rubric points.

    Constructs synthetic rubric items by pairing reference met/unmet decisions
    with the rubric point values from generation data.
    """
    items = [
        {"criteria_met": met, "points": rub["points"]}
        for met, rub in zip(follow_list_bools, gen_rubrics)
    ]
    return instance_score(items, scoring_mode)
