"""
Data loading module for IFEval judge prompting methods analysis.

Loads experimental data from ducttape output directories and provides
structured access to generation, reference, and evaluation data for
SR (Single Rubric), AR (All Rubrics), DA (Direct Assessment), and
PWC (Pairwise Comparison) methods.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

DATA_ROOT = Path("/mnt/data/jpombal/checklist-bias/experiments/ducttape_outputs")

# 12 generators whose outputs are evaluated
GENERATORS = [
    "gemma_3_27b_it", "gemma_3_12b_it", "gemma_3_4b_it",
    "llama_4_maverick_17b_128e_instruct", "llama_4_scout_17b_16e_instruct",
    "qwen3_235b_instruct", "qwen3_30b_instruct", "qwen3_4b_instruct",
    "gpt_120b_oss", "gpt_5",
    "claude_4_5_haiku", "claude_4_5_sonnet",
]

# 9 default judges (excludes gpt_5, claude_4_5_haiku, claude_4_5_sonnet)
JUDGES = [
    "gemma_3_27b_it", "gemma_3_12b_it", "gemma_3_4b_it",
    "llama_4_maverick_17b_128e_instruct", "llama_4_scout_17b_16e_instruct",
    "qwen3_235b_instruct", "qwen3_30b_instruct", "qwen3_4b_instruct",
    "gpt_120b_oss",
]

# All 12 models can serve as judges (some have partial data)
ALL_JUDGES = list(GENERATORS)

# Extra judges beyond the default 9 (SR-only + partial AR)
EXTRA_JUDGES = [j for j in ALL_JUDGES if j not in JUDGES]

# Model families for self-preference analysis
FAMILIES = {
    "Gemma": ["gemma_3_27b_it", "gemma_3_12b_it", "gemma_3_4b_it"],
    "Llama": ["llama_4_maverick_17b_128e_instruct", "llama_4_scout_17b_16e_instruct"],
    "Qwen": ["qwen3_235b_instruct", "qwen3_30b_instruct", "qwen3_4b_instruct"],
    "GPT": ["gpt_120b_oss", "gpt_5"],
    "Claude": ["claude_4_5_haiku", "claude_4_5_sonnet"],
}

# Reverse mapping: model name → family name
MODEL_TO_FAMILY = {}
for _family, _models in FAMILIES.items():
    for _model in _models:
        MODEL_TO_FAMILY[_model] = _family

N_INSTANCES = 541  # Expected number of instances per generator


def get_family(model: str) -> str:
    """Get the family name for a model. Raises KeyError if unknown."""
    return MODEL_TO_FAMILY[model]


def is_same_family(model_a: str, model_b: str) -> bool:
    """Check if two models belong to the same family."""
    return get_family(model_a) == get_family(model_b)


def get_other_generators(judge: str) -> List[str]:
    """Get generators that are NOT the judge itself and NOT in the judge's family."""
    judge_family = get_family(judge)
    return [g for g in GENERATORS if get_family(g) != judge_family]


def get_committee_families(members: List[str]) -> set:
    """Get the set of all family names represented in a committee."""
    return {get_family(m) for m in members}


def get_family_generators(judge: str, include_self: bool = True) -> List[str]:
    """Get generators in the same family as the judge."""
    judge_family = get_family(judge)
    gens = [g for g in GENERATORS if get_family(g) == judge_family]
    if not include_self:
        gens = [g for g in gens if g != judge]
    return gens


# ============================================================
# Reference data loading
# ============================================================

def load_reference_data() -> Dict[str, Dict]:
    """
    Load ground-truth (programmatic) evaluation results for all generators.

    Returns:
        Dict mapping generator name to:
            - "follow_all": List[bool] of length N_INSTANCES
              (True if all rubrics met for that instance)
            - "follow_list": List[List[bool]] of length N_INSTANCES
              (per-rubric True/False for each instance)
    """
    ref_data = {}
    for gen in GENERATORS:
        path = DATA_ROOT / "EvaluateIFEvalTrue" / f"Generator.{gen}" / "eval_results_strict.jsonl"
        assert path.exists(), f"Reference file not found: {path}"

        follow_all = []
        follow_list = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                follow_all.append(row["follow_all_instructions"])
                follow_list.append(row["follow_instruction_list"])

        assert len(follow_all) == N_INSTANCES, (
            f"Expected {N_INSTANCES} instances for {gen}, got {len(follow_all)}"
        )

        # Validate consistency: follow_all should equal all(follow_list[i])
        for i in range(N_INSTANCES):
            computed = all(follow_list[i])
            assert follow_all[i] == computed, (
                f"Inconsistency at instance {i} for {gen}: "
                f"follow_all={follow_all[i]} but all(follow_list)={computed}"
            )

        ref_data[gen] = {"follow_all": follow_all, "follow_list": follow_list}

    logger.info(f"Loaded reference data for {len(ref_data)} generators")
    return ref_data


# ============================================================
# Generation data loading (for rubric counts)
# ============================================================

def load_generation_data(ref_data: Dict[str, Dict]) -> Dict[str, List[int]]:
    """
    Load generation data to extract the number of rubrics per instance.

    Args:
        ref_data: Reference data (used for cross-validation of rubric counts).

    Returns:
        Dict mapping generator name to list of rubric counts per instance.
    """
    gen_data = {}
    for gen in GENERATORS:
        path = DATA_ROOT / "GenerateIFEval" / f"Generator.{gen}" / "generation.jsonl"
        assert path.exists(), f"Generation file not found: {path}"

        rubric_counts = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                rubric_counts.append(len(row["rubrics"]))

        assert len(rubric_counts) == N_INSTANCES, (
            f"Expected {N_INSTANCES} instances for {gen}, got {len(rubric_counts)}"
        )

        # Cross-validate: rubric count should match reference follow_list length
        for i in range(N_INSTANCES):
            ref_len = len(ref_data[gen]["follow_list"][i])
            assert rubric_counts[i] == ref_len, (
                f"Rubric count mismatch at instance {i} for {gen}: "
                f"generation has {rubric_counts[i]}, reference has {ref_len}"
            )

        gen_data[gen] = rubric_counts

    logger.info(f"Loaded generation data for {len(gen_data)} generators")
    return gen_data


# ============================================================
# SR and AR data loading (rubric-level evaluations)
# ============================================================

def _load_rubric_eval_data(
    eval_dir: str,
    ref_data: Dict[str, Dict],
    method_name: str,
    judges: Optional[List[str]] = None,
) -> Dict[Tuple[str, str], List[List[bool]]]:
    """
    Shared loader for SR and AR evaluation data (both have identical format).

    Args:
        eval_dir: Directory name under DATA_ROOT (e.g., "EvaluateIFEval").
        ref_data: Reference data for cross-validation.
        method_name: "SR" or "AR" for logging.
        judges: List of judges to load. Defaults to JUDGES.

    Returns:
        Dict mapping (judge, generator) to list of per-instance rubric evaluations.
        Each inner list contains booleans for each rubric's criteria_met value.
    """
    if judges is None:
        judges = JUDGES

    data = {}
    loaded = 0
    missing = 0

    for judge in judges:
        for gen in GENERATORS:
            dir_name = f"Evaluator.{judge}+Generator.{gen}"
            path = DATA_ROOT / eval_dir / dir_name / "evaluation_allresults.json"

            if not path.exists():
                missing += 1
                logger.debug(f"{method_name} missing: {dir_name}")
                continue

            with open(path) as f:
                raw = json.load(f)

            metadata = raw["metadata"]["example_level_metadata"]
            assert len(metadata) == N_INSTANCES, (
                f"{method_name} instance count mismatch for {dir_name}: "
                f"got {len(metadata)}, expected {N_INSTANCES}"
            )

            instance_rubrics = []
            for i, item in enumerate(metadata):
                rubric_items = item["rubric_items"]
                criteria = [r["criteria_met"] for r in rubric_items]

                # Validate rubric count matches reference
                ref_len = len(ref_data[gen]["follow_list"][i])
                assert len(criteria) == ref_len, (
                    f"{method_name} rubric count mismatch at instance {i} for {dir_name}: "
                    f"got {len(criteria)}, expected {ref_len}"
                )

                instance_rubrics.append(criteria)

            data[(judge, gen)] = instance_rubrics
            loaded += 1

    logger.info(f"Loaded {method_name} data: {loaded} judge-generator pairs ({missing} missing)")
    return data


def load_extra_judge_data(ref_data: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Load SR and AR data for the extra judges (gpt_5, claude_4_5_haiku, claude_4_5_sonnet).

    Returns dict with keys "sr" and "ar", each mapping (judge, gen) -> rubric data.
    """
    sr = _load_rubric_eval_data("EvaluateIFEval", ref_data, "SR-extra", judges=EXTRA_JUDGES)
    ar = _load_rubric_eval_data("EvaluateIFEvalAR", ref_data, "AR-extra", judges=EXTRA_JUDGES)
    return {"sr": sr, "ar": ar}


def load_sr_data(ref_data: Dict[str, Dict], judges: Optional[List[str]] = None) -> Dict[Tuple[str, str], List[List[bool]]]:
    """Load Single Rubric (SR) evaluation data."""
    return _load_rubric_eval_data("EvaluateIFEval", ref_data, "SR", judges=judges)


def load_ar_data(ref_data: Dict[str, Dict], judges: Optional[List[str]] = None) -> Dict[Tuple[str, str], List[List[bool]]]:
    """Load All Rubrics (AR) evaluation data."""
    return _load_rubric_eval_data("EvaluateIFEvalAR", ref_data, "AR", judges=judges)


# ============================================================
# DA data loading
# ============================================================

def load_da_data() -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    """
    Load Direct Assessment (DA) evaluation data.

    Note: The DA judge reports score_raw as "N/M" where M is the judge's
    perceived total rubric count. M may NOT match the actual rubric count
    (the judge can hallucinate extra criteria). For analysis, we only care
    whether N == M (judge says all criteria met) vs N < M (some not met).

    Returns:
        Dict mapping (judge, generator) to list of (n_met, n_total) tuples per instance.
        n_met = number of rubrics the judge says are met.
        n_total = total rubrics according to the judge (may differ from ground truth).
    """
    data = {}
    loaded = 0
    missing = 0

    for judge in JUDGES:
        for gen in GENERATORS:
            dir_name = f"Evaluator.{judge}+Generator.{gen}"
            path = DATA_ROOT / "EvaluateIFEvalDA" / dir_name / "evaluation_allresults.json"

            if not path.exists():
                missing += 1
                logger.debug(f"DA missing: {dir_name}")
                continue

            with open(path) as f:
                raw = json.load(f)

            metadata = raw["metadata"]["example_level_metadata"]
            assert len(metadata) == N_INSTANCES, (
                f"DA instance count mismatch for {dir_name}: "
                f"got {len(metadata)}, expected {N_INSTANCES}"
            )

            scores = []
            for i, item in enumerate(metadata):
                score_raw = item["score_raw"]
                # Parse "N/M" format
                parts = score_raw.split("/")
                assert len(parts) == 2, (
                    f"DA score_raw format error at instance {i} for {dir_name}: '{score_raw}'"
                )
                n_met = int(parts[0])
                n_total = int(parts[1])

                assert n_total > 0, (
                    f"DA score_raw denominator is 0 at instance {i} for {dir_name}"
                )
                assert n_met >= 0, (
                    f"DA score_raw n_met is negative at instance {i} for {dir_name}: "
                    f"n_met={n_met}"
                )
                # Note: n_met > n_total can happen (DA judge hallucination).
                # This is treated the same as n_met == n_total (all criteria met).

                scores.append((n_met, n_total))

            data[(judge, gen)] = scores
            loaded += 1

    logger.info(f"Loaded DA data: {loaded} judge-generator pairs ({missing} missing)")
    return data


# ============================================================
# PWC data loading
# ============================================================

def load_pwc_data() -> Dict[Tuple[str, str, str], List[str]]:
    """
    Load raw Pairwise Comparison (PWC) evaluation data (before positional bias resolution).

    Returns:
        Dict mapping (judge, gen_a, gen_b) to list of outcome strings per instance.
        Outcomes are: "A is better", "B is better", or "tie".
        gen_a is the generator whose response is presented first (as A).
    """
    data = {}
    loaded = 0
    missing = 0
    valid_outcomes = {"A is better", "B is better", "tie"}

    for judge in JUDGES:
        for gen_a in GENERATORS:
            for gen_b in GENERATORS:
                dir_name = f"Evaluator.{judge}+Generator.{gen_a}+GeneratorB.{gen_b}"
                path = DATA_ROOT / "EvaluateIFEvalPWC" / dir_name / "evaluation_allresults.json"

                if not path.exists():
                    missing += 1
                    logger.debug(f"PWC missing: {dir_name}")
                    continue

                with open(path) as f:
                    raw = json.load(f)

                metadata = raw["metadata"]["example_level_metadata"]
                assert len(metadata) == N_INSTANCES, (
                    f"PWC instance count mismatch for {dir_name}: "
                    f"got {len(metadata)}, expected {N_INSTANCES}"
                )

                outcomes = []
                for i, item in enumerate(metadata):
                    outcome = item["outcome"]
                    assert outcome in valid_outcomes, (
                        f"PWC invalid outcome at instance {i} for {dir_name}: '{outcome}'"
                    )
                    outcomes.append(outcome)

                data[(judge, gen_a, gen_b)] = outcomes
                loaded += 1

    logger.info(f"Loaded PWC data: {loaded} judge-gen_a-gen_b triples ({missing} missing)")
    return data


def resolve_pwc_positional_bias(
    raw_data: Dict[Tuple[str, str, str], List[str]],
) -> Dict[Tuple[str, str, str], List[str]]:
    """
    Resolve positional bias in PWC evaluations by combining both presentation orders.

    For each judge and each unordered pair {X, Y}, we look at the outcome when
    X is presented as A (and Y as B) and when Y is presented as A (and X as B).
    The resolution logic:

    If the judge consistently favors X regardless of position → X wins.
    If the judge's preference flips with position → tie (positional bias).
    If both orderings yield a tie → tie.

    Implementation uses a numeric encoding:
    - For X: outcome_AB gives +1 if "A is better" (X wins), -1 if "B is better" (Y wins), 0 if tie
    - For X: outcome_BA gives +1 if "B is better" (X wins), -1 if "A is better" (Y wins), 0 if tie
    - Sum > 0 → X wins; sum < 0 → Y wins; sum == 0 → tie

    Returns:
        Dict mapping (judge, gen_x, gen_y) to resolved outcomes per instance.
        Outcomes: "X wins", "Y wins", or "tie".
        Only canonical pairs where gen_x < gen_y (lexicographic) are included.
        Self-pairs (gen_x == gen_y) are excluded.
    """
    resolved = {}

    for judge in JUDGES:
        for i, gen_x in enumerate(GENERATORS):
            for gen_y in GENERATORS[i + 1:]:  # gen_x < gen_y lexicographically by list order
                # We need both orderings
                key_xy = (judge, gen_x, gen_y)  # X as A, Y as B
                key_yx = (judge, gen_y, gen_x)  # Y as A, X as B

                if key_xy not in raw_data or key_yx not in raw_data:
                    logger.debug(
                        f"PWC resolution skipped for judge={judge}, "
                        f"pair=({gen_x}, {gen_y}): missing ordering(s)"
                    )
                    continue

                outcomes_xy = raw_data[key_xy]  # X is A, Y is B
                outcomes_yx = raw_data[key_yx]  # Y is A, X is B

                resolved_outcomes = []
                for inst_idx in range(N_INSTANCES):
                    o_xy = outcomes_xy[inst_idx]
                    o_yx = outcomes_yx[inst_idx]

                    # Encode from X's perspective
                    # When X is A: "A is better" → X wins (+1), "B is better" → Y wins (-1)
                    x_from_xy = 1 if o_xy == "A is better" else (-1 if o_xy == "B is better" else 0)
                    # When Y is A: "B is better" → X wins (+1), "A is better" → Y wins (-1)
                    x_from_yx = 1 if o_yx == "B is better" else (-1 if o_yx == "A is better" else 0)

                    total = x_from_xy + x_from_yx
                    if total > 0:
                        resolved_outcomes.append("X wins")
                    elif total < 0:
                        resolved_outcomes.append("Y wins")
                    else:
                        resolved_outcomes.append("tie")

                resolved[(judge, gen_x, gen_y)] = resolved_outcomes

    n_pairs = len(resolved)
    logger.info(f"Resolved PWC positional bias: {n_pairs} (judge, gen_x, gen_y) triples")
    return resolved


# ============================================================
# Convenience: load everything
# ============================================================

def load_all_data() -> dict:
    """
    Load all data needed for the analysis.

    Returns a dict with keys:
        - "ref": reference data
        - "gen": generation data (rubric counts)
        - "sr": SR evaluation data
        - "ar": AR evaluation data
        - "da": DA evaluation data
        - "pwc_raw": raw PWC evaluation data
        - "pwc": resolved PWC evaluation data (positional bias removed)
    """
    logger.info("Loading reference data...")
    ref = load_reference_data()

    logger.info("Loading generation data...")
    gen = load_generation_data(ref)

    logger.info("Loading SR data...")
    sr = load_sr_data(ref)

    logger.info("Loading AR data...")
    ar = load_ar_data(ref)

    logger.info("Loading DA data...")
    da = load_da_data()

    logger.info("Loading PWC data...")
    pwc_raw = load_pwc_data()

    logger.info("Resolving PWC positional bias...")
    pwc = resolve_pwc_positional_bias(pwc_raw)

    return {
        "ref": ref,
        "gen": gen,
        "sr": sr,
        "ar": ar,
        "da": da,
        "pwc_raw": pwc_raw,
        "pwc": pwc,
    }
