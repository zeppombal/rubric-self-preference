"""
Data loading module for HealthBench SR analysis.

Loads experimental data from ducttape output directories and provides
structured access to generation and evaluation data. Same 12 models
as IFEval, but with HealthBench-specific data format (weighted rubrics,
5000 instances).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

DATA_ROOT = Path("/mnt/data/jpombal/checklist-bias/experiments/ducttape_outputs")

# Same 12 generators as IFEval
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

# All 12 models can serve as judges
ALL_JUDGES = list(GENERATORS)

# Extra judges beyond the default 9
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

N_INSTANCES = 5000

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


# ============================================================
# Family helpers
# ============================================================


def get_family(model: str) -> str:
    """Get the family name for a model."""
    return MODEL_TO_FAMILY[model]


def is_same_family(model_a: str, model_b: str) -> bool:
    """Check if two models belong to the same family."""
    return get_family(model_a) == get_family(model_b)


def get_other_generators(judge: str) -> List[str]:
    """Get generators NOT in the judge's family."""
    judge_family = get_family(judge)
    return [g for g in GENERATORS if get_family(g) != judge_family]


def get_family_generators(judge: str, include_self: bool = True) -> List[str]:
    """Get generators in the same family as the judge."""
    judge_family = get_family(judge)
    gens = [g for g in GENERATORS if get_family(g) == judge_family]
    if not include_self:
        gens = [g for g in gens if g != judge]
    return gens


def get_committee_families(members: List[str]) -> set:
    """Get the set of all family names represented in a committee."""
    return {get_family(m) for m in members}


# ============================================================
# Generation data loading (rubric structure + points)
# ============================================================


def load_generation_data() -> Dict[str, List[List[Dict]]]:
    """
    Load generation data to extract rubric structure (criterion, points, tags).

    Returns:
        Dict mapping generator name to list of instances, each a list of
        rubric dicts: {"criterion": str, "points": int, "tags": list}.
    """
    gen_data = {}
    for gen in GENERATORS:
        path = DATA_ROOT / "GenerateHealthBench" / f"Generator.{gen}" / "generation.jsonl"
        assert path.exists(), f"Generation file not found: {path}"

        instances = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                rubrics = []
                for r in row["rubrics"]:
                    rubrics.append({
                        "criterion": r["criterion"],
                        "points": r["points"],
                        "tags": r.get("tags", []),
                    })
                instances.append(rubrics)

        assert len(instances) == N_INSTANCES, (
            f"Expected {N_INSTANCES} instances for {gen}, got {len(instances)}"
        )
        gen_data[gen] = instances

    # Validate rubric structure is identical across generators (same prompts)
    ref_gen = GENERATORS[0]
    for gen in GENERATORS[1:]:
        for i in range(N_INSTANCES):
            assert len(gen_data[gen][i]) == len(gen_data[ref_gen][i]), (
                f"Rubric count mismatch at instance {i}: "
                f"{gen} has {len(gen_data[gen][i])}, {ref_gen} has {len(gen_data[ref_gen][i])}"
            )
            for j in range(len(gen_data[gen][i])):
                assert gen_data[gen][i][j]["points"] == gen_data[ref_gen][i][j]["points"], (
                    f"Points mismatch at instance {i}, rubric {j}: "
                    f"{gen} has {gen_data[gen][i][j]['points']}, {ref_gen} has {gen_data[ref_gen][i][j]['points']}"
                )

    logger.info(f"Loaded generation data for {len(gen_data)} generators")
    return gen_data


# ============================================================
# SR evaluation data loading
# ============================================================


def _load_sr_data(
    judges: List[str],
    gen_data: Optional[Dict[str, List[List[Dict]]]] = None,
    label: str = "SR",
) -> Dict[Tuple[str, str], List[List[Dict]]]:
    """
    Load HealthBench SR evaluation data.

    Returns:
        Dict mapping (judge, generator) to list of instances, each a list of
        rubric item dicts: {"criteria_met": bool, "points": int, ...}.
    """
    data = {}
    loaded = 0
    missing = 0

    for judge in judges:
        for gen in GENERATORS:
            dir_name = f"Evaluator.{judge}+Generator.{gen}"
            path = DATA_ROOT / "EvaluateHealthBench" / dir_name / "evaluation_allresults.json"

            if not path.exists():
                missing += 1
                logger.debug(f"{label} missing: {dir_name}")
                continue

            with open(path) as f:
                raw = json.load(f)

            metadata = raw["metadata"]["example_level_metadata"]
            assert len(metadata) == N_INSTANCES, (
                f"{label} instance count mismatch for {dir_name}: "
                f"got {len(metadata)}, expected {N_INSTANCES}"
            )

            instance_rubrics = []
            for i, item in enumerate(metadata):
                rubric_items = item["rubric_items"]
                items = []
                for r in rubric_items:
                    items.append({
                        "criteria_met": r["criteria_met"],
                        "points": r["points"],
                    })

                # Validate rubric count matches generation data
                if gen_data is not None:
                    gen_len = len(gen_data[gen][i])
                    assert len(items) == gen_len, (
                        f"{label} rubric count mismatch at instance {i} for {dir_name}: "
                        f"got {len(items)}, expected {gen_len}"
                    )

                instance_rubrics.append(items)

            data[(judge, gen)] = instance_rubrics
            loaded += 1

    logger.info(f"Loaded {label} data: {loaded} judge-generator pairs ({missing} missing)")
    return data


def load_sr_data(
    gen_data: Optional[Dict[str, List[List[Dict]]]] = None,
    judges: Optional[List[str]] = None,
) -> Dict[Tuple[str, str], List[List[Dict]]]:
    """Load SR data for default judges."""
    if judges is None:
        judges = JUDGES
    return _load_sr_data(judges, gen_data, "SR")


def load_extra_judge_data(
    gen_data: Optional[Dict[str, List[List[Dict]]]] = None,
) -> Dict[Tuple[str, str], List[List[Dict]]]:
    """Load SR data for extra judges (gpt_5, claude_4_5_haiku, claude_4_5_sonnet)."""
    return _load_sr_data(EXTRA_JUDGES, gen_data, "SR-extra")


def load_instance_tags() -> list:
    """Load example_tags per instance from generation data (first generator)."""
    ref_gen = GENERATORS[0]
    path = DATA_ROOT / "GenerateHealthBench" / f"Generator.{ref_gen}" / "generation.jsonl"
    assert path.exists(), f"Generation file not found: {path}"
    tags_list = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            tags_list.append(row.get("example_tags", []))
    assert len(tags_list) == N_INSTANCES, (
        f"Expected {N_INSTANCES} instances, got {len(tags_list)}"
    )
    logger.info(f"Loaded instance tags for {len(tags_list)} instances")
    return tags_list


def load_all_data() -> dict:
    """
    Load all HealthBench data.

    Returns dict with keys:
        - "gen": generation data (rubric structure)
        - "sr": SR evaluation data (all judges including extras)
    """
    logger.info("Loading generation data...")
    gen = load_generation_data()

    logger.info("Loading SR data (default judges)...")
    sr = load_sr_data(gen)

    logger.info("Loading SR data (extra judges)...")
    extra = load_extra_judge_data(gen)
    sr.update(extra)
    logger.info(f"Merged extra judge data: {len(extra)} pairs")

    return {"gen": gen, "sr": sr}
