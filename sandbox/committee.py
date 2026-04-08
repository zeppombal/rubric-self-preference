"""
Committee-based judge evaluation module.

A committee is a group of 2+ individual judges whose evaluations are
aggregated into a single "virtual judge." Aggregation rules:
- SR/AR: majority voting at rubric level (ties → not met)
- DA: average instance-level scores across members
- PWC: majority voting over resolved verdicts (ties → tie)

Self-preference for committees is computed per-member:
- "raw" targets: the committee member itself (or family, depending on variant)
- "other" targets: generators not in ANY family represented in the committee
- Opponents (for overestimation): all generators except committee members
- Self-preference = raw_overestimation - other_overestimation
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from data_loading import (
    GENERATORS,
    MODEL_TO_FAMILY,
    N_INSTANCES,
    get_committee_families,
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
    judges: List[str], min_size: int = 2
) -> List[Tuple[str, List[str]]]:
    """Return list of (committee_name, member_list) for all C(N,k), k=min_size..N."""
    committees = []
    for k in range(min_size, len(judges) + 1):
        for combo in combinations(sorted(judges), k):
            members = list(combo)
            committees.append((committee_name(members), members))
    return committees


# ============================================================
# Aggregation functions
# ============================================================


def aggregate_sr_ar(
    rubric_data: Dict[Tuple[str, str], List[List[bool]]],
    members: List[str],
    generators: List[str],
    n_instances: int,
) -> Dict[Tuple[str, str], List[List[bool]]]:
    """
    Majority voting at rubric level for SR or AR data.

    For each (gen, instance, rubric): count True votes across members.
    If votes > len(available)/2, result is True. Ties (exactly 50%) → False.

    Returns data in same format as individual judge data, keyed by
    (committee_name, gen).
    """
    cname = committee_name(members)
    result = {}
    for gen in generators:
        available = [m for m in members if (m, gen) in rubric_data]
        if len(available) < 2:
            continue
        instance_rubrics = []
        for i in range(n_instances):
            member_rubrics = [rubric_data[(m, gen)][i] for m in available]
            n_rubrics = len(member_rubrics[0])
            aggregated = []
            for r in range(n_rubrics):
                votes = sum(1 for mr in member_rubrics if mr[r])
                # Strict majority: more than half
                aggregated.append(votes > len(available) / 2)
            instance_rubrics.append(aggregated)
        result[(cname, gen)] = instance_rubrics
    return result


def aggregate_da(
    da_data: Dict[Tuple[str, str], List[Tuple[int, int]]],
    members: List[str],
    generators: List[str],
    n_instances: int,
) -> Dict[Tuple[str, str], List[float]]:
    """
    Average per-instance float scores across committee members for DA.

    Returns List[float] per instance (NOT List[Tuple[int,int]]).
    Downstream metrics must use the _float variant functions.
    """
    cname = committee_name(members)
    result = {}
    for gen in generators:
        available = [m for m in members if (m, gen) in da_data]
        if len(available) < 2:
            continue
        instance_scores = []
        for i in range(n_instances):
            scores = []
            for m in available:
                n_met, n_total = da_data[(m, gen)][i]
                scores.append(min(n_met, n_total) / n_total)
            instance_scores.append(sum(scores) / len(scores))
        result[(cname, gen)] = instance_scores
    return result


def aggregate_pwc(
    resolved_data: Dict[Tuple[str, str, str], List[str]],
    members: List[str],
    n_instances: int,
) -> Dict[Tuple[str, str, str], List[str]]:
    """
    Majority vote over resolved PWC verdicts per instance per pair.

    For each canonical pair and instance: sum signs (+1 X wins, -1 Y wins,
    0 tie) across members. sum > 0 → "X wins", sum < 0 → "Y wins",
    sum == 0 → "tie".
    """
    cname = committee_name(members)
    result = {}

    # Collect all canonical pairs that appear in the data
    seen_pairs = set()
    for (judge, gx, gy) in resolved_data:
        seen_pairs.add((gx, gy))

    for (gx, gy) in seen_pairs:
        available = [m for m in members if (m, gx, gy) in resolved_data]
        if len(available) < 2:
            continue
        outcomes = []
        for i in range(n_instances):
            total = 0
            for m in available:
                o = resolved_data[(m, gx, gy)][i]
                if o == "X wins":
                    total += 1
                elif o == "Y wins":
                    total -= 1
                # "tie" adds 0
            if total > 0:
                outcomes.append("X wins")
            elif total < 0:
                outcomes.append("Y wins")
            else:
                outcomes.append("tie")
        result[(cname, gx, gy)] = outcomes
    return result


# ============================================================
# Committee self-preference helpers
# ============================================================


def get_committee_sp_generators(
    member: str,
    committee_members: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Get (self_gens, other_gens) for a committee member's self-preference.

    other = generators not in ANY family represented in the committee.
    This is the same set for all members.
    """
    committee_families = get_committee_families(committee_members)
    other_gens = [g for g in GENERATORS if MODEL_TO_FAMILY[g] not in committee_families]
    return [member], other_gens


# ============================================================
# Committee self-preference metrics
# ============================================================


def compute_committee_mispb(
    method: str,
    committee_data,
    ref_data: Dict[str, Dict],
    cname: str,
    members: List[str],
    error_denom: bool = False,
    instance_scores=None,
    ref_inst=None,
) -> Dict:
    """
    Compute MISPB for a committee, with per-member self-preference.

    For each member M:
    - Use the committee's aggregated evaluations (keyed by cname)
    - "raw" = overestimation rate for member as target
    - "other" = overestimation rate for non-committee-family generators
    - Opponents exclude committee members (no head-to-head)

    Args:
        instance_scores: Pre-computed instance scores (pass to avoid recomputation).
        ref_inst: Pre-computed reference instance scores (pass to avoid recomputation).

    Returns dict with per-member MISPB and median across members.
    """
    from metrics import (
        _compute_instance_scores_da_float,
        _compute_instance_scores_rubric,
        _compute_overestimation_rate_non_pwc,
        _compute_overestimation_rate_pwc,
    )

    # Prepare instance scores for non-PWC methods (if not pre-computed)
    if instance_scores is None and method != "pwc":
        if method in ("sr", "ar"):
            instance_scores = _compute_instance_scores_rubric(committee_data)
        elif method == "da":
            instance_scores = _compute_instance_scores_da_float(committee_data)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Exclude committee members from opponent set (no head-to-head)
    non_committee_gens = [g for g in GENERATORS if g not in members]

    per_member = {}
    per_member_raw = {}
    per_member_other = {}
    per_member_ratio = {}
    per_member_n_overest_self = {}
    per_member_n_total_self = {}
    per_member_n_overest_other = {}
    per_member_n_total_other = {}

    for member in members:
        self_gens, other_gens = get_committee_sp_generators(member, members)

        if not other_gens:
            continue

        # Compute overestimation rate for self (target = member)
        raw_rates = []
        total_n_overest_self = 0
        total_n_total_self = 0
        for tg in self_gens:
            if method == "pwc":
                result = _compute_overestimation_rate_pwc(
                    committee_data, ref_data, cname, tg, error_denom,
                    generators=non_committee_gens, ref_inst=ref_inst,
                )
            else:
                result = _compute_overestimation_rate_non_pwc(
                    instance_scores, ref_data, cname, tg, error_denom,
                    generators=non_committee_gens, ref_inst=ref_inst,
                )
            if result is not None:
                raw_rates.append(result["rate"])
                total_n_overest_self += result["n_overest"]
                total_n_total_self += result["n_total"]

        if not raw_rates:
            continue
        mispb_raw = np.mean(raw_rates)

        # Compute overestimation rate for "other" generators
        other_rates = []
        total_n_overest_other = 0
        total_n_total_other = 0
        for og in other_gens:
            if method == "pwc":
                result = _compute_overestimation_rate_pwc(
                    committee_data, ref_data, cname, og, error_denom,
                    generators=non_committee_gens, ref_inst=ref_inst,
                )
            else:
                result = _compute_overestimation_rate_non_pwc(
                    instance_scores, ref_data, cname, og, error_denom,
                    generators=non_committee_gens, ref_inst=ref_inst,
                )
            if result is not None:
                other_rates.append(result["rate"])
                total_n_overest_other += result["n_overest"]
                total_n_total_other += result["n_total"]

        if not other_rates:
            continue
        mispb_other = np.mean(other_rates)

        mispb = mispb_raw - mispb_other
        ratio = mispb_raw / mispb_other if mispb_other > 0 else float("inf")

        per_member[member] = mispb
        per_member_raw[member] = mispb_raw
        per_member_other[member] = mispb_other
        per_member_ratio[member] = ratio
        per_member_n_overest_self[member] = total_n_overest_self
        per_member_n_total_self[member] = total_n_total_self
        per_member_n_overest_other[member] = total_n_overest_other
        per_member_n_total_other[member] = total_n_total_other

    vals = list(per_member.values())
    median_val = float(np.median(vals)) if vals else float("nan")
    mean_val = float(np.mean(vals)) if vals else float("nan")

    return {
        "median": median_val,
        "mean": mean_val,
        "per_member": per_member,
        "per_member_raw": per_member_raw,
        "per_member_other": per_member_other,
        "per_member_ratio": per_member_ratio,
        "per_member_n_overest_self": per_member_n_overest_self,
        "per_member_n_total_self": per_member_n_total_self,
        "per_member_n_overest_other": per_member_n_overest_other,
        "per_member_n_total_other": per_member_n_total_other,
    }


def compute_committee_msd_sp(
    system_scores: Dict[Tuple[str, str], float],
    ref_scores: Dict[str, float],
    cname: str,
    members: List[str],
    normalize: bool = False,
) -> Dict:
    """
    Compute MSD-SP for a committee, with per-member self-preference.

    For each member M:
    - delta_self = committee_score(M) - ref_score(M)
    - delta_other = mean committee_score(g) - ref_score(g) for g in other_gens
    - MSD-SP_M = delta_self - delta_other

    Returns per-member values and median.
    """
    per_member = {}
    per_member_d_self = {}
    per_member_d_other = {}

    for member in members:
        _, other_gens = get_committee_sp_generators(member, members)

        if (cname, member) not in system_scores or member not in ref_scores:
            continue

        def delta(g):
            d = system_scores[(cname, g)] - ref_scores[g]
            if normalize and ref_scores[g] > 0:
                d /= ref_scores[g]
            return d

        d_self = delta(member)

        others_with_data = [
            g for g in other_gens
            if (cname, g) in system_scores and g in ref_scores
        ]
        if not others_with_data:
            continue

        d_other = np.mean([delta(g) for g in others_with_data])

        per_member[member] = d_self - d_other
        per_member_d_self[member] = d_self
        per_member_d_other[member] = d_other

    vals = list(per_member.values())
    median_val = float(np.median(vals)) if vals else float("nan")
    mean_val = float(np.mean(vals)) if vals else float("nan")

    return {
        "median": median_val,
        "mean": mean_val,
        "per_member": per_member,
        "per_member_d_self": per_member_d_self,
        "per_member_d_other": per_member_d_other,
    }


def compute_committee_mrspb(
    rubric_data: Dict[Tuple[str, str], List[List[bool]]],
    ref_data: Dict[str, Dict],
    cname: str,
    members: List[str],
    error_denom: bool = False,
) -> Dict:
    """
    Compute MRSPB for a committee, with per-member self-preference.

    For each member M:
    - FPR_self = false positive rate for M's rubrics (judge says met, ref says not)
    - FPR_other = FPR for non-committee-family generators
    - MRSPB_M = FPR_self - FPR_other

    Returns per-member values and median.
    """
    per_member = {}
    per_member_raw = {}
    per_member_other = {}
    per_member_ratio = {}

    for member in members:
        _, other_gens = get_committee_sp_generators(member, members)

        if not other_gens:
            continue

        # Compute FPR for self
        self_result = _rubric_overest_rate(rubric_data, ref_data, cname, member, error_denom)
        if self_result is None:
            continue
        fpr_self = self_result[0]

        # Compute FPR for others
        other_rates = []
        for og in other_gens:
            result = _rubric_overest_rate(rubric_data, ref_data, cname, og, error_denom)
            if result is not None:
                other_rates.append(result[0])

        if not other_rates:
            continue
        fpr_other = np.mean(other_rates)

        mrspb = fpr_self - fpr_other
        ratio = fpr_self / fpr_other if fpr_other > 0 else float("inf")

        per_member[member] = mrspb
        per_member_raw[member] = fpr_self
        per_member_other[member] = fpr_other
        per_member_ratio[member] = ratio

    vals = list(per_member.values())
    median_val = float(np.median(vals)) if vals else float("nan")
    mean_val = float(np.mean(vals)) if vals else float("nan")

    return {
        "median": median_val,
        "mean": mean_val,
        "per_member": per_member,
        "per_member_raw": per_member_raw,
        "per_member_other": per_member_other,
        "per_member_ratio": per_member_ratio,
    }


def _rubric_overest_rate(
    rubric_data, ref_data, judge, target_gen, error_denom=False
):
    """Compute rubric-level false positive rate for a given judge and target generator."""
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
                if r_val:
                    continue
            n_total += 1
            if j_val and not r_val:
                n_overest += 1

    if n_total == 0:
        return None
    return n_overest / n_total, n_overest, n_total


# ============================================================
# Fast vectorized batch committee analysis
# ============================================================


def run_all_committees_fast(
    data, ref_data, judges, generators, n_instances,
    ref_scores, ref_pwc_scores, short_fn, log,
):
    """
    Vectorized batch committee analysis.

    Pre-computes all data as numpy arrays, then iterates committees
    using vectorized operations for aggregation and metric computation.

    Supports mixed judge availability: different judges may have data for
    different methods (e.g., extra judges may only have SR/AR data).
    Per-method judge lists and index mappings are used.

    Returns (accuracy_rows, bias_rows, sp_detail_rows, rubric_rows).
    """
    from scipy import stats as scipy_stats

    n_gens = len(generators)
    g2i = {g: i for i, g in enumerate(generators)}
    gfam = {g: MODEL_TO_FAMILY[g] for g in generators}

    # ==========================================================
    # Phase 1: Determine per-method judge availability
    # ==========================================================
    log.info("Determining per-method judge availability...")

    # Which judges have data for each method?
    judges_for = {}
    judges_for["sr"] = [j for j in judges if any((j, g) in data["sr"] for g in generators)]
    judges_for["ar"] = [j for j in judges if any((j, g) in data["ar"] for g in generators)]
    judges_for["da"] = [j for j in judges if any((j, g) in data["da"] for g in generators)]

    seen_pwc_judges = set(j for (j, _, _) in data["pwc"])
    judges_for["pwc"] = [j for j in judges if j in seen_pwc_judges]

    j2i_for = {mk: {j: i for i, j in enumerate(jl)} for mk, jl in judges_for.items()}

    for mk, jl in judges_for.items():
        log.info(f"  {mk.upper()}: {len(jl)} judges")

    # ==========================================================
    # Phase 2: Pre-compute numpy arrays (per-method dimensions)
    # ==========================================================
    log.info("Pre-computing numpy arrays for committee analysis...")

    # 2a. Reference instance scores: (n_gens, n_instances)
    ref_inst = np.zeros((n_gens, n_instances))
    for gi, gen in enumerate(generators):
        for i in range(n_instances):
            rub = ref_data[gen]["follow_list"][i]
            ref_inst[gi, i] = sum(rub) / len(rub)

    # 2b. Reference sign matrix: (n_gens, n_gens, n_instances)
    ref_sgn = np.sign(ref_inst[:, None, :] - ref_inst[None, :, :])

    # 2c. SR/AR flattened rubric arrays (per-method judge count)
    flat_arrs = {}
    inst_bounds = {}
    cum_bounds_arr = {}

    for mk in ("sr", "ar"):
        nj_mk = len(judges_for[mk])
        flat_arrs[mk] = {}
        inst_bounds[mk] = {}
        cum_bounds_arr[mk] = {}
        for gi, gen in enumerate(generators):
            sample_key = next((k for k in data[mk] if k[1] == gen), None)
            if sample_key is None:
                continue
            sample = data[mk][sample_key]
            n_rub = np.array([len(sample[i]) for i in range(n_instances)])
            total_rub = int(n_rub.sum())
            inst_bounds[mk][gi] = n_rub
            cb = np.zeros(n_instances + 1, dtype=np.int64)
            cb[1:] = np.cumsum(n_rub)
            cum_bounds_arr[mk][gi] = cb

            arr = np.zeros((nj_mk, total_rub), dtype=np.int8)
            for ji, judge in enumerate(judges_for[mk]):
                if (judge, gen) in data[mk]:
                    rubrics = data[mk][(judge, gen)]
                    flat = []
                    for inst_r in rubrics:
                        flat.extend(inst_r)
                    arr[ji] = flat
            flat_arrs[mk][gi] = arr

    # 2d. Reference rubric flat arrays (for MRA / MRSPB)
    ref_flat = {}
    for mk in ("sr", "ar"):
        ref_flat[mk] = {}
        for gi, gen in enumerate(generators):
            flat = []
            for i in range(n_instances):
                flat.extend(ref_data[gen]["follow_list"][i])
            ref_flat[mk][gi] = np.array(flat, dtype=np.int8)

    # 2e. DA instance scores: (n_da_judges, n_gens, n_instances)
    nj_da = len(judges_for["da"])
    da_all = np.zeros((nj_da, n_gens, n_instances))
    for ji, judge in enumerate(judges_for["da"]):
        for gi, gen in enumerate(generators):
            if (judge, gen) in data["da"]:
                for i in range(n_instances):
                    n_met, n_total = data["da"][(judge, gen)][i]
                    da_all[ji, gi, i] = min(n_met, n_total) / n_total

    # 2f. PWC resolved signs: (n_pwc_judges, n_pairs, n_instances)
    seen_pairs = sorted(set((gx, gy) for (_, gx, gy) in data["pwc"]))
    p2i = {p: i for i, p in enumerate(seen_pairs)}
    n_pairs = len(seen_pairs)
    nj_pwc = len(judges_for["pwc"])

    pwc_all = np.zeros((nj_pwc, n_pairs, n_instances), dtype=np.int8)
    for (j, gx, gy), outcomes in data["pwc"].items():
        if j not in j2i_for["pwc"]:
            continue
        ji = j2i_for["pwc"][j]
        pi = p2i[(gx, gy)]
        for i in range(n_instances):
            if outcomes[i] == "X wins":
                pwc_all[ji, pi, i] = 1
            elif outcomes[i] == "Y wins":
                pwc_all[ji, pi, i] = -1

    # 2g. Target-pair mapping for PWC overestimation
    target_pairs = {gi: [] for gi in range(n_gens)}
    for (gx, gy), pi in p2i.items():
        target_pairs[g2i[gx]].append((pi, 1, g2i[gy]))
        target_pairs[g2i[gy]].append((pi, -1, g2i[gx]))

    # Reference system scores as arrays
    ref_sys = np.array([ref_scores[g] for g in generators])
    ref_pwc_sys = np.array([ref_pwc_scores[g] for g in generators])

    log.info("Pre-computation complete.")

    # ==========================================================
    # Phase 3: Iterate committees
    # ==========================================================
    committees = enumerate_committees(judges, min_size=2)
    n_comms = len(committees)
    log.info(f"Processing {n_comms} committees...")

    accuracy_rows = []
    bias_rows = []
    sp_detail_rows = []
    rubric_rows = []

    for ci, (cname, members) in enumerate(committees):
        if (ci + 1) % 500 == 0:
            log.info(f"  Committee {ci + 1}/{n_comms}...")

        nm = len(members)
        members_short = ", ".join(short_fn(m) for m in members)

        # --- Per-method member filtering ---
        mk_members = {}  # method_key -> list of members with data
        mk_midxs = {}    # method_key -> np array of indices into per-method arrays
        for mk in ("sr", "ar", "da", "pwc"):
            mm = [m for m in members if m in j2i_for[mk]]
            mk_members[mk] = mm
            if len(mm) >= 2:
                mk_midxs[mk] = np.array([j2i_for[mk][m] for m in mm])

        # --- Aggregate + compute instance scores ---
        method_inst = {}   # method_label -> (n_gens, n_instances)
        method_flat = {}   # method_key -> {gi: boolean array}
        active_methods = []

        for mk, ml in [("sr", "SR"), ("ar", "AR")]:
            if mk not in mk_midxs:
                continue
            midxs = mk_midxs[mk]
            nm_mk = len(mk_members[mk])
            inst = np.zeros((n_gens, n_instances))
            flat_maj = {}
            for gi in range(n_gens):
                if gi not in flat_arrs[mk]:
                    continue
                votes = flat_arrs[mk][gi][midxs].sum(axis=0)
                maj = votes > nm_mk / 2
                flat_maj[gi] = maj
                cb = cum_bounds_arr[mk][gi]
                nb = inst_bounds[mk][gi]
                inst_met = np.add.reduceat(maj.astype(np.float64), cb[:-1])
                inst[gi] = inst_met / nb
            method_inst[ml] = inst
            method_flat[mk] = flat_maj
            active_methods.append(ml)

        if "da" in mk_midxs:
            method_inst["DA"] = da_all[mk_midxs["da"]].mean(axis=0)
            active_methods.append("DA")

        if "pwc" in mk_midxs:
            pwc_c = np.sign(pwc_all[mk_midxs["pwc"]].sum(axis=0))
            active_methods.append("PWC")
        else:
            pwc_c = None

        if not active_methods:
            continue

        # --- System scores ---
        method_sys = {}
        for ml in ("SR", "AR", "DA"):
            if ml in method_inst:
                method_sys[ml] = method_inst[ml].mean(axis=1)

        if pwc_c is not None:
            pwc_win = np.zeros(n_gens)
            pwc_cnt = np.zeros(n_gens)
            for (gx, gy), pi in p2i.items():
                gxi, gyi = g2i[gx], g2i[gy]
                signs = pwc_c[pi]
                xw = (signs > 0).sum()
                yw = (signs < 0).sum()
                ti = (signs == 0).sum()
                pwc_win[gxi] += xw + 0.5 * ti
                pwc_win[gyi] += yw + 0.5 * ti
                pwc_cnt[gxi] += n_instances
                pwc_cnt[gyi] += n_instances
            method_sys["PWC"] = pwc_win / np.maximum(pwc_cnt, 1)

        # --- Accuracy metrics per active method ---
        for method in active_methods:
            ss = method_sys[method]
            rs = ref_pwc_sys if method == "PWC" else ref_sys

            # MPA
            n_conc = 0
            n_tot = 0
            for gi in range(n_gens):
                for gj in range(gi + 1, n_gens):
                    jd = ss[gi] - ss[gj]
                    rd = rs[gi] - rs[gj]
                    n_tot += 1
                    if (jd > 0 and rd > 0) or (jd < 0 and rd < 0) or (jd == 0 and rd == 0):
                        n_conc += 1
            mpa = n_conc / n_tot if n_tot > 0 else float("nan")

            # MRD
            j_ranks = scipy_stats.rankdata(-ss, method="average")
            r_ranks = scipy_stats.rankdata(-rs, method="average")
            mrd = float(np.mean(np.abs(j_ranks - r_ranks)))

            # MSD
            msd = float(np.mean(ss - rs))

            # MSD-norm
            mask = rs > 0
            msd_norm = float(np.mean((ss[mask] - rs[mask]) / rs[mask])) if mask.any() else float("nan")

            # MIPA
            if method == "PWC":
                n_ag = 0
                n_tm = 0
                for (gx, gy), pi in p2i.items():
                    gxi, gyi = g2i[gx], g2i[gy]
                    js = pwc_c[pi]
                    rd = ref_inst[gxi] - ref_inst[gyi]
                    rsgn = np.sign(rd)
                    n_ag += int((js == rsgn).sum())
                    n_tm += n_instances
                mipa = n_ag / n_tm if n_tm > 0 else float("nan")
            else:
                inst = method_inst[method]
                n_ag = 0
                n_tm = 0
                for gi in range(n_gens):
                    for gj in range(gi + 1, n_gens):
                        jd = inst[gi] - inst[gj]
                        rd = ref_inst[gi] - ref_inst[gj]
                        jsgn = np.sign(jd)
                        rsgn = np.sign(rd)
                        n_ag += int((jsgn == rsgn).sum())
                        n_tm += n_instances
                mipa = n_ag / n_tm if n_tm > 0 else float("nan")

            accuracy_rows.append({
                "Committee": cname, "Size": nm, "Members": members_short,
                "Method": method, "MPA": mpa, "MRD": mrd,
                "MSD": msd, "MSD_norm": msd_norm, "MIPA": mipa,
            })

        # --- Overestimation rates (all targets, computed once per method) ---
        # Exclude committee members from opponent set (no head-to-head)
        member_idxs = set(g2i[m] for m in members)
        method_overest = {}
        for method in ("SR", "AR", "DA"):
            if method not in method_inst:
                continue
            inst = method_inst[method]
            rates = np.full(n_gens, np.nan)
            rates_err = np.full(n_gens, np.nan)
            for t in range(n_gens):
                jd = inst[t] - inst
                js = np.sign(jd)
                rs = ref_sgn[t]
                ov = js > rs

                opp = np.ones(n_gens, dtype=bool)
                for mi_idx in member_idxs:
                    opp[mi_idx] = False
                opp[t] = False

                n_ov = int(ov[opp].sum())
                n_tl = int(opp.sum()) * n_instances
                if n_tl > 0:
                    rates[t] = n_ov / n_tl

                err_mask = rs[opp] < 0
                n_ov_e = int((ov[opp] & err_mask).sum())
                n_tl_e = int(err_mask.sum())
                if n_tl_e > 0:
                    rates_err[t] = n_ov_e / n_tl_e

            method_overest[method] = (rates, rates_err)

        if pwc_c is not None:
            pwc_rates = np.full(n_gens, np.nan)
            pwc_rates_err = np.full(n_gens, np.nan)
            for t in range(n_gens):
                n_ov = 0
                n_tl = 0
                n_ov_e = 0
                n_tl_e = 0
                for pi, sm, oi in target_pairs[t]:
                    if oi in member_idxs:
                        continue  # skip committee member opponents
                    js = pwc_c[pi] * sm
                    rd = ref_inst[t] - ref_inst[oi]
                    rs = np.sign(rd)
                    ov = js > rs
                    n_ov += int(ov.sum())
                    n_tl += n_instances
                    ev = rs < 0
                    n_ov_e += int((ov & ev).sum())
                    n_tl_e += int(ev.sum())
                if n_tl > 0:
                    pwc_rates[t] = n_ov / n_tl
                if n_tl_e > 0:
                    pwc_rates_err[t] = n_ov_e / n_tl_e
            method_overest["PWC"] = (pwc_rates, pwc_rates_err)

        # --- Self-preference per member ---
        committee_fams = {gfam[m] for m in members}
        oi_all = [g2i[g] for g in generators if gfam[g] not in committee_fams]

        for method in active_methods:
            ss = method_sys[method]
            rs = ref_pwc_sys if method == "PWC" else ref_sys
            if method not in method_overest:
                continue
            rates, rates_err = method_overest[method]

            bias_row = {
                "Committee": cname, "Size": nm,
                "Members": members_short, "Method": method,
            }

            m_mispb = {}
            m_hspp = {}
            m_msd_sp = {}
            m_msd_sp_n = {}

            for member in members:
                mi = g2i[member]

                if not oi_all:
                    continue
                oia = np.array(oi_all)

                # MISPB
                mispb_raw = float(rates[mi])
                mispb_other = float(np.nanmean(rates[oia]))
                mispb = mispb_raw - mispb_other

                # HSPP
                hspp_raw = float(rates_err[mi])
                hspp_other = float(np.nanmean(rates_err[oia]))
                hspp = hspp_raw - hspp_other

                # MSD-SP
                d_self = float(ss[mi] - rs[mi])
                d_oth = ss[oia] - rs[oia]
                msd_sp = d_self - float(np.mean(d_oth))

                # MSD-SP-norm
                d_self_n = d_self / rs[mi] if rs[mi] > 0 else 0.0
                r_oi = rs[oia]
                d_oth_n = np.where(r_oi > 0, (ss[oia] - r_oi) / r_oi, 0.0)
                msd_sp_n = d_self_n - float(np.mean(d_oth_n))

                m_mispb[member] = mispb
                m_hspp[member] = hspp
                m_msd_sp[member] = msd_sp
                m_msd_sp_n[member] = msd_sp_n

                sp_detail_rows.append({
                    "Committee": cname, "Size": nm, "Method": method,
                    "Member": short_fn(member), "Member_full": member,
                    "MISPB": mispb, "MISPB_raw": mispb_raw,
                    "MISPB_other": mispb_other, "HSPP": hspp,
                    "MSD_SP": msd_sp, "MSD_SP_norm": msd_sp_n,
                })

            def _mm(d):
                v = list(d.values())
                if not v:
                    return float("nan"), float("nan")
                return float(np.median(v)), float(np.mean(v))

            for metric, vals in [("MISPB", m_mispb), ("HSPP", m_hspp),
                                 ("MSD_SP", m_msd_sp), ("MSD_SP_norm", m_msd_sp_n)]:
                med, mn = _mm(vals)
                bias_row[f"{metric}_median"] = med
                bias_row[f"{metric}_mean"] = mn

            bias_rows.append(bias_row)

        # --- Rubric-level metrics (SR/AR) ---
        for mk, ml in [("sr", "SR"), ("ar", "AR")]:
            if mk not in method_flat:
                continue
            cfm = method_flat[mk]

            # MRA
            n_correct = 0
            n_total_rub = 0
            for gi in range(n_gens):
                if gi not in cfm:
                    continue
                cr = cfm[gi]
                rr = ref_flat[mk][gi].astype(bool)
                n_correct += int((cr == rr).sum())
                n_total_rub += len(cr)
            mra = n_correct / n_total_rub if n_total_rub > 0 else float("nan")

            rubric_row = {"Committee": cname, "Size": nm, "Method": ml, "MRA": mra}

            # MRSPB per member
            m_mrspb = {}

            for member in members:
                mi = g2i[member]
                if not oi_all:
                    continue

                if mi not in cfm:
                    continue
                cs = cfm[mi]
                rs_r = ref_flat[mk][mi].astype(bool)
                ns = len(cs)
                if ns == 0:
                    continue
                fpr_self = int((cs & ~rs_r).sum()) / ns

                o_fprs = []
                for og in oi_all:
                    if og not in cfm:
                        continue
                    co = cfm[og]
                    ro = ref_flat[mk][og].astype(bool)
                    no = len(co)
                    if no > 0:
                        o_fprs.append(int((co & ~ro).sum()) / no)
                if o_fprs:
                    m_mrspb[member] = fpr_self - np.mean(o_fprs)

            vals = list(m_mrspb.values())
            rubric_row["MRSPB_median"] = float(np.median(vals)) if vals else float("nan")
            rubric_row["MRSPB_mean"] = float(np.mean(vals)) if vals else float("nan")

            rubric_rows.append(rubric_row)

    return accuracy_rows, bias_rows, sp_detail_rows, rubric_rows
