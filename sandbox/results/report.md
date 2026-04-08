# IFEval Judge Prompting Methods: Analysis Report

This report analyzes four LLM-as-a-judge prompting methods—Single Rubric (SR), All Rubrics (AR), Direct Assessment (DA), and Pairwise Comparison (PWC)—on the IFEval benchmark (541 instances, 12 generators, 9 judges). All rubrics are programmatically verifiable, providing ground-truth labels free from subjectivity.

Instance-level scoring uses the **fraction of rubrics met** per instance (not binary all-or-nothing). System-level scores are the mean of instance-level fractions across all instances.

## Reference System Scores

Ground-truth system-level scores (mean fraction of rubrics met per instance):

| Generator | Family | Score |
|-----------|--------|-------|
| GPT-5 | GPT | 0.9627 |
| Claude-Sonnet | Claude | 0.9273 |
| Qwen-235B | Qwen | 0.9251 |
| Llama-Mav | Llama | 0.9214 |
| GPT-120B | GPT | 0.8993 |
| Llama-Scout | Llama | 0.8980 |
| Qwen-30B | Qwen | 0.8909 |
| Qwen-4B | Qwen | 0.8903 |
| Claude-Haiku | Claude | 0.8879 |
| Gemma-27B | Gemma | 0.8725 |
| Gemma-12B | Gemma | 0.8669 |
| Gemma-4B | Gemma | 0.8222 |

**GPT-5** is the strongest generator (0.9627). **Gemma-4B** is the weakest (0.8222).

---

## RQ1: Which Prompting Method Yields the Most Accurate Evaluations?

### System-Level Accuracy

| Method | MPA | MRD | MSD | MSD-norm |
|--------|-----|-----|-----|----------|
| SR | 0.6919 | 2.66 | 0.0268 | 0.0309 |
| AR | 0.6785 | 2.72 | 0.0428 | 0.0489 |
| DA | 0.6549 | 2.77 | 0.0498 | 0.0567 |
| PWC | 0.5842 | 3.39 | 0.0000 | 0.0012 |

**Key**: MPA = Mean Pairwise Accuracy (higher = better); MRD = Mean Ranking Difference (lower = better); MSD = Mean Score Delta (closer to 0 = better).

**Observations**:
- **SR** achieves the highest MPA (0.6919), meaning it most accurately ranks generators in their correct order.
- **SR** achieves the lowest MRD (2.66), indicating the smallest average rank displacement.
- **PWC** performs worst on MPA (0.5842) and **PWC** worst on MRD (3.39 vs ~2.7 for others).
- **MSD** is near zero for PWC (relative scoring cancels systematic bias). SR, AR, DA show positive MSD (0.0268–0.0498), indicating systematic overestimation of generator quality.

### Instance-Level Accuracy

| Method | MIPA |
|--------|------|
| SR | 0.8058 |
| AR | 0.8139 |
| DA | 0.8154 |
| PWC | 0.4722 |

*Sample sizes (n_agree / n_total comparisons per judge):*

| Judge | SR | AR | DA | PWC |
|-------|---|---|---|---|
| Gemma-27B | 29369/35706 | 29483/35706 | 29538/35706 | 14857/35706 |
| Gemma-12B | 28917/35706 | 29550/35706 | 29539/35706 | 13802/35706 |
| Gemma-4B | 29575/35706 | 29651/35706 | 29670/35706 | 12763/35706 |
| Llama-Mav | 28184/35706 | 28201/35706 | 27896/35706 | 18362/35706 |
| Llama-Scout | 27532/35706 | 28428/35706 | 28114/35706 | 16727/35706 |
| Qwen-235B | 28693/35706 | 28880/35706 | 29228/35706 | 15846/35706 |
| Qwen-30B | 28864/35706 | 29321/35706 | 29501/35706 | 16078/35706 |
| Qwen-4B | 27953/35706 | 28158/35706 | 28524/35706 | 14362/35706 |
| GPT-120B | 29855/35706 | 29874/35706 | 30008/35706 | 28959/35706 |

**Observations**:
- SR, AR, and DA all achieve similar MIPA (0.8058–0.8154), with **DA** marginally best.
- **PWC is dramatically worse** (0.4722), performing below chance (0.5). This means PWC judges more often disagree with the reference on instance-level pairwise comparisons than agree. The likely cause: PWC must compare two full responses simultaneously, which is a harder cognitive task for the judge.

### Rubric-Level Accuracy (SR and AR only)

| Method | MRA |
|--------|-----|
| SR | 0.8798 |
| AR | 0.8880 |

*Sample sizes (n_correct / n_total rubric evaluations per judge):*

| Judge | SR | AR |
|-------|----|----|
| Gemma-27B | 8952/10008 | 8988/10008 |
| Gemma-12B | 8878/10008 | 8984/10008 |
| Gemma-4B | 8947/10008 | 8967/10008 |
| Llama-Mav | 8772/10008 | 8787/10008 |
| Llama-Scout | 8425/10008 | 8595/10008 |
| Qwen-235B | 8880/10008 | 8911/10008 |
| Qwen-30B | 8827/10008 | 8993/10008 |
| Qwen-4B | 8504/10008 | 8688/10008 |
| GPT-120B | 9060/10008 | 9069/10008 |

**AR** slightly outperforms SR at rubric-level accuracy (0.8880 vs 0.8798). Seeing all rubrics together may help the judge contextualize each criterion.

### Per-Judge MPA

| Judge | SR | AR | DA | PWC |
|-------|----|----|----|----|
| Gemma-27B | 0.6515 | 0.7121 | 0.5606 | 0.5152 |
| Gemma-12B | 0.4091 | 0.5758 | 0.5303 | 0.4242 |
| Gemma-4B | 0.4697 | 0.3636 | 0.3939 | 0.3788 |
| Llama-Mav | 0.8030 | 0.7424 | 0.7424 | 0.5909 |
| Llama-Scout | 0.6212 | 0.5606 | 0.5606 | 0.5758 |
| Qwen-235B | 0.8636 | 0.8788 | 0.8182 | 0.6667 |
| Qwen-30B | 0.7727 | 0.6818 | 0.6818 | 0.6364 |
| Qwen-4B | 0.7424 | 0.6818 | 0.6970 | 0.6364 |
| GPT-120B | 0.8939 | 0.9091 | 0.9091 | 0.8333 |

**GPT-120B** is the most accurate judge across methods (avg MPA = 0.8864). **Gemma-4B** is the least accurate (avg MPA = 0.4015).

### RQ1 Answer

**No single method dominates across all metrics.** The overall picture:

| Metric | Best Method | Interpretation |
|--------|-------------|----------------|
| MPA | SR | Best at ranking generators correctly |
| MRD | SR | Smallest average rank displacement |
| MIPA | DA | Best instance-level discrimination |
| MRA | AR | Best rubric-level judgment |

- **SR** is best at: system-level ranking (MPA), ranking displacement (MRD)
- **AR** is best at: rubric-level accuracy (MRA)
- **DA** is best at: instance-level discrimination (MIPA)

**PWC is consistently the worst** by a large margin, especially at instance level. Among rubric-aware methods, AR has a slight edge over SR.

---

## RQ2: Which Methods Are Most Sensitive to Self-Preference Bias?

### System-Level Bias Metrics

| Method | MRD-SP | d_self | d_other | MRD-FSP | d_fam | d_nonfam | MSD-SP | d_self | d_other | MSD-SP-norm | MSD-FSP | d_fam | d_nonfam | MSD-FSP-norm |
|--------|--------|--------|---------|---------|-------|----------|--------|--------|---------|-------------|---------|-------|----------|--------------|
| SR | -3.930 | -3.111 | 0.819 | -3.319 | -2.500 | 0.819 | 0.0269 | 0.0476 | 0.0207 | 0.0322 | 0.0245 | 0.0451 | 0.0207 | 0.0295 |
| AR | -3.823 | -3.056 | 0.767 | -2.925 | -2.157 | 0.767 | 0.0247 | 0.0621 | 0.0374 | 0.0298 | 0.0220 | 0.0594 | 0.0374 | 0.0268 |
| DA | -4.052 | -3.333 | 0.719 | -2.830 | -2.111 | 0.719 | 0.0205 | 0.0656 | 0.0451 | 0.0251 | 0.0186 | 0.0636 | 0.0451 | 0.0230 |
| PWC | -3.548 | -2.667 | 0.881 | -2.993 | -2.111 | 0.881 | 0.0888 | 0.0679 | -0.0208 | 0.1851 | 0.0725 | 0.0516 | -0.0208 | 0.1514 |

**Key**:
- **MRD-SP**: Signed rank difference for self minus that for others. Negative = judge ranks itself better than reference says, relative to how it treats others.
- **MSD-SP**: Score delta for self minus score delta for others. Positive = judge inflates its own score more than others' scores.
- **-FSP** variants: Same but for the judge's entire model family.

*Per-judge MSD-SP detail (delta_self, delta_other, n_others):*

| Judge | SR (d_self / d_other / n) | AR (d_self / d_other / n) | DA (d_self / d_other / n) | PWC (d_self / d_other / n) |
|-------|---|---|---|---|
| Gemma-27B | 0.104 / 0.062 / 9 | 0.113 / 0.071 / 9 | 0.116 / 0.073 / 9 | 0.137 / -0.038 / 9 |
| Gemma-12B | 0.105 / 0.051 / 9 | 0.122 / 0.072 / 9 | 0.122 / 0.077 / 9 | 0.178 / -0.053 / 9 |
| Gemma-4B | 0.171 / 0.079 / 9 | 0.174 / 0.083 / 9 | 0.176 / 0.086 / 9 | 0.155 / -0.055 / 9 |
| Llama-Mav | 0.026 / 0.020 / 10 | 0.023 / 0.023 / 10 | 0.014 / 0.018 / 10 | -0.103 / 0.022 / 10 |
| Llama-Scout | -0.024 / -0.013 / 10 | 0.021 / 0.031 / 10 | 0.005 / 0.013 / 10 | -0.141 / 0.031 / 10 |
| Qwen-235B | 0.029 / 0.011 / 9 | 0.030 / 0.021 / 9 | 0.044 / 0.045 / 9 | 0.126 / -0.022 / 9 |
| Qwen-30B | 0.059 / 0.048 / 9 | 0.083 / 0.068 / 9 | 0.084 / 0.080 / 9 | 0.154 / -0.039 / 9 |
| Qwen-4B | 0.016 / -0.009 / 9 | 0.039 / 0.027 / 9 | 0.078 / 0.066 / 9 | 0.091 / -0.030 / 9 |
| GPT-120B | -0.058 / -0.062 / 10 | -0.046 / -0.060 / 10 | -0.050 / -0.052 / 10 | 0.014 / -0.004 / 10 |

**Observations**:
- **MRD-SP** is negative for all methods, confirming universal self-preference in rankings. DA shows the strongest ranking inflation (-4.05), PWC the least (-3.55).
- **MSD-SP** is positive for all methods. PWC shows the largest MSD-SP (0.0888), vs ~0.0240 for others.
- **Family bias** (MRD-FSP, MSD-FSP) follows similar patterns but is attenuated, indicating that self-preference is strongest for the exact model and weaker for same-family models.

### Instance-Level Self-Preference Bias (MISPB)

| Method | MISPB | raw | other | ratio | MISPB-F | F_ratio | MISPB-FO | FO_ratio |
|--------|-------|-----|-------|-------|---------|---------|----------|----------|
| SR | 0.0213 | 0.1137 | 0.0924 | 1.28 | 0.0186 | 1.25 | 0.0172 | 1.23 |
| AR | 0.0199 | 0.1087 | 0.0888 | 1.28 | 0.0166 | 1.24 | 0.0146 | 1.22 |
| DA | 0.0163 | 0.1049 | 0.0886 | 1.24 | 0.0141 | 1.21 | 0.0124 | 1.20 |
| PWC | 0.0809 | 0.3264 | 0.2454 | 1.33 | 0.0650 | 1.27 | 0.0563 | 1.24 |

*Per-judge MISPB sample sizes with overestimation sub-types (t2w=tie→win, l2w=loss→win, l2t=loss→tie):*

| Judge | Method | MISPB | overest_self/total_self | self t2w/l2w/l2t | overest_other/total_other | other t2w/l2w/l2t |
|-------|--------|-------|------------------------|------------------|--------------------------|-------------------|
| Gemma-27B | SR | 0.0359 | 662/5951 | 75/12/575 | 4036/53559 | 699/87/3250 |
| Gemma-27B | AR | 0.0375 | 658/5951 | 67/9/582 | 3915/53559 | 532/64/3319 |
| Gemma-27B | DA | 0.0332 | 632/5951 | 36/15/581 | 3912/53559 | 443/54/3415 |
| Gemma-27B | PWC | 0.1601 | 2492/5951 | 2005/249/238 | 13854/53559 | 11941/803/1110 |
| Gemma-12B | SR | 0.0453 | 746/5951 | 148/27/571 | 4289/53559 | 979/105/3205 |
| Gemma-12B | AR | 0.0447 | 693/5951 | 78/11/604 | 3845/53559 | 496/29/3320 |
| Gemma-12B | DA | 0.0402 | 668/5951 | 49/11/608 | 3859/53559 | 392/69/3398 |
| Gemma-12B | PWC | 0.2147 | 2818/5951 | 2302/264/252 | 13861/53559 | 11681/859/1321 |
| Gemma-4B | SR | 0.0974 | 1008/5951 | 26/19/963 | 3856/53559 | 324/51/3481 |
| Gemma-4B | AR | 0.0982 | 1003/5951 | 15/13/975 | 3767/53559 | 129/42/3596 |
| Gemma-4B | DA | 0.0976 | 1000/5951 | 14/8/978 | 3775/53559 | 150/32/3593 |
| Gemma-4B | PWC | 0.1668 | 2628/5951 | 1810/459/359 | 14720/53559 | 12437/1049/1234 |
| Llama-Mav | SR | -0.0017 | 613/5951 | 312/10/291 | 6229/59510 | 2271/201/3757 |
| Llama-Mav | AR | -0.0066 | 585/5951 | 279/18/288 | 6240/59510 | 2243/215/3782 |
| Llama-Mav | DA | -0.0089 | 600/5951 | 292/20/288 | 6532/59510 | 2399/248/3885 |
| Llama-Mav | PWC | -0.0988 | 966/5951 | 792/29/145 | 15537/59510 | 12261/922/2354 |
| Llama-Scout | SR | -0.0126 | 621/5951 | 241/26/354 | 6961/59510 | 2769/262/3930 |
| Llama-Scout | AR | -0.0070 | 582/5951 | 178/13/391 | 6238/59510 | 1821/222/4195 |
| Llama-Scout | DA | -0.0072 | 610/5951 | 188/20/402 | 6527/59510 | 2079/273/4175 |
| Llama-Scout | PWC | -0.1393 | 896/5951 | 679/34/183 | 17250/59510 | 13845/1120/2285 |
| Qwen-235B | SR | 0.0036 | 593/5951 | 333/21/239 | 5143/53559 | 1866/163/3114 |
| Qwen-235B | AR | -0.0038 | 540/5951 | 293/24/223 | 5066/53559 | 1675/168/3223 |
| Qwen-235B | DA | -0.0129 | 461/5951 | 197/18/246 | 4840/53559 | 1162/112/3566 |
| Qwen-235B | PWC | 0.1271 | 2304/5951 | 2056/107/141 | 13927/53559 | 11704/698/1525 |
| Qwen-30B | SR | 0.0022 | 592/5951 | 128/13/451 | 5208/53559 | 1113/120/3975 |
| Qwen-30B | AR | 0.0048 | 572/5951 | 97/9/466 | 4893/53559 | 646/99/4148 |
| Qwen-30B | DA | -0.0035 | 513/5951 | 47/7/459 | 4807/53559 | 452/65/4290 |
| Qwen-30B | PWC | 0.1796 | 2500/5951 | 2107/184/209 | 12881/53559 | 10412/749/1720 |
| Qwen-4B | SR | 0.0125 | 717/5951 | 277/52/388 | 5785/53559 | 1833/218/3734 |
| Qwen-4B | AR | 0.0028 | 656/5951 | 209/41/406 | 5755/53559 | 1692/199/3864 |
| Qwen-4B | DA | 0.0052 | 638/5951 | 161/32/445 | 5466/53559 | 1184/149/4133 |
| Qwen-4B | PWC | 0.1020 | 2236/5951 | 1858/191/187 | 14662/53559 | 11997/862/1803 |
| GPT-120B | SR | 0.0092 | 538/5951 | 347/19/172 | 4831/59510 | 3224/135/1472 |
| GPT-120B | AR | 0.0091 | 535/5951 | 348/12/175 | 4811/59510 | 3144/101/1566 |
| GPT-120B | DA | 0.0033 | 496/5951 | 314/5/177 | 4761/59510 | 3029/118/1614 |
| GPT-120B | PWC | 0.0162 | 641/5951 | 495/19/127 | 5443/59510 | 4064/145/1234 |

**Observations**:
- All methods show positive MISPB (judges overestimate themselves more than others).
- **PWC** has the highest MISPB (0.0809). The other methods range from 0.0163–0.0213.
- MISPB_ratio shows judges are 1.24x–1.33x more likely to overestimate themselves than other generators.
- Family bias (MISPB-F) is consistently lower than self bias, indicating same-model preference is stronger than same-family preference.

### Harmful Self-Preference Propensity (HSPP)

HSPP restricts the denominator to instances where the reference says the other generator is better, measuring how often self-preference causes actual errors.

| Method | HSPP | raw | other | ratio | HSPP-F | F_ratio | HSPP-FO | FO_ratio |
|--------|------|-----|-------|-------|--------|---------|---------|----------|
| SR | 0.0578 | 0.8437 | 0.7859 | 1.08 | 0.0518 | 1.09 | 0.0535 | 1.11 |
| AR | 0.0481 | 0.8541 | 0.8060 | 1.06 | 0.0415 | 1.08 | 0.0416 | 1.10 |
| DA | 0.0394 | 0.8702 | 0.8308 | 1.04 | 0.0410 | 1.07 | 0.0451 | 1.10 |
| PWC | 0.1252 | 0.6686 | 0.5434 | 1.24 | 0.1050 | 1.22 | 0.0976 | 1.23 |

**PWC** has the highest HSPP (0.1252): when the reference says the other generator is better, judges still overestimate themselves at a higher net rate. For the other methods, HSPP ranges from 0.0394–0.0578.

### Notable Per-Judge Patterns

From the per-judge MISPB breakdown (see `mispb_per_judge.csv`):

- **SR**: Most biased judge = Gemma-4B (0.0974), least biased = Llama-Scout (-0.0126)
- **AR**: Most biased judge = Gemma-4B (0.0982), least biased = Llama-Scout (-0.0070)
- **DA**: Most biased judge = Gemma-4B (0.0976), least biased = Qwen-235B (-0.0129)

### RQ2 Answer

**PWC is by far the most sensitive to self-preference bias** across all metrics (MISPB = 0.0809, MSD-SP = 0.0888, HSPP = 0.1252).
**DA is the least biased** (MISPB = 0.0163).
Ordering from most to least biased: **PWC > SR > AR > DA**.

---

## RQ2.1: Are Rubric-Based Methods Sensitive to Self-Preference Bias?

### Rubric-Level Self-Preference Bias (MRSPB)

| Method | MRSPB | raw | other | ratio | MRSPB-F | F_ratio | MRSPB-FO | FO_ratio | MRSPB-err | err_ratio | MRSPB-err-F | errF_ratio | MRSPB-err-FO | errFO_ratio |
|--------|-------|-----|-------|-------|---------|---------|----------|----------|-----------|-----------|-------------|------------|--------------|-------------|
| SR | 0.0180 | 0.0881 | 0.0701 | 1.24 | 0.0156 | 1.19 | 0.0141 | 1.14 | 0.0625 | 1.11 | 0.0541 | 1.12 | 0.0522 | 1.13 |
| AR | 0.0174 | 0.0914 | 0.0740 | 1.20 | 0.0146 | 1.16 | 0.0129 | 1.13 | 0.0565 | 1.08 | 0.0465 | 1.10 | 0.0453 | 1.13 |

*Per-judge MRSPB sample sizes (n_overest_self / n_total_self | n_overest_other / n_total_other):*

| Judge | Method | MRSPB | n_overest_self/n_total_self | n_overest_other/n_total_other |
|-------|--------|-------|----------------------------|-------------------------------|
| Gemma-27B | SR | 0.0330 | 94/834 | 598/7506 |
| Gemma-27B | AR | 0.0326 | 95/834 | 610/7506 |
| Gemma-12B | SR | 0.0378 | 96/834 | 580/7506 |
| Gemma-12B | AR | 0.0430 | 103/834 | 604/7506 |
| Gemma-4B | SR | 0.0823 | 141/834 | 651/7506 |
| Gemma-4B | AR | 0.0803 | 140/834 | 657/7506 |
| Llama-Mav | SR | -0.0113 | 53/834 | 624/8340 |
| Llama-Mav | AR | -0.0095 | 55/834 | 629/8340 |
| Llama-Scout | SR | 0.0017 | 61/834 | 596/8340 |
| Llama-Scout | AR | 0.0013 | 72/834 | 709/8340 |
| Qwen-235B | SR | -0.0016 | 52/834 | 480/7506 |
| Qwen-235B | AR | -0.0080 | 51/834 | 519/7506 |
| Qwen-30B | SR | 0.0068 | 75/834 | 624/7506 |
| Qwen-30B | AR | 0.0083 | 79/834 | 649/7506 |
| Qwen-4B | SR | 0.0084 | 68/834 | 549/7506 |
| Qwen-4B | AR | 0.0084 | 72/834 | 585/7506 |
| GPT-120B | SR | 0.0047 | 21/834 | 171/8340 |
| GPT-120B | AR | 0.0001 | 19/834 | 189/8340 |

**Observations**:
- **Yes, rubric-based methods are sensitive to self-preference bias.** Both SR and AR show positive MRSPB (SR=0.0180, AR=0.0174), meaning judges are more likely to say their own rubrics are met (false positive) compared to how they treat other generators' rubrics.
- The bias levels are similar for SR (0.0180) and AR (0.0174), suggesting that the presentation format (single vs. all rubrics) does not meaningfully affect self-preference at the rubric level.
- The error-denominator variant (MRSPB-err) is higher (SR=0.0625, AR=0.0565), meaning that among rubrics where the reference says 'not met', judges incorrectly say 'met' for their own outputs more often than for others.
- **SR** rubric bias: most biased = Gemma-4B (0.0823), least biased = Llama-Mav (-0.0113)
- **AR** rubric bias: most biased = Gemma-4B (0.0803), least biased = Llama-Mav (-0.0095)

### RQ2.1 Answer

**Yes, rubric-based methods (SR and AR) are sensitive to self-preference bias**, with MRSPB of 0.0180 (SR) and 0.0174 (AR), and MRSPB-err of 0.0625 (SR) and 0.0565 (AR). The bias is modest but consistent, and the two rubric formats show very similar bias levels.

---

## RQ2.2: Is Self-Preference Bias Related to Method Quality?

### Method-Level Analysis

| Method | MPA (accuracy) | |MISPB| (bias) |
|--------|----------------|----------------|
| SR | 0.6919 | 0.0213 |
| AR | 0.6785 | 0.0199 |
| DA | 0.6549 | 0.0163 |
| PWC | 0.5842 | 0.0809 |

Pearson correlation (all 4 methods): r = -0.924, p = 0.076

Pearson correlation (excluding PWC): r = 0.995, p = 0.063

**Observations**:
- When all four methods are included, the correlation is r = -0.924 (p = 0.076). With only 4 data points, statistical significance is limited.
- **Excluding PWC**, the correlation is r = 0.995. Among SR, AR, and DA, the relationship between accuracy and bias may differ from the overall pattern driven by PWC as an outlier.
- The relationship is **not straightforward**: it depends heavily on whether PWC is included, and the non-PWC methods have very similar bias levels making the correlation unstable.

### RQ2.2 Answer

**The relationship between accuracy and bias is nuanced.** PWC stands out as both the least accurate and most biased method by a wide margin, creating an apparent negative correlation. However, among the more comparable methods (SR, AR, DA), the differences in bias are small and the correlation may reverse. Self-preference bias appears to be a relatively stable phenomenon across SR/AR/DA, with magnitude driven more by individual judge quality than by the prompting method itself.

---

## Figures

1. `figures/accuracy_comparison.png` — RQ1: Accuracy metrics across methods
2. `figures/bias_comparison.png` — RQ2: Bias metrics across methods
3. `figures/mispb_heatmap.png` — MISPB per judge and method
4. `figures/mispb_family_heatmap.png` — Family self-preference per judge and method
5. `figures/accuracy_vs_bias.png` — RQ2.2: Accuracy vs. bias scatter
6. `figures/mrspb_heatmap.png` — MRSPB per judge (SR vs AR)
7. `figures/hspp_comparison.png` — Harmful Self-Preference Propensity

## Tables

All CSV tables saved in `tables/`:
- `system_scores.csv` — System-level scores per generator and method
- `system_accuracy.csv` — MPA, MRD, MSD per method
- `system_bias.csv` — MRD-SP, MSD-SP, MSD-FSP per method
- `mpa_per_judge.csv` — MPA broken down by judge (with n_concordant, n_total)
- `msd_sp_per_judge.csv` — MSD-SP broken down by judge (with d_self, d_other, n_others)
- `mipa.csv` — MIPA per method
- `mipa_per_judge.csv` — MIPA broken down by judge (with n_agree, n_total)
- `mispb.csv` — MISPB summary per method
- `mispb_per_judge.csv` — MISPB broken down by judge (with overestimation counts)
- `rubric_metrics.csv` — MRA and MRSPB (SR, AR)
- `mra_per_judge.csv` — MRA broken down by judge (with n_correct, n_total)
- `mrspb_per_judge.csv` — MRSPB broken down by judge (with overestimation counts)
