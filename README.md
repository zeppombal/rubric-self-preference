# Overview

Repository for the paper [Self-Preference Bias in Rubric-Based Evaluation
of Large Language Models](https://arxiv.org/pdf/2604.06996).

# Reproduce paper experiments and analysis

To reproduce the experiments and analysis, install [ducttape](https://github.com/CoderPat/ducttape) and run:

```bash
ducttape ducttape/tasks/main.tape -C ducttape/tconfs/test.tconf -p reproduce_paper
```
> Setting up paths, and model details and credentials is required at `ducttape/tconfs/base.tconf`

The `sandbox` folder contains csv's and dashboards built based on the experiments from the command above. To jump straight to the dashboards, check `sandbox/dashboard.html` (IFEval) and `sandbox/hb/hb_dashboard.html` (HealthBench).


# Citation

```bibtex
@misc{pombal2026selfpreferencebiasrubricbasedevaluation,
      title={Self-Preference Bias in Rubric-Based Evaluation of Large Language Models}, 
      author={José Pombal and Ricardo Rei and André F. T. Martins},
      year={2026},
      eprint={2604.06996},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.06996}, 
}
```
