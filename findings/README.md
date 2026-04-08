# Findings

This folder contains **only confirmed findings** — results that survived robustness checks and that we believe to be real, reproducible, and at least mildly novel.

The bar for promotion to this folder:

1. **Reproducible across seeds.** Multiple training runs give consistent results.
2. **Robust to obvious nulls.** A baseline (e.g. untrained network, random data) gives a clearly different answer.
3. **Robust to nuisance parameters.** Wiggling clustering thresholds, learning rates, etc. doesn't kill the result.
4. **Replicates on a second setting.** A different task / dataset / architecture gives the same qualitative pattern.
5. **At least one honest novelty check** that the result isn't trivially in the literature.

Each finding has its own markdown file with:

- A one-sentence summary
- A "simple words" explanation
- The data table that backs it
- The robustness checks that survived
- Honest scope limits
- How to reproduce (which experiment script, which command, what to look for)

## Index

| # | Title | Status |
|---|---|---|
| 1 | [Concept multiplicity invariance](./01_concept_multiplicity_invariance.md) | confirmed |

## Things that did NOT make it into findings

These were investigated but failed at least one of the criteria above:

- **Cross-architecture compression (experiment.py).** Showed that 3-neuron students can mimic any 4-to-10-neuron random ReLU teacher. Real, but rediscovers the well-known "neural networks are massively redundant" result. Not novel enough to promote.
- **Composition cost bimodality (experiment2.py).** Showed that random function composition is sometimes sub-additive and sometimes super-additive in essential complexity. Interesting, but only 4 seeds per cell — too noisy to claim with confidence, and the qualitative finding is implied by depth-separation theorems.
- **Selection vs optimization (experiment3.py).** Quantified that 4 well-placed ReLU hinges beat ~4000 random ones by 50–100x on natural functions. Real, but rediscovers the random-features-vs-trained-features gap from Rahimi-Recht 2007.
- **Multiplicity predicts adversarial robustness (experiment6.py).** Hypothesis: higher dream attractor count → more competing internal templates → more "weak points" for FGSM. **REJECTED.** The first run looked strong (Pearson −0.98) but the apparent correlation was driven entirely by the unregularized `wd=0` outlier. With proper controls (80 models across 8 weight-decay levels, 2 training-set sizes, 5 seeds), the correlation excluding the outlier collapses to Pearson **−0.01 to −0.22** at every FGSM epsilon. In the regime where models actually generalize, multiplicity varies from 3 to 10 but robustness is flat at 87–90%. The naive hypothesis is wrong.

These all live in `experiments/` for inspection but are *not* claims we stand behind as findings.
