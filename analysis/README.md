# The estimator

`estimator.py` is how every interval in the paper was computed. It is here so a reader can check
the method, and so the same estimator can be applied to evaluation records from a fresh run.

The unit of analysis is the problem, scored as Avg@4 over the four samples. A generation that
reached its cap was regenerated under a larger budget, and the rescue-aware rule takes the final
output. Intervals come from one paired bootstrap over problem clusters, 10,000 resamples with seed
20260810, stratified by the three split-construction groups, which are 128 problems each in the
frozen test split. The interval is the 2.5 and 97.5 percentile of the resampled difference, and the
two-sided bootstrap p is twice the smaller tail, floored at one over the number of resamples. A
contrast is resolved when its interval excludes zero. Where a comparison has several training
seeds, per-problem effects are averaged over seeds before resampling.

The external tier works differently, because the benchmarks are small. The 90 problems of
AIME 2024, AIME 2025 and HMMT February 2025 are pooled into one paired resample against the frozen
base, with no regeneration pass.

Holm's correction is applied where the paper reports it.

The paper also reports seed-level uncertainty beside the problem bootstrap, and those are here
too. `seed_mean_interval` gives a t interval over per-seed effects, and `welch_interval` compares
two configurations whose seed counts differ. The two answer different questions: the seed-level
check conditions on the test set, the problem bootstrap on the observed seeds. `seed_aware`
averages per-problem effects over seeds before resampling, keeping the problems every seed
scored.

## Using it

```python
import sys; sys.path.insert(0, "analysis")
import estimator, json
from opsd import artifact_layout as layout

cell, provenance = estimator.read_cell(layout.path(layout.CONTROL_EVAL_THINKING),
                                       "fu_", "<run name>", 100)
split = json.load(open(estimator.SPLITS_FILE))
boot = estimator.Boot(list(split["test"]), split["group"])
```

`read_cell` turns one evaluation cell into per-problem Avg@4 and reports whether the cell is
complete. `Boot` holds the shared resample: `vec` puts a cell's per-problem values in the split's
order, `mean` gives a level with its interval, and `delta` gives a paired contrast with its interval
and p-value. It reads the stratum sizes from the split, so a different split works unchanged.
`holm` adjusts a family of p-values. `ext_cell`, `ext_anchor` and `ext_boot` do the same for the
external benchmarks, pooling all 90 problems in one resample without benchmark strata.

The evaluation records themselves are not part of this release. Produce them with `training/` and
`evaluation/`, and point `OPSD_ARTIFACTS` at the result.
