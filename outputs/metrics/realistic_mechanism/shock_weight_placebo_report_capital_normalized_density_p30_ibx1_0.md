# Shock-weight placebo robustness check

This diagnostic compares the real sector-shock mapping with two benchmarks:

- aggregate firm exposure only;
- a uniform-shock exposure index, where every sector receives the same scenario-average shock;
- randomized placebo mappings, where real sector shocks are permuted across sector labels.

The target is not regenerated here. This is therefore a placebo/alignment test on the existing scenarios, not a full counterfactual clearing simulation.

Random placebo draws: 1000
Randomization groups: graph_id

## Main feature comparison

| feature | all_spearman | test_spearman | test_r2 | test_mae | test_rmse |
| --- | --- | --- | --- | --- | --- |
| aggregate_firm_exposure_to_capital | 0.332692 | 0.293419 | 0.130609 | 0.411753 | 0.612525 |
| actual_shock_weighted_sector_exposure_to_capital | 0.864657 | 0.855287 | 0.794638 | 0.186357 | 0.297698 |
| uniform_shock_weighted_sector_exposure_to_capital | 0.818399 | 0.800929 | 0.670749 | 0.234485 | 0.376947 |

## Random placebo summary

| metric | actual_value | random_mean | random_std | random_p05 | random_p50 | random_p95 | placebo_p_value_as_good_or_better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_pearson | 0.914445 | 0.822172 | 0.00234247 | 0.8184 | 0.822205 | 0.825946 | 0.000999001 |
| all_spearman | 0.864657 | 0.78782 | 0.00133809 | 0.785613 | 0.787782 | 0.790112 | 0.000999001 |
| test_pearson | 0.89611 | 0.796685 | 0.00586305 | 0.78747 | 0.796703 | 0.806122 | 0.000999001 |
| test_spearman | 0.855287 | 0.774412 | 0.00315269 | 0.769197 | 0.774358 | 0.779485 | 0.000999001 |
| test_r2 | 0.794638 | 0.623056 | 0.0105218 | 0.606035 | 0.622931 | 0.640409 | 0.000999001 |
| test_mae | 0.186357 | 0.24587 | 0.00221989 | 0.242212 | 0.245943 | 0.249453 | 0.000999001 |
| test_rmse | 0.297698 | 0.403285 | 0.00563171 | 0.393932 | 0.403391 | 0.41233 | 0.000999001 |

## Interpretation guide

Evidence is stronger if the actual shock-weighted sector exposure beats aggregate exposure, beats the uniform-shock benchmark, and lies above the randomized placebo distribution for correlation/R2 metrics.

If the uniform benchmark performs almost the same as the actual shock-weighted measure, the result is mostly aggregate exposure plus average scenario severity.

If randomized mappings perform almost as well as the actual mapping, the conclusion should be weakened because sector labels are not adding much information.
