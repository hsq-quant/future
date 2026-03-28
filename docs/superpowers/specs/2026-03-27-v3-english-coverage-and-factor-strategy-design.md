# INE SC V3 English Coverage And Factor Strategy Design

**Date:** 2026-03-27

**Status:** Drafted for next-session execution

## Goal

Use the next iteration (`V3`) to answer two more practical research questions:

1. Can we expand and rebalance the English global news pipeline so that the English sample covers the full weekly research horizon more evenly, instead of stopping early and over-concentrating in a subset of weeks?
2. Can we turn the current weekly LLM feature stack into a more realistic factor-style trading signal by moving from binary classification to continuous return prediction, then mapping the signal into position size rather than simple `+1/-1` switching?

This iteration is not just about improving one metric. It is about making the research design more interpretable, more comparable across languages, and more useful as a realistic systematic signal.

## Why V3 Is Needed

### Current Findings

We already have three useful baselines:

- `V1 中文主链`
  - full supervised horizon
  - weak but non-random predictive power
  - slightly positive excess return versus always-long benchmark
- `V2 中英增强`
  - full supervised horizon
  - weaker than `V1`
  - simple Chinese+English concatenation did not help
- `V2 衍生版纯英文`
  - much stronger metrics numerically
  - but only `131` out-of-sample weeks
  - ends at `2025-09-05`
  - not directly comparable with `V1` and `V2`

### Main Interpretation

The current evidence suggests:

- English global energy news may contain strong information.
- But our current English coverage is not yet full-horizon and evenly distributed.
- Simple multilingual mixing adds noise.
- A binary up/down prediction is easy to explain, but it is still a blunt instrument for real trading use.

So `V3` should do two things:

- make English coverage fairer and more complete
- convert the model output into a continuous factor signal that is closer to real quant workflow

## Workstream A: Full-Coverage English Pipeline

### Research Question

If the English global energy news pipeline is expanded to cover the full horizon more evenly, does the English-only signal remain strong, and does it become more comparable to the Chinese baseline?

### Current Limitation

The current English-only version stops at `2025-09-05` because:

- the Hugging Face source was only partially materialized into the local research store during `V2`
- the current energy keyword filter is relatively strict
- after weekly alignment and relevance filtering, many weeks remain sparse or empty

This means the present English-only results are informative, but not yet a fair “full-history” experiment.

### V3 Design

#### A.1 Source Expansion

Use the Hugging Face English global financial news source as the primary English input, but move from partial ingestion to full ingestion:

- fetch all available parquet shards, not just the subset used in `V2`
- store them under the same English archive path
- keep the ingestion process resumable and idempotent

The key operational change is:

- use the existing fetcher with `--all-files`
- then rebuild the normalized English archive from the full local shard set

#### A.2 Smarter Energy Filtering

Do not just make the English filter looser. Make it broader but more structured.

The filtering logic should move from “small keyword set” to “energy-topic buckets” such as:

- crude oil / oil price
- OPEC / OPEC+
- SPR / petroleum reserve
- refinery / refining margins
- inventories / stockpiles
- shale / rigs / drilling
- gasoline / diesel / distillates
- shipping / tanker / Red Sea / freight if clearly oil-linked

This should improve coverage without turning the English set into generic macro noise.

#### A.3 Coverage Audit

Before rebuilding the model, explicitly measure English weekly coverage:

- total English article count
- article count by year
- article count by week
- number of supervised weeks with at least 1 aligned English article
- number of supervised weeks with at least 3 aligned English articles
- longest gap of zero-article weeks

This audit should produce a compact report so we can answer:

- did English become full-horizon?
- is it only technically full, or actually reasonably dense?

### Success Criteria For Workstream A

Minimum:

- English-only weekly model table reaches at least `280` supervised weeks
- no long tail ending at `2025-09-05`
- final English-only sample is clearly more comparable to `V1`

Preferred:

- English-only reaches the full `316` supervised weeks
- weekly coverage is not dominated by a small recent block

## Workstream B: Regression Factor Strategy

### Research Question

If we treat the LLM feature stack as a return-prediction factor rather than just a classification signal, can we build a more realistic and interpretable strategy layer?

### Why Move Beyond Binary Classification

The current classifier only answers:

- up or down next week

But for trading, that throws away useful information:

- expected magnitude
- confidence ranking
- signal strength distribution

A more realistic quant path is:

- predict next-week return directly
- standardize the prediction into a factor exposure
- map that exposure into position size

### B.1 Target Definition

Replace the classification target:

- old: `y_t = 1[r_{t+1} > 0]`

with a regression target:

- new: `target_t = r_{t+1}`

where `r_{t+1}` is the realized next-week log return of continuous `INE SC`.

### B.2 Model Objective

Use LightGBM regression instead of classification.

Primary optimization target:

- `IC`

Secondary diagnostics:

- `MSE`
- `MAE`
- directional hit rate from `sign(predicted_return)`

The training structure remains:

- `5-fold`
- `expanding window`
- fully out-of-sample predictions

### B.3 IC Definition And Interpretation

For this project, `IC` is the Spearman rank correlation between the model score and the realized next-week return:

`IC = corr_rank(ŷ_t, r_{t+1})`

where:

- `ŷ_t` is the out-of-sample prediction for week `t`
- `r_{t+1}` is the realized next-week return

Interpretation:

- `IC > 0` means higher predicted scores tend to align with higher realized returns
- `IC < 0` means the ranking is wrong-way
- small positive `IC` can still matter in systematic strategies if breadth is high

This is important because a factor model is often judged more by ranking quality than by exact return magnitude.

### B.4 Signal Standardization

After producing out-of-sample predicted returns, standardize them using only past information.

Base version:

- rolling z-score of the prediction

Formula:

- `z_t = (ŷ_t - mean(ŷ_{1:t-1})) / std(ŷ_{1:t-1})`

This must be computed causally:

- only past predicted values may be used
- no future leakage

### B.5 Position Mapping

The strategy should no longer be just `+1/-1`.

We should test at least these two mappings:

#### Mapping 1: Linear Clip

- `position_t = clip(0.5 * z_t, -1, +1)`

Interpretation:

- moderate signal -> moderate position
- very strong signal -> capped full position

#### Mapping 2: Tanh Compression

- `position_t = tanh(k * z_t)`

with a small grid such as:

- `k = 0.8`
- `k = 1.0`
- `k = 1.2`

Interpretation:

- keeps signal direction
- compresses extreme exposures
- more robust to outliers

Optional later extension, not required for tomorrow:

- volatility-targeted scaling

### B.6 Strategy Accounting

The new regression strategy should still be evaluated on the same weekly horizon and compared with:

- always-long benchmark
- the current binary strategy

Outputs should include:

- gross return
- realistic `0 tick`
- realistic `1 tick`
- realistic `2 tick`

And metrics should include:

- cumulative return
- annualized return
- annualized volatility
- Sharpe ratio
- max drawdown
- Calmar ratio
- win rate
- benchmark excess return
- information ratio

### B.7 Regime Analysis

Run the same regime split already used in `V1`:

- `Recovery Bull`
- `War Spike`
- `Post-Spike Bear`
- `OPEC-Supported Range`
- `Oversupply Bear`

But now report them for the regression factor strategy as well.

This lets us answer:

- does the factor-style strategy reduce the “all or nothing” behavior we saw in classification?
- does it improve bear-phase alpha while giving up less in bull phases?

## Output Structure

`V3` should remain isolated from `V1` and `V2`.

Recommended naming:

- `2026-03-28-english-full-v3`
- `2026-03-28-english-full-v3-regression`

At minimum, isolate:

- scored articles
- weekly model table
- predictions
- strategy outputs
- reports

## Deliverables For Tomorrow

### Deliverable 1: English Coverage Upgrade

- full English shard fetch completed or fully resumed
- rebuilt English clean table
- coverage audit report
- new English-only weekly model table with improved week coverage

### Deliverable 2: Regression Factor Prototype

- LightGBM regression CV run
- out-of-sample predicted return series
- z-score signal series
- linear and tanh position-mapping strategies
- scenario and regime reports

### Deliverable 3: Comparison Summary

One compact comparison between:

- `V1 中文主链`
- `V2 中英增强`
- `V2 衍生版纯英文`
- `V3 全量英文`
- `V3 全量英文回归因子策略`

## Guardrails

- Do not overwrite `V1` or `V2`.
- Do not treat partial English coverage as directly comparable with full-horizon Chinese results.
- Keep all z-score and signal normalization causal.
- Keep benchmark and excess-return reporting mandatory.
- Treat `IC` as a primary research metric, not just a side metric.

## Short Conclusion

Tomorrow’s work should not just “try more English.”

It should answer two sharper questions:

1. If English news is made full-horizon and better filtered, does it still outperform the Chinese-only baseline?
2. If we convert the current LLM news stack into a continuous factor signal, does it become more realistic and more useful as a trading input than the current binary classifier?
