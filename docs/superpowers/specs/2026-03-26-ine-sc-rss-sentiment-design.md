# INE SC RSS Sentiment Prediction Design

**Date:** 2026-03-26

**Status:** Approved for implementation planning

## Goal

Reproduce the core methodology of the paper *Beyond Polarity: Multi-Dimensional LLM Sentiment Signals for WTI Crude Oil Futures Return Prediction* and migrate it to Shanghai International Energy Exchange (INE) crude oil futures (`SC`) using only Chinese RSS news sources.

The objective is to fully run the end-to-end research pipeline rather than maximize final predictive performance. The project should prioritize clarity, reproducibility, and completion over squeezing out marginal model improvements.

## Scope

### In Scope

- Build a continuous `SC` futures price series with an explicit roll rule.
- Generate weekly returns and next-week binary direction labels.
- Collect Chinese energy-related news from RSS feeds only.
- Normalize and clean RSS article data into a unified research dataset.
- Use an external Qwen-family model to score each article on five sentiment dimensions.
- Construct the 11 weekly features used for a single five-dimension LLM source in the paper.
- Train and evaluate a LightGBM classifier with five-fold expanding-window time-series cross-validation.
- Report `AUROC`, `Accuracy`, `IC`, and `SHAP` feature importance.
- Run a simple weekly paper-trading simulation on top of the prediction output.
- Deliver scripts, config, and documentation needed to rerun the pipeline.

### Out of Scope

- Overseas news sources.
- Full web crawling of arbitrary sites.
- Multi-LLM ensembles in phase one.
- Performance tuning beyond what is needed to complete a valid run.
- Sophisticated execution assumptions, portfolio optimization, and production-grade backtesting.
- Fancy dashboards or production deployment.

## Research Framing

This project is a **replication plus migration**, not a strict one-to-one reproduction.

- Replication:
  - Preserve the paper's core workflow: article scoring, weekly aggregation, 11 features, LightGBM, expanding-window CV, and SHAP.
- Migration:
  - Replace WTI with INE `SC`.
  - Replace the paper's English and mixed-source news setup with Chinese RSS-only sources.
  - Replace GPT-4o with a lower-cost Qwen-family model supplied by the user.

The expected outcome is a complete and defensible process showing that the paper's method can be carried over to a Chinese crude oil futures setting.

## Label Definition

### Contract Construction

Use a **continuous near-month `SC` series** with a fixed roll rule:

- Start from individual INE `SC` contract daily bars.
- Roll from the current contract to the next contract **5 trading days before the current contract's last trading day**.
- Use the active contract's daily close as the continuous series close.

This rule is preferred because it is explicit, stable, and easy to explain in a course assignment. It also avoids relying on vendor-specific "main contract" definitions.

### Weekly Return

Construct weekly close observations using the last trading day of each calendar week, typically Friday. Compute weekly log return:

`r_t = log(P_t / P_{t-1})`

where `P_t` is the weekly close of the continuous `SC` series.

### Prediction Target

Define the binary target as:

`y_t = 1[r_{t+1} > 0]`, otherwise `0`

This means week `t` news is used to predict whether week `t+1` has a positive return.

### Week Alignment Rule

To mirror the paper's weekly design, every observation must be assigned to a weekly bucket before feature construction:

- define each prediction week by its **last INE trading day of the week**
- assign an article to week `t` if its `published_at` timestamp falls within that Shanghai calendar week
- use only articles published **no later than the end of week `t`**
- use those week `t` articles to predict the return direction of week `t+1`

This should be implemented so that there is no ambiguity about news timing and no look-ahead leakage into the next week.

## Simple Trading Layer

The project should add one deliberately simple trading simulation after the predictive evaluation is complete.

### Strategy Definition

Use a weekly **always-in directional strategy**:

- if the model predicts week `t+1` will be up, take a `+1` long position for week `t+1`
- if the model predicts week `t+1` will be down, take a `-1` short position for week `t+1`

The signal is generated using information available by the end of week `t`, and the resulting position is applied to week `t+1`.

The trading signal must be derived from the **out-of-sample class prediction** produced by the expanding-window validation process, not from in-sample fitted values.

### Strategy Return

Weekly strategy return is:

`strategy_return_{t+1} = position_t * r_{t+1}`

where `position_t` is `+1` or `-1`, and `r_{t+1}` is the realized week `t+1` futures return.

### Simplifying Assumptions

To keep the assignment lightweight, phase one may assume:

- no transaction costs
- no slippage
- full notional long or short exposure
- rebalance once per week

These assumptions must be stated clearly in the final report.

## News Data Design

### Source Policy

Use **Chinese RSS feeds only**. No direct browser scraping is required in phase one unless an RSS item lacks readable body text and a simple fetch can recover it.

Selection principles:

- Chinese-language source
- Stable RSS or Atom feed
- Energy, macro, policy, or commodities relevance
- Reasonable availability of article title, link, publish time, and ideally summary/body

### Initial Feed Set

The project should start with a small, stable set of feeds so the pipeline can run end to end:

- National Energy Administration or other official China energy-policy feeds
- Xinhua or similar general Chinese news feed with energy coverage
- One or two Chinese financial news feeds with commodities or macro coverage

The feed list should be stored in a config file rather than hard-coded in scripts.

### Raw Article Schema

Each ingested article should be normalized to:

- `article_id`
- `title`
- `body`
- `summary`
- `source`
- `source_type`
- `published_at`
- `published_at_tz`
- `url`
- `language`
- `rss_feed`
- `raw_tags`
- `ingested_at`

### Time Zone

Convert all timestamps to `Asia/Shanghai`.

This simplifies weekly alignment with INE data and removes cross-market timing ambiguity.

## News Cleaning and Filtering

### Deduplication

Remove duplicates using one or more of:

- exact `url`
- exact `(published_at, title)`
- normalized-title hash

Keep the first valid record when duplicates conflict.

### Text Cleaning

Apply lightweight text normalization only:

- strip HTML
- normalize whitespace
- convert Traditional Chinese to Simplified Chinese
- preserve numerals, dates, and domain terms

### Language Handling

Only keep Chinese-language articles in phase one. Language detection can be used as a guardrail, but source-level assumptions are acceptable when feeds are known Chinese sources.

### Topic Filtering

Keep only energy-relevant items using simple rules:

- feed-level relevance where possible
- title/body keyword matching for terms such as `原油`, `油价`, `INE`, `上海原油`, `能源`, `库存`, `OPEC`, `炼厂`, `成品油`, `供应`, `需求`

The keyword filter should remain deliberately simple because the LLM relevance score is the main semantic filter later in the pipeline.

## LLM Scoring Design

### Model

Use a user-provided Qwen-family model through external API access.

The scoring step must be implemented behind a model adapter so the exact Qwen endpoint can be swapped without changing pipeline logic.

### Prompt

Use the paper's Appendix A prompt structure with only minimal wording adaptation where needed for Chinese input. The required output fields remain:

- `relevance`
- `polarity`
- `intensity`
- `uncertainty`
- `forwardness`

The scoring rules must preserve the paper's behavior:

- evaluate `relevance` first
- if `relevance < 0.1`, set all other fields to `null`
- return JSON only

### Exact Dimension Semantics

Each article must be scored on the same numeric ranges used in the paper:

- `relevance`: `0.0` to `1.0`
  - `0.0` means unrelated to oil, gas, coal, energy policy, or energy markets
  - `1.0` means directly relevant to crude oil or broader energy-market pricing
- `polarity`: `-1.0` to `1.0`
  - negative values mean bearish for oil or energy markets
  - positive values mean bullish
- `intensity`: `0.0` to `1.0`
  - low values mean weak or neutral wording
  - high values mean strong conviction or strong emotional tone
- `uncertainty`: `0.0` to `1.0`
  - low values mean certainty and clear factual tone
  - high values mean ambiguity, hedging, risk, or unclear outlook
- `forwardness`: `0.0` to `1.0`
  - low values mean mostly backward-looking reporting
  - high values mean future-oriented expectations, forecasts, or projections

### Operational Scoring Rules

The prompt and parser must encode the paper's judgment rules explicitly:

- score `relevance` first
- if `relevance < 0.1`, force `polarity`, `intensity`, `uncertainty`, and `forwardness` to `null`
- treat `polarity` and `intensity` as independent
  - a strongly worded negative article can have `polarity = -0.9` and `intensity = 0.9`
- interpret `uncertainty` through hedge or ambiguity language
  - examples include expressions like `可能`, `或许`, `仍不确定`, `存在风险`, `尚不明确`
- interpret `forwardness` through future-oriented wording
  - forecasts, expectations, policy outlooks, guidance, planned supply changes, demand outlooks, and projected inventory moves

### Expected JSON Schema

The model adapter should expect exactly this logical schema:

```json
{
  "relevance": 0.0,
  "polarity": 0.0,
  "intensity": 0.0,
  "uncertainty": 0.0,
  "forwardness": 0.0
}
```

When `relevance < 0.1`, the valid shape becomes:

```json
{
  "relevance": 0.05,
  "polarity": null,
  "intensity": null,
  "uncertainty": null,
  "forwardness": null
}
```

### Article-Level Scoring Output

Store one row per article with:

- article identifiers and metadata
- model name
- prompt version
- all five dimensions
- raw JSON response
- parse status
- request timestamp

Scoring should be cacheable so reruns do not repeatedly call the API for already scored articles.

## Weekly Feature Construction

Phase one uses the paper's **single-model 11-feature design**.

### Article Set for Week `t`

Let `A_t` be the set of scored articles assigned to week `t`.

For each article `i` in week `t`, the Qwen scorer returns:

`s_{t,i} = (re_{t,i}, p_{t,i}, intensity_{t,i}, u_{t,i}, f_{t,i})`

where:

- `re` is relevance
- `p` is polarity
- `intensity` is sentiment intensity
- `u` is uncertainty
- `f` is forwardness

Articles with `relevance < 0.1` remain in the weekly article count, but their four non-relevance fields are `null` by prompt rule.

### Weekly Base Fields

- `article_count`
- `relevance_mean`
- `polarity_mean`
- `intensity_mean`
- `uncertainty_mean`
- `forwardness_mean`

Define these fields exactly as follows:

- `article_count_t`
  - number of scored articles in `A_t`
- `relevance_mean_t`
  - simple arithmetic mean of `re_{t,i}` over all articles in `A_t`
- `polarity_mean_t`
- `intensity_mean_t`
- `uncertainty_mean_t`
- `forwardness_mean_t`

For `polarity_mean`, `intensity_mean`, `uncertainty_mean`, and `forwardness_mean`, use the paper's relevance-weighted mean over articles with non-null values:

`mean_w(t) = sum(relevance_i * w_i) / sum(relevance_i)`

where `w ∈ {polarity, intensity, uncertainty, forwardness}`.

No extra relevance filtering is applied beyond the prompt rule and this weighting. In other words:

- do **not** drop low-relevance articles before counting
- do **not** apply an extra hard relevance threshold during aggregation
- use relevance only as the weighting term exactly as in the paper

If a week has no valid denominator for a weighted mean, set that weekly feature to missing and exclude that row later only if the final modeling table cannot support it. The implementation should log these cases.

### Dispersion Features

- `polarity_std`
- `uncertainty_std`

These capture within-week disagreement across articles.

Define them as unweighted within-week standard deviations over non-null article-level scores:

- `polarity_std_t = std({p_{t,i}})`
- `uncertainty_std_t = std({u_{t,i}})`

The point of these features is to measure disagreement or dispersion in the week's news flow, not average tone.

### Momentum Features

- `polarity_momentum`
- `uncertainty_momentum`
- `forwardness_momentum`

Each is defined as the first difference of the weekly mean:

`delta_w(t) = mean_w(t) - mean_w(t-1)`

Only these three dimensions receive momentum features. This follows the paper directly:

- include momentum for `polarity`, `uncertainty`, and `forwardness`
- exclude momentum for `relevance`
- exclude momentum for `intensity`

The reason is that polarity, uncertainty, and forwardness have interpretable temporal directionality, while week-to-week changes in relevance and intensity do not have a clear market interpretation in the paper.

### Final 11 Features

- `article_count`
- `relevance_mean`
- `polarity_mean`
- `intensity_mean`
- `uncertainty_mean`
- `forwardness_mean`
- `polarity_std`
- `uncertainty_std`
- `polarity_momentum`
- `uncertainty_momentum`
- `forwardness_momentum`

### Weekly Modeling Row

After feature construction, each usable modeling row should contain:

- `week_end_date`
- the 11 weekly sentiment features
- realized `r_t`
- target label `y_t = 1[r_{t+1} > 0]`

The first week without a lagged return and the last week without a forward label should be excluded from the final supervised dataset.

## Modeling Design

### Model Choice

Use `LightGBM` binary classification, matching the paper's tabular prediction setup.

The model input for phase one is the 11-feature weekly table from a single Qwen-based sentiment source.

### Hyperparameter Search

Use `Optuna` with TPE search. The optimization target is mean validation `AUROC`.

The tuned objective should operate on the same expanding-window weekly validation design used for final evaluation.

### Cross-Validation

Use **5-fold expanding-window time-series cross-validation**.

Design requirements:

- no look-ahead leakage
- each fold trains on earlier weeks and validates on later weeks
- folds should be deterministic and saved for reproducibility

Implementation expectations:

- sort all observations by `week_end_date`
- split chronologically into 5 validation folds
- for fold `k`, train on all weeks strictly before the validation block
- validate on the next contiguous block of future weeks
- save out-of-sample predicted probabilities for every validation observation

At the end of evaluation, the out-of-sample predictions from all folds should be concatenated into one prediction table covering the full validation horizon.

### Thresholding

For `Accuracy`, follow the paper rather than using a naive `0.5` threshold.

The paper states that positive-return weeks account for about `53.03%` of the sample and sets the classification threshold to match the empirical class distribution. To stay aligned, phase one should:

- compute the positive-class share from the training sample for each fold
- choose a probability cutoff so the predicted positive rate approximately matches that share
- use that cutoff for fold-level `Accuracy` and for the simple trading signal

If an exact class-ratio-matching rule proves awkward in implementation, the fallback may be a fixed threshold, but the report must say so explicitly.

### Metrics

Report the same core metrics as the paper:

- `AUROC`
  - threshold-free classification discrimination
- `Accuracy`
  - binary directional hit rate using the class-ratio-aligned threshold
- `IC`
  - Spearman rank correlation between predicted positive probability and realized next-week return

For `IC`, use the paper's interpretation:

- higher predicted probabilities should be associated with higher realized next-week returns
- `IC` evaluates the ranking quality of the full probability output, not just the binary decision

### SHAP Interpretation

Use SHAP on the fitted LightGBM model to measure feature contribution.

The implementation should at minimum produce:

- global mean absolute SHAP ranking
- a summary plot or equivalent summary table

This is important because the paper's main qualitative takeaway is not only whether prediction works, but also which sentiment dimensions matter most.

## Evaluation Outputs

The completed pipeline should produce:

- fold-by-fold `AUROC`
- fold-by-fold `Accuracy`
- fold-by-fold `IC`
- aggregate mean and standard deviation across folds
- an out-of-sample weekly prediction table with `week_end_date`, `pred_prob`, `pred_label`, `actual_return`, and `actual_label`
- SHAP global importance plot or ranking table
- a concise experiment summary table
- a weekly simulated strategy return series
- simple trading summary metrics such as cumulative return, win rate, and max drawdown

The final interpretation should answer:

- Did the multi-dimensional Qwen sentiment pipeline run successfully on INE?
- Do the features show any directional predictive content above random?
- Which dimensions appear most influential under SHAP?
- What does the simplest possible weekly long-short simulation look like under these predictions?

## Project Structure

The implementation should stay simple and assignment-friendly.

Suggested structure:

- `README.md`
- `configs/feeds.yaml`
- `configs/model.yaml`
- `data/raw/`
- `data/intermediate/`
- `data/processed/`
- `notebooks/` or `reports/`
- `src/data/`
- `src/features/`
- `src/models/`
- `src/pipeline/`
- `src/utils/`
- `scripts/`

## Dependencies

Expected Python dependencies:

- `pandas`
- `numpy`
- `pyarrow`
- `requests` or `httpx`
- `feedparser`
- `beautifulsoup4`
- `trafilatura`
- `opencc`
- `pydantic`
- `tenacity`
- `lightgbm`
- `optuna`
- `scikit-learn`
- `shap`
- `matplotlib`
- `seaborn`

Optional if needed:

- `langdetect`
- `duckdb`
- `pytest`

## External Requirements

The project needs:

- user-provided Qwen API key and endpoint details
- a usable INE historical price source for individual `SC` contracts
- a short list of Chinese RSS feed URLs

If free INE contract history is incomplete, the implementation may begin with the best available accessible source, but the contract roll logic must remain explicit in code and docs.

## Success Criteria

The project is successful if all of the following are true:

- the full pipeline runs from RSS ingestion to model evaluation
- weekly labels for INE `SC` are generated correctly and reproducibly
- Qwen article scoring is cached and reusable
- the 11 weekly sentiment features are constructed exactly as specified
- 5-fold expanding-window LightGBM evaluation completes without leakage
- fold-level prediction probabilities and labels are saved for inspection
- the simple weekly always-in trading simulation runs on out-of-sample predictions
- results are summarized in a clean, reproducible report

Predictive performance does **not** need to beat the original paper or reach any specific threshold.

## Risks and Simplifications

### Main Risks

- Chinese RSS feeds may have shallow article bodies or inconsistent formatting.
- Historical RSS coverage may be limited depending on the source.
- Qwen JSON output may occasionally be malformed and require retry logic.
- INE contract history access may be the main data bottleneck.

### Chosen Simplifications

- use only Chinese RSS feeds
- use one LLM source in phase one
- use only a simple weekly always-in strategy
- skip advanced feature expansion unless the baseline pipeline is already complete

These simplifications are intentional and aligned with the project's assignment-first goal.

## Implementation Order

The implementation plan should follow this order:

1. Build the `SC` contract stitching and weekly label pipeline.
2. Build RSS ingestion and normalization.
3. Build cleaning, filtering, and deduplication.
4. Build Qwen scoring with cache and retry logic.
5. Build weekly 11-feature aggregation.
6. Build LightGBM plus Optuna and 5-fold expanding-window evaluation.
7. Build the simple weekly always-in strategy simulation.
8. Build SHAP reporting and final summary artifacts.
