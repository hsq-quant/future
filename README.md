# INE SC Sentiment Pipeline

Weekly news-sentiment research pipeline for INE crude oil futures (`SC`).

## Reference paper and timeliness

This project is directly inspired by a very recent arXiv working paper:

- **Beyond Polarity: Multi-Dimensional LLM Sentiment Signals for WTI Crude Oil Futures Return Prediction**
- Authors: Dehao Dai, Ding Ma, Dou Liu, Kerui Geng, Yiqing Wang
- arXiv: [`2603.11408`](https://arxiv.org/abs/2603.11408)
- Submitted: **March 2026**

Why this matters:

- The paper is extremely recent, which gives this project strong topical relevance.
- Its main contribution is to move beyond simple polarity and construct five LLM-based sentiment dimensions for weekly crude oil futures prediction.
- Our project adapts that framework from WTI to **INE SC**, and extends it with both classification and regression pipelines plus trading-layer experiments.

## What it does

- builds a continuous `SC` near-month series
- creates weekly returns and next-week labels
- ingests historical Chinese and English news sources
- scores articles with a Qwen-compatible API on five paper-aligned sentiment dimensions
- aggregates article scores into 11 weekly features
- trains LightGBM classification and regression models with 5-fold expanding-window validation
- compares simple classification execution and factor-style regression execution
- exports final aligned comparison tables, workbook draft, and reporting notes

## Final delivery

This repository is now organized around two final reporting artifacts:

- an Excel workbook draft
- a Markdown reporting notes document

The final class discussion should prioritize the **aligned** outputs, which clip every model/strategy to the same common trading window:

- `2021-01-22`
- `2025-09-05`

That prevents unfair comparisons caused by non-overlapping late-period performance.

## Key reporting outputs

- `/Users/hsq/Desktop/codex/future/reports/final/model_family_comparison_aligned.csv`
- `/Users/hsq/Desktop/codex/future/reports/final/model_family_comparison_aligned.md`
- `/Users/hsq/Desktop/codex/future/reports/final/regime_comparison_aligned.csv`
- `/Users/hsq/Desktop/codex/future/reports/final/regime_comparison_aligned.md`
- `/Users/hsq/Desktop/codex/future/reports/final/ine_sc_report_draft_aligned.xlsx`
- `/Users/hsq/Desktop/codex/future/reports/final/ine_sc_report_notes_aligned.md`
- `/Users/hsq/Desktop/codex/future/reports/final/charts/`

## Main scripts

### Training and strategy

- classification / regression training:

```bash
python3 scripts/train_eval.py --config configs/model_v1_baseline.yaml
python3 scripts/train_eval.py --config configs/model_v1_regression.yaml
python3 scripts/train_eval.py --config configs/model_v3_english_classification_sample20.yaml
python3 scripts/train_eval.py --config configs/model_v3_english_regression_sample20.yaml
```

- simple classification strategy:

```bash
python3 scripts/run_strategy.py \
  --config configs/model_v3_english_classification_sample20.yaml \
  --strategy-config configs/strategy.yaml
```

### Mapping comparisons

- classification execution variants:

```bash
python3 scripts/run_classification_mapping_comparison.py \
  --config configs/model_v1_baseline.yaml \
  --strategy-config configs/strategy_classification_variants.yaml \
  --output-dir reports/iterations/2026-03-28-v1-classification-mapping-comparison
```

- regression execution variants:

```bash
python3 scripts/run_factor_mapping_comparison.py \
  --config configs/model_v3_english_regression_sample20.yaml \
  --strategy-config configs/strategy_factor_variants.yaml \
  --output-dir reports/iterations/2026-03-28-v3-english-mapping-comparison
```

### Final aligned reporting

```bash
python3 scripts/build_final_reporting_bundle.py
```

This single script refreshes:

- `model_family_comparison_aligned.csv/.md`
- `regime_comparison_aligned.csv/.md`
- `ine_sc_report_draft_aligned.xlsx`
- `ine_sc_report_notes_aligned.md`
- `reports/final/charts/*.png`

## Workbook structure

The final workbook is now a draft-book, not just a raw dump. It includes:

- `01_Method`
- `02_Features`
- `03_Coverage`
- `04_IC_Dist`
- `05_Model_Table`
- `06_IC_Acc`
- `07_Market`
- `08_CumRet`
- `09_PerfCost`
- `10_Position`
- `11_RegimeBar`
- `12_RegimeHeat`
- `13_Notes`

## Notes

- `V1` is the Chinese mainline.
- `V3` is the English mainline.
- Classification is the intuitive “predict up/down” story.
- Regression is the “continuous factor” extension.
- Older non-aligned iteration outputs are still kept for traceability, but final reporting should use the aligned files above.
