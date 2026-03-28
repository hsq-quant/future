# V3 English Coverage And Regression Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the English source to full local coverage, then build and compare regression-factor strategy variants for V3 English and V1 Chinese.

**Architecture:** Keep the current binary pipeline intact and add a parallel regression path. First, complete the local Hugging Face shard ingestion and rebuild the English clean table with a broader but structured energy filter plus coverage audit. Then add a regression training/evaluation layer and continuous-position strategy layer, isolating outputs into new V3 iteration directories so V1/V2 remain untouched.

**Tech Stack:** Python 3.11/3.9, pandas, LightGBM, Optuna, shap, matplotlib, parquet/pyarrow, existing project scripts.

---

### Task 1: English Coverage Audit And Full-Download Support

**Files:**
- Modify: `/Users/hsq/Desktop/codex/future/scripts/fetch_hf_financial_multisource.py`
- Modify: `/Users/hsq/Desktop/codex/future/src/data/hf_dataset_fetch.py`
- Create: `/Users/hsq/Desktop/codex/future/src/data/english_coverage_audit.py`
- Create: `/Users/hsq/Desktop/codex/future/scripts/audit_english_coverage.py`
- Test: `/Users/hsq/Desktop/codex/future/tests/test_hf_dataset_fetch.py`
- Test: `/Users/hsq/Desktop/codex/future/tests/test_english_energy_candidates.py`

- [ ] **Step 1: Write failing tests for full-file selection and coverage audit**

Add tests for:
- selecting all parquet URLs when `--all-files` is requested
- computing weekly coverage summary and longest zero-article gap from aligned English data

- [ ] **Step 2: Run the targeted tests and verify they fail**

Run:
```bash
pytest -q /Users/hsq/Desktop/codex/future/tests/test_hf_dataset_fetch.py /Users/hsq/Desktop/codex/future/tests/test_english_energy_candidates.py
```

- [ ] **Step 3: Implement minimal coverage-audit helpers and any missing fetch behavior**

Implement a small audit helper that outputs:
- total rows
- min/max publish date
- supervised weeks covered
- weeks with >=1 article
- weeks with >=3 articles
- longest zero-article gap

- [ ] **Step 4: Re-run the targeted tests and verify they pass**

Run:
```bash
pytest -q /Users/hsq/Desktop/codex/future/tests/test_hf_dataset_fetch.py /Users/hsq/Desktop/codex/future/tests/test_english_energy_candidates.py
```

### Task 2: Broaden English Energy Filtering

**Files:**
- Modify: `/Users/hsq/Desktop/codex/future/src/data/english_energy_candidates.py`
- Modify: `/Users/hsq/Desktop/codex/future/scripts/build_english_global_clean.py`
- Test: `/Users/hsq/Desktop/codex/future/tests/test_english_energy_candidates.py`

- [ ] **Step 1: Write failing tests for broader structured topic buckets**

Add tests covering:
- OPEC / SPR / refinery / inventories / tanker-style oil-linked text
- avoiding obvious false positives

- [ ] **Step 2: Run the focused test and verify failure**

Run:
```bash
pytest -q /Users/hsq/Desktop/codex/future/tests/test_english_energy_candidates.py
```

- [ ] **Step 3: Implement broader structured phrase sets and preserve stable ids**

Keep the filter explicit and explainable, not fuzzy.

- [ ] **Step 4: Re-run the focused test and verify pass**

Run:
```bash
pytest -q /Users/hsq/Desktop/codex/future/tests/test_english_energy_candidates.py
```

### Task 3: Regression Training Path

**Files:**
- Modify: `/Users/hsq/Desktop/codex/future/src/models/train_eval.py`
- Modify: `/Users/hsq/Desktop/codex/future/scripts/train_eval.py`
- Create: `/Users/hsq/Desktop/codex/future/configs/model_v3_english_regression.yaml`
- Create: `/Users/hsq/Desktop/codex/future/configs/model_v1_regression.yaml`
- Test: `/Users/hsq/Desktop/codex/future/tests/test_scoring_and_cv.py`

- [ ] **Step 1: Write failing tests for regression prep and regression metric output**

Add tests for:
- preparing a regression training table
- computing IC from continuous predictions and returns
- emitting regression prediction columns distinct from classifier columns

- [ ] **Step 2: Run regression-targeted tests and verify failure**

Run:
```bash
pytest -q /Users/hsq/Desktop/codex/future/tests/test_scoring_and_cv.py
```

- [ ] **Step 3: Implement minimal regression-capable training path**

Support both:
- binary classification (existing path)
- regression (`next_week_return` target)

Keep the existing classifier outputs unchanged.

- [ ] **Step 4: Re-run regression-targeted tests and verify pass**

Run:
```bash
pytest -q /Users/hsq/Desktop/codex/future/tests/test_scoring_and_cv.py
```

### Task 4: Continuous Factor Strategy Layer

**Files:**
- Modify: `/Users/hsq/Desktop/codex/future/src/models/strategy.py`
- Modify: `/Users/hsq/Desktop/codex/future/scripts/run_strategy.py`
- Modify: `/Users/hsq/Desktop/codex/future/configs/strategy.yaml`
- Test: `/Users/hsq/Desktop/codex/future/tests/test_modeling_and_strategy.py`

- [ ] **Step 1: Write failing tests for rolling z-score and continuous position mapping**

Add tests for:
- causal z-score using only past predictions
- linear clipped position mapping
- tanh position mapping
- continuous strategy returns vs benchmark/excess metrics

- [ ] **Step 2: Run strategy tests and verify failure**

Run:
```bash
pytest -q /Users/hsq/Desktop/codex/future/tests/test_modeling_and_strategy.py
```

- [ ] **Step 3: Implement minimal continuous factor strategy support**

Support:
- binary strategy mode (existing)
- regression factor mode
- mapping methods: `linear_clip`, `tanh`

- [ ] **Step 4: Re-run strategy tests and verify pass**

Run:
```bash
pytest -q /Users/hsq/Desktop/codex/future/tests/test_modeling_and_strategy.py
```

### Task 5: Run V3 English And V1 Regression Iterations

**Files:**
- Use: `/Users/hsq/Desktop/codex/future/scripts/fetch_hf_financial_multisource.py`
- Use: `/Users/hsq/Desktop/codex/future/scripts/build_english_global_clean.py`
- Use: `/Users/hsq/Desktop/codex/future/scripts/archive_ingest.py`
- Use: `/Users/hsq/Desktop/codex/future/scripts/score_articles.py`
- Use: `/Users/hsq/Desktop/codex/future/scripts/build_weekly_dataset.py`
- Use: `/Users/hsq/Desktop/codex/future/scripts/train_eval.py`
- Use: `/Users/hsq/Desktop/codex/future/scripts/run_strategy.py`

- [ ] **Step 1: Fetch remaining English shards**

Run the HF fetcher with all-file support and skip existing files where possible.

- [ ] **Step 2: Rebuild English clean table and run coverage audit**

Generate:
- refreshed English clean file
- coverage audit markdown/csv

- [ ] **Step 3: Materialize a V3 English-only scored/model input set**

Reuse prior scored rows where possible and score only new English rows.

- [ ] **Step 4: Train the V3 English regression model**

Export:
- predictions
- CV metrics
- SHAP outputs

- [ ] **Step 5: Run regression factor strategy scenarios**

Export:
- strategy table
- strategy metrics
- scenario comparison
- regime comparison

- [ ] **Step 6: Run V1 Chinese regression baseline**

Produce the same artifacts for fair comparison.

### Task 6: Update Comparison Artifacts And Docs

**Files:**
- Modify: `/Users/hsq/Desktop/codex/future/scripts/build_iteration_comparison.py`
- Modify: `/Users/hsq/Desktop/codex/future/docs/project/ine-sc-project-overview.md`
- Modify: `/Users/hsq/Desktop/codex/future/docs/project/ine-sc-presentation-script.md`

- [ ] **Step 1: Extend comparison script to include regression variants**
- [ ] **Step 2: Regenerate iteration comparison table**
- [ ] **Step 3: Update overview and presentation docs with V3 and V1-regression findings**

### Task 7: Final Verification

**Files:**
- Verify only

- [ ] **Step 1: Run full targeted test suite**

Run:
```bash
pytest -q /Users/hsq/Desktop/codex/future/tests/test_scoring_and_cv.py /Users/hsq/Desktop/codex/future/tests/test_modeling_and_strategy.py /Users/hsq/Desktop/codex/future/tests/test_english_energy_candidates.py /Users/hsq/Desktop/codex/future/tests/test_hf_dataset_fetch.py /Users/hsq/Desktop/codex/future/tests/test_rss_ingest.py
```

- [ ] **Step 2: Run syntax verification**

Run:
```bash
python3 -m py_compile /Users/hsq/Desktop/codex/future/scripts/build_weekly_dataset.py /Users/hsq/Desktop/codex/future/scripts/train_eval.py /Users/hsq/Desktop/codex/future/scripts/run_strategy.py /Users/hsq/Desktop/codex/future/scripts/fetch_hf_financial_multisource.py /Users/hsq/Desktop/codex/future/scripts/build_iteration_comparison.py /Users/hsq/Desktop/codex/future/src/models/train_eval.py /Users/hsq/Desktop/codex/future/src/models/strategy.py /Users/hsq/Desktop/codex/future/src/data/english_coverage_audit.py
```
