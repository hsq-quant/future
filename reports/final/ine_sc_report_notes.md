# INE SC Workbook Notes

## Purpose
- This workbook is the report draft only. It is the main delivery artifact, not a PPT source file.
- The classification section is the intuitive trading story: predict up/down, then translate directly into a weekly trading action. It now includes multiple execution mappings, not only the naive always-in long/short switch.
- The regression section is the factor story: predict next-week return, then map the signal into a smoother position for practical quant usage.

## Sheets
- `Summary`: all model-task-execution combinations in one comparison table.
- `Classification`: up/down model results plus fairer execution variants such as long-only classification.
- `Regression`: all regression training results, including factor IC and directional accuracy.
- `Mappings`: the regression execution variants that test whether the factor or the trading layer is the bottleneck.

## Current Readout
- Best classification strategy in the current table: `V1 Chinese classification / long_only` with cumulative return `344.11%` and Sharpe `1.3923`.
- Best regression execution in the current table: `V3 English regression / long_only_factor` with cumulative return `96.65%` and Sharpe `0.8922`.
- The main narrative is to compare classification vs regression, then show that regression can be useful as a factor even when naive execution underperforms a strong always-long crude benchmark.

## Workbook Usage
- Use `Summary` for the one-page total comparison.
- Use `Classification` when explaining the simple, intuitive trading logic.
- Use `Regression` and `Mappings` when explaining IC, factor ranking ability, and why execution design matters.
