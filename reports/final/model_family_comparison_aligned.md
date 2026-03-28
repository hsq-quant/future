# Model Family Comparison

| Label | Task | Execution | Weeks | AUC | Accuracy | IC | Directional Acc | Cum Return | Sharpe | Benchmark Cum | Cum Diff |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 Chinese classification | classification | always_in | 237 | 0.5366 | 0.5435 | 0.0614 | 0.5435 | 155.07% | 0.6345 | 168.72% | -13.66% |
| V1 Chinese classification | classification | long_only | 237 | 0.5366 | 0.5435 | 0.0614 | 0.5435 | 210.96% | 1.2149 | 168.72% | 42.24% |
| V1 Chinese classification | classification | threshold_short_only | 237 | 0.5366 | 0.5435 | 0.0614 | 0.5435 | 210.96% | 1.2149 | 168.72% | 42.24% |
| V3 English classification | classification | always_in | 237 | 0.5810 | 0.5515 | 0.0950 | 0.5515 | -28.69% | -0.1978 | 168.72% | -197.41% |
| V3 English classification | classification | long_only | 237 | 0.5810 | 0.5515 | 0.0950 | 0.5515 | 60.04% | 0.4201 | 168.72% | -108.68% |
| V3 English classification | classification | threshold_short_only | 237 | 0.5810 | 0.5515 | 0.0950 | 0.5515 | -13.61% | -0.1109 | 168.72% | -182.33% |
| V1 Chinese regression | regression | asymmetric_long_short | 237 | 0.0000 | 0.0000 | -0.0673 | 0.4673 | 34.70% | 0.4176 | 168.72% | -134.02% |
| V1 Chinese regression | regression | baseline_tanh | 237 | 0.0000 | 0.0000 | -0.0673 | 0.4673 | 2.72% | 0.0267 | 168.72% | -166.00% |
| V1 Chinese regression | regression | long_only_factor | 237 | 0.0000 | 0.0000 | -0.0673 | 0.4673 | 64.98% | 0.6494 | 168.72% | -103.74% |
| V1 Chinese regression | regression | threshold_short_only | 237 | 0.0000 | 0.0000 | -0.0673 | 0.4673 | 41.01% | 0.4975 | 168.72% | -127.72% |
| V3 English regression | regression | asymmetric_long_short | 237 | 0.0000 | 0.0000 | 0.1186 | 0.5308 | 71.91% | 0.7445 | 168.72% | -96.82% |
| V3 English regression | regression | baseline_tanh | 237 | 0.0000 | 0.0000 | 0.1186 | 0.5308 | 49.15% | 0.3798 | 168.72% | -119.57% |
| V3 English regression | regression | long_only_factor | 237 | 0.0000 | 0.0000 | 0.1186 | 0.5308 | 96.65% | 0.9010 | 168.72% | -72.07% |
| V3 English regression | regression | threshold_short_only | 237 | 0.0000 | 0.0000 | 0.1186 | 0.5308 | 58.42% | 0.6381 | 168.72% | -110.30% |
