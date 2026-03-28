# Model Family Comparison

| Label | Task | Execution | Weeks | AUC | Accuracy | IC | Directional Acc | Cum Return | Sharpe | Benchmark Cum | Cum Diff |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 Chinese classification | classification | always_in | 263 | 0.5366 | 0.5435 | 0.0614 | 0.5435 | 284.98% | 0.8491 | 260.92% | 24.06% |
| V1 Chinese classification | classification | long_only | 263 | 0.5366 | 0.5435 | 0.0614 | 0.5435 | 344.11% | 1.3923 | 260.92% | 83.19% |
| V1 Chinese classification | classification | threshold_short_only | 263 | 0.5366 | 0.5435 | 0.0614 | 0.5435 | 344.11% | 1.3923 | 260.92% | 83.19% |
| V3 English classification | classification | always_in | 241 | 0.5810 | 0.5515 | 0.0950 | 0.5515 | -6.20% | -0.0371 | 185.10% | -191.30% |
| V3 English classification | classification | long_only | 241 | 0.5810 | 0.5515 | 0.0950 | 0.5515 | 89.43% | 0.5475 | 185.10% | -95.66% |
| V3 English classification | classification | threshold_short_only | 241 | 0.5810 | 0.5515 | 0.0950 | 0.5515 | 2.26% | 0.0164 | 185.10% | -182.84% |
| V1 Chinese regression | regression | asymmetric_long_short | 263 | 0.0000 | 0.0000 | -0.0673 | 0.4673 | 45.96% | 0.4543 | 260.92% | -214.96% |
| V1 Chinese regression | regression | baseline_tanh | 263 | 0.0000 | 0.0000 | -0.0673 | 0.4673 | 5.63% | 0.0486 | 260.92% | -255.29% |
| V1 Chinese regression | regression | long_only_factor | 263 | 0.0000 | 0.0000 | -0.0673 | 0.4673 | 88.42% | 0.6894 | 260.92% | -172.50% |
| V1 Chinese regression | regression | threshold_short_only | 263 | 0.0000 | 0.0000 | -0.0673 | 0.4673 | 57.07% | 0.5594 | 260.92% | -203.85% |
| V3 English regression | regression | asymmetric_long_short | 241 | 0.0000 | 0.0000 | 0.1186 | 0.5308 | 75.15% | 0.7642 | 185.10% | -109.95% |
| V3 English regression | regression | baseline_tanh | 241 | 0.0000 | 0.0000 | 0.1186 | 0.5308 | 57.25% | 0.4276 | 185.10% | -127.84% |
| V3 English regression | regression | long_only_factor | 241 | 0.0000 | 0.0000 | 0.1186 | 0.5308 | 96.65% | 0.8922 | 185.10% | -88.45% |
| V3 English regression | regression | threshold_short_only | 241 | 0.0000 | 0.0000 | 0.1186 | 0.5308 | 59.90% | 0.6455 | 185.10% | -125.20% |
