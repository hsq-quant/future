# INE SC 底稿说明

- 本底稿只使用 aligned 共同窗口：2021-01-22 至 2025-09-05。
- `V1 classification / long_only` 是最直观、最适合课堂展示的分类结果。
- `V3 regression / long_only_factor` 是最适合解释量化因子扩展价值的结果。
- `V3 classification` 模型指标更高，但最朴素的 always-in 多空执行很差，说明执行映射非常重要。
- 分阶段结果请优先看 Regime 柱状图和热力图。

## Workbook Sheet Guide

- 01_Method: 论文与本项目的方法流程对比。
- 02_Features: 11 个周频特征列表。
- 03_IC_Dist: 四个模型族的 IC 折间分布。
- 04_Model_Table: V1/V3、分类/回归核心指标表。
- 05_IC_Acc: IC 与准确率横向对比。
- 06_Market: INE SC 连续价格与累计收益曲线。
- 07_CumRet: 关键策略与 benchmark 的累计收益曲线。
- 08_PerfCost: 带成本/滑点的绩效对比表与图。
- 09_Position: 各策略 long/short/flat 占比。
- 10_RegimeBar: Regime 分阶段柱状对比。
- 11_RegimeHeat: Regime 绩效热力图。
- 12_Notes: 汇报口径与说明。
