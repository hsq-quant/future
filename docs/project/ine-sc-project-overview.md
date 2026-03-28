# INE 原油期货新闻情绪项目说明

## 1. 项目目标

这个项目的目标不是做高频交易，也不是追求极致收益，而是：

1. 尽量复刻论文的方法框架。
2. 把论文的标的迁移到中国 `INE SC` 原油期货。
3. 评估新闻情绪能否形成可用的周频预测信号。
4. 再看这个信号是更适合“直白的分类交易”，还是更适合“回归因子式扩展”。

一句话概括：

- 我们把新闻文本变成结构化情绪因子，
- 再用这些因子预测下一周 `SC` 的变化，
- 最后把预测映射成不同风格的交易执行方式，做公平比较。

## 2. 整体流程

整个项目可以拆成 6 步：

1. 准备 `INE SC` 各月份合约日线数据。
2. 构造连续主力价格序列和周度标签。
3. 收集历史新闻并清洗。
4. 用 Qwen 给每篇新闻打 5 个情绪分数。
5. 按周聚合成 11 个论文因子。
6. 用 `LightGBM + 5-fold expanding window` 做分类与回归建模，并接交易层。

## 3. 最终交付形式

这个项目最后的正式交付，不以 PPT 为主，而以两份底稿为主：

- 一个 Excel 底稿 workbook
- 一个 Markdown 版底稿说明文档

最终最重要的产物是：

- [model_family_comparison_aligned.csv](/Users/hsq/Desktop/codex/future/reports/final/model_family_comparison_aligned.csv)
- [regime_comparison_aligned.csv](/Users/hsq/Desktop/codex/future/reports/final/regime_comparison_aligned.csv)
- [ine_sc_report_draft_aligned.xlsx](/Users/hsq/Desktop/codex/future/reports/final/ine_sc_report_draft_aligned.xlsx)
- [ine_sc_report_notes_aligned.md](/Users/hsq/Desktop/codex/future/reports/final/ine_sc_report_notes_aligned.md)

现在收口时，统一用这个脚本重刷最终交付：

```bash
python3 /Users/hsq/Desktop/codex/future/scripts/build_final_reporting_bundle.py
```

这个脚本会同时刷新：

- aligned 总对比表
- aligned regime 分阶段表
- 底稿 workbook
- 底稿说明文档
- 图表目录 `reports/final/charts/`

最终 workbook 已经按照汇报顺序整理成 13 个 sheet：

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

## 4. 市场数据怎么处理

### 4.1 为什么不能直接用单个合约

`SC` 是期货，不同月份合约会到期。

如果直接拿单个合约一路往后看，会断；
如果简单把不同月份直接拼接，又会在换月时出现假跳空。

所以必须先做连续价格序列。

### 4.2 当前连续合约规则

- 默认跟踪近月合约
- 在最后交易日前 `5` 个交易日滚到下一合约
- 换月时做价差调整，消除价格台阶

### 4.3 周度标签

我们是按周预测，不是按天预测。

- `weekly_return_t = log(P_t / P_{t-1})`
- `y_t = 1[weekly_return_{t+1} > 0]`

意思是：

- 第 `t` 周结束前的新闻
- 用来预测第 `t+1` 周的收益方向

## 5. 新闻数据怎么处理

### 5.1 当前两条主新闻线

项目最后收敛成两条主新闻线：

- `V1`
  - 中文主链
- `V3`
  - 英文全球能源新闻主链

### 5.2 新闻如何存储

新闻不会直接拿去喂模型，而是分层存储：

- `raw_articles`
- `clean_articles`
- `article_scores`
- `weekly_features`

这样做的好处是：

1. 每一步都能检查。
2. 出错时可以单独重跑。
3. 不必每次都重新调用大模型。

### 5.3 为什么最后采用 aligned 口径

项目中间不同版本的新闻覆盖区间并不完全相同，尤其英文版较早只到 `2025-09-05`。

所以最后汇报统一采用 **aligned 口径**：

- 起点：`2021-01-22`
- 终点：`2025-09-05`

这样做的目的，是让 `V1 / V3 / 分类 / 回归` 都在同一交易期里公平比较。

## 6. 大模型怎么打分

每篇新闻都交给 `Qwen2.5` 打 5 个维度：

- `relevance`
- `polarity`
- `intensity`
- `uncertainty`
- `forwardness`

其中论文里最关键的一条规则是：

- 如果 `relevance < 0.1`
- 其他 4 个维度必须为 `null`

这一步的意义是：把原始新闻文本变成可以做统计和机器学习的结构化信号。

## 7. 11 个周频特征怎么构建

按论文定义，我们对每一周构造 11 个特征：

1. `article_count`
2. `relevance_mean`
3. `polarity_mean`
4. `intensity_mean`
5. `uncertainty_mean`
6. `forwardness_mean`
7. `polarity_std`
8. `uncertainty_std`
9. `polarity_momentum`
10. `uncertainty_momentum`
11. `forwardness_momentum`

其中：

- `relevance_mean` 用普通平均
- 其他四个均值用 `relevance` 加权平均

## 8. 模型怎么训练

我们最终跑了两条模型线：

### 8.1 分类

- 输出：下一周上涨概率
- 主要指标：`AUC`、`Accuracy`、`IC`

### 8.2 回归

- 输出：下一周真实收益率预测值
- 主要指标：`IC`、方向准确率

### 8.3 时间序列验证

模型使用 `LightGBM`，并采用：

- `5-fold`
- `expanding window`

也就是：

- 每一折只能用过去训练
- 再去预测更后面的数据

## 9. 当前最终结果

这里我们只讲 aligned 结果。

### 9.1 分类结果

`V1 Chinese classification`

- `AUC = 0.5366`
- `Accuracy = 0.5435`
- `IC = 0.0614`

在共同窗口下：

- `always_in`：累计收益 `155.07%`
- `long_only`：累计收益 `210.96%`

`V3 English classification`

- `AUC = 0.5810`
- `Accuracy = 0.5515`
- `IC = 0.0950`

在共同窗口下：

- `always_in`：累计收益 `-28.69%`
- `long_only`：累计收益 `60.04%`

这说明：

1. `V3` 分类模型指标更好。
2. 但“预测跌就做空”的最朴素交易层很差。
3. 分类模型好，不等于分类交易一定好。

### 9.2 回归结果

`V1 Chinese regression`

- `IC = -0.0673`
- 因子本身较弱
- `long_only_factor`：累计收益 `64.98%`

`V3 English regression`

- `IC = 0.1186`
- 平均方向准确率 `53.08%`
- `long_only_factor`：累计收益 `96.65%`

所以从“量化因子扩展”的角度，当前最值得保留的是：

- `V3 regression / long_only_factor`

## 10. 分阶段压力测试

我们把共同窗口拆成 5 个阶段：

- `Recovery Bull`
- `War Spike`
- `Post-Spike Bear`
- `OPEC-Supported Range`
- `Oversupply Bear`

结果文件在：

- [regime_comparison_aligned.csv](/Users/hsq/Desktop/codex/future/reports/final/regime_comparison_aligned.csv)
- [regime_comparison_aligned.md](/Users/hsq/Desktop/codex/future/reports/final/regime_comparison_aligned.md)

最值得讲的结论：

1. `V1 classification / long_only`
   - 是分类里最稳的一版
2. `V3 classification`
   - 指标高，但交易执行弱
3. `V3 regression / long_only_factor`
   - 是当前最平衡、最像“可扩展因子”的版本
4. `War Spike`
   - 是所有策略都最难做的压力阶段

## 11. 最后结论

最后这版项目可以收成 4 句话：

1. 我们成功把论文的方法迁移到了 `INE SC`。
2. 中文新闻主链能提供弱但真实的预测力。
3. 英文 `V3` 在模型层更强，但分类交易层不够好。
4. 如果继续往量化方向扩展，最值得保留的是：
   - 分类主线做直观解释
   - 回归因子主线做实盘扩展
