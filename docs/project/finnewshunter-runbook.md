# FinnewsHunter 主线运行手册

## 1. 主线目标

我们现在坚持的主线不是：

- 直接把 `FinnewsHunter` 所有站点和所有模块原样搬进来

而是：

- 用 `FinnewsHunter` 已经验证过的中文历史新闻抓取路线
- 优先拿到一批真实历史新闻输出
- 再接入我们已经写好的 `Qwen -> 周因子 -> LightGBM` 管线

## 2. 当前推荐站点优先级

按仓库说明和我们这次项目目标，建议优先顺序是：

1. `CNStock`
2. `JRJ`
3. `NBD`
4. `Sina Finance`
5. `Yicai`
6. `Caixin`

暂不作为主抓取源：

- `STCN`

## 3. 仓库里最关键的信息

从 `FinnewsHunter` README 可以确认：

- 历史新闻会先落到 `MongoDB`
- 单条新闻核心字段包含：
  - `Date`
  - `Url`
  - `Title`
  - `Article`
  - `Category`

这正是我们当前导入器已经兼容的字段。

## 4. 推荐操作顺序

### 第一步：在 FinnewsHunter 侧抓历史新闻

优先尝试这些 README 里明确给出的历史抓取入口：

- `CnStockSpyder.get_historical_news(...)`
- `JrjSpyder.get_historical_news(...)`
- `NbdSpyder.get_historical_news(...)`

如果能跑通，就让数据先进入它自己的 `MongoDB`。

### 第二步：从 MongoDB 导出结果

目标不是保留它的整套数据库依赖，而是把结果导成：

- `CSV`
- `JSONL`
- 或 `Parquet`

至少保留这些字段：

- `Date`
- `Url`
- `Title`
- `Article`
- `Category`

如果你已经从 MongoDB 导出了 `JSON` 或 `JSONL`，现在可以直接运行：

```bash
python3 /Users/hsq/Desktop/codex/future/scripts/export_finnewshunter_mongo.py \
  --source-name CNStock \
  --input-json "data/examples/finnewshunter_mongo_sample.json" \
  --output "data/raw/finnewshunter_exports/cnstock_export.parquet"
```

这一步会把 Mongo 风格文档转成我们项目后面能直接读取的表。

## 5. 导入到我们项目

导出文件放到：

`data/raw/finnewshunter_exports/`

然后运行：

```bash
python3 /Users/hsq/Desktop/codex/future/scripts/import_finnewshunter_export.py
python3 /Users/hsq/Desktop/codex/future/scripts/archive_ingest.py --max-articles 100
```

## 6. 进入 Qwen 打分

归档清洗完成后，直接运行：

```bash
python3 /Users/hsq/Desktop/codex/future/scripts/score_articles.py --max-articles 20
```

如果想先小样本验证，也可以：

```bash
python3 /Users/hsq/Desktop/codex/future/scripts/score_articles.py --max-articles 5
```

## 7. 进入周因子与训练

打分后继续：

```bash
python3 /Users/hsq/Desktop/codex/future/scripts/build_weekly_dataset.py
python3 /Users/hsq/Desktop/codex/future/scripts/train_eval.py
python3 /Users/hsq/Desktop/codex/future/scripts/run_strategy.py
```

## 8. 什么时候才切备线

只有当下面任一情况发生时，才切 `Common Crawl` 备线：

1. `CNStock / JRJ / NBD / Sina Finance` 这 4 个核心源里，拿不到足够历史文章
2. 抓取或导出成本明显高于我们作业可承受范围
3. 导出后的时间覆盖无法覆盖 `2020-01` 到 `2026-03`
