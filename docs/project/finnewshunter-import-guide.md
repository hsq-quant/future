# FinnewsHunter 导入说明

## 1. 我们为什么先走这条线

这次历史新闻主线优先走 `FinnewsHunter` 风格的中文财经源，不再依赖 live RSS。

原因很简单：

- 它本身就是中文财经多站点抓取思路
- 站点更贴近中国市场
- 导出的文章结构比较清楚，适合直接接到我们的 `raw_articles` 流程

## 2. 当前导入器支持什么格式

导入器现在兼容两类常见字段风格：

### A. 常规风格

- `title`
- `body` / `content` / `maintext`
- `published_at` / `publish_time`
- `url`
- `source`
- `language`

### B. FinnewsHunter 旧版风格

- `Title`
- `Article`
- `Date`
- `Url`
- `Category`

也就是说，只要本地导出文件里至少有“标题、正文、时间、网址、来源”这些信息，脚本一般都能识别。

## 3. 怎么导入

把本地导出文件放到一个目录里，然后运行：

```bash
python3 /Users/hsq/Desktop/codex/future/scripts/import_finnewshunter_export.py \
  --input-glob "data/raw/finnewshunter_exports/*" \
  --output "data/raw/archive/finnewshunter/normalized.parquet"
```

脚本会做两件事：

1. 统一不同列名
2. 输出成项目后续 `archive_ingest` 能直接读取的表

## 4. 样例文件

项目里放了一份最小样例：

- [finnewshunter_sample.csv](/Users/hsq/Desktop/codex/future/data/examples/finnewshunter_sample.csv)

如果要先做联调，可以直接这样跑：

```bash
python3 /Users/hsq/Desktop/codex/future/scripts/import_finnewshunter_export.py \
  --input-glob "data/examples/finnewshunter_sample.csv" \
  --output "data/tmp/finnewshunter_smoke.parquet"
```

## 5. 后续怎么接主流程

一旦 `normalized.parquet` 生成，就可以继续走：

```bash
python3 /Users/hsq/Desktop/codex/future/scripts/archive_ingest.py --max-articles 100
```

后面这些阶段就都不需要再关心原始站点格式了，只看统一后的文章表。

## 6. 接到 Qwen 打分

现在后续的打分脚本也已经能优先读取归档清洗结果。

如果 `articles_archive_clean.parquet` 已经生成，可以直接运行：

```bash
python3 /Users/hsq/Desktop/codex/future/scripts/score_articles.py --max-articles 5
```

如果你想显式指定输入文件，也可以：

```bash
python3 /Users/hsq/Desktop/codex/future/scripts/score_articles.py \
  --input "data/intermediate/articles_archive_clean.parquet" \
  --max-articles 5
```

这样做的目的，是先小批量验证：

- Qwen API 能不能通
- JSON 格式是否稳定
- 五维分数是否能正常写回 `articles_scored.parquet`
