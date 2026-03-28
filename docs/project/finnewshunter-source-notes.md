# FinnewsHunter 来源说明

## 1. 这个仓库在我们项目里扮演什么角色

`FinnewsHunter` 对我们最重要的价值，不是直接把它整套系统搬过来，而是：

- 提供中文财经多站点抓取思路
- 提供一类比较典型的历史新闻导出结构
- 帮我们确定中文主语料优先站点

我们现在把它当作：

- **中文历史新闻主线的参考实现**
- **本地历史导出文件的兼容目标**

## 2. 从仓库信息里能确定什么

从它的 README 和示例描述可以确认几件事：

- 它抓的是中文财经新闻站点
- 它会把新闻先存到数据库
- 单条新闻的核心字段至少包括：
  - 时间
  - 网址
  - 标题
  - 正文

在旧版示例里，还能看到类似这种字段命名：

- `Title`
- `Article`
- `Date`
- `Url`
- `Category`

所以我们当前导入器已经专门兼容这组字段。

## 3. 为什么我们不强依赖它原仓库直接跑

原因主要有三个：

1. 这次作业重点是“新闻 -> 因子 -> 预测”，不是复刻它整套爬虫环境
2. 它原始工程依赖、站点结构和运行环境可能已经变化
3. 我们更需要的是“能稳定导入历史新闻结果”，而不是把抓取系统完全绑定在单一仓库上

所以现在的策略是：

- **借它的站点清单和字段形态**
- **兼容它的导出结果**
- **把导出后的文章统一接入我们自己的研究管线**

## 4. 我们实际怎么接

如果你手里拿到的是 `FinnewsHunter` 风格导出文件，可以直接放到：

`data/raw/finnewshunter_exports/`

支持的常见字段有两组：

### 常规字段

- `title`
- `body` / `content` / `maintext`
- `published_at` / `publish_time`
- `url`
- `source`
- `language`

### FinnewsHunter 旧版风格

- `Title`
- `Article`
- `Date`
- `Url`
- `Category`

然后运行：

```bash
python3 /Users/hsq/Desktop/codex/future/scripts/import_finnewshunter_export.py
python3 /Users/hsq/Desktop/codex/future/scripts/archive_ingest.py --max-articles 100
```

这样就会进入我们后面的统一文章表。

## 5. 如果未来真要从它的数据库导出

如果后面需要，我们可以再单独补一个“从 MongoDB 导出为 CSV/JSONL”的脚本或说明。

但当前阶段，最重要的是：

- 先拿到任何一个可用的历史导出文件
- 让它进入 `articles_archive_clean`
- 再接到 Qwen 打分和周特征构建
