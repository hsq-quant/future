# 历史新闻源路线评估

## 结论先说

如果目标是做 `2020-01` 到 `2026-03` 的 **中文原油相关新闻历史样本**，最合适的路线不是继续盲扫 `Common Crawl` 单个 WARC，而是：

1. **中文主语料：站点适配器 / 历史页面爬取**
2. **国际补充或发现层：Common Crawl / GDELT**
3. **live RSS：只做联调，不进入主流程**

## 1. GitHub 上搜到的主要路线

### A. RSSHub

仓库：
- [DIYgod/RSSHub](https://github.com/DIYgod/RSSHub)

适合做什么：
- 把中文网站转成 RSS
- 快速做 live 数据接入

不适合什么：
- 不适合 `2020-2026` 的历史回填

结论：
- 适合作为联调和近期增量源
- 不适合作为本项目的历史主语料

### B. news-please

仓库：
- [fhamborg/news-please](https://github.com/fhamborg/news-please)

GitHub README 明确支持：
- old, archived articles
- Common Crawl archive extraction
- JSON / PostgreSQL / ElasticSearch 等多种存储

适合做什么：
- 作为历史新闻归档抓取的基础工具
- 提取结构化文章字段

当前实际问题：
- 它的 Common Crawl 路线会先扫 WARC，再抽正文，再过滤
- 对中文财经这种窄主题，命中率不够高时会很慢

结论：
- 可以保留，适合作为“国际/补充历史源”
- 不适合作为中文主语料的唯一方案

### C. Common Crawl / CC-NEWS

仓库：
- [commoncrawl/news-crawl](https://github.com/commoncrawl/news-crawl)
- [commoncrawl](https://github.com/commoncrawl)

GitHub 信息说明：
- CC-NEWS 是通过 RSS/Atom feeds 和 news sitemaps 发现新闻链接
- 最终产出 WARC

适合做什么：
- 做大规模历史新闻归档
- 适合英文和全球新闻范围

当前实际问题：
- 单个 WARC 文件很大
- 如果不先知道目标域名在哪个 WARC 包里，抽取成本高
- 我们本地扫描 `2020-03-09` 某个 WARC 前 20000 条 URI 时，看到 `Bloomberg/Reuters`，但没看到预期中的中国财经域名

结论：
- 可以作为补充来源
- 但不能继续盲扫，必须先做 WARC scout

### D.1 Common Crawl 索引优先工具

如果我们仍然想走 `WARC` 路线，更合适的不是“先下载大包再扫正文”，而是先从索引层筛：

仓库：
- [cocrawler/cdx_toolkit](https://github.com/cocrawler/cdx_toolkit)
- [commoncrawl/cc-index-table](https://github.com/commoncrawl/cc-index-table)

这些工具适合做什么：
- 先按 `domain / date / status` 在 Common Crawl 索引里筛 URL
- 先确认目标站点和时间窗口里到底有没有数据
- 再把命中的记录导出为较小的 WARC 子集或 URL 列表

为什么这条路更适合我们：
- 我们之前已经实测过，盲扫单个 `CC-NEWS` WARC 很可能长时间没有目标中文财经域名命中
- 索引优先可以先做“有没有”和“在哪个 WARC”的判断
- 这样比直接抽整包正文更节省时间，也更适合写进作业汇报

结论：
- 如果继续保留 `Common Crawl`，主方法应该改成“索引优先 + 定向导出”，而不是“整包下载 + 后过滤”
- 这条线适合做补充或实验性历史发现，不适合当前项目的中文主语料主线

### E. GDELT

仓库：
- [linwoodc3/gdeltPyR](https://github.com/linwoodc3/gdeltPyR)
- [gdelt/gdelt.github.io](https://github.com/gdelt/gdelt.github.io)

适合做什么：
- 历史新闻发现
- 事件检索
- 做“这段时间哪些主题和域名最活跃”的发现层

不适合什么：
- 不适合作为完整正文主存储

结论：
- 更像“索引层 / 发现层”
- 可以帮助我们找到高命中的日期窗口和站点，再去定向抓取

### F. 中文站点整套流程范例

仓库：
- [DemonDamon/FinnewsHunter](https://github.com/DemonDamon/FinnewsHunter)

这个仓库最有参考价值的点不是模型，而是它明确把中文财经源拆成了多站点抓取流程，覆盖：
- 新浪财经
- 每经网
- 金融界
- 中国证券网
- 证券时报

结论：
- 这是“中文源整套流程”的最好参考
- 非常适合我们借它的站点清单和适配器思路，做自己的轻量历史抓取版

## 2. 为什么当前没有历史样本落盘

当前不是“代码完全没写”，而是卡在：

1. `news-please` 依赖问题已经逐个修掉了
   - `nltk punkt / punkt_tab`
   - `jieba`
2. 真正剩下的主问题是：
   - 选中的 WARC 包里，不一定有我们想要的中文财经域名
   - `news-please` 会先抽正文，再做主题过滤
   - 所以当中文目标文章占比很低时，效率会很差

## 3. 对我们最合适的路线

### 推荐主线

中文主语料改成：

- 新浪财经
- 中国证券网
- 证券时报
- 每经网
- 财新
- 第一财经

优先理由：
- 中国市场相关性更强
- 历史页面结构通常比 live RSS 更稳定
- 比盲扫 CC-NEWS 更容易解释和汇报

### 推荐辅助线

把 `Common Crawl / news-please` 留作：

- 补国际新闻
- 做英文补充
- 或者在明确知道目标域名和时间窗口后定向抽样

### 当前不推荐

- 继续把 `live RSS` 当历史主源
- 继续盲扫 `Common Crawl` 整包正文

## 4. 下一步实现建议

先做这两步：

1. 加一个 `WARC scout` 脚本
   - 只扫 `WARC-Target-URI`
   - 统计每个 WARC 中有哪些目标域名
   - 先把高命中的包挑出来

2. 并行搭一个 `china_site_adapters` 骨架
   - 每个中文站点一个适配器
   - 统一输出到 `raw_articles`
   - 优先跑单站历史样本

这样主线和备线都在，不会继续卡在单一路径上。

## 5. 当前推荐决策

结合 GitHub 路线和我们已经踩到的坑，这个项目最合适的方案是：

1. 中文主语料：
   参考 `FinnewsHunter` 的多站点抓取思路，优先做中文财经站点适配器
2. 历史归档补充：
   若需要归档/WARC，只走 `cdx_toolkit / cc-index-table` 这类索引优先路线
3. `news-please` 的角色：
   继续保留，但更适合作为正文抽取工具或归档补充工具，而不是中文主语料入口

一句话总结：

- **中文主线：站点适配器**
- **WARC 备线：索引优先**
- **不再继续盲扫整包 Common Crawl**
