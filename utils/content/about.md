## 关于这个论文库

这里是 AI Infra 方向系统论文的中文精读与原文对照。每篇论文都保留原始 PDF、英文原文、
逐节中文译文，以及从原文裁切的图表；公式、表格和章节编号尽量与原文一致。

本站是一个纯静态站点：内容是可移植的 Markdown，页面、目录、搜索索引和资源清单都由
`utils/` 下的构建器生成。没有数据库，也没有运行时后端。

## 支持的格式

### 数学公式

行内公式：$U(b)=r/(\delta+\varepsilon)$。块级公式：

$$
U(b)=\frac{b.r}{b.\delta+\varepsilon},
\qquad \varepsilon=10^{-3}
$$

### Mermaid 图

```mermaid title="BidKV 的一次压力事件"
flowchart LR
  P["KV 使用率<br/>超过阈值"] --> D["压力检测"]
  D --> B["为每个请求<br/>生成投标"]
  B --> R["按效用 U = r/(δ+ε)<br/>排序"]
  R --> V["选择最高效用<br/>牺牲者"]
  V --> E["框架原生抢占"]
  E --> Q["请求重新排队"]
  Q -.-> P
```

### 代码块

带文件名和语法高亮：

```python title="utils/paperlib/utility.py"
def utility(freed_tokens: int, disruption: float, eps: float = 1e-3) -> float:
    """Rank a request by reclaimable capacity per unit disruption."""
    return freed_tokens / (disruption + eps)
```

### 表格

表 1：格式能力一览。
{: .table-caption #tab-format }

| 能力 | 写法 | 页面效果 |
| --- | --- | --- |
| 数学 | `$...$` / `$$...$$` | KaTeX 渲染，可横向滚动 |
| 图表 | ` ```mermaid ` | 本地 Mermaid 渲染，可放大 |
| 代码 | 三反引号 + 语言 | Pygments 高亮、复制按钮 |
| 图片 | `![图注](assets/...)` | 语义化 figure、灯箱预览 |
| 视频 | ` ```video ` | 本地 MP4 或 YouTube/Bilibili |
| 目录 | 自动从 h2/h3 生成 | 滚动高亮、移动端抽屉 |

### 图片

![图 1：BidKV 的四层调度架构。图片会自动包成 figure，并支持点击放大。](../bidkv/assets/figures/fig2.png)

### 视频

本地视频会自动响应式铺满容器，支持封面图和说明文字：

```video
src: assets/demo.mp4
poster: assets/demo-poster.jpg
caption: 示例视频：KV 压力下的一次抢占与重算循环。
credit: 由 ffmpeg 生成，仅用于演示视频块。
```

### 提示块

!!! note "写作约定"
    术语第一次出现时保留英文原文，例如 prefill（预填充）、victim（被抢占请求）。
    公式中的符号与原文保持一致，方便逐条核对。

## 文件结构

```text
papers/
  bidkv/
    meta.yaml          # 标题、作者、摘要、标签、封面
    zh.md              # 中文译文
    original.md        # 英文原文
    assets/
      source.pdf       # 原始 PDF
      figures/         # 从 PDF 裁切的图
  index.html           # 论文库首页（生成）
  bidkv/index.html     # 译文页（生成）
  bidkv/original/      # 原文页（生成）
```

## 新增一篇论文

```bash
python3 utils/new_paper.py \
  --slug my-paper \
  --title "Paper Title" \
  --title-zh "论文中文标题" \
  --url "https://example.com/paper.pdf" \
  --year 2026

# 编辑 papers/my-paper/zh.md、original.md 和 meta.yaml
./utils/build.sh
```

构建器会校验中英文标题层级、图片路径、重复 id 和缺失资源。若两版章节结构不一致，
语言切换会报错，而不是悄悄跳到错误位置。

## 设计取舍

- **静态优先**：所有页面可直接用浏览器打开，不需要服务器；KaTeX、Mermaid 全部本地化。
- **章节 id 对齐**：中英文使用同一套 `sec-*` id，切换语言时保留当前阅读位置。
- **内容与样式分离**：论文只写 Markdown；版式、目录、灯箱、搜索都在 `utils/` 中维护。
- **可校验**：构建失败比发布坏链接好；`--strict` 会把警告升级为错误。
