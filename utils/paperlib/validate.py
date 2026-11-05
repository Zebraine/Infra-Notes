"""Source and output validation for the paper library."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .discover import Paper
from .config import CONTENT_DIR, PAPERS_DIR

MD_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_SRC_RE = re.compile(r'(?:src|poster)=["\'](?P<src>[^"\']+)["\']')
YAML_SRC_RE = re.compile(r"^\s*(?:src|poster):\s*[\"']?(?P<src>[^\"'\n]+)[\"']?\s*$", re.M)
HEADING_RE = re.compile(r"^(#{2,4})\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass(slots=True)
class Issue:
    level: str
    message: str
    paper: str = ""

    def format(self) -> str:
        prefix = f"[{self.paper}] " if self.paper else ""
        return f"{prefix}{self.message}"


def _local_path(url: str) -> str | None:
    url = url.strip()
    if not url or url.startswith(("#", "http://", "https://", "//", "data:", "mailto:")):
        return None
    return url.split("#", 1)[0].split("?", 1)[0]


def _headings(text: str) -> list[tuple[int, str]]:
    items: list[tuple[int, str]] = []
    in_fence = False
    marker = ""
    for line in text.splitlines():
        fence = FENCE_RE.match(line)
        if fence:
            char = fence.group(1)[0]
            if not in_fence:
                in_fence, marker = True, char
            elif char == marker:
                in_fence, marker = False, ""
            continue
        if in_fence:
            continue
        match = HEADING_RE.match(line)
        if match:
            items.append((len(match.group(1)), match.group(2).strip()))
    return items


def _strip_fences(text: str) -> str:
    """Remove fenced code blocks so examples are not treated as real assets."""
    out: list[str] = []
    in_fence = False
    marker = ""
    for line in text.splitlines():
        fence = FENCE_RE.match(line)
        if fence:
            char = fence.group(1)[0]
            if not in_fence:
                in_fence, marker = True, char
            elif char == marker:
                in_fence, marker = False, ""
            continue
        if not in_fence:
            out.append(line)
    return "\n".join(out)


def _strip_inline_code(text: str) -> str:
    return re.sub(r"`[^`\n]*`", "", text)


def _iter_video_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        match = re.match(r"^\s*(`{3,}|~{3,})\s*([A-Za-z0-9_+-]+)?", lines[index])
        if not match:
            index += 1
            continue
        marker = match.group(1)
        language = (match.group(2) or "").lower()
        close = re.compile(r"^\s*" + re.escape(marker[0]) + r"{" + str(len(marker)) + r",}\s*$")
        body: list[str] = []
        index += 1
        while index < len(lines) and not close.match(lines[index]):
            body.append(lines[index])
            index += 1
        if language == "video":
            blocks.append("\n".join(body))
        index += 1
    return blocks


def _check_assets(
    text: str,
    base_dir: Path,
    issues: list[Issue],
    *,
    label: str,
    source_name: str,
    prefix: str = "",
) -> None:
    visible = _strip_inline_code(_strip_fences(text))
    for match in MD_IMAGE_RE.finditer(visible):
        src = _local_path(match.group("src"))
        if src and not (base_dir / src).exists():
            issues.append(Issue("error", f"{source_name}图片不存在：{src}", label))
        if not match.group("alt").strip():
            issues.append(Issue("warn", f"{source_name}图片缺少 alt 文本：{src}", label))
    for match in HTML_SRC_RE.finditer(visible):
        src = _local_path(match.group("src"))
        if src and not src.startswith(("http://", "https://", "//")) and not (base_dir / src).exists():
            issues.append(Issue("error", f"{source_name}资源不存在：{src}", label))
    for block in _iter_video_blocks(text):
        for match in YAML_SRC_RE.finditer(block):
            src = _local_path(match.group("src"))
            if src and not src.startswith(("http://", "https://", "//")) and not (base_dir / src).exists():
                issues.append(Issue("error", f"{source_name}视频/音频资源不存在：{src}", label))


def validate_sources(papers: list[Paper]) -> list[Issue]:
    issues: list[Issue] = []
    for paper in papers:
        label = paper.slug
        if not paper.abstract_zh or len(paper.abstract_zh) < 30:
            issues.append(Issue("warn", "abstract_zh 太短，论文卡片会缺少摘要。", label))
        if not paper.authors or paper.authors == ["Anonymous"]:
            issues.append(Issue("info", "作者为匿名或缺失。", label))
        for rel, kind in ((paper.pdf, "PDF"), (paper.cover, "封面图")):
            if not rel:
                continue
            path = paper.dir / rel
            if not path.exists():
                issues.append(Issue("error", f"{kind}不存在：{rel}", label))
        for field, value in (
            ("source_url", paper.source_url),
            ("conference_url", paper.conference_url),
            ("code_url", paper.code_url),
        ):
            if value and not value.startswith(("http://", "https://")):
                issues.append(Issue("warn", f"{field} 不是 http(s) 链接：{value}", label))

        zh_text = paper.zh_path.read_text(encoding="utf-8")
        en_text = paper.original_path.read_text(encoding="utf-8")
        zh_heads = _headings(zh_text)
        en_heads = _headings(en_text)
        if [level for level, _ in zh_heads] != [level for level, _ in en_heads]:
            issues.append(
                Issue(
                    "error",
                    f"中英文标题层级不一致（译文 {len(zh_heads)} 个，原文 {len(en_heads)} 个），"
                    "切换语言时无法保持章节位置。",
                    label,
                )
            )

        _check_assets(zh_text, paper.dir, issues, label=label, source_name="译文")
        _check_assets(en_text, paper.dir, issues, label=label, source_name="原文")

        if not (paper.dir / "assets" / "source.pdf").exists() and paper.pdf.endswith("source.pdf"):
            issues.append(Issue("warn", "未找到 assets/source.pdf，PDF 链接可能失效。", label))

    about = CONTENT_DIR / "about.md"
    if about.exists():
        about_dir = PAPERS_DIR / "about"
        _check_assets(
            about.read_text(encoding="utf-8"),
            about_dir,
            issues,
            label="about",
            source_name="格式手册",
        )
    return issues


def validate_rendered(renderer) -> list[Issue]:  # type: ignore[no-untyped-def]
    issues: list[Issue] = []
    for slug, rendered in renderer.rendered.items():
        for lang, item in (("译文", rendered.zh), ("原文", rendered.en)):
            if "build-error" in item.html:
                issues.append(Issue("error", f"{lang}渲染出现 build-error。", slug))
            if not item.headings:
                issues.append(Issue("warn", f"{lang}正文没有 h2/h3 标题，目录为空。", slug))
            ids = re.findall(r'\bid="([^"]+)"', item.html)
            duplicates = sorted({value for value in ids if ids.count(value) > 1})
            if duplicates:
                issues.append(
                    Issue("error", f"{lang}重复 id：{', '.join(duplicates)}", slug)
                )
        zh_ids = [h["id"] for h in rendered.zh.headings]
        en_ids = [h["id"] for h in rendered.en.headings]
        if zh_ids != en_ids:
            missing_zh = sorted(set(en_ids) - set(zh_ids))
            issues.append(
                Issue(
                    "warn",
                    "中英文章节 id 不完全对应"
                    + (f"（译文缺少：{', '.join(missing_zh)}）" if missing_zh else ""),
                    slug,
                )
            )
        if '<pre class="mermaid"' in rendered.zh.html:
            issues.append(Issue("warn", "Mermaid 未转换为容器。", slug))
    return issues


def print_issues(issues: list[Issue]) -> tuple[int, int]:
    errors = [issue for issue in issues if issue.level == "error"]
    warnings = [issue for issue in issues if issue.level == "warn"]
    infos = [issue for issue in issues if issue.level == "info"]
    for issue in errors + warnings + infos:
        marker = {"error": "✗", "warn": "!", "info": "·"}[issue.level]
        print(f"  {marker} {issue.format()}")
    return len(errors), len(warnings)
