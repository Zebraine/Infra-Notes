"""Markdown pipeline: math, mermaid, video, algorithms, figures and tables."""

from __future__ import annotations

import html
import json
import re
import unicodedata
import xml.etree.ElementTree as etree
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import yaml
from markdown import Markdown
from markdown.extensions import Extension
from markdown.postprocessors import Postprocessor
from markdown.preprocessors import Preprocessor
from markdown.treeprocessors import Treeprocessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
HEADING_RE = re.compile(r"^(#{2,4})\s+(.+?)\s*$")
TAG_RE = re.compile(r"<[^>]+>")
CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def _escape(text: str) -> str:
    return html.escape(text, quote=False)


def _read_meta_block(source: str) -> tuple[dict[str, Any], str]:
    """Read an optional leading ``---`` YAML block from a fenced block."""
    text = source.strip("\n")
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if lines[0].strip() != "---":
        return {}, text
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            raw = "\n".join(lines[1:idx])
            body = "\n".join(lines[idx + 1 :]).lstrip("\n")
            try:
                data = yaml.safe_load(raw) or {}
            except yaml.YAMLError:
                return {}, text
            if isinstance(data, dict):
                return data, body
    return {}, text


def _mermaid_format(source: str, language: str, class_name: str, options: dict, md: Markdown, **kwargs: Any) -> str:
    code = source.strip("\n")
    attrs = kwargs.get("attrs") or {}
    title = str(attrs.get("title") or "").strip()
    caption = f'<figcaption class="mermaid-caption">{_escape(title)}</figcaption>' if title else ""
    encoded = html.escape(json.dumps(code, ensure_ascii=False), quote=True)
    return (
        '<figure class="mermaid-figure" data-diagram>'
        '<div class="mermaid-toolbar"><span class="mermaid-label"><span class="dot"></span>Mermaid</span>'
        '<span class="mermaid-hint">点击图表可放大</span></div>'
        f'<div class="mermaid" data-src="{encoded}"></div>'
        f"{caption}</figure>"
    )


VIDEO_HOSTS = {
    "youtube.com": "youtube",
    "www.youtube.com": "youtube",
    "youtu.be": "youtube",
    "bilibili.com": "bilibili",
    "www.bilibili.com": "bilibili",
    "player.bilibili.com": "bilibili",
    "vimeo.com": "vimeo",
    "player.vimeo.com": "vimeo",
}


def _video_embed(url: str, title: str, start: int | None) -> tuple[str, str] | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    kind = VIDEO_HOSTS.get(host)
    if not kind:
        return None
    if kind == "youtube":
        if host == "youtu.be":
            vid = parsed.path.strip("/")
        else:
            vid = parsed.query.split("v=")[-1].split("&")[0] if "v=" in parsed.query else parsed.path.rstrip("/").split("/")[-1]
        if not vid:
            return None
        suffix = f"?start={int(start)}" if start else ""
        return f"https://www.youtube-nocookie.com/embed/{vid}{suffix}", "youtube"
    if kind == "bilibili":
        if parsed.netloc.startswith("player."):
            return "https:" + url if url.startswith("//") else url, "bilibili"
        path = parsed.path.strip("/")
        bvid = next((part for part in path.split("/") if part.startswith("BV")), "")
        if bvid:
            return (
                f"//player.bilibili.com/player.html?bvid={bvid}&page=1&high_quality=1",
                "bilibili",
            )
        av = next((part for part in path.split("/") if part.startswith("av")), "")
        if av:
            return f"//player.bilibili.com/player.html?aid={av[2:]}&page=1", "bilibili"
        return None
    if kind == "vimeo":
        vid = parsed.path.strip("/").split("/")[-1]
        if not vid.isdigit():
            return None
        return f"https://player.vimeo.com/video/{vid}", "vimeo"
    return None


def _video_format(source: str, language: str, class_name: str, options: dict, md: Markdown, **kwargs: Any) -> str:
    attrs = kwargs.get("attrs") or {}
    data, body = _read_meta_block(source)
    if not data:
        stripped = source.strip()
        if stripped and not stripped.splitlines()[0].startswith(("http://", "https://", "//")):
            try:
                parsed = yaml.safe_load(stripped)
            except yaml.YAMLError:
                parsed = None
            if isinstance(parsed, dict):
                data = parsed
                body = ""
    if not data:
        data = {
            "src": attrs.get("src", body.strip()),
            "caption": attrs.get("caption", ""),
        }
    for key in ("src", "poster", "caption", "credit", "width", "type", "start", "autoplay", "loop", "muted"):
        if key in attrs and key not in data:
            data[key] = attrs[key]
    src = str(data.get("src") or "").strip()
    if not src:
        return '<div class="build-error">video 块缺少 src。</div>'
    caption = str(data.get("caption") or "").strip()
    credit = str(data.get("credit") or "").strip()
    poster = str(data.get("poster") or "").strip()
    width = str(data.get("width") or "").strip()
    start = data.get("start")
    kind = str(data.get("type") or "").strip().lower()
    embed = _video_embed(src, caption, start) if src.startswith(("http://", "https://", "//")) else None
    if not kind and embed:
        kind = embed[1]
    if not kind:
        ext = src.rsplit(".", 1)[-1].lower()
        kind = "audio" if ext in {"mp3", "wav", "m4a", "ogg"} else "video"

    cap_html = ""
    if caption or credit:
        cap_html = "<figcaption>"
        if caption:
            cap_html += f'<span class="video-caption">{_escape(caption)}</span>'
        if credit:
            cap_html += f'<span class="video-credit">{_escape(credit)}</span>'
        cap_html += "</figcaption>"

    style = f' style="max-width:{html.escape(width, quote=True)}"' if width else ""

    if kind in {"youtube", "bilibili", "vimeo"}:
        if not embed:
            return f'<div class="build-error">无法识别视频链接：{_escape(src)}</div>'
        embed_url, _ = embed
        return (
            f'<figure class="paper-video">'
            '<div class="video-frame"><iframe '
            f'src="{html.escape(embed_url, quote=True)}" title="{html.escape(caption or "嵌入式视频", quote=True)}" '
            'loading="lazy" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" '
            "allowfullscreen></iframe></div>"
            f"{cap_html}</figure>"
        )

    if kind == "audio":
        return (
            f'<figure class="paper-audio"><audio controls preload="metadata" '
            f'src="{html.escape(src, quote=True)}"></audio>{cap_html}</figure>'
        )

    attrs = [
        "controls",
        "playsinline",
        'preload="metadata"',
    ]
    if poster:
        attrs.append(f'poster="{html.escape(poster, quote=True)}"')
    if data.get("autoplay"):
        attrs.append("autoplay")
    if data.get("loop"):
        attrs.append("loop")
    if data.get("muted", True):
        attrs.append("muted")
    if start:
        attrs.append(f'data-start="{int(start)}"')
    ext = src.rsplit(".", 1)[-1].lower()
    mime = {
        "mp4": "video/mp4",
        "webm": "video/webm",
        "ogv": "video/ogg",
        "mov": "video/quicktime",
    }.get(ext, "video/mp4")
    return (
        f'<figure class="paper-video"{style}>'
        '<div class="video-frame"><video ' + " ".join(attrs) + ">"
        f'<source src="{html.escape(src, quote=True)}" type="{mime}">'
        "你的浏览器不支持内嵌视频，请下载后观看。"
        "</video></div>"
        f"{cap_html}</figure>"
    )


def _algorithm_format(source: str, language: str, class_name: str, options: dict, md: Markdown, **kwargs: Any) -> str:
    attrs = kwargs.get("attrs") or {}
    title = str(attrs.get("title") or "").strip()
    label = str(attrs.get("label") or "ALGORITHM").strip()
    rows: list[str] = []
    for raw_line in source.strip("\n").splitlines():
        line = raw_line.rstrip()
        match = re.match(r"^\s*(\d+):\s?(.*)$", line)
        if match:
            number, body = match.group(1), match.group(2)
        else:
            number, body = "", line
        body = _escape(body)
        body = re.sub(
            r"(⊲|//\s|#\s)(.*)$",
            lambda m: f'<span class="algo-comment">{m.group(0)}</span>',
            body,
        )
        rows.append(
            f'<div class="algo-line"><span class="algo-ln">{number}</span>'
            f'<code class="algo-code">{body or "&nbsp;"}</code></div>'
        )
    head = ""
    if title or label:
        head = (
            '<div class="algo-head">'
            f'<span class="algo-label">{_escape(label)}</span>'
            f'<span class="algo-title">{_escape(title)}</span>'
            "</div>"
        )
    return (
        f'<figure class="algorithm">{head}'
        f'<div class="algo-body">{"".join(rows)}</div></figure>'
    )


# ---------------------------------------------------------------------------
# Preprocessors / tree processors / postprocessors
# ---------------------------------------------------------------------------

CUSTOM_FENCE_LANGS = {"mermaid", "video", "algorithm"}
INFO_ATTR_RE = re.compile(r"""([A-Za-z_][\w-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s]+))""")


def _parse_info(info: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in INFO_ATTR_RE.finditer(info):
        attrs[match.group(1)] = match.group(2) or match.group(3) or match.group(4) or ""
    return attrs


class CustomFencePreprocessor(Preprocessor):
    """Render mermaid/video/algorithm fences before SuperFences sees them.

    SuperFences treats a ``title=`` option as a normal code-block option and
    falls back to syntax highlighting, so custom fences with captions are
    intercepted here. The rendered HTML is stashed and restored after parsing.
    """

    OPEN_RE = re.compile(r"^(?P<indent>\s*)(?P<fence>`{3,}|~{3,})\s*(?P<lang>[A-Za-z0-9_+.-]+)(?P<info>.*)$")

    def run(self, lines: list[str]) -> list[str]:
        out: list[str] = []
        index = 0
        while index < len(lines):
            match = self.OPEN_RE.match(lines[index])
            if not match or match.group("lang").lower() not in CUSTOM_FENCE_LANGS:
                out.append(lines[index])
                index += 1
                continue
            marker = match.group("fence")
            lang = match.group("lang").lower()
            info = _parse_info(match.group("info"))
            close = len(marker) * marker[0]
            end = index + 1
            close_re = re.compile(r"^\s*" + re.escape(marker[0]) + r"{" + str(len(marker)) + r",}\s*$")
            while end < len(lines) and not close_re.match(lines[end]):
                end += 1
            if end >= len(lines):
                out.append(lines[index])
                index += 1
                continue
            body = "\n".join(lines[index + 1 : end])
            html = self._render(lang, body, info)
            out.append(self.md.htmlStash.store(html))
            index = end + 1
        return out

    def _render(self, lang: str, body: str, info: dict[str, str]) -> str:
        if lang == "mermaid":
            meta, source = _read_meta_block(body)
            title = info.get("title") or meta.get("title") or meta.get("caption") or ""
            attrs = {"title": str(title)} if title else {}
            return _mermaid_format(source.strip("\n"), lang, "mermaid", {}, self.md, attrs=attrs)
        if lang == "video":
            return _video_format(body, lang, "video", {}, self.md, attrs=info)
        meta, source = _read_meta_block(body)
        title = info.get("title") or meta.get("title") or ""
        label = info.get("label") or meta.get("label") or "ALGORITHM"
        attrs = {}
        if title:
            attrs["title"] = str(title)
        if label:
            attrs["label"] = str(label)
        return _algorithm_format(source.strip("\n"), lang, "algorithm", {}, self.md, attrs=attrs)


class AutoSectionIds(Preprocessor):
    """Give every h2-h4 a language-independent ``sec-*`` id.

    The Chinese and English versions of a paper share the same heading
    structure, so the generated ids let readers switch languages while
    staying on the same section.
    """

    def run(self, lines: list[str]) -> list[str]:
        counters: dict[int, int] = {2: 0, 3: 0, 4: 0}
        in_fence = False
        fence_marker = ""
        out: list[str] = []
        for line in lines:
            fence = FENCE_RE.match(line)
            if fence:
                marker = fence.group(1)[0]
                if not in_fence:
                    in_fence, fence_marker = True, marker
                elif marker == fence_marker:
                    in_fence, fence_marker = False, ""
                out.append(line)
                continue
            if in_fence:
                out.append(line)
                continue
            match = HEADING_RE.match(line)
            if not match or "{#" in line:
                out.append(line)
                continue
            level = len(match.group(1))
            text = match.group(2)
            counters[level] += 1
            for deeper in range(level + 1, 5):
                counters[deeper] = 0
            section = "-".join(str(counters[item]) for item in range(2, level + 1))
            out.append(f"{match.group(1)} {text} {{#sec-{section}}}")
        return out


class FigureTreeprocessor(Treeprocessor):
    """Turn standalone images into semantic ``<figure>`` blocks."""

    def __init__(self, md: Markdown, label: str = "查看大图：") -> None:
        super().__init__(md)
        self.label = label

    def run(self, root: etree.Element) -> etree.Element:
        for parent in list(root.iter()):
            for index, child in list(enumerate(list(parent))):
                if child.tag != "p":
                    continue
                images = [node for node in child.iter("img")]
                if len(images) != 1:
                    continue
                others = [node for node in child if node.tag != "img"]
                if others or (child.text or "").strip() or (child.tail or "").strip():
                    continue
                image = images[0]
                classes = (image.get("class") or "").split()
                if "no-figure" in classes:
                    continue
                src = image.get("src") or ""
                alt = (image.get("alt") or "").strip()
                title = (image.get("title") or "").strip()
                caption = alt or title
                for attr in ("class", "id"):
                    if image.get(attr):
                        image.attrib.pop(attr, None)
                image.set("loading", "lazy")
                image.set("decoding", "async")
                figure_classes = ["paper-figure"]
                for cls in ("wide", "narrow", "plain"):
                    if cls in classes:
                        figure_classes.append(cls)
                figure = etree.Element("figure", {"class": " ".join(figure_classes)})
                link = etree.SubElement(
                    figure,
                    "a",
                    {
                        "class": "figure-media",
                        "href": src,
                        "aria-label": self.label.rstrip("：: ") or "查看大图",
                    },
                )
                link.append(image)
                if caption:
                    figcaption = etree.SubElement(figure, "figcaption")
                    figcaption.text = caption
                parent.remove(child)
                parent.insert(index, figure)
        return root


def _merge_first_column(table: etree.Element) -> None:
    """Merge empty first-column cells into the preceding group label."""
    tbody = table.find("tbody")
    if tbody is None:
        return
    anchor = None
    span = 0
    for row in list(tbody):
        cells = row.findall("td")
        if not cells:
            continue
        first = cells[0]
        text = "".join(first.itertext()).strip()
        if not text and anchor is not None:
            span += 1
            row.remove(first)
        else:
            if anchor is not None and span > 1:
                anchor.set("rowspan", str(span))
            anchor = first
            span = 1
    if anchor is not None and span > 1:
        anchor.set("rowspan", str(span))


class TableCaptionTreeprocessor(Treeprocessor):
    """Wrap ``p.table-caption + table`` in a figure for captions/overflow."""

    def run(self, root: etree.Element) -> etree.Element:
        for parent in list(root.iter()):
            index = 0
            while index < len(parent):
                child = parent[index]
                if child.tag != "p":
                    index += 1
                    continue
                classes = (child.get("class") or "").split()
                if "table-caption" not in classes:
                    index += 1
                    continue
                if index + 1 >= len(parent) or parent[index + 1].tag != "table":
                    index += 1
                    continue
                table = parent[index + 1]
                caption_id = child.get("id") or f"table-caption-{id(child)}"
                figure = etree.Element("figure", {"class": "paper-table"})
                figcaption = etree.SubElement(figure, "figcaption", {"id": caption_id})
                figcaption.text = child.text
                for node in list(child):
                    child.remove(node)
                    figcaption.append(node)
                scroll = etree.SubElement(figure, "div", {"class": "table-scroll"})
                parent.remove(table)
                scroll.append(table)
                table.set("aria-labelledby", caption_id)
                parent.remove(child)
                parent.insert(index, figure)
                index += 1

        for table in root.iter("table"):
            _merge_first_column(table)
        return root


class LinkAndAssetPostprocessor(Postprocessor):
    """Rewrite paper-relative assets and annotate external links."""

    def __init__(self, md: Markdown, asset_prefix: str = "") -> None:
        super().__init__(md)
        self.asset_prefix = asset_prefix

    def _rewrite_url(self, url: str) -> str:
        if not url or url.startswith(("#", "/", "mailto:", "tel:", "data:", "http://", "https://", "//")):
            return url
        if url.endswith(".md"):
            url = url[:-3] + ".html"
        if url.startswith("./"):
            url = url[2:]
        if url.startswith("assets/") and self.asset_prefix:
            return self.asset_prefix + url
        return url

    def run(self, text: str) -> str:
        attr_re = re.compile(r'(?P<attr>\b(?:src|href|poster))=(["\'])(?P<url>[^"\']+)\2')

        def replace_attr(match: re.Match[str]) -> str:
            url = self._rewrite_url(match.group("url"))
            return f'{match.group("attr")}={match.group(2)}{url}{match.group(2)}'

        text = attr_re.sub(replace_attr, text)

        def external(match: re.Match[str]) -> str:
            tag = match.group(0)
            if " target=" in tag or ' href="#' in tag:
                return tag
            if not re.search(r'href="(?:https?:)?//', tag):
                return tag
            tag = tag[:-1].rstrip()
            if 'class="' in tag:
                tag = re.sub(r'class="([^"]*)"', r'class="\1 external"', tag)
                return tag + ' target="_blank" rel="noopener noreferrer">'
            return tag + ' target="_blank" rel="noopener noreferrer" class="external">'

        return re.sub(r"<a\b[^>]*>", external, text)


# ---------------------------------------------------------------------------
# Public rendering API
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RenderedMarkdown:
    html: str
    toc: str
    headings: list[dict[str, Any]]
    has_math: bool
    has_mermaid: bool
    has_code: bool
    has_video: bool
    word_count: int
    cjk_count: int


class PaperMarkdown(Extension):
    """Small extension that installs the custom processors."""

    def __init__(self, asset_prefix: str = "", figure_label: str = "查看大图：") -> None:
        self.asset_prefix = asset_prefix
        self.figure_label = figure_label
        super().__init__()

    def extendMarkdown(self, md: Markdown) -> None:
        md.registerExtension(self)
        md.preprocessors.register(AutoSectionIds(md), "auto_section_ids", 32)
        md.preprocessors.register(CustomFencePreprocessor(md), "custom_fences", 28)
        md.treeprocessors.register(FigureTreeprocessor(md, self.figure_label), "paper_figures", 7)
        md.treeprocessors.register(TableCaptionTreeprocessor(md), "paper_tables", 6)
        md.postprocessors.register(
            LinkAndAssetPostprocessor(md, self.asset_prefix), "paper_links", 8
        )


def _slugify(text: str, sep: str) -> str:
    text = TAG_RE.sub("", text).strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\-\u3400-\u9fff ]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", sep, text).strip(sep).lower()


def _html_text(fragment: str) -> str:
    return html.unescape(TAG_RE.sub("", fragment)).strip()


def _count_words(text: str) -> tuple[int, int]:
    plain = html.unescape(TAG_RE.sub(" ", text))
    cjk = len(CJK_RE.findall(plain))
    latin = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-]*", plain)
    return len(latin) + cjk, cjk


def render_markdown(text: str, *, asset_prefix: str = "", figure_label: str = "查看大图：", toc_depth: str = "2-3") -> RenderedMarkdown:
    """Render one paper body, including custom fenced blocks."""
    md = Markdown(
        extensions=[
            PaperMarkdown(asset_prefix, figure_label),
            "tables",
            "attr_list",
            "def_list",
            "abbr",
            "footnotes",
            "md_in_html",
            "admonition",
            "toc",
            "pymdownx.details",
            "pymdownx.highlight",
            "pymdownx.superfences",
            "pymdownx.arithmatex",
            "pymdownx.tasklist",
            "pymdownx.tilde",
            "pymdownx.mark",
        ],
        extension_configs={
            "toc": {
                "toc_depth": toc_depth,
                "permalink": "§",
                "permalink_title": "复制本节链接",
                "permalink_class": "headerlink",
                "slugify": _slugify,
            },
            "pymdownx.highlight": {
                "anchor_linenums": False,
                "linenums": False,
                "pygments_lang_class": True,
                "auto_title": False,
                "css_class": "codehilite",
                "pygments_style": "friendly",
            },
            "pymdownx.superfences": {"css_class": "codehilite"},
            "pymdownx.arithmatex": {
                "generic": True,
                "smart_dollar": True,
                "block_syntax": ["dollar", "square", "begin"],
                "inline_syntax": ["dollar", "round"],
            },
            "pymdownx.tasklist": {"custom_checkbox": True},
        },
        output_format="html5",
    )
    html_output = md.convert(text)
    headings = []
    for match in re.finditer(r"<h([2-4]) id=\"([^\"]+)\">(.*?)</h\1>", html_output, re.S):
        headings.append(
            {
                "level": int(match.group(1)),
                "id": match.group(2),
                "text": _html_text(match.group(3).replace("§", "")),
            }
        )
    word_count, cjk_count = _count_words(html_output)
    return RenderedMarkdown(
        html=html_output,
        toc=md.toc,
        headings=headings,
        has_math='class="arithmatex"' in html_output,
        has_mermaid='class="mermaid"' in html_output,
        has_code='class="codehilite"' in html_output,
        has_video='class="paper-video"' in html_output or 'class="paper-audio"' in html_output,
        word_count=word_count,
        cjk_count=cjk_count,
    )
