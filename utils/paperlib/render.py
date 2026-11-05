"""Render the static paper library with Jinja2."""

from __future__ import annotations

import datetime as dt
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import jinja2
from markupsafe import Markup

from .config import CONTENT_DIR, PAPERS_DIR, TEMPLATES_DIR, SiteConfig
from .discover import Paper, collect_tags, tag_colour
from .markdown_ext import RenderedMarkdown, render_markdown


@dataclass(slots=True)
class RenderedPaper:
    paper: Paper
    zh: RenderedMarkdown
    en: RenderedMarkdown
    reading_zh: int
    reading_en: int


def _reading_minutes(cjk: int, latin: int) -> int:
    return max(1, math.ceil(cjk / 360 + latin / 240))


def _json_for_script(value: Any) -> Markup:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    raw = raw.replace("</", "<\\/")
    return Markup(raw)


def _date_display(value: dt.date) -> str:
    return f"{value.year}.{value.month:02d}.{value.day:02d}"


def _duration(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} 分钟"
    hours, mins = divmod(minutes, 60)
    return f"{hours} 小时 {mins} 分" if mins else f"{hours} 小时"


class SiteRenderer:
    def __init__(self, site: SiteConfig, papers: list[Paper], asset_version: str) -> None:
        self.site = site
        self.papers = papers
        self.asset_version = asset_version
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.env.filters.update(
            {
                "date_display": _date_display,
                "duration": _duration,
                "tag_colour": tag_colour,
                "json_script": _json_for_script,
                "urlencode": lambda value: quote(str(value), safe=""),
            }
        )
        self.rendered: dict[str, RenderedPaper] = {}
        self.tags = collect_tags(papers)

    # ------------------------------------------------------------------ data
    def prepare(self) -> None:
        for paper in self.papers:
            original = paper.original_path.read_text(encoding="utf-8")
            translation = paper.zh_path.read_text(encoding="utf-8")
            zh = render_markdown(translation, asset_prefix="")
            en = render_markdown(original, asset_prefix="../", figure_label="View larger image: ")
            reading_zh = _reading_minutes(zh.cjk_count, zh.word_count - zh.cjk_count)
            reading_en = _reading_minutes(0, en.word_count)
            paper.meta["_reading_minutes"] = reading_zh
            paper.meta["_reading_en_minutes"] = reading_en
            paper.meta["_word_count"] = zh.word_count
            self.rendered[paper.slug] = RenderedPaper(
                paper=paper, zh=zh, en=en, reading_zh=reading_zh, reading_en=reading_en
            )
        # A featured paper is pinned first; the rest stay newest-first.
        self.papers.sort(
            key=lambda p: (
                not p.featured,
                -p.date.toordinal(),
                -p.order,
                p.slug,
            )
        )
        for index, paper in enumerate(self.papers, start=1):
            paper.meta["_number"] = index
        self.tags = collect_tags(self.papers)

    # --------------------------------------------------------------- helpers
    def _base(
        self,
        *,
        root: str,
        page: dict[str, Any],
        nav_active: str,
    ) -> dict[str, Any]:
        return {
            "site": self.site,
            "root": root,
            "page": page,
            "nav_active": nav_active,
            "asset_version": self.asset_version,
            "papers": self.papers,
            "tags": self.tags,
            "today": dt.date.today(),
            "has_math": False,
            "has_mermaid": False,
            "has_video": False,
        }

    def _paper_url(self, paper: Paper, root: str) -> str:
        return f"{root}{paper.slug}/index.html"

    def _paper_json(self, paper: Paper) -> dict[str, Any]:
        return {
            "slug": paper.slug,
            "number": paper.number,
            "title": paper.title,
            "title_zh": paper.title_zh,
            "short_title": paper.short_title,
            "authors": paper.authors,
            "affiliation": paper.affiliation,
            "year": paper.year,
            "date": paper.date.isoformat(),
            "venue": paper.venue,
            "venue_zh": paper.venue_zh,
            "venue_full_zh": paper.venue_full_zh,
            "conference_url": paper.conference_url,
            "code_url": paper.code_url,
            "tags": paper.tags,
            "abstract": paper.abstract,
            "abstract_zh": paper.abstract_zh,
            "summary": paper.summary,
            "cover": f"{paper.slug}/{paper.cover}" if paper.cover else "",
            "cover_alt": paper.cover_alt,
            "pdf": f"{paper.slug}/{paper.pdf}",
            "source_url": paper.source_url,
            "pages": paper.pages,
            "reading_minutes": paper.reading_minutes,
            "url": f"{paper.slug}/index.html",
            "original_url": f"{paper.slug}/original/index.html",
        }

    def _related(self, paper: Paper, limit: int = 2) -> list[dict[str, str]]:
        items = []
        for other in self.papers:
            if other.slug == paper.slug:
                continue
            items.append(
                {
                    "slug": other.slug,
                    "title_zh": other.title_zh,
                    "title": other.title,
                    "year": str(other.year),
                    "cover": other.cover,
                    "reading": str(other.reading_minutes),
                }
            )
            if len(items) >= limit:
                break
        return items

    def _prev_next(self, paper: Paper) -> tuple[Paper | None, Paper | None]:
        index = next(
            (i for i, item in enumerate(self.papers) if item.slug == paper.slug), -1
        )
        if index < 0:
            return None, None
        prev_paper = self.papers[index - 1] if index > 0 else None
        next_paper = self.papers[index + 1] if index + 1 < len(self.papers) else None
        return prev_paper, next_paper

    def _write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    # ----------------------------------------------------------------- pages
    def render_index(self) -> None:
        featured = self.papers[0] if self.papers else None
        others = self.papers[1:]
        total_pages = sum(p.pages for p in self.papers if p.pages)
        years = sorted({p.year for p in self.papers}, reverse=True)
        payload = [self._paper_json(p) for p in self.papers]
        ctx = self._base(
            root="",
            nav_active="library",
            page={
                "title": f"{self.site.title} · 系统论文精读",
                "description": self.site.description,
                "type": "website",
            },
        )
        ctx.update(
            {
                "featured": featured,
                "others": others,
                "total_pages": total_pages,
                "years": years,
                "papers_json": payload,
                "featured_json": self._paper_json(featured) if featured else None,
            }
        )
        html = self.env.get_template("index.html").render(**ctx)
        self._write(PAPERS_DIR / "index.html", html)
        self._write(
            PAPERS_DIR / "index.json",
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )

    def render_paper(self, paper: Paper) -> None:
        rendered = self.rendered[paper.slug]
        prev_paper, next_paper = self._prev_next(paper)
        common = {
            "paper": paper,
            "tags": self.tags,
            "citation": paper.bibtex,
            "related": self._related(paper),
            "prev_paper": prev_paper,
            "next_paper": next_paper,
            "translation_note": "译文由 AI 辅助完成，术语保留英文缩写；技术细节请以原文为准。",
            "source_pdf": f"{paper.slug}/{paper.pdf}",
        }

        zh_ctx = self._base(
            root="../",
            nav_active="library",
            page={
                "title": f"{paper.title_zh} · {self.site.title}",
                "description": paper.abstract_zh[:160],
                "type": "article",
                "published": paper.date.isoformat(),
                "author": ", ".join(paper.authors),
            },
        )
        zh_ctx.update(common)
        zh_ctx.update(
            {
                "lang": "zh",
                "body": Markup(rendered.zh.html),
                "toc": Markup(rendered.zh.toc),
                "headings": rendered.zh.headings,
                "asset_prefix": "",
                "translation_url": "index.html",
                "original_url": "original/index.html",
                "pdf_url": paper.pdf,
                "cover_url": paper.cover,
                "reading_minutes": rendered.reading_zh,
                "has_math": rendered.zh.has_math,
                "has_mermaid": rendered.zh.has_mermaid,
                "has_video": rendered.zh.has_video,
                "has_code": rendered.zh.has_code,
                "paper_title": paper.title_zh,
                "alt_title": paper.title,
                "lang_label": "中文译文",
                "alt_lang_label": "English Original",
            }
        )
        html = self.env.get_template("paper.html").render(**zh_ctx)
        self._write(paper.dir / "index.html", html)

        en_ctx = self._base(
            root="../../",
            nav_active="library",
            page={
                "title": f"{paper.title} · {self.site.title}",
                "description": paper.abstract[:160],
                "type": "article",
                "published": paper.date.isoformat(),
                "author": ", ".join(paper.authors),
            },
        )
        en_ctx.update(common)
        en_ctx.update(
            {
                "lang": "en",
                "body": Markup(rendered.en.html),
                "toc": Markup(rendered.en.toc),
                "headings": rendered.en.headings,
                "asset_prefix": "../",
                "translation_url": "../index.html",
                "original_url": "index.html",
                "pdf_url": f"../{paper.pdf}",
                "cover_url": f"../{paper.cover}" if paper.cover else "",
                "reading_minutes": rendered.reading_en,
                "has_math": rendered.en.has_math,
                "has_mermaid": rendered.en.has_mermaid,
                "has_video": rendered.en.has_video,
                "has_code": rendered.en.has_code,
                "paper_title": paper.title,
                "alt_title": paper.title_zh,
                "lang_label": "English Original",
                "alt_lang_label": "中文译文",
            }
        )
        html = self.env.get_template("paper.html").render(**en_ctx)
        self._write(paper.dir / "original" / "index.html", html)

    def render_about(self) -> None:
        source = CONTENT_DIR / "about.md"
        if not source.exists():
            return
        rendered = render_markdown(source.read_text(encoding="utf-8"), asset_prefix="")
        ctx = self._base(
            root="../",
            nav_active="about",
            page={
                "title": f"格式手册 · {self.site.title}",
                "description": "论文库支持的 Markdown、数学、Mermaid、代码、图片与视频写法。",
                "type": "article",
            },
        )
        ctx.update(
            {
                "body": Markup(rendered.html),
                "toc": Markup(rendered.toc),
                "headings": rendered.headings,
                "has_math": rendered.has_math,
                "has_mermaid": rendered.has_mermaid,
                "has_video": rendered.has_video,
                "has_code": True,
            }
        )
        html = self.env.get_template("about.html").render(**ctx)
        self._write(PAPERS_DIR / "about" / "index.html", html)

    def render_all(self) -> None:
        self.prepare()
        self.render_index()
        for paper in self.papers:
            self.render_paper(paper)
        self.render_about()


def clean_generated(papers: list[Paper]) -> None:
    """Remove generated HTML/assets but keep paper sources and images."""
    targets = [
        PAPERS_DIR / "index.html",
        PAPERS_DIR / "index.json",
        PAPERS_DIR / "about" / "index.html",
        PAPERS_DIR / "assets" / "site",
        PAPERS_DIR / "assets" / "vendor",
    ]
    for paper in papers:
        targets.append(paper.dir / "index.html")
        targets.append(paper.dir / "original" / "index.html")
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
