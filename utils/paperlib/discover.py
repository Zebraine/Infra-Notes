"""Discover and normalise papers stored under ``papers/<slug>/``."""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import PAPERS_DIR, SiteConfig

REQUIRED_FIELDS = ("title", "title_zh", "year", "tags", "abstract_zh")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class PaperError(ValueError):
    """Raised when a paper folder is invalid."""


@dataclass(slots=True)
class Paper:
    slug: str
    dir: Path
    meta: dict[str, Any]
    meta_path: Path
    original_path: Path
    zh_path: Path
    order: int = 0
    _html_cache: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- convenience accessors -------------------------------------------------
    @property
    def title(self) -> str:
        return str(self.meta["title"])

    @property
    def title_zh(self) -> str:
        return str(self.meta["title_zh"])

    @property
    def short_title(self) -> str:
        return str(self.meta.get("short_title") or self.title_zh)

    @property
    def year(self) -> int:
        return int(self.meta["year"])

    @property
    def date(self) -> dt.date:
        raw = self.meta.get("date")
        if isinstance(raw, dt.date):
            return raw
        if isinstance(raw, str):
            try:
                return dt.date.fromisoformat(raw)
            except ValueError:
                pass
        return dt.date(self.year, 1, 1)

    @property
    def authors(self) -> list[str]:
        authors = self.meta.get("authors") or ["Anonymous"]
        return [str(a) for a in authors]

    @property
    def authors_line(self) -> str:
        return " · ".join(self.authors)

    @property
    def affiliation(self) -> str:
        return str(self.meta.get("affiliation", ""))

    @property
    def venue(self) -> str:
        return str(self.meta.get("venue", ""))

    @property
    def venue_zh(self) -> str:
        return str(self.meta.get("venue_zh") or self.venue)

    @property
    def venue_full_zh(self) -> str:
        return str(self.meta.get("venue_full_zh") or self.venue_zh)

    @property
    def conference_url(self) -> str:
        return str(self.meta.get("conference_url", ""))

    @property
    def code_url(self) -> str:
        return str(self.meta.get("code_url", ""))

    @property
    def provenance_note(self) -> str:
        return str(self.meta.get("provenance_note", ""))

    @property
    def provenance_note_en(self) -> str:
        return str(self.meta.get("provenance_note_en") or self.provenance_note)

    @property
    def tags(self) -> list[str]:
        return [str(t) for t in (self.meta.get("tags") or [])]

    @property
    def abstract(self) -> str:
        return str(self.meta.get("abstract", ""))

    @property
    def abstract_zh(self) -> str:
        return str(self.meta.get("abstract_zh", ""))

    @property
    def summary(self) -> str:
        return str(self.meta.get("summary") or self.abstract_zh)

    @property
    def cover(self) -> str:
        return str(self.meta.get("cover", ""))

    @property
    def cover_alt(self) -> str:
        return str(self.meta.get("cover_alt") or f"{self.title_zh} 配图")

    @property
    def pdf(self) -> str:
        return str(self.meta.get("pdf", "assets/source.pdf"))

    @property
    def source_url(self) -> str:
        return str(self.meta.get("source_url", ""))

    @property
    def pages(self) -> int:
        return int(self.meta.get("pages", 0))

    @property
    def featured(self) -> bool:
        return bool(self.meta.get("featured", False))

    @property
    def number(self) -> int:
        """Stable 1-based display number, newest first."""
        return int(self.meta.get("_number", 0))

    @property
    def reading_minutes(self) -> int:
        return int(self.meta.get("_reading_minutes", 0))

    @property
    def word_count(self) -> int:
        return int(self.meta.get("_word_count", 0))

    @property
    def citation_key(self) -> str:
        if self.meta.get("citation_key"):
            return str(self.meta["citation_key"])
        first = re.sub(r"[^A-Za-z]", "", self.authors[0].split()[-1]).lower()
        if not first or self.authors[0].lower().startswith("anonymous"):
            first = re.sub(r"[^a-z0-9]", "", self.slug.lower())[:12] or "paper"
        return f"{first}{self.year}"

    @property
    def bibtex(self) -> str:
        key = self.citation_key
        authors = " and ".join(self.authors)
        fields = [
            f"  author = {{{authors}}}",
            f"  title = {{{self.title}}}",
            f"  year = {{{self.year}}}",
        ]
        if self.venue:
            fields.append(f"  howpublished = {{{self.venue}}}")
        if self.source_url:
            fields.append(f"  url = {{{self.source_url}}}")
        fields.append(f"  note = {{AI Infra Paper Library, {self.slug}}}")
        return "@misc{" + key + ",\n" + ",\n".join(fields) + "\n}"

    @property
    def fingerprint(self) -> str:
        """Short fingerprint used for deterministic page ids."""
        return hashlib.sha1(self.slug.encode("utf-8")).hexdigest()[:8]

    @property
    def rel_dir(self) -> str:
        return f"{self.slug}/"


def _normalise_authors(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(";") if part.strip()]
    return [str(part).strip() for part in raw if str(part).strip()]


def _load_one(directory: Path) -> Paper | None:
    if not directory.is_dir() or directory.name.startswith((".", "_")):
        return None
    meta_path = directory / "meta.yaml"
    if not meta_path.exists():
        return None
    try:
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - surfaced by CLI
        raise PaperError(f"{meta_path}: YAML 解析失败：{exc}") from exc
    if not isinstance(meta, dict):
        raise PaperError(f"{meta_path}: 顶层必须是映射（mapping）")

    slug = str(meta.get("slug") or directory.name)
    if not SLUG_RE.match(slug):
        raise PaperError(
            f"{meta_path}: slug 只能包含小写字母、数字和连字符，当前为 {slug!r}"
        )
    if slug != directory.name:
        raise PaperError(
            f"{meta_path}: slug={slug!r} 与目录名 {directory.name!r} 不一致"
        )

    missing = [f for f in REQUIRED_FIELDS if not meta.get(f)]
    if missing:
        raise PaperError(f"{meta_path}: 缺少必填字段：{', '.join(missing)}")

    if not isinstance(meta.get("tags"), list):
        raise PaperError(f"{meta_path}: tags 必须是列表")

    meta["authors"] = _normalise_authors(meta.get("authors"))

    original = directory / str(meta.get("original", "original.md"))
    zh = directory / str(meta.get("translation", "zh.md"))
    for path in (original, zh):
        if not path.exists():
            raise PaperError(f"{meta_path}: 缺少正文文件 {path.relative_to(directory)}")

    paper = Paper(
        slug=slug,
        dir=directory,
        meta=meta,
        meta_path=meta_path,
        original_path=original,
        zh_path=zh,
        order=int(meta.get("order", 0)),
    )
    return paper


def discover_papers(papers_dir: Path = PAPERS_DIR) -> list[Paper]:
    """Return all valid paper folders, newest first."""
    papers: list[Paper] = []
    if not papers_dir.exists():
        return papers
    for directory in sorted(papers_dir.iterdir()):
        paper = _load_one(directory)
        if paper:
            papers.append(paper)
    papers.sort(key=lambda p: (p.date, p.order, p.slug), reverse=True)
    for index, paper in enumerate(papers, start=1):
        paper.meta["_number"] = index
    return papers


def collect_tags(papers: list[Paper]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for paper in papers:
        for tag in paper.tags:
            counts[tag] = counts.get(tag, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))


def tag_colour(tag: str) -> str:
    """Deterministic pastel hue for tag chips."""
    digest = hashlib.md5(tag.encode("utf-8")).hexdigest()
    hue = int(digest[:4], 16) % 360
    return f"hsl({hue} 38% 46%)"


def citation_text(paper: Paper, site: SiteConfig) -> str:
    authors = ", ".join(paper.authors)
    venue = paper.venue or "preprint"
    return (
        f"{authors}. {paper.title}. {venue}, {paper.year}. "
        f"AI Infra Paper Library."
    )
