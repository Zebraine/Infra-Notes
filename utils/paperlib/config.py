"""Configuration and shared paths for the paper library."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = ROOT / "utils"
PAPERS_DIR = ROOT / "papers"
TEMPLATES_DIR = UTILS_DIR / "templates"
STATIC_DIR = UTILS_DIR / "static"
VENDOR_DIR = UTILS_DIR / "vendor"
CONTENT_DIR = UTILS_DIR / "content"
SITE_CONFIG = UTILS_DIR / "site.yaml"
BUILD_CACHE = UTILS_DIR / ".cache"

ASSET_DIR = PAPERS_DIR / "assets"
SITE_ASSET_DIR = ASSET_DIR / "site"
VENDOR_ASSET_DIR = ASSET_DIR / "vendor"


@dataclass(slots=True)
class SiteConfig:
    """Site-wide settings loaded from ``utils/site.yaml``."""

    title: str
    short_title: str
    description: str
    author: str
    lang: str
    footer_note: str
    nav: list[dict[str, str]]
    accent: str = "#0f766e"
    repo_url: str = ""

    @classmethod
    def load(cls, path: Path = SITE_CONFIG) -> "SiteConfig":
        data: dict[str, Any] = {}
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            title=data.get("title", "AI Infra 论文库"),
            short_title=data.get("short_title", "AI Infra Papers"),
            description=data.get(
                "description", "AI Infra 方向系统论文的中文精读与原文对照。"
            ),
            author=data.get("author", "AI Infra Notes"),
            lang=data.get("lang", "zh-CN"),
            footer_note=data.get(
                "footer_note", "内容以论文原文为准。译文由 AI 辅助完成，欢迎校对。"
            ),
            nav=data.get("nav")
            or [
                {"label": "论文库", "href": "index.html"},
                {"label": "格式手册", "href": "about/index.html"},
            ],
            accent=data.get("accent", "#0f766e"),
            repo_url=data.get("repo_url", ""),
        )
