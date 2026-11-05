#!/usr/bin/env python3
"""Scaffold a new paper folder under papers/.

The tool deliberately does not translate anything. It creates the metadata,
downloads the PDF (optional), extracts a plain-text reference, and leaves two
matching Markdown skeletons for the original and the translation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "papers"
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

META_TEMPLATE = """\
slug: {slug}
title: {title}
title_zh: {title_zh}
short_title: {short_title}
authors:
{authors}
affiliation: {affiliation}
year: {year}
date: {date}
venue: {venue}
venue_zh: {venue_zh}
type: {type}
tags:
{tags}
abstract: >-
{abstract}
abstract_zh: >-
{abstract_zh}
summary: >-
{summary}
cover: ""
pdf: "assets/source.pdf"
source_url: {source_url}
pages: {pages}
featured: {featured}
order: {order}
citation_key: {citation_key}
"""

ORIGINAL_TEMPLATE = """\
## Abstract

TODO: paste the original abstract here.

## 1 Introduction

TODO: transcribe the original paper here.

## References

1. TODO: add references.
"""

ZH_TEMPLATE = """\
## 摘要

TODO: 在这里写摘要译文。

## 1 引言

TODO: 在这里写正文译文。

## 参考文献

1. TODO: 添加参考文献。
"""


def q(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="new_paper.py",
        description="在 papers/ 下创建一篇新论文的目录骨架。",
    )
    parser.add_argument("--slug", required=True, help="目录名，如 bidkv")
    parser.add_argument("--title", required=True, help="英文标题")
    parser.add_argument("--title-zh", required=True, help="中文标题")
    parser.add_argument("--url", default="", help="PDF URL（可选，会下载到 assets/source.pdf）")
    parser.add_argument("--year", type=int, default=dt.date.today().year)
    parser.add_argument("--authors", default="Anonymous Authors", help="用分号分隔")
    parser.add_argument("--affiliation", default="")
    parser.add_argument("--venue", default="Preprint")
    parser.add_argument("--venue-zh", default="预印本")
    parser.add_argument("--tags", default="LLM Serving,KV Cache", help="用逗号分隔")
    parser.add_argument("--order", type=int, default=99)
    parser.add_argument("--force", action="store_true", help="覆盖已存在的目录")
    parser.add_argument("--no-download", action="store_true", help="不下载 PDF")
    return parser.parse_args(argv)


def download_pdf(url: str, destination: Path) -> bool:
    if not url:
        return False
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ai-infra-paper-library/0.1 (+https://example.invalid)",
            "Accept": "application/pdf,*/*",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            destination.write_bytes(response.read())
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  ! PDF 下载失败：{exc}")
        return False


def pdf_pages(path: Path) -> int:
    if not path.exists() or not shutil.which("pdfinfo"):
        return 0
    result = subprocess.run(
        ["pdfinfo", str(path)], capture_output=True, text=True, check=False
    )
    match = re.search(r"^Pages:\s+(\d+)", result.stdout, re.MULTILINE)
    return int(match.group(1)) if match else 0


def extract_text(path: Path) -> None:
    if not path.exists() or not shutil.which("pdftotext"):
        return
    target = path.with_name("source.txt")
    subprocess.run(
        ["pdftotext", "-layout", str(path), str(target)],
        capture_output=True,
        check=False,
    )
    if target.exists():
        print(f"  · 纯文本参考：{target.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not SLUG_RE.match(args.slug):
        print("✗ slug 只能包含小写字母、数字和连字符。", file=sys.stderr)
        return 1
    directory = PAPERS / args.slug
    if directory.exists() and not args.force:
        print(f"✗ {directory.relative_to(ROOT)} 已存在；使用 --force 覆盖。", file=sys.stderr)
        return 1
    if directory.exists():
        shutil.rmtree(directory)
    (directory / "assets" / "figures").mkdir(parents=True)
    (directory / "assets" / ".gitkeep").touch()

    pdf_path = directory / "assets" / "source.pdf"
    downloaded = False
    if args.url and not args.no_download:
        print(f"→ 下载 PDF：{args.url}")
        downloaded = download_pdf(args.url, pdf_path)
        if downloaded:
            extract_text(pdf_path)

    authors = [part.strip() for part in args.authors.split(";") if part.strip()]
    if not authors:
        authors = ["Anonymous Authors"]
    tags = [part.strip() for part in args.tags.split(",") if part.strip()]
    pages = pdf_pages(pdf_path)

    today = dt.date.today().isoformat()
    meta = META_TEMPLATE.format(
        slug=args.slug,
        title=q(args.title),
        title_zh=q(args.title_zh),
        short_title=q(args.title_zh[:24]),
        authors="\n".join(f"  - {q(author)}" for author in authors),
        affiliation=q(args.affiliation),
        year=args.year,
        date=today,
        venue=q(args.venue),
        venue_zh=q(args.venue_zh),
        type=q("系统论文"),
        tags="\n".join(f"  - {q(tag)}" for tag in tags),
        abstract="  TODO: paste the original abstract here.",
        abstract_zh="  TODO: 在这里写摘要译文。",
        summary="  TODO: 一句话概括这篇论文。",
        source_url=q(args.url),
        pages=pages,
        featured="false",
        order=args.order,
        citation_key=q(f"{args.slug.replace('-', '')}{args.year}"),
    )
    (directory / "meta.yaml").write_text(meta, encoding="utf-8")
    (directory / "original.md").write_text(ORIGINAL_TEMPLATE, encoding="utf-8")
    (directory / "zh.md").write_text(ZH_TEMPLATE, encoding="utf-8")

    print(f"\n✓ 已创建 {directory.relative_to(ROOT)}")
    print("  meta.yaml     元数据（标题、作者、标签、摘要）")
    print("  original.md   英文原文")
    print("  zh.md         中文译文（标题层级必须与原文一致）")
    print("  assets/       PDF、图表、视频")
    if downloaded:
        print("  assets/source.pdf")
    print("\n下一步：")
    print(f"  1. 编辑 {directory.relative_to(ROOT)}/meta.yaml")
    print("  2. 填写 original.md 与 zh.md（保持相同的 h2/h3 层级）")
    print("  3. 将图表放到 assets/figures/，用 ![图注](assets/figures/x.png) 引用")
    print("  4. ./utils/build.sh --check && ./utils/build.sh\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
