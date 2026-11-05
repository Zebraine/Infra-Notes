"""Build the static AI Infra paper library.

Usage:
    python utils/build.py                 # build everything
    python utils/build.py --check         # validate sources only
    python utils/build.py --serve         # build and preview on :8765
    python utils/build.py --clean         # remove generated output
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paperlib.assets import prepare_assets
from utils.paperlib.config import PAPERS_DIR, SiteConfig
from utils.paperlib.discover import PaperError, discover_papers
from utils.paperlib.render import SiteRenderer, clean_generated
from utils.paperlib.validate import (
    print_issues,
    validate_rendered,
    validate_sources,
)

BANNER = """\
┌──────────────────────────────────────────────┐
│  AI Infra · Paper Library                    │
└──────────────────────────────────────────────┘"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build.py",
        description="构建 AI Infra 论文库静态站点。",
    )
    parser.add_argument("--check", action="store_true", help="只校验，不写入文件")
    parser.add_argument("--strict", action="store_true", help="把警告视为错误")
    parser.add_argument("--clean", action="store_true", help="清理生成的文件")
    parser.add_argument("--no-assets", action="store_true", help="跳过静态资源复制")
    parser.add_argument("--serve", action="store_true", help="构建后启动本地预览服务器")
    parser.add_argument("--port", type=int, default=8765, help="预览端口，默认 8765")
    return parser.parse_args(argv)


def serve(port: int) -> None:
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(PAPERS_DIR), **kwargs)

        def log_message(self, fmt: str, *args: object) -> None:
            if "404" in (fmt % args):
                super().log_message(fmt, *args)

    with socketserver.ThreadingTCPServer(("127.0.0.1", port), QuietHandler) as httpd:
        print(f"\n  预览：http://127.0.0.1:{port}/index.html")
        print("  Ctrl+C 停止\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  已停止。")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(BANNER)

    try:
        papers = discover_papers()
    except PaperError as exc:
        print(f"\n✗ 论文元数据错误：{exc}\n", file=sys.stderr)
        return 1

    if args.clean:
        clean_generated(papers)
        print(f"\n  已清理生成文件（保留论文源文件与图片）。\n")
        return 0

    if not papers:
        print("\n  papers/ 下还没有论文。可运行 utils/new_paper.py 新建一篇。\n")
        return 1

    if not args.no_assets and not args.check:
        manifest = prepare_assets()
        asset_version = manifest["version"]
    else:
        asset_version = "dev"

    site = SiteConfig.load()
    renderer = SiteRenderer(site, papers, asset_version)
    renderer.prepare()

    issues = validate_sources(papers) + validate_rendered(renderer)
    errors, warnings = print_issues(issues)
    if errors or (args.strict and warnings):
        print(
            f"\n✗ 校验未通过：{errors} 个错误，{warnings} 个警告。未写入页面。\n",
            file=sys.stderr,
        )
        return 1

    if args.check:
        print(f"\n✓ 校验通过：{len(papers)} 篇论文，{warnings} 个警告。\n")
        return 0

    renderer.render_index()
    for paper in papers:
        renderer.render_paper(paper)
    renderer.render_about()

    total_reading = sum(p.reading_minutes for p in papers)
    print(
        f"\n  论文      {len(papers)} 篇"
        f"\n  页面      {len(papers) * 2 + 2} 个"
        f"\n  阅读量    约 {total_reading} 分钟"
        f"\n  入口      {PAPERS_DIR / 'index.html'}"
        f"\n  版本      {asset_version}"
        f"\n\n✓ 构建完成。\n"
    )

    if args.serve:
        serve(args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
