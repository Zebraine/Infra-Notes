# Vendored assets

The paper library ships its reading-time dependencies locally so pages work offline.

| Asset | Version | Upstream | License |
| --- | --- | --- | --- |
| KaTeX (`katex/`) | 0.16.22 | https://katex.org/ | MIT |
| Mermaid (`mermaid/`) | 11.12.1 | https://mermaid.js.org/ | MIT |

The build only copies `katex.min.css`, `katex.min.js`, the KaTeX font files, and
`mermaid.min.js` into `papers/assets/vendor/`. The full license texts are kept next
to the assets (`katex/LICENSE`, `mermaid/LICENSE`).

Build-time Python dependencies (not redistributed with the generated site):

- Python-Markdown — BSD
- pymdown-extensions — MIT
- PyYAML — MIT
- Jinja2 — BSD
- Pygments — BSD (its generated CSS is included in `papers/assets/site/pygments.css`)
