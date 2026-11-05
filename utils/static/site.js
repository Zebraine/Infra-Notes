/* AI Infra Paper Library — progressive enhancement only. */
(function () {
  "use strict";

  var docEl = document.documentElement;
  var $ = function (selector, root) {
    return (root || document).querySelector(selector);
  };
  var $$ = function (selector, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(selector));
  };
  var THEME_KEY = "aiinfra-theme";
  var FONT_KEY = "aiinfra-font-size";

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function safeStorageGet(key) {
    try {
      return localStorage.getItem(key);
    } catch (error) {
      return null;
    }
  }

  function safeStorageSet(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch (error) {
      /* private mode / file:// — preferences simply do not persist */
    }
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.top = "-1000px";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      try {
        var ok = document.execCommand("copy");
        document.body.removeChild(area);
        if (ok) resolve();
        else reject(new Error("copy failed"));
      } catch (error) {
        document.body.removeChild(area);
        reject(error);
      }
    });
  }

  var toastTimer = null;
  function toast(message) {
    var node = $(".copy-toast");
    if (!node) {
      node = document.createElement("div");
      node.className = "copy-toast";
      node.setAttribute("role", "status");
      document.body.appendChild(node);
    }
    node.textContent = message;
    node.classList.add("is-visible");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(function () {
      node.classList.remove("is-visible");
    }, 1800);
  }

  /* ---------------------------------------------------------------- theme */
  function setTheme(theme, persist) {
    docEl.dataset.theme = theme;
    $$(".theme-toggle").forEach(function (button) {
      var dark = theme === "dark";
      button.setAttribute("aria-label", dark ? "切换浅色模式" : "切换深色模式");
      button.setAttribute("title", dark ? "切换浅色模式" : "切换深色模式");
    });
    var meta = $('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", theme === "dark" ? "#0c1116" : "#f6f4ee");
    if (persist) safeStorageSet(THEME_KEY, theme);
    document.dispatchEvent(
      new CustomEvent("themechange", { detail: { theme: theme } })
    );
  }

  function initTheme() {
    var initial = docEl.dataset.theme || "light";
    setTheme(initial, false);
    $$(".theme-toggle").forEach(function (button) {
      button.addEventListener("click", function () {
        setTheme(docEl.dataset.theme === "dark" ? "light" : "dark", true);
      });
    });
  }

  /* ----------------------------------------------------------------- math */
  function renderMath() {
    if (!window.katex) return;
    $$(".arithmatex").forEach(function (element) {
      if (element.dataset.rendered === "1") return;
      var raw = (element.textContent || "").trim();
      if (!raw) return;
      var display = element.tagName === "DIV" || /^\\\[/.test(raw);
      var tex = raw
        .replace(/^\\\[/, "")
        .replace(/\\\]$/, "")
        .replace(/^\\\(/, "")
        .replace(/\\\)$/, "")
        .trim();
      try {
        window.katex.render(tex, element, {
          displayMode: display,
          throwOnError: false,
          strict: "ignore",
          trust: false
        });
        element.dataset.rendered = "1";
      } catch (error) {
        element.classList.add("math-error");
        element.setAttribute("title", error.message);
      }
    });
  }

  /* -------------------------------------------------------------- mermaid */
  var mermaidCounter = 0;
  var mermaidRunning = false;
  var mermaidQueued = false;

  function mermaidVariables(theme) {
    if (theme === "dark") {
      return {
        background: "#121920",
        mainBkg: "#121920",
        primaryColor: "#12332f",
        primaryTextColor: "#e7edf3",
        primaryBorderColor: "#5eead4",
        secondaryColor: "#1a252e",
        tertiaryColor: "#0f161c",
        lineColor: "#5f8b84",
        textColor: "#d2dbe4",
        nodeBorder: "#5eead4",
        clusterBkg: "#101820",
        clusterBorder: "#2a3a44",
        edgeLabelBackground: "#121920",
        fontFamily:
          '"Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif'
      };
    }
    return {
      background: "#fffdf8",
      mainBkg: "#fffdf8",
      primaryColor: "#dbefeb",
      primaryTextColor: "#182024",
      primaryBorderColor: "#0f766e",
      secondaryColor: "#f3eee3",
      tertiaryColor: "#f6f4ee",
      lineColor: "#7c9a95",
      textColor: "#2b3238",
      nodeBorder: "#0f766e",
      clusterBkg: "#f6f4ee",
      clusterBorder: "#d8d2c5",
      edgeLabelBackground: "#fffdf8",
      fontFamily:
        '"Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif'
    };
  }

  function renderMermaid() {
    var nodes = $$(".mermaid[data-src]");
    if (!nodes.length) return;
    if (!window.mermaid) {
      nodes.forEach(function (node) {
        node.innerHTML =
          '<div class="mermaid-error">Mermaid 资源未加载，请检查 assets/vendor/mermaid/。</div>';
      });
      return;
    }
    if (mermaidRunning) {
      mermaidQueued = true;
      return;
    }
    mermaidRunning = true;
    var theme = docEl.dataset.theme === "dark" ? "dark" : "light";
    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      themeVariables: mermaidVariables(theme),
      flowchart: { htmlLabels: true, curve: "basis" },
      sequence: { useMaxWidth: true },
      er: { useMaxWidth: true }
    });

    var queue = nodes.slice();
    function next() {
      if (!queue.length) {
        mermaidRunning = false;
        if (mermaidQueued) {
          mermaidQueued = false;
          renderMermaid();
        }
        return;
      }
      var node = queue.shift();
      var source;
      try {
        source = JSON.parse(node.getAttribute("data-src"));
      } catch (error) {
        node.innerHTML =
          '<div class="mermaid-error">无法读取 Mermaid 源码。</div>';
        next();
        return;
      }
      mermaidCounter += 1;
      var id = "mermaid-svg-" + mermaidCounter;
      window.mermaid
        .render(id, source)
        .then(function (result) {
          node.innerHTML = result.svg;
          node.dataset.rendered = "1";
          var svg = $("svg", node);
          if (svg) {
            svg.removeAttribute("height");
            svg.style.maxWidth = "100%";
            svg.style.height = "auto";
          }
        })
        .catch(function () {
          node.innerHTML =
            '<div class="mermaid-error">图表渲染失败，源码如下：\n' +
            escapeHtml(source) +
            "</div>";
        })
        .then(next);
    }
    next();
  }

  function scheduleMermaid() {
    window.clearTimeout(scheduleMermaid.timer);
    scheduleMermaid.timer = window.setTimeout(renderMermaid, 60);
  }

  /* -------------------------------------------------------------- lightbox */
  var lightbox = null;
  var lightboxIndex = 0;
  var lightboxTrigger = null;
  var gallery = [];

  function collectGallery() {
    gallery = [];
    $$(".paper-figure .figure-media").forEach(function (link) {
      var figure = link.closest(".paper-figure");
      var caption = figure && $("figcaption", figure);
      gallery.push({
        type: "image",
        href: link.getAttribute("href"),
        alt: link.querySelector("img")
          ? link.querySelector("img").alt
          : "",
        caption: caption ? caption.textContent.trim() : ""
      });
    });
    $$(".mermaid-figure").forEach(function (figure) {
      var svg = $(".mermaid svg", figure);
      if (!svg) return;
      var caption = $(".mermaid-caption", figure);
      gallery.push({
        type: "svg",
        svg: svg,
        caption: caption ? caption.textContent.trim() : "Mermaid 图表"
      });
    });
  }

  function buildLightbox() {
    lightbox = document.createElement("div");
    lightbox.className = "lightbox";
    lightbox.hidden = true;
    lightbox.setAttribute("role", "dialog");
    lightbox.setAttribute("aria-modal", "true");
    lightbox.setAttribute("aria-label", "图片预览");
    lightbox.innerHTML =
      '<button class="lightbox-close" type="button" aria-label="关闭预览">' +
      '<svg class="icon" width="20" height="20" aria-hidden="true"><use href="#i-x"></use></svg>' +
      "</button>" +
      '<button class="lightbox-prev" type="button" aria-label="上一张">' +
      '<svg class="icon" width="20" height="20" aria-hidden="true"><use href="#i-arrow-left"></use></svg>' +
      "</button>" +
      '<div class="lightbox-stage"><div class="lightbox-media"></div><p class="lightbox-caption"></p></div>' +
      '<button class="lightbox-next" type="button" aria-label="下一张">' +
      '<svg class="icon" width="20" height="20" aria-hidden="true"><use href="#i-arrow-right"></use></svg>' +
      "</button>";
    document.body.appendChild(lightbox);
    lightbox.addEventListener("click", function (event) {
      if (event.target === lightbox) closeLightbox();
    });
    $(".lightbox-close", lightbox).addEventListener("click", closeLightbox);
    $(".lightbox-prev", lightbox).addEventListener("click", function () {
      showLightbox(lightboxIndex - 1);
    });
    $(".lightbox-next", lightbox).addEventListener("click", function () {
      showLightbox(lightboxIndex + 1);
    });
  }

  function showLightbox(index) {
    if (!gallery.length) return;
    if (index < 0) index = gallery.length - 1;
    if (index >= gallery.length) index = 0;
    lightboxIndex = index;
    var item = gallery[index];
    var media = $(".lightbox-media", lightbox);
    var caption = $(".lightbox-caption", lightbox);
    media.innerHTML = "";
    if (item.type === "image") {
      var image = document.createElement("img");
      image.src = item.href;
      image.alt = item.alt || "";
      media.appendChild(image);
    } else if (item.type === "svg" && item.svg) {
      var clone = item.svg.cloneNode(true);
      clone.removeAttribute("width");
      clone.removeAttribute("height");
      media.appendChild(clone);
    }
    caption.textContent = item.caption || "";
    caption.hidden = !item.caption;
    var multiple = gallery.length > 1;
    $(".lightbox-prev", lightbox).hidden = !multiple;
    $(".lightbox-next", lightbox).hidden = !multiple;
    lightbox.hidden = false;
    document.body.classList.add("lightbox-open");
    $(".lightbox-close", lightbox).focus();
  }

  function openLightbox(index, trigger) {
    if (!lightbox) buildLightbox();
    collectGallery();
    lightboxTrigger = trigger || null;
    showLightbox(index);
  }

  function closeLightbox() {
    if (!lightbox || lightbox.hidden) return;
    lightbox.hidden = true;
    document.body.classList.remove("lightbox-open");
    $(".lightbox-media", lightbox).innerHTML = "";
    if (lightboxTrigger) {
      lightboxTrigger.focus();
      lightboxTrigger = null;
    }
  }

  function initLightbox() {
    $$(".paper-figure .figure-media").forEach(function (link) {
      link.addEventListener("click", function (event) {
        if (event.metaKey || event.ctrlKey || event.shiftKey) return;
        collectGallery();
        var index = gallery.findIndex(function (item) {
          return item.type === "image" && item.href === link.getAttribute("href");
        });
        if (index < 0) return;
        event.preventDefault();
        openLightbox(index, link);
      });
    });
    $$(".mermaid-figure").forEach(function (figure) {
      figure.addEventListener("click", function (event) {
        if (event.target.closest(".mermaid-toolbar")) return;
        collectGallery();
        var svg = $(".mermaid svg", figure);
        if (!svg) return;
        var index = gallery.findIndex(function (item) {
          return item.type === "svg" && item.svg === svg;
        });
        if (index < 0) return;
        openLightbox(index, figure);
      });
    });
    document.addEventListener("keydown", function (event) {
      if (!lightbox || lightbox.hidden) return;
      if (event.key === "Escape") closeLightbox();
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        showLightbox(lightboxIndex - 1);
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        showLightbox(lightboxIndex + 1);
      }
    });
  }

  /* ------------------------------------------------------------ toc / toc */
  function initToc() {
    var toc = $(".toc");
    var article = $("#paper-body");
    if (!toc) return;
    var links = $$('a[href^="#"]', toc);
    var headings = links
      .map(function (link) {
        var id = decodeURIComponent(link.hash.slice(1));
        return document.getElementById(id);
      })
      .filter(Boolean);

    function setActive(id) {
      links.forEach(function (link) {
        var active = decodeURIComponent(link.hash.slice(1)) === id;
        var item = link.closest("li");
        if (item) item.classList.toggle("is-active", active);
        if (active) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    }

    function update() {
      if (!headings.length) return;
      var marker = window.scrollY + 200;
      var current = headings[0];
      for (var i = 0; i < headings.length; i += 1) {
        if (headings[i].offsetTop <= marker) current = headings[i];
        else break;
      }
      var nearBottom =
        window.innerHeight + window.scrollY >=
        document.documentElement.scrollHeight - 80;
      if (nearBottom) current = headings[headings.length - 1];
      setActive(current.id);
    }

    var ticking = false;
    function onScroll() {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(function () {
        update();
        updateProgress(article);
        ticking = false;
      });
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    update();
  }

  /* --------------------------------------------------------- reading bar */
  function updateProgress(article) {
    var bar = $("#reading-progress-bar");
    if (!bar) return;
    var height = document.documentElement.scrollHeight - window.innerHeight;
    var ratio = height > 0 ? Math.min(1, Math.max(0, window.scrollY / height)) : 0;
    bar.style.width = (ratio * 100).toFixed(2) + "%";
  }

  /* --------------------------------------------------------- mobile drawer */
  function initTocDrawer() {
    var toggle = $(".toc-toggle");
    var drawer = $("#toc-drawer");
    if (!toggle || !drawer) return;
    var backdrop = $(".toc-backdrop");
    var close = $(".toc-close", drawer);
    var lastFocus = null;

    function open() {
      lastFocus = document.activeElement;
      document.body.classList.add("toc-open");
      toggle.setAttribute("aria-expanded", "true");
      if (backdrop) backdrop.hidden = false;
      if (close) close.focus();
    }

    function shut() {
      document.body.classList.remove("toc-open");
      toggle.setAttribute("aria-expanded", "false");
      if (backdrop) backdrop.hidden = true;
      if (lastFocus && lastFocus.focus) lastFocus.focus();
    }

    toggle.addEventListener("click", function () {
      if (document.body.classList.contains("toc-open")) shut();
      else open();
    });
    if (close) close.addEventListener("click", shut);
    if (backdrop) backdrop.addEventListener("click", shut);
    $$('a[href^="#"]', drawer).forEach(function (link) {
      link.addEventListener("click", function () {
        if (window.matchMedia("(max-width: 920px)").matches) shut();
      });
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && document.body.classList.contains("toc-open")) {
        shut();
      }
    });
    window.addEventListener("resize", function () {
      if (
        window.innerWidth > 920 &&
        document.body.classList.contains("toc-open")
      ) {
        shut();
      }
    });
  }

  /* -------------------------------------------------------------- font size */
  function applyFontSize(value, persist) {
    var valid = ["sm", "md", "lg"];
    if (valid.indexOf(value) < 0) value = "md";
    docEl.dataset.readingSize = value;
    $$("[data-font-size]").forEach(function (button) {
      button.classList.toggle("is-active", button.dataset.fontSize === value);
    });
    if (persist) safeStorageSet(FONT_KEY, value);
  }

  function initFontSize() {
    var saved = safeStorageGet(FONT_KEY);
    if (saved) applyFontSize(saved, false);
    $$("[data-font-size]").forEach(function (button) {
      button.addEventListener("click", function () {
        applyFontSize(button.dataset.fontSize, true);
      });
    });
  }

  /* --------------------------------------------------------------- copying */
  function initCopyButtons() {
    $$("[data-copy-target]").forEach(function (button) {
      button.addEventListener("click", function () {
        var target = document.getElementById(button.dataset.copyTarget);
        if (!target) return;
        copyText(target.innerText || target.textContent || "")
          .then(function () {
            toast("已复制到剪贴板");
          })
          .catch(function () {
            toast("复制失败，请手动选择");
          });
      });
    });
    $$("[data-copy-url]").forEach(function (button) {
      button.addEventListener("click", function () {
        copyText(window.location.href)
          .then(function () {
            toast("本页链接已复制");
          })
          .catch(function () {
            toast("复制失败");
          });
      });
    });
  }

  function initCodeCopy() {
    $$(".codehilite").forEach(function (block) {
      if ($(".code-block-head", block)) return;
      var pre = $("pre", block);
      if (!pre) return;
      var code = $("code", pre);
      var className = ((code && code.className) || block.className || "");
      var match = className.match(/language-([\w+#.-]+)/);
      var language = match ? match[1] : "";
      var header = document.createElement("div");
      header.className = "code-block-head";
      header.innerHTML =
        '<span class="code-lang">' +
        escapeHtml(language || "code") +
        "</span>" +
        '<button class="code-copy" type="button">' +
        '<svg class="icon" width="13" height="13" aria-hidden="true"><use href="#i-copy"></use></svg>' +
        "<span>复制</span></button>";
      block.insertBefore(header, block.firstChild);
      var button = $(".code-copy", header);
      button.addEventListener("click", function () {
        copyText(pre.innerText)
          .then(function () {
            button.classList.add("is-copied");
            $("span", button).textContent = "已复制";
            window.setTimeout(function () {
              button.classList.remove("is-copied");
              $("span", button).textContent = "复制";
            }, 1600);
          })
          .catch(function () {
            toast("复制失败，请手动选择");
          });
      });
    });
  }

  /* -------------------------------------------------------------- language */
  function initLanguageSwitch() {
    $$("[data-lang]").forEach(function (link) {
      link.addEventListener("click", function () {
        if (!window.location.hash || window.location.hash.length < 2) return;
        var href = link.getAttribute("href");
        if (href.indexOf("#") < 0) {
          link.setAttribute("href", href + window.location.hash);
        }
      });
    });
  }

  /* ------------------------------------------------------------ library UI */
  function normalize(value) {
    return String(value || "")
      .normalize("NFKC")
      .toLocaleLowerCase()
      .replace(/\s+/g, " ")
      .trim();
  }

  function initLibrary() {
    var dataNode = document.getElementById("paper-data");
    if (!dataNode) return;
    var papers;
    try {
      papers = JSON.parse(dataNode.textContent || "[]");
    } catch (error) {
      return;
    }
    var bySlug = {};
    papers.forEach(function (paper) {
      bySlug[paper.slug] = paper;
      paper._search = normalize(
        [
          paper.title,
          paper.title_zh,
          paper.short_title,
          paper.abstract,
          paper.abstract_zh,
          paper.summary,
          paper.authors.join(" "),
          paper.venue,
          paper.venue_zh,
          paper.tags.join(" "),
          paper.year
        ].join(" ")
      );
    });

    var input = $("#paper-search");
    var sort = $("#paper-sort");
    var count = $("#result-count");
    var clear = $("#clear-filters");
    var empty = $("#no-results");
    var emptyClear = $("#empty-clear");
    var grid = $(".paper-grid");
    var featured = $(".featured-card");
    var cards = $$(".paper-card");
    var tagButtons = $$(".tag-filter [data-tag]");
    var state = { query: "", tag: "", sort: "latest" };

    var params = new URLSearchParams(window.location.search);
    if (params.get("q")) state.query = params.get("q");
    if (params.get("tag")) state.tag = params.get("tag");
    if (input) input.value = state.query;

    function match(paper) {
      if (!paper) return false;
      if (state.tag && paper.tags.indexOf(state.tag) < 0) return false;
      if (!state.query) return true;
      var terms = normalize(state.query).split(" ").filter(Boolean);
      return terms.every(function (term) {
        return paper._search.indexOf(term) !== -1;
      });
    }

    function setCount(visible) {
      if (count) {
        if (!state.query && !state.tag) {
          count.textContent = "共 " + papers.length + " 篇论文";
        } else {
          count.textContent =
            "找到 " + visible + " 篇 / 共 " + papers.length + " 篇";
        }
      }
      if (clear) clear.hidden = !state.query && !state.tag;
      if (empty) empty.hidden = visible !== 0;
      if (grid) grid.hidden = visible === 0;
    }

    function apply() {
      var visible = 0;
      cards.forEach(function (card) {
        var paper = bySlug[card.dataset.slug];
        var show = match(paper);
        card.hidden = !show;
        if (show) visible += 1;
      });
      if (featured) {
        var featuredPaper = bySlug[featured.dataset.slug];
        var showFeatured = match(featuredPaper);
        featured.hidden = !showFeatured;
        if (showFeatured) visible += 1;
      }
      setCount(visible);
      tagButtons.forEach(function (button) {
        var active = (button.dataset.tag || "") === state.tag;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", active ? "true" : "false");
      });
    }

    function syncUrl() {
      var params = new URLSearchParams();
      if (state.query) params.set("q", state.query);
      if (state.tag) params.set("tag", state.tag);
      var query = params.toString();
      try {
        var url = window.location.pathname + (query ? "?" + query : "") + window.location.hash;
        window.history.replaceState(null, "", url);
      } catch (error) {
        /* file:// may reject history updates; filtering still works */
      }
    }

    function sortCards() {
      if (!grid) return;
      var nodes = cards.slice();
      nodes.sort(function (a, b) {
        var left = bySlug[a.dataset.slug];
        var right = bySlug[b.dataset.slug];
        if (!left || !right) return 0;
        if (state.sort === "oldest") return left.date.localeCompare(right.date);
        if (state.sort === "title") {
          return left.title_zh.localeCompare(right.title_zh, "zh-Hans-CN");
        }
        if (state.sort === "reading") {
          return left.reading_minutes - right.reading_minutes;
        }
        return right.date.localeCompare(left.date);
      });
      nodes.forEach(function (node) {
        grid.appendChild(node);
      });
    }

    var timer = null;
    if (input) {
      input.addEventListener("input", function () {
        window.clearTimeout(timer);
        timer = window.setTimeout(function () {
          state.query = input.value.trim();
          apply();
          syncUrl();
        }, 110);
      });
      input.addEventListener("keydown", function (event) {
        if (event.key === "Escape") {
          input.value = "";
          state.query = "";
          apply();
          syncUrl();
          input.blur();
        }
      });
    }

    tagButtons.forEach(function (button) {
      button.addEventListener("click", function () {
        state.tag = button.dataset.tag || "";
        apply();
        syncUrl();
      });
    });

    if (sort) {
      sort.addEventListener("change", function () {
        state.sort = sort.value;
        sortCards();
      });
    }

    function reset() {
      state.query = "";
      state.tag = "";
      if (input) input.value = "";
      apply();
      syncUrl();
      if (input) input.focus();
    }

    if (clear) clear.addEventListener("click", reset);
    if (emptyClear) emptyClear.addEventListener("click", reset);

    document.addEventListener("keydown", function (event) {
      var target = event.target;
      var typing =
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable);
      if (event.key === "/" && !typing && input) {
        event.preventDefault();
        input.focus();
        input.select();
      }
    });

    sortCards();
    apply();
  }

  /* ---------------------------------------------------------------- boot */
  function boot() {
    initTheme();
    initToc();
    initTocDrawer();
    initFontSize();
    initCopyButtons();
    initCodeCopy();
    initLanguageSwitch();
    initLightbox();
    renderMath();
    scheduleMermaid();
    initLibrary();

    document.addEventListener("themechange", function () {
      scheduleMermaid();
    });

    var article = $("#paper-body");
    if (article) updateProgress(article);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
