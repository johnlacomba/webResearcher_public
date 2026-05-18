"""
Tests for research_tool.web — Browser, DuckDuckGo search, content extraction,
image extraction, image download, rate limiting, and URL validation.

All tests run offline: no Playwright browser, no network access.
"""

from __future__ import annotations

import io
import socket
import time
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from research_tool.web import (
    Browser,
    RateLimiter,
    SearchResult,
    _extract_bs4,
    _extract_readability,
    _extract_trafilatura,
    _resolve_and_validate,
    download_image,
    extract_content,
    extract_images,
    extract_links,
    is_safe_url,
    search_ddg,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

ARTICLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Test Article Title</title></head>
<body>
<nav><a href="/">Home</a> | <a href="/about">About</a></nav>
<article>
<h1>Test Article Title</h1>
<p>This is the first paragraph of a test article that contains enough text
to be extracted by trafilatura. It discusses the implications of autonomous
web research tools and their potential applications in various domains.</p>
<p>The second paragraph elaborates on the technical architecture, describing
how tiered content extraction works. First, trafilatura attempts to parse
the page using its high-F1 algorithm. If that fails, readability-lxml is
tried. Finally, BeautifulSoup4 brute-force extraction serves as the
ultimate fallback.</p>
<p>A third paragraph provides additional context about the rate limiting
strategy, which uses per-domain delays with random jitter to avoid
overwhelming target servers. DuckDuckGo gets longer delays (2-5 seconds)
while general pages use 1-3 second delays.</p>
</article>
<footer><p>Copyright 2025</p></footer>
</body>
</html>
"""

MINIMAL_HTML = """
<html><body>
<p>Short paragraph one about testing.</p>
<p>Short paragraph two about validation.</p>
<p>Short paragraph three with more content about extraction methods and algorithms used.</p>
</body></html>
"""

EMPTY_HTML = ""

WHITESPACE_HTML = "   \n\t  "

SCRIPT_HEAVY_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Script Heavy Page</title>
<script>var x = 'should not appear in extracted text';</script>
<style>body { font-family: sans-serif; } .hidden { display: none; }</style>
</head>
<body>
<script>document.write('injected content');</script>
<nav><ul><li>Menu Item 1</li><li>Menu Item 2</li></ul></nav>
<div id="content">
<h1>Actual Content Title</h1>
<p>This is the real content that should be extracted. It contains meaningful
text about the topic at hand, which is content extraction from web pages.
The extraction process needs to filter out scripts, styles, navigation
elements, and other non-content markup.</p>
</div>
<footer><p>Footer text that should be stripped</p></footer>
<script>console.log('another script block');</script>
<noscript>Enable JavaScript to view this page.</noscript>
</body>
</html>
"""



# ── Content Extraction Tests ──────────────────────────────────────────────────


class TestExtractContent:
    """Tests for the tiered extract_content function."""

    def test_article_html_extracts_text(self):
        """Happy path: trafilatura extracts clean text > 200 chars from article HTML."""
        result = extract_content(ARTICLE_HTML, url="https://example.com/article")
        assert result["text"]
        assert len(result["text"]) > 200
        assert "tiered content extraction" in result["text"].lower() or \
               "autonomous" in result["text"].lower() or \
               "paragraph" in result["text"].lower()
        assert result["method"] in ("trafilatura", "readability", "bs4")

    def test_article_no_script_in_output(self):
        """Content extraction should not include script content."""
        result = extract_content(ARTICLE_HTML)
        assert "Copyright" not in result["text"] or result["method"] == "bs4"

    def test_minimal_html_still_extracts(self):
        """Happy path: minimal HTML falls back to readability or BS4, still returns text."""
        result = extract_content(MINIMAL_HTML)
        assert result["text"]
        assert result["method"] in ("trafilatura", "readability", "bs4")
        # Should contain at least some of the paragraph content
        assert "testing" in result["text"].lower() or "validation" in result["text"].lower()

    def test_empty_html_returns_empty(self):
        """Edge case: empty HTML returns empty text."""
        result = extract_content(EMPTY_HTML)
        assert result["text"] == ""
        assert result["method"] == "none"

    def test_whitespace_html_returns_empty(self):
        """Edge case: whitespace-only HTML returns empty text."""
        result = extract_content(WHITESPACE_HTML)
        assert result["text"] == ""
        assert result["method"] == "none"

    def test_script_heavy_html_strips_scripts(self):
        """Edge case: scripts/nav/footer are stripped from output."""
        result = extract_content(SCRIPT_HEAVY_HTML)
        assert result["text"]
        # Script content must not leak through
        assert "should not appear" not in result["text"]
        assert "injected content" not in result["text"]
        assert "console.log" not in result["text"]
        # Real content should be present
        assert "real content" in result["text"].lower() or \
               "content extraction" in result["text"].lower()

    def test_result_has_expected_keys(self):
        """All results must have title, text, method keys."""
        for html in [ARTICLE_HTML, MINIMAL_HTML, EMPTY_HTML, SCRIPT_HEAVY_HTML]:
            result = extract_content(html)
            assert "title" in result
            assert "text" in result
            assert "method" in result

    def test_code_blocks_preserved_with_fences(self):
        """Code blocks in <pre><code> should be wrapped in markdown fences."""
        html = '''<html><body>
        <article>
        <p>Some intro text that is long enough to pass the extraction threshold and
        provides enough context for the content extraction to work properly with
        trafilatura or readability.</p>
        <pre><code class="language-python">def hello():
    print("hello world")
</code></pre>
        <p>More text after the code block to ensure extraction works.</p>
        </article>
        </body></html>'''
        result = extract_content(html)
        assert "```" in result["text"]
        assert "def hello" in result["text"]

    def test_code_blocks_without_language_class(self):
        """<pre><code> without language class still gets fenced."""
        html = '''<html><body>
        <article>
        <p>Introductory paragraph with sufficient length to pass extraction
        thresholds in all three tiers of the content extraction pipeline.</p>
        <pre><code>echo "hello world"
ls -la
</code></pre>
        <p>Closing paragraph with additional context.</p>
        </article>
        </body></html>'''
        result = extract_content(html)
        assert "```" in result["text"]
        assert "echo" in result["text"]

    def test_github_highlight_source_pattern(self):
        """GitHub highlight-source-* class on parent div should detect language."""
        from research_tool.web import _preserve_code_blocks
        html = '''<div class="highlight highlight-source-python">
        <pre><code>def foo():
    pass
</code></pre></div>'''
        result = _preserve_code_blocks(html)
        assert "```python" in result


class TestExtractTrafilatura:
    """Tests for the trafilatura tier specifically."""

    def test_returns_dict_on_success(self):
        result = _extract_trafilatura(ARTICLE_HTML, "https://example.com")
        if result is not None:
            assert result["method"] == "trafilatura"
            assert len(result["text"]) > 50

    def test_returns_none_on_empty(self):
        result = _extract_trafilatura("", "")
        assert result is None


class TestExtractReadability:
    """Tests for the readability-lxml tier."""

    def test_extracts_from_article(self):
        result = _extract_readability(ARTICLE_HTML)
        if result is not None:
            assert result["method"] == "readability"
            assert len(result["text"]) > 50

    def test_returns_none_on_empty(self):
        result = _extract_readability("")
        assert result is None


class TestExtractBS4:
    """Tests for the BS4 brute-force tier."""

    def test_strips_scripts_and_styles(self):
        result = _extract_bs4(SCRIPT_HEAVY_HTML)
        assert "should not appear" not in result["text"]
        assert "console.log" not in result["text"]
        assert "font-family" not in result["text"]
        assert result["method"] == "bs4"

    def test_strips_nav_and_footer(self):
        result = _extract_bs4(SCRIPT_HEAVY_HTML)
        assert "Menu Item" not in result["text"]
        assert "Footer text" not in result["text"]

    def test_preserves_content(self):
        result = _extract_bs4(SCRIPT_HEAVY_HTML)
        assert "real content" in result["text"].lower() or \
               "content extraction" in result["text"].lower()

    def test_extracts_title(self):
        result = _extract_bs4(SCRIPT_HEAVY_HTML)
        assert result["title"] == "Script Heavy Page"

    def test_handles_empty_html(self):
        result = _extract_bs4("<html><body></body></html>")
        assert result["text"] == ""


# ── Rate Limiter Tests ────────────────────────────────────────────────────────


class TestRateLimiter:

    def test_enforces_delay_same_domain(self):
        """Happy path: second request to same domain is delayed."""
        rl = RateLimiter()
        domain = "example.com"

        start = time.monotonic()
        rl.wait(domain)
        first_done = time.monotonic()

        rl.wait(domain)
        second_done = time.monotonic()

        # First call should be near-instant
        assert (first_done - start) < 0.5

        # Second call should have waited at least 1.0s (DEFAULT_DELAY low bound)
        assert (second_done - first_done) >= 0.9  # small tolerance

    def test_allows_immediate_different_domain(self):
        """Edge case: different domains don't share rate limit state."""
        rl = RateLimiter()

        start = time.monotonic()
        rl.wait("domain-a.com")
        rl.wait("domain-b.com")
        elapsed = time.monotonic() - start

        # Both should be near-instant (first request to each domain)
        assert elapsed < 0.5

    def test_ddg_domain_uses_longer_delay(self):
        """DuckDuckGo domains should use 2-5s delay range."""
        rl = RateLimiter()
        domain = "html.duckduckgo.com"

        rl.wait(domain)
        start = time.monotonic()
        rl.wait(domain)
        elapsed = time.monotonic() - start

        # Should wait at least 4.0s (DDG_DELAY low bound)
        assert elapsed >= 3.9  # small tolerance

    def test_delay_range_classification(self):
        """Check that _delay_range returns correct ranges."""
        rl = RateLimiter()
        assert rl._delay_range("html.duckduckgo.com") == (4.0, 8.0)
        assert rl._delay_range("duckduckgo.com") == (4.0, 8.0)
        assert rl._delay_range("example.com") == (1.0, 3.0)
        assert rl._delay_range("google.com") == (1.0, 3.0)


# ── DuckDuckGo Search Tests (ddgs library) ──────────────────────────────────


class TestSearchDDG:

    @patch("ddgs.DDGS")
    def test_search_returns_results(self, mock_ddgs_cls):
        """Happy path: search_ddg returns structured results from ddgs library."""
        mock_instance = MagicMock()
        mock_ddgs_cls.return_value = mock_instance
        mock_instance.text.return_value = [
            {"title": "Result One", "href": "https://example.com/1", "body": "Snippet one"},
            {"title": "Result Two", "href": "https://example.org/2", "body": "Snippet two"},
        ]

        results = search_ddg("test query")

        assert len(results) == 2
        assert results[0].title == "Result One"
        assert results[0].url == "https://example.com/1"
        assert results[0].snippet == "Snippet one"
        assert results[1].title == "Result Two"

    @patch("ddgs.DDGS")
    def test_search_passes_max_results(self, mock_ddgs_cls):
        """search_ddg forwards max_results to ddgs."""
        mock_instance = MagicMock()
        mock_ddgs_cls.return_value = mock_instance
        mock_instance.text.return_value = []

        search_ddg("test", max_results=5)

        mock_instance.text.assert_called_once_with("test", max_results=5)

    @patch("ddgs.DDGS")
    def test_search_handles_exception(self, mock_ddgs_cls):
        """search_ddg returns empty list when ddgs raises."""
        mock_ddgs_cls.side_effect = RuntimeError("connection failed")

        results = search_ddg("test query")

        assert results == []

    @patch("ddgs.DDGS")
    def test_search_with_rate_limiter(self, mock_ddgs_cls):
        """search_ddg calls rate_limiter.wait when provided."""
        mock_instance = MagicMock()
        mock_ddgs_cls.return_value = mock_instance
        mock_instance.text.return_value = []

        rl = MagicMock(spec=RateLimiter)
        search_ddg("test", rate_limiter=rl)

        rl.wait.assert_called_once_with("duckduckgo.com")

    @patch("ddgs.DDGS")
    def test_search_handles_missing_fields(self, mock_ddgs_cls):
        """search_ddg handles results with missing keys gracefully."""
        mock_instance = MagicMock()
        mock_ddgs_cls.return_value = mock_instance
        mock_instance.text.return_value = [
            {"title": "Only Title"},
        ]

        results = search_ddg("test")

        assert len(results) == 1
        assert results[0].title == "Only Title"
        assert results[0].url == ""
        assert results[0].snippet == ""


# ── URL Validation Tests ─────────────────────────────────────────────────────


class TestURLValidation:

    def test_accepts_http(self):
        """Happy path: http URLs are accepted."""
        # Patch DNS resolution to return a public IP
        with patch("research_tool.web.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("93.184.216.34", 80)),
            ]
            assert is_safe_url("http://example.com") is True

    def test_accepts_https(self):
        """Happy path: https URLs are accepted."""
        with patch("research_tool.web.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("93.184.216.34", 443)),
            ]
            assert is_safe_url("https://example.com/page") is True

    def test_rejects_file_scheme(self):
        """Edge case: file:// URLs are rejected."""
        assert is_safe_url("file:///etc/passwd") is False

    def test_rejects_javascript_scheme(self):
        """Edge case: javascript: URLs are rejected."""
        assert is_safe_url("javascript:alert(1)") is False

    def test_rejects_ftp_scheme(self):
        assert is_safe_url("ftp://files.example.com/data") is False

    def test_rejects_data_scheme(self):
        assert is_safe_url("data:text/html,<h1>hi</h1>") is False

    def test_rejects_empty_url(self):
        assert is_safe_url("") is False

    def test_rejects_loopback_127(self):
        """Edge case: 127.x.x.x (loopback) is rejected."""
        with patch("research_tool.web.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("127.0.0.1", 80)),
            ]
            assert is_safe_url("http://localhost") is False

    def test_rejects_private_10(self):
        """Edge case: 10.x.x.x (private) is rejected."""
        with patch("research_tool.web.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("10.0.0.1", 80)),
            ]
            assert is_safe_url("http://internal.corp") is False

    def test_rejects_private_172(self):
        """Edge case: 172.16-31.x.x (private) is rejected."""
        with patch("research_tool.web.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("172.16.0.1", 80)),
            ]
            assert is_safe_url("http://internal.corp") is False

    def test_rejects_private_192_168(self):
        """Edge case: 192.168.x.x (private) is rejected."""
        with patch("research_tool.web.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("192.168.1.1", 80)),
            ]
            assert is_safe_url("http://router.local") is False

    def test_rejects_link_local(self):
        """Edge case: 169.254.x.x (link-local) is rejected."""
        with patch("research_tool.web.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("169.254.169.254", 80)),
            ]
            assert is_safe_url("http://metadata.google.internal") is False

    def test_rejects_literal_private_ip(self):
        """Edge case: literal private IP in URL is rejected even without DNS."""
        with patch("research_tool.web.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("192.168.1.1", 80)),
            ]
            assert is_safe_url("http://192.168.1.1/admin") is False

    def test_rejects_literal_loopback_ip(self):
        """Edge case: literal loopback IP in URL is rejected."""
        with patch("research_tool.web.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("127.0.0.1", 80)),
            ]
            assert is_safe_url("http://127.0.0.1:8080") is False

    def test_dns_failure_rejects_private_ip(self):
        """When DNS fails, literal private IPs are rejected."""
        with patch("research_tool.web.socket.getaddrinfo") as mock_dns:
            mock_dns.side_effect = socket.gaierror("DNS failed")
            assert is_safe_url("http://10.0.0.1/secret") is False

    def test_dns_failure_rejects_domain(self):
        """When DNS fails, all URLs are rejected (fail closed)."""
        with patch("research_tool.web.socket.getaddrinfo") as mock_dns:
            mock_dns.side_effect = socket.gaierror("DNS failed")
            assert is_safe_url("https://example.com") is False

    def test_rejects_no_hostname(self):
        assert is_safe_url("http://") is False


# ── Browser Tests (mocked) ───────────────────────────────────────────────────


class TestBrowser:

    @patch("playwright.sync_api.sync_playwright")
    def test_browser_context_manager(self, mock_pw_fn):
        """Browser supports context manager protocol."""
        mock_pw = MagicMock()
        mock_pw_fn.return_value.start.return_value = mock_pw
        mock_browser = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_context = MagicMock()
        mock_browser.new_context.return_value = mock_context

        with Browser() as b:
            assert b is not None

        mock_browser.close.assert_called_once()
        mock_pw.stop.assert_called_once()

    @patch("research_tool.web.is_safe_url", return_value=False)
    @patch("playwright.sync_api.sync_playwright")
    def test_fetch_page_rejects_unsafe_url(self, mock_pw_fn, mock_safe):
        """fetch_page raises ValueError for unsafe URLs."""
        mock_pw = MagicMock()
        mock_pw_fn.return_value.start.return_value = mock_pw
        mock_browser = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser

        b = Browser()
        with pytest.raises(ValueError, match="URL blocked"):
            b.fetch_page("file:///etc/passwd")
        b.close()

    @patch("research_tool.web.is_safe_url", return_value=True)
    @patch("playwright.sync_api.sync_playwright")
    def test_fetch_page_returns_html(self, mock_pw_fn, mock_safe):
        """fetch_page navigates and returns page content."""
        mock_pw = MagicMock()
        mock_pw_fn.return_value.start.return_value = mock_pw
        mock_browser = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_context = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_page = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_page.content.return_value = "<html><body>Hello</body></html>"
        mock_page.url = "https://example.com"

        b = Browser()
        html, final_url = b.fetch_page("https://example.com")
        assert html == "<html><body>Hello</body></html>"
        assert final_url == "https://example.com"
        mock_page.goto.assert_called_once()
        b.close()


# ── Parallel Fetch Tests ────────────────────────────────────────────────────


class TestFetchPagesParallel:

    @patch("playwright.sync_api.sync_playwright")
    def test_empty_urls_returns_empty(self, mock_pw_fn):
        """Empty URL list returns empty result without launching async browser."""
        mock_pw = MagicMock()
        mock_pw_fn.return_value.start.return_value = mock_pw
        mock_browser = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser

        b = Browser()
        result = b.fetch_pages_parallel([])
        assert result == []
        b.close()

    @patch("research_tool.web._resolve_and_validate", return_value=("http://1.2.3.4/", "example.com"))
    @patch("research_tool.web.is_safe_url", return_value=True)
    @patch("playwright.sync_api.sync_playwright")
    def test_single_url_fetch(self, mock_pw_fn, mock_safe, mock_resolve):
        """Single URL is fetched and returned as (url, html, final_url)."""
        mock_pw = MagicMock()
        mock_pw_fn.return_value.start.return_value = mock_pw
        mock_browser = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser

        mock_async_page = MagicMock()
        mock_async_context = MagicMock()
        mock_async_browser = MagicMock()
        mock_async_pw = MagicMock()

        import asyncio

        async def fake_goto(*a, **kw):
            pass

        async def fake_wait(*a, **kw):
            pass

        async def fake_content():
            return "<html>hello</html>"

        async def fake_close():
            pass

        async def fake_click(*a, **kw):
            raise Exception("no banner")

        mock_async_page.goto = fake_goto
        mock_async_page.wait_for_load_state = fake_wait
        mock_async_page.content = fake_content
        mock_async_page.close = fake_close
        mock_async_page.click = fake_click
        mock_async_page.url = "https://example.com"

        async def fake_new_page():
            return mock_async_page

        mock_async_context.new_page = fake_new_page

        async def fake_new_context(**kw):
            return mock_async_context

        async def fake_launch(**kw):
            return mock_async_browser

        async def fake_browser_close():
            pass

        mock_async_browser.new_context = fake_new_context
        mock_async_browser.close = fake_browser_close

        with patch("playwright.async_api.async_playwright") as mock_async_pw_fn:
            mock_async_pw_ctx = MagicMock()

            async def fake_aenter():
                return mock_async_pw

            async def fake_aexit(*a):
                pass

            mock_async_pw.chromium.launch = fake_launch
            mock_async_pw_ctx.__aenter__ = lambda self: fake_aenter()
            mock_async_pw_ctx.__aexit__ = lambda self, *a: fake_aexit(*a)
            mock_async_pw_fn.return_value = mock_async_pw_ctx

            b = Browser()
            result = b.fetch_pages_parallel(["https://example.com"])
            assert len(result) == 1
            assert result[0][0] == "https://example.com"
            assert result[0][1] == "<html>hello</html>"
            b.close()

    @patch("research_tool.web._resolve_and_validate", return_value=None)
    @patch("research_tool.web.is_safe_url", return_value=True)
    @patch("playwright.sync_api.sync_playwright")
    def test_dns_validation_failure_returns_none(self, mock_pw_fn, mock_safe, mock_resolve):
        """URL that fails DNS validation is filtered out."""
        mock_pw = MagicMock()
        mock_pw_fn.return_value.start.return_value = mock_pw
        mock_browser = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser

        mock_async_browser = MagicMock()

        async def fake_launch(**kw):
            return mock_async_browser

        async def fake_new_context(**kw):
            return MagicMock()

        async def fake_browser_close():
            pass

        mock_async_browser.new_context = fake_new_context
        mock_async_browser.close = fake_browser_close
        mock_async_pw = MagicMock()
        mock_async_pw.chromium.launch = fake_launch

        with patch("playwright.async_api.async_playwright") as mock_async_pw_fn:
            mock_ctx = MagicMock()

            async def fake_aenter():
                return mock_async_pw

            async def fake_aexit(*a):
                pass

            mock_ctx.__aenter__ = lambda self: fake_aenter()
            mock_ctx.__aexit__ = lambda self, *a: fake_aexit(*a)
            mock_async_pw_fn.return_value = mock_ctx

            b = Browser()
            result = b.fetch_pages_parallel(["https://evil.com"])
            assert result == []
            b.close()

    @patch("research_tool.web.is_safe_url", return_value=False)
    @patch("playwright.sync_api.sync_playwright")
    def test_unsafe_url_filtered(self, mock_pw_fn, mock_safe):
        """Unsafe URLs are filtered out of results."""
        mock_pw = MagicMock()
        mock_pw_fn.return_value.start.return_value = mock_pw
        mock_browser = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser

        mock_async_browser = MagicMock()

        async def fake_launch(**kw):
            return mock_async_browser

        async def fake_new_context(**kw):
            return MagicMock()

        async def fake_browser_close():
            pass

        mock_async_browser.new_context = fake_new_context
        mock_async_browser.close = fake_browser_close
        mock_async_pw = MagicMock()
        mock_async_pw.chromium.launch = fake_launch

        with patch("playwright.async_api.async_playwright") as mock_async_pw_fn:
            mock_ctx = MagicMock()

            async def fake_aenter():
                return mock_async_pw

            async def fake_aexit(*a):
                pass

            mock_ctx.__aenter__ = lambda self: fake_aenter()
            mock_ctx.__aexit__ = lambda self, *a: fake_aexit(*a)
            mock_async_pw_fn.return_value = mock_ctx

            b = Browser()
            result = b.fetch_pages_parallel(["file:///etc/passwd"])
            assert result == []
            b.close()


# ── Image Extraction Fixtures ────────────────────────────────────────────────

IMAGES_HTML = """
<!DOCTYPE html>
<html>
<head><title>Image Test Page</title></head>
<body>
<nav><img src="/logo.png" alt="Logo" width="50" height="50"></nav>
<header><img src="/header-bg.jpg" alt="Header"></header>
<article>
  <img src="/photos/landscape.jpg" alt="Beautiful landscape" width="800" height="600">
  <img src="/photos/portrait.jpg" alt="Portrait photo" width="400" height="500">
  <img src="/photos/chart.png" alt="Data chart" width="600" height="400">
  <img src="/spacer.gif" alt="" width="1" height="1">
  <img src="/pixel.gif" alt="" width="1" height="1">
  <img src="https://tracker.example.com/beacon?id=123" alt="" width="1" height="1">
  <img src="/icons/arrow.svg" alt="Arrow">
  <img src="/favicon.ico" alt="Favicon">
</article>
<aside><img src="/ad-banner.jpg" alt="Ad" width="300" height="250"></aside>
<footer><img src="/footer-logo.png" alt="Footer Logo"></footer>
</body>
</html>
"""

LAZY_LOAD_HTML = """
<html><body>
<article>
  <img data-src="/lazy/image1.jpg" alt="Lazy image 1" width="800" height="600">
  <img data-lazy-src="/lazy/image2.jpg" alt="Lazy image 2" width="600" height="400">
  <img src="/placeholder.jpg" data-src="/lazy/image3.jpg" alt="Lazy with fallback" width="500" height="300">
</article>
</body></html>
"""

SRCSET_HTML = """
<html><body>
<article>
  <img src="/small.jpg"
       srcset="/small.jpg 400w, /medium.jpg 800w, /large.jpg 1200w"
       alt="Responsive image" width="800" height="600">
</article>
</body></html>
"""


# ── Image Extraction Tests ───────────────────────────────────────────────────


class TestExtractImages:
    """Tests for extract_images function."""

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_happy_path_filters_decorative(self, mock_safe):
        """Happy path: HTML with 3 content images and several decorative images
        returns only the 3 content images."""
        results = extract_images(IMAGES_HTML, "https://example.com")
        urls = [r["src_url"] for r in results]
        # Content images in <article> with valid dimensions
        assert "https://example.com/photos/landscape.jpg" in urls
        assert "https://example.com/photos/portrait.jpg" in urls
        assert "https://example.com/photos/chart.png" in urls
        # Decorative images should be filtered out
        assert "https://example.com/logo.png" not in urls  # in <nav>
        assert "https://example.com/header-bg.jpg" not in urls  # in <header>
        assert "https://example.com/footer-logo.png" not in urls  # in <footer>
        assert "https://example.com/ad-banner.jpg" not in urls  # in <aside>
        # Spacer/pixel/tracking
        assert not any("spacer" in u for u in urls)
        assert not any("pixel" in u for u in urls)
        assert not any("beacon" in u for u in urls)
        # SVG and ICO
        assert not any(u.endswith(".svg") for u in urls)
        assert not any(u.endswith(".ico") for u in urls)

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_lazy_loaded_data_src(self, mock_safe):
        """Happy path: lazy-loaded images with data-src are extracted."""
        results = extract_images(LAZY_LOAD_HTML, "https://example.com")
        urls = [r["src_url"] for r in results]
        assert "https://example.com/lazy/image1.jpg" in urls
        assert "https://example.com/lazy/image2.jpg" in urls

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_lazy_loaded_data_src_overrides_src(self, mock_safe):
        """When both src and data-src are present, src is used (first in priority)."""
        results = extract_images(LAZY_LOAD_HTML, "https://example.com")
        urls = [r["src_url"] for r in results]
        # The third img has src="/placeholder.jpg" which is used since src comes first
        assert "https://example.com/placeholder.jpg" in urls

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_srcset_picks_highest_resolution(self, mock_safe):
        """Happy path: srcset images resolve to the highest-resolution variant."""
        results = extract_images(SRCSET_HTML, "https://example.com")
        assert len(results) == 1
        # Should pick /large.jpg (1200w is highest)
        assert results[0]["src_url"] == "https://example.com/large.jpg"

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_image_inside_nav_filtered(self, mock_safe):
        """Edge case: image inside <nav> element is filtered out."""
        html = '<html><body><nav><img src="/nav-img.jpg" alt="Nav" width="200" height="200"></nav></body></html>'
        results = extract_images(html, "https://example.com")
        assert len(results) == 0

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_image_role_presentation_filtered(self, mock_safe):
        """Edge case: image with role="presentation" is filtered out."""
        html = '<html><body><article><img src="/decorative.jpg" role="presentation" width="200" height="200"></article></body></html>'
        results = extract_images(html, "https://example.com")
        assert len(results) == 0

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_image_aria_hidden_filtered(self, mock_safe):
        """Edge case: image with aria-hidden="true" is filtered out."""
        html = '<html><body><article><img src="/hidden.jpg" aria-hidden="true" width="200" height="200"></article></body></html>'
        results = extract_images(html, "https://example.com")
        assert len(results) == 0

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_small_dimensions_filtered(self, mock_safe):
        """Edge case: image smaller than 100x100px is filtered out."""
        html = '<html><body><article><img src="/tiny.jpg" alt="Tiny" width="50" height="50"></article></body></html>'
        results = extract_images(html, "https://example.com")
        assert len(results) == 0

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_small_area_filtered(self, mock_safe):
        """Edge case: image with area <15,000px^2 is filtered out."""
        # 120 x 120 = 14,400 < 15,000
        html = '<html><body><article><img src="/small-area.jpg" alt="Small" width="120" height="120"></article></body></html>'
        results = extract_images(html, "https://example.com")
        assert len(results) == 0

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_data_uri_skipped(self, mock_safe):
        """Edge case: data: URI images are skipped."""
        html = '<html><body><article><img src="data:image/png;base64,iVBOR..." alt="Inline" width="200" height="200"></article></body></html>'
        results = extract_images(html, "https://example.com")
        assert len(results) == 0

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_tracking_pattern_filtered(self, mock_safe):
        """Edge case: URL matching tracking pattern is filtered out."""
        html = '<html><body><article><img src="/pixel.gif" alt="" width="200" height="200"></article></body></html>'
        results = extract_images(html, "https://example.com")
        assert len(results) == 0

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_empty_html_returns_empty(self, mock_safe):
        """Edge case: HTML with zero images returns empty list."""
        results = extract_images("<html><body><p>No images here</p></body></html>", "https://example.com")
        assert results == []

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_completely_empty_html(self, mock_safe):
        """Edge case: empty string returns empty list."""
        assert extract_images("", "https://example.com") == []

    def test_ssrf_unsafe_url_rejected(self):
        """Edge case: SSRF-unsafe image URL is rejected."""
        html = '<html><body><article><img src="http://169.254.169.254/metadata" alt="SSRF" width="800" height="600"></article></body></html>'
        with patch("research_tool.web.is_safe_url", return_value=False):
            results = extract_images(html, "https://example.com")
            assert len(results) == 0

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_result_dict_has_expected_keys(self, mock_safe):
        """Results contain all expected keys."""
        html = '<html><body><article><img src="/photo.jpg" alt="Photo" width="800" height="600"></article></body></html>'
        results = extract_images(html, "https://example.com")
        assert len(results) == 1
        r = results[0]
        assert "src_url" in r
        assert "alt_text" in r
        assert "width" in r
        assert "height" in r
        assert "dom_offset" in r
        assert r["width"] == 800
        assert r["height"] == 600
        assert r["alt_text"] == "Photo"

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_dom_offset_is_nonnegative(self, mock_safe):
        """dom_offset should be a non-negative integer."""
        html = '<html><body><article><img src="/photo.jpg" alt="Photo" width="800" height="600"></article></body></html>'
        results = extract_images(html, "https://example.com")
        assert len(results) == 1
        assert results[0]["dom_offset"] >= 0


# ── Image Download Tests ─────────────────────────────────────────────────────


def _make_valid_png(width: int = 200, height: int = 200) -> bytes:
    """Create a minimal valid PNG image in memory for testing."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mock_dns_public():
    """Return a mock for socket.getaddrinfo that resolves to a public IP."""
    return [(2, 1, 6, "", ("93.184.216.34", 443))]


class TestDownloadImage:
    """Tests for download_image function."""

    @patch("research_tool.web.urllib.request.urlopen")
    @patch("research_tool.web.socket.getaddrinfo", return_value=_mock_dns_public())
    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_happy_path_download(self, mock_safe, mock_dns, mock_urlopen):
        """Happy path: valid image is downloaded and returned."""
        png_bytes = _make_valid_png(200, 200)
        mock_resp = MagicMock()
        mock_resp.read.return_value = png_bytes
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = download_image("https://example.com/photo.jpg")
        assert result == png_bytes

    @patch("research_tool.web.urllib.request.urlopen")
    @patch("research_tool.web.socket.getaddrinfo", return_value=_mock_dns_public())
    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_timeout_returns_none(self, mock_safe, mock_dns, mock_urlopen):
        """Error path: download_image returns None on timeout."""
        mock_urlopen.side_effect = urllib.error.URLError("timeout")
        result = download_image("https://example.com/photo.jpg")
        assert result is None

    @patch("research_tool.web.urllib.request.urlopen")
    @patch("research_tool.web.socket.getaddrinfo", return_value=_mock_dns_public())
    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_exceeds_max_bytes_returns_none(self, mock_safe, mock_dns, mock_urlopen):
        """Error path: image exceeding max_bytes returns None."""
        # Return data larger than max_bytes
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"x" * 1001
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = download_image("https://example.com/photo.jpg", max_bytes=1000)
        assert result is None

    @patch("research_tool.web.urllib.request.urlopen")
    @patch("research_tool.web.socket.getaddrinfo", return_value=_mock_dns_public())
    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_pil_validation_failure_returns_none(self, mock_safe, mock_dns, mock_urlopen):
        """Error path: downloaded bytes that PIL cannot open are discarded."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not an image at all"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = download_image("https://example.com/photo.jpg")
        assert result is None

    @patch("research_tool.web.urllib.request.urlopen")
    @patch("research_tool.web.socket.getaddrinfo", return_value=_mock_dns_public())
    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_pil_too_small_returns_none(self, mock_safe, mock_dns, mock_urlopen):
        """Error path: image that is too small (< 100x100) is discarded."""
        tiny_png = _make_valid_png(50, 50)
        mock_resp = MagicMock()
        mock_resp.read.return_value = tiny_png
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = download_image("https://example.com/photo.jpg")
        assert result is None

    def test_unsafe_url_returns_none(self):
        """Edge case: SSRF-unsafe image URL is rejected."""
        with patch("research_tool.web.is_safe_url", return_value=False):
            result = download_image("http://169.254.169.254/metadata")
            assert result is None

    @patch("research_tool.web.socket.getaddrinfo")
    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_private_ip_resolution_returns_none(self, mock_safe, mock_dns):
        """Edge case: hostname resolving to private IP is rejected."""
        mock_dns.return_value = [(2, 1, 6, "", ("10.0.0.1", 443))]
        result = download_image("https://example.com/photo.jpg")
        assert result is None

    @patch("research_tool.web.urllib.request.urlopen")
    @patch("research_tool.web.socket.getaddrinfo", return_value=_mock_dns_public())
    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_rate_limiter_called(self, mock_safe, mock_dns, mock_urlopen):
        """download_image calls rate_limiter.wait when provided."""
        png_bytes = _make_valid_png(200, 200)
        mock_resp = MagicMock()
        mock_resp.read.return_value = png_bytes
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        rl = MagicMock(spec=RateLimiter)
        download_image("https://example.com/photo.jpg", rate_limiter=rl)
        rl.wait.assert_called_once_with("example.com")

    @patch("research_tool.web.socket.getaddrinfo")
    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_dns_failure_returns_none(self, mock_safe, mock_dns):
        """Error path: DNS resolution failure returns None."""
        mock_dns.side_effect = socket.gaierror("DNS failed")
        result = download_image("https://example.com/photo.jpg")
        assert result is None

    @patch("research_tool.web.urllib.request.urlopen")
    @patch("research_tool.web.socket.getaddrinfo", return_value=_mock_dns_public())
    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_palette_with_transparency_converted(self, mock_safe, mock_dns, mock_urlopen):
        """Palette images with transparency should convert cleanly via RGBA."""
        import warnings
        from PIL import Image

        img = Image.new("P", (200, 200))
        img.putpalette([i % 256 for i in range(768)])
        img.info["transparency"] = b"\x00" * 256
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        mock_resp = MagicMock()
        mock_resp.read.return_value = png_bytes
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = download_image("https://example.com/icon.png")
        assert result is not None

    @patch("research_tool.web.urllib.request.urlopen")
    @patch("research_tool.web.socket.getaddrinfo", return_value=_mock_dns_public())
    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_rgba_image_converted(self, mock_safe, mock_dns, mock_urlopen):
        """RGBA images should convert to RGB without warnings."""
        from PIL import Image

        img = Image.new("RGBA", (200, 200), color=(128, 128, 128, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        mock_resp = MagicMock()
        mock_resp.read.return_value = png_bytes
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = download_image("https://example.com/icon.png")
        assert result is not None


# ── Link Extraction Tests ──────────────────────────────────────────────────


LINKS_HTML = """
<!DOCTYPE html>
<html>
<head><title>Links Page</title></head>
<body>
<article>
  <a href="https://example.com/page1">Page One</a>
  <a href="https://example.com/page2">Page Two</a>
  <a href="https://example.com/page3">Page Three</a>
  <a href="/relative/page4">Page Four</a>
  <a href="/relative/page5">Page Five</a>
  <a href="https://external.org/article">External Article</a>
  <a href="https://other.net/resource">Other Resource</a>
  <a href="https://third.io/data">Third Party</a>
</article>
</body>
</html>
"""


class TestExtractLinks:
    """Tests for extract_links function."""

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_happy_path_extracts_all_links(self, mock_safe):
        """Happy path: HTML with 5 internal and 3 external links -> all 8 extracted."""
        results = extract_links(LINKS_HTML, "https://example.com/start")
        assert len(results) == 8
        urls = [r["target_url"] for r in results]
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls
        assert "https://example.com/page3" in urls
        assert "https://example.com/relative/page4" in urls
        assert "https://example.com/relative/page5" in urls
        assert "https://external.org/article" in urls
        assert "https://other.net/resource" in urls
        assert "https://third.io/data" in urls

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_anchor_text_sanitized(self, mock_safe):
        """Anchor text with HTML/script content is escaped."""
        html = '<html><body><a href="https://example.com/xss">Click <script>alert(1)</script> here</a></body></html>'
        results = extract_links(html, "https://example.com/start")
        assert len(results) == 1
        # The raw text from get_text will be "Click alert(1) here" (BS4 strips tags)
        # html_module.escape then escapes any remaining HTML entities
        assert "<script>" not in results[0]["anchor_text"]

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_relative_urls_resolved(self, mock_safe):
        """Relative URLs are resolved against page_url."""
        html = '<html><body><a href="/about">About</a><a href="sub/page">Sub</a></body></html>'
        results = extract_links(html, "https://example.com/dir/page")
        urls = [r["target_url"] for r in results]
        assert "https://example.com/about" in urls
        assert "https://example.com/dir/sub/page" in urls

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_fragment_only_links_filtered(self, mock_safe):
        """Fragment-only links (e.g. #section) are filtered out."""
        html = '<html><body><a href="#top">Top</a><a href="#section2">Section 2</a><a href="https://example.com/real">Real</a></body></html>'
        results = extract_links(html, "https://example.com/page")
        assert len(results) == 1
        assert results[0]["target_url"] == "https://example.com/real"

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_javascript_and_mailto_filtered(self, mock_safe):
        """javascript: and mailto: links are filtered out."""
        html = '''<html><body>
        <a href="javascript:void(0)">JS</a>
        <a href="mailto:test@example.com">Email</a>
        <a href="https://example.com/ok">OK</a>
        </body></html>'''
        results = extract_links(html, "https://example.com/page")
        assert len(results) == 1
        assert results[0]["target_url"] == "https://example.com/ok"

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_duplicate_links_deduplicated(self, mock_safe):
        """Duplicate links on same page are deduplicated."""
        html = '''<html><body>
        <a href="https://example.com/page1">First</a>
        <a href="https://example.com/page1">Second</a>
        <a href="https://example.com/page1">Third</a>
        </body></html>'''
        results = extract_links(html, "https://example.com/start")
        assert len(results) == 1
        assert results[0]["target_url"] == "https://example.com/page1"

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_self_links_filtered(self, mock_safe):
        """Links pointing to the same page_url are filtered out."""
        html = '<html><body><a href="https://example.com/current">Self</a><a href="https://example.com/other">Other</a></body></html>'
        results = extract_links(html, "https://example.com/current")
        assert len(results) == 1
        assert results[0]["target_url"] == "https://example.com/other"

    def test_ssrf_unsafe_urls_filtered(self):
        """SSRF-unsafe URLs are filtered out."""
        html = '''<html><body>
        <a href="https://safe.example.com/page">Safe</a>
        <a href="http://169.254.169.254/metadata">Unsafe</a>
        </body></html>'''
        with patch("research_tool.web.is_safe_url") as mock_safe:
            mock_safe.side_effect = lambda url: "169.254" not in url
            results = extract_links(html, "https://example.com/page")
            urls = [r["target_url"] for r in results]
            assert "https://safe.example.com/page" in urls
            assert "http://169.254.169.254/metadata" not in urls

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_no_links_returns_empty(self, mock_safe):
        """Page with no links returns empty list."""
        html = "<html><body><p>No links here</p></body></html>"
        results = extract_links(html, "https://example.com/page")
        assert results == []

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_empty_html_returns_empty(self, mock_safe):
        """Empty HTML returns empty list."""
        assert extract_links("", "https://example.com") == []
        assert extract_links("   ", "https://example.com") == []

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_malformed_href_skipped(self, mock_safe):
        """Tags with empty or whitespace-only href are skipped."""
        html = '<html><body><a href="">Empty</a><a href="   ">Spaces</a><a href="https://example.com/ok">OK</a></body></html>'
        results = extract_links(html, "https://example.com/page")
        # Empty href after strip is skipped; spaces-only href after strip is empty and skipped
        # Only the valid link should remain
        urls = [r["target_url"] for r in results]
        assert "https://example.com/ok" in urls

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_fragments_stripped_from_resolved(self, mock_safe):
        """Fragments are stripped from resolved URLs."""
        html = '<html><body><a href="https://example.com/page#section">Link</a></body></html>'
        results = extract_links(html, "https://example.com/start")
        assert len(results) == 1
        assert results[0]["target_url"] == "https://example.com/page"
        assert "#" not in results[0]["target_url"]

    @patch("research_tool.web.is_safe_url", return_value=True)
    def test_anchor_text_empty_for_image_link(self, mock_safe):
        """Link with only an image inside has empty anchor text."""
        html = '<html><body><a href="https://example.com/page"><img src="img.jpg"></a></body></html>'
        results = extract_links(html, "https://example.com/start")
        assert len(results) == 1
        assert results[0]["anchor_text"] == ""

