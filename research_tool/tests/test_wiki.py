"""Tests for research_tool.wiki — BFS wiki crawler."""

import asyncio
import hashlib
import os
import stat
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from research_tool.wiki import (
    AsyncRateLimiter,
    WikiCrawler,
    build_auth_headers,
    canonicalize_url,
    classify_url,
    load_auth_config,
    match_credentials,
    merge_cli_auth,
    _is_repo_url,
    _needs_playwright_fallback,
    _repo_prefix,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

SEED_URL = "https://docs.example.com/start"
SEED_NETLOC = "docs.example.com"


def _make_html(title: str, body: str, links: list[str] | None = None) -> str:
    """Build a minimal HTML page with optional link tags."""
    link_tags = ""
    if links:
        link_tags = "\n".join(f'<a href="{u}">{u}</a>' for u in links)
    return f"<html><head><title>{title}</title></head><body><h1>{title}</h1><p>{body}</p>{link_tags}</body></html>"


def _make_response(status_code: int, text: str, url: str) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.url = url
    return resp


# ── Test: canonicalize_url ───────────────────────────────────────────────────


class TestCanonicalizeUrl:
    """canonicalize_url strips fragments, sorts query params, lowercases scheme/host,
    and normalizes trailing slashes."""

    def test_strip_fragment(self):
        result = canonicalize_url("https://example.com/page#section")
        assert "#" not in result
        assert result == "https://example.com/page"

    def test_sort_query_params(self):
        result = canonicalize_url("https://example.com/page?z=1&a=2")
        assert result == "https://example.com/page?a=2&z=1"

    def test_lowercase_scheme_and_host(self):
        result = canonicalize_url("HTTPS://EXAMPLE.COM/Path")
        assert result == "https://example.com/Path"

    def test_strip_trailing_slash(self):
        result = canonicalize_url("https://example.com/page/")
        assert result == "https://example.com/page"

    def test_keep_root_trailing_slash(self):
        result = canonicalize_url("https://example.com/")
        assert result == "https://example.com/"

    def test_no_query_no_fragment(self):
        result = canonicalize_url("https://example.com/page")
        assert result == "https://example.com/page"

    def test_percent_encoding_normalized(self):
        # %2f should remain encoded (it's a slash), but hex digits uppercase
        result = canonicalize_url("https://example.com/path%20to%20page")
        assert "%20" in result or " " not in result

    def test_combined(self):
        result = canonicalize_url("HTTP://Example.COM/docs/page/#section?b=2&a=1")
        # Fragment stripped, scheme/host lowered, params sorted
        assert result.startswith("http://example.com/")
        assert "#" not in result


# ── Test: classify_url ───────────────────────────────────────────────────────


class TestClassifyUrl:
    """classify_url identifies same-domain, external, github repo, and bitbucket repo."""

    def test_same_domain(self):
        assert classify_url("https://docs.example.com/page", "docs.example.com") == "same-domain"

    def test_external(self):
        assert classify_url("https://other.com/page", "docs.example.com") == "external"

    def test_github_repo(self):
        assert classify_url("https://github.com/org/repo", "docs.example.com") == "repo"

    def test_github_repo_with_path(self):
        assert classify_url("https://github.com/org/repo/tree/main", "docs.example.com") == "repo"

    def test_bitbucket_repo(self):
        assert classify_url("https://bitbucket.org/team/project", "docs.example.com") == "repo"

    def test_github_non_repo(self):
        # Just github.com without org/repo is external
        assert classify_url("https://github.com/", "docs.example.com") == "external"

    def test_case_insensitive_netloc(self):
        assert classify_url("https://DOCS.EXAMPLE.COM/page", "docs.example.com") == "same-domain"


# ── Test: BFS same-domain and external ───────────────────────────────────────


class TestBFSSameDomainAndExternal:
    """Mock 5 same-domain + 2 external pages, verify all same-domain visited,
    external links not followed."""

    def test_bfs_same_domain_and_external(self):
        asyncio.run(self._run())

    async def _run(self):
        # Page map: seed links to p1, p2; p1 links to p3, ext1; p2 links to p4, ext2
        pages = {
            "https://docs.example.com/start": _make_html(
                "Start", "Welcome",
                ["https://docs.example.com/p1", "https://docs.example.com/p2"],
            ),
            "https://docs.example.com/p1": _make_html(
                "Page 1", "Content 1",
                ["https://docs.example.com/p3", "https://external.com/ext1"],
            ),
            "https://docs.example.com/p2": _make_html(
                "Page 2", "Content 2",
                ["https://docs.example.com/p4", "https://external2.com/ext2"],
            ),
            "https://docs.example.com/p3": _make_html("Page 3", "Content 3"),
            "https://docs.example.com/p4": _make_html("Page 4", "Content 4"),
        }

        visited_urls = []

        def mock_extract_links(html, page_url):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                links.append({"target_url": a["href"], "anchor_text": a.get_text()})
            return links

        def mock_extract_content(html, url=""):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.string if soup.title else ""
            text = soup.get_text()
            return {"title": title, "text": text, "method": "mock"}

        async def mock_get(url, **kwargs):
            visited_urls.append(str(url))
            html = pages.get(str(url), "<html><body>Not found</body></html>")
            resp = _make_response(200, html, str(url))
            return resp

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("research_tool.wiki.is_safe_url", return_value=True), \
             patch("research_tool.wiki.extract_content", side_effect=mock_extract_content), \
             patch("research_tool.wiki.extract_links", side_effect=mock_extract_links), \
             patch("research_tool.wiki.chunk_wiki_content", return_value=[]), \
             patch("research_tool.wiki._make_embedding", return_value=None), \
             patch("research_tool.wiki.httpx.AsyncClient", return_value=mock_client), \
             patch("research_tool.wiki.ResearchStore") as MockStore:

            store_inst = MockStore.return_value
            store_inst.get_content_hash.return_value = None
            store_inst.store_page = MagicMock()
            store_inst.store_chunks = MagicMock()
            store_inst.store_links = MagicMock()
            store_inst.mark_domain_pages_stale = MagicMock()
            store_inst.get_stale_pages = MagicMock(return_value=[])
            store_inst.mark_page_active = MagicMock()

            crawler = WikiCrawler()
            stats = await crawler.crawl(
                seed_url=SEED_URL,
                db_path=":memory:",
                concurrency=2,
                progress_callback=lambda s: None,
            )

        # All 5 same-domain pages should have been visited
        same_domain_visited = [u for u in visited_urls if "docs.example.com" in u]
        assert len(same_domain_visited) == 5

        # External pages should NOT have been fetched
        external_visited = [u for u in visited_urls if "external" in u]
        assert len(external_visited) == 0


# ── Test: unchanged content hash skipped ──────────────────────────────────────


class TestUnchangedContentHashSkipped:
    """Page with same hash as stored is not re-chunked."""

    def test_unchanged_content_hash_skipped(self):
        asyncio.run(self._run())

    async def _run(self):
        html = _make_html("Test", "Same content as before")
        content_text = "Same content as before"
        content_hash = hashlib.sha256(content_text.encode()).hexdigest()

        async def mock_get(url, **kwargs):
            return _make_response(200, html, str(url))

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        chunk_mock = MagicMock()

        with patch("research_tool.wiki.is_safe_url", return_value=True), \
             patch("research_tool.wiki.extract_content", return_value={
                 "title": "Test", "text": content_text, "method": "mock",
             }), \
             patch("research_tool.wiki.extract_links", return_value=[]), \
             patch("research_tool.wiki.chunk_wiki_content", side_effect=chunk_mock) as chunk_fn, \
             patch("research_tool.wiki._make_embedding", return_value=None), \
             patch("research_tool.wiki.httpx.AsyncClient", return_value=mock_client), \
             patch("research_tool.wiki.ResearchStore") as MockStore:

            store_inst = MockStore.return_value
            # Return matching hash => content unchanged
            store_inst.get_content_hash.return_value = content_hash
            store_inst.store_page = MagicMock()
            store_inst.store_chunks = MagicMock()
            store_inst.store_links = MagicMock()
            store_inst.mark_domain_pages_stale = MagicMock()
            store_inst.get_stale_pages = MagicMock(return_value=[])
            store_inst.mark_page_active = MagicMock()

            crawler = WikiCrawler()
            stats = await crawler.crawl(
                seed_url=SEED_URL,
                db_path=":memory:",
                concurrency=1,
                progress_callback=lambda s: None,
            )

        assert stats["skipped_unchanged"] == 1
        # chunk_wiki_content should NOT have been called
        chunk_mock.assert_not_called()
        # store_page should NOT have been called (page was skipped)
        store_inst.store_page.assert_not_called()


# ── Test: circular links no infinite loop ─────────────────────────────────────


class TestCircularLinksNoInfiniteLoop:
    """A -> B -> A stops after visiting each once."""

    def test_circular_links_no_infinite_loop(self):
        asyncio.run(self._run())

    async def _run(self):
        page_a = _make_html("A", "Page A", ["https://docs.example.com/b"])
        page_b = _make_html("B", "Page B", ["https://docs.example.com/start"])

        visit_count = {"a": 0, "b": 0}

        def mock_extract_links(html, page_url):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                links.append({"target_url": a["href"], "anchor_text": a.get_text()})
            return links

        def mock_extract_content(html, url=""):
            return {"title": "T", "text": "content " + url, "method": "mock"}

        async def mock_get(url, **kwargs):
            url_str = str(url)
            if "start" in url_str:
                visit_count["a"] += 1
                return _make_response(200, page_a, url_str)
            else:
                visit_count["b"] += 1
                return _make_response(200, page_b, url_str)

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("research_tool.wiki.is_safe_url", return_value=True), \
             patch("research_tool.wiki.extract_content", side_effect=mock_extract_content), \
             patch("research_tool.wiki.extract_links", side_effect=mock_extract_links), \
             patch("research_tool.wiki.chunk_wiki_content", return_value=[]), \
             patch("research_tool.wiki._make_embedding", return_value=None), \
             patch("research_tool.wiki.httpx.AsyncClient", return_value=mock_client), \
             patch("research_tool.wiki.ResearchStore") as MockStore:

            store_inst = MockStore.return_value
            store_inst.get_content_hash.return_value = None
            store_inst.store_page = MagicMock()
            store_inst.store_chunks = MagicMock()
            store_inst.store_links = MagicMock()
            store_inst.mark_domain_pages_stale = MagicMock()
            store_inst.get_stale_pages = MagicMock(return_value=[])
            store_inst.mark_page_active = MagicMock()

            crawler = WikiCrawler()
            stats = await crawler.crawl(
                seed_url=SEED_URL,
                db_path=":memory:",
                concurrency=1,
                progress_callback=lambda s: None,
            )

        # Each page visited exactly once
        assert visit_count["a"] == 1
        assert visit_count["b"] == 1


# ── Test: URL normalization dedup ─────────────────────────────────────────────


class TestUrlNormalizationDedup:
    """page.html#section and page.html treated as same URL."""

    def test_url_normalization_dedup(self):
        url1 = "https://example.com/page.html"
        url2 = "https://example.com/page.html#section"
        assert canonicalize_url(url1) == canonicalize_url(url2)

    def test_trailing_slash_dedup(self):
        url1 = "https://example.com/page"
        url2 = "https://example.com/page/"
        assert canonicalize_url(url1) == canonicalize_url(url2)

    def test_case_dedup(self):
        url1 = "https://Example.COM/page"
        url2 = "https://example.com/page"
        assert canonicalize_url(url1) == canonicalize_url(url2)

    def test_query_order_dedup(self):
        url1 = "https://example.com/page?b=2&a=1"
        url2 = "https://example.com/page?a=1&b=2"
        assert canonicalize_url(url1) == canonicalize_url(url2)


# ── Test: fetch failure continues crawl ───────────────────────────────────────


class TestFetchFailureContinuesCrawl:
    """404/500/timeout logged, crawl continues to other pages."""

    def test_fetch_failure_continues_crawl(self):
        asyncio.run(self._run())

    async def _run(self):
        # Seed has two links: one returns 500, the other succeeds
        seed_html = _make_html(
            "Start", "Welcome",
            ["https://docs.example.com/broken", "https://docs.example.com/good"],
        )
        good_html = _make_html("Good", "This page works")

        def mock_extract_links(html, page_url):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                links.append({"target_url": a["href"], "anchor_text": a.get_text()})
            return links

        def mock_extract_content(html, url=""):
            return {"title": "T", "text": "content", "method": "mock"}

        async def mock_get(url, **kwargs):
            url_str = str(url)
            if "broken" in url_str:
                return _make_response(500, "Server Error", url_str)
            elif "good" in url_str:
                return _make_response(200, good_html, url_str)
            else:
                return _make_response(200, seed_html, url_str)

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("research_tool.wiki.is_safe_url", return_value=True), \
             patch("research_tool.wiki.extract_content", side_effect=mock_extract_content), \
             patch("research_tool.wiki.extract_links", side_effect=mock_extract_links), \
             patch("research_tool.wiki.chunk_wiki_content", return_value=[]), \
             patch("research_tool.wiki._make_embedding", return_value=None), \
             patch("research_tool.wiki.httpx.AsyncClient", return_value=mock_client), \
             patch("research_tool.wiki.ResearchStore") as MockStore:

            store_inst = MockStore.return_value
            store_inst.get_content_hash.return_value = None
            store_inst.store_page = MagicMock()
            store_inst.store_chunks = MagicMock()
            store_inst.store_links = MagicMock()
            store_inst.mark_domain_pages_stale = MagicMock()
            store_inst.get_stale_pages = MagicMock(return_value=[])
            store_inst.mark_page_active = MagicMock()

            crawler = WikiCrawler()
            stats = await crawler.crawl(
                seed_url=SEED_URL,
                db_path=":memory:",
                concurrency=1,
                progress_callback=lambda s: None,
            )

        # One page failed (the 500), but the good page was still visited
        assert stats["failed"] >= 1
        assert stats["visited"] >= 1


# ── Test: invalid URL skipped ─────────────────────────────────────────────────


class TestInvalidUrlSkipped:
    """Malformed href doesn't crash the crawler."""

    def test_invalid_url_skipped(self):
        # canonicalize_url should not crash on malformed URLs
        result = canonicalize_url("not-a-valid-url")
        assert isinstance(result, str)

    def test_empty_url(self):
        result = canonicalize_url("")
        assert isinstance(result, str)

    def test_url_with_only_fragment(self):
        result = canonicalize_url("#fragment")
        assert "#" not in result


# ── Test: progress callback ──────────────────────────────────────────────────


class TestProgressCallback:
    """Callback receives accurate stats."""

    def test_progress_callback(self):
        asyncio.run(self._run())

    async def _run(self):
        html = _make_html("Test", "Content here")

        callback_calls = []

        def progress_cb(stats):
            callback_calls.append(dict(stats))

        async def mock_get(url, **kwargs):
            return _make_response(200, html, str(url))

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("research_tool.wiki.is_safe_url", return_value=True), \
             patch("research_tool.wiki.extract_content", return_value={
                 "title": "Test", "text": "Content here", "method": "mock",
             }), \
             patch("research_tool.wiki.extract_links", return_value=[]), \
             patch("research_tool.wiki.chunk_wiki_content", return_value=[]), \
             patch("research_tool.wiki._make_embedding", return_value=None), \
             patch("research_tool.wiki.httpx.AsyncClient", return_value=mock_client), \
             patch("research_tool.wiki.ResearchStore") as MockStore:

            store_inst = MockStore.return_value
            store_inst.get_content_hash.return_value = None
            store_inst.store_page = MagicMock()
            store_inst.store_chunks = MagicMock()
            store_inst.store_links = MagicMock()
            store_inst.mark_domain_pages_stale = MagicMock()
            store_inst.get_stale_pages = MagicMock(return_value=[])
            store_inst.mark_page_active = MagicMock()

            crawler = WikiCrawler()
            stats = await crawler.crawl(
                seed_url=SEED_URL,
                db_path=":memory:",
                concurrency=1,
                progress_callback=progress_cb,
            )

        # Callback was called at least once
        assert len(callback_calls) >= 1

        # Final callback should have the right keys
        last_call = callback_calls[-1]
        assert "visited" in last_call
        assert "discovered" in last_call
        assert "skipped_unchanged" in last_call
        assert "failed" in last_call
        assert "elapsed_s" in last_call
        assert last_call["elapsed_s"] > 0


# ── Test: repo URL detection ────────────────────────────────────────────────


class TestRepoUrlDetection:
    """Github/bitbucket URLs detected and subordinate URLs excluded from BFS."""

    def test_github_repo_detected(self):
        assert _is_repo_url("https://github.com/org/repo")
        assert _is_repo_url("https://github.com/org/repo/tree/main/src")

    def test_bitbucket_repo_detected(self):
        assert _is_repo_url("https://bitbucket.org/team/project")

    def test_non_repo_not_detected(self):
        assert not _is_repo_url("https://example.com/page")
        assert not _is_repo_url("https://github.com/")

    def test_repo_prefix_extraction(self):
        prefix = _repo_prefix("https://github.com/org/repo/tree/main")
        assert prefix == "github.com/org/repo/"

    def test_repo_prefix_none_for_non_repo(self):
        assert _repo_prefix("https://example.com/page") is None

    def test_subordinate_urls_excluded(self):
        """Verify that once a repo is found, its subordinate pages are not enqueued."""
        asyncio.run(self._run_subordinate())

    async def _run_subordinate(self):
        seed_html = _make_html(
            "Start", "Welcome",
            [
                "https://github.com/org/repo",
                "https://github.com/org/repo/blob/main/README.md",
            ],
        )

        def mock_extract_links(html, page_url):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                links.append({"target_url": a["href"], "anchor_text": a.get_text()})
            return links

        def mock_extract_content(html, url=""):
            return {"title": "T", "text": "content", "method": "mock"}

        fetched_urls = []

        async def mock_get(url, **kwargs):
            url_str = str(url)
            fetched_urls.append(url_str)
            return _make_response(200, seed_html, url_str)

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("research_tool.wiki.is_safe_url", return_value=True), \
             patch("research_tool.wiki.extract_content", side_effect=mock_extract_content), \
             patch("research_tool.wiki.extract_links", side_effect=mock_extract_links), \
             patch("research_tool.wiki.chunk_wiki_content", return_value=[]), \
             patch("research_tool.wiki._make_embedding", return_value=None), \
             patch("research_tool.wiki.httpx.AsyncClient", return_value=mock_client), \
             patch("research_tool.wiki.ResearchStore") as MockStore:

            store_inst = MockStore.return_value
            store_inst.get_content_hash.return_value = None
            store_inst.store_page = MagicMock()
            store_inst.store_chunks = MagicMock()
            store_inst.store_links = MagicMock()
            store_inst.mark_domain_pages_stale = MagicMock()
            store_inst.get_stale_pages = MagicMock(return_value=[])
            store_inst.mark_page_active = MagicMock()

            crawler = WikiCrawler()
            stats = await crawler.crawl(
                seed_url=SEED_URL,
                db_path=":memory:",
                concurrency=1,
                progress_callback=lambda s: None,
            )

        # github.com URLs should NOT have been fetched
        github_fetched = [u for u in fetched_urls if "github.com" in u]
        assert len(github_fetched) == 0


# ── Test: async rate limiter ─────────────────────────────────────────────────


class TestAsyncRateLimiter:
    """Delays are actually applied between requests."""

    def test_async_rate_limiter(self):
        asyncio.run(self._run())

    async def _run(self):
        limiter = AsyncRateLimiter()
        # Override delays to something small but measurable
        limiter.DEFAULT_DELAY = (0.05, 0.05)

        domain = "test.example.com"

        t0 = time.monotonic()
        await limiter.wait(domain)
        t1 = time.monotonic()
        await limiter.wait(domain)
        t2 = time.monotonic()

        # First wait may be immediate (no previous request)
        # Second wait should have at least the minimum delay
        second_gap = t2 - t1
        assert second_gap >= 0.04, f"Expected >= 0.04s delay, got {second_gap:.4f}s"

    def test_different_domains_independent(self):
        asyncio.run(self._run_independent())

    async def _run_independent(self):
        limiter = AsyncRateLimiter()
        limiter.DEFAULT_DELAY = (0.1, 0.1)

        # Requests to different domains should not block each other
        t0 = time.monotonic()
        await limiter.wait("domain-a.com")
        await limiter.wait("domain-b.com")
        t1 = time.monotonic()

        # Both first requests should be near-instant (no prior history)
        total = t1 - t0
        # Should be much less than 0.2s (which would mean serial delays)
        assert total < 0.15, f"Different domains should not block each other, took {total:.4f}s"


# ── Test: Playwright fallback triggers ──────────────────────────────────────


class TestPlaywrightFallbackTriggers:
    """Page with < 200 chars extracted text and > 5KB HTML triggers Playwright,
    returns full content after re-extraction."""

    def test_playwright_fallback_triggers(self):
        asyncio.run(self._run())

    async def _run(self):
        # Initial HTML is > 5KB but httpx extraction yields < 200 chars
        padding = "x" * 6000
        sparse_html = f"<html><head><title>App</title></head><body><!-- {padding} --><div id='root'></div></body></html>"

        # Playwright returns rich HTML
        rich_html = _make_html("App", "This is the full rendered content " * 20)

        call_log = {"playwright_called": False}

        async def mock_get(url, **kwargs):
            return _make_response(200, sparse_html, str(url))

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        extract_call_count = {"n": 0}

        def mock_extract_content(html, url=""):
            extract_call_count["n"] += 1
            if extract_call_count["n"] == 1:
                # First call (httpx HTML): sparse content
                return {"title": "App", "text": "Loading...", "method": "mock"}
            else:
                # Second call (Playwright HTML): full content
                return {"title": "App", "text": "This is the full rendered content " * 20, "method": "mock"}

        mock_browser_instance = MagicMock()

        def mock_fetch_page(url):
            call_log["playwright_called"] = True
            return (rich_html, url)

        mock_browser_instance.fetch_page = mock_fetch_page
        mock_browser_instance.close = MagicMock()

        with patch("research_tool.wiki.is_safe_url", return_value=True), \
             patch("research_tool.wiki.extract_content", side_effect=mock_extract_content), \
             patch("research_tool.wiki.extract_links", return_value=[]), \
             patch("research_tool.wiki.chunk_wiki_content", return_value=[]), \
             patch("research_tool.wiki._make_embedding", return_value=None), \
             patch("research_tool.wiki.httpx.AsyncClient", return_value=mock_client), \
             patch("research_tool.wiki.ResearchStore") as MockStore, \
             patch("research_tool.web.Browser", return_value=mock_browser_instance):

            store_inst = MockStore.return_value
            store_inst.get_content_hash.return_value = None
            store_inst.store_page = MagicMock()
            store_inst.store_chunks = MagicMock()
            store_inst.store_links = MagicMock()
            store_inst.mark_domain_pages_stale = MagicMock()
            store_inst.get_stale_pages = MagicMock(return_value=[])
            store_inst.mark_page_active = MagicMock()

            crawler = WikiCrawler()
            # Bypass rate-limit delay for tests
            crawler._pw_last_fetch = 0.0

            stats = await crawler.crawl(
                seed_url=SEED_URL,
                db_path=":memory:",
                concurrency=1,
                progress_callback=lambda s: None,
            )

        assert call_log["playwright_called"], "Playwright should have been called for sparse page"
        # extract_content called twice: once for httpx, once for Playwright HTML
        assert extract_call_count["n"] == 2
        assert stats["visited"] == 1


# ── Test: Atlassian always triggers Playwright ──────────────────────────────


class TestAtlassianAlwaysTriggersPlaywright:
    """URL matching *.atlassian.net always triggers Playwright regardless of content length."""

    def test_atlassian_always_triggers_playwright(self):
        asyncio.run(self._run())

    async def _run(self):
        atlassian_url = "https://myteam.atlassian.net/wiki/spaces/DOC/page"
        # Even with plenty of text, Atlassian should trigger fallback
        html = _make_html("Confluence Page", "Sufficient content " * 50)

        call_log = {"playwright_called": False}

        async def mock_get(url, **kwargs):
            return _make_response(200, html, str(url))

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        def mock_extract_content(html_arg, url=""):
            # Plenty of text — normally would NOT trigger fallback
            return {"title": "Confluence Page", "text": "Sufficient content " * 50, "method": "mock"}

        mock_browser_instance = MagicMock()

        def mock_fetch_page(url):
            call_log["playwright_called"] = True
            return (html, url)

        mock_browser_instance.fetch_page = mock_fetch_page
        mock_browser_instance.close = MagicMock()

        with patch("research_tool.wiki.is_safe_url", return_value=True), \
             patch("research_tool.wiki.extract_content", side_effect=mock_extract_content), \
             patch("research_tool.wiki.extract_links", return_value=[]), \
             patch("research_tool.wiki.chunk_wiki_content", return_value=[]), \
             patch("research_tool.wiki._make_embedding", return_value=None), \
             patch("research_tool.wiki.httpx.AsyncClient", return_value=mock_client), \
             patch("research_tool.wiki.ResearchStore") as MockStore, \
             patch("research_tool.web.Browser", return_value=mock_browser_instance):

            store_inst = MockStore.return_value
            store_inst.get_content_hash.return_value = None
            store_inst.store_page = MagicMock()
            store_inst.store_chunks = MagicMock()
            store_inst.store_links = MagicMock()
            store_inst.mark_domain_pages_stale = MagicMock()
            store_inst.get_stale_pages = MagicMock(return_value=[])
            store_inst.mark_page_active = MagicMock()

            crawler = WikiCrawler()
            crawler._pw_last_fetch = 0.0

            stats = await crawler.crawl(
                seed_url=atlassian_url,
                db_path=":memory:",
                concurrency=1,
                progress_callback=lambda s: None,
            )

        assert call_log["playwright_called"], "Playwright should always trigger for *.atlassian.net"

        # Verify the detection function directly
        assert _needs_playwright_fallback(
            "https://team.atlassian.net/wiki/page",
            "Lots of content " * 100,
            "<html></html>",
        )


# ── Test: Playwright failure continues crawl ────────────────────────────────


class TestPlaywrightFailureContinuesCrawl:
    """Playwright fallback fails (exception) -> page counted as failed,
    crawl continues."""

    def test_playwright_failure_continues_crawl(self):
        asyncio.run(self._run())

    async def _run(self):
        # JS-heavy page (sparse content, big HTML) — will trigger fallback
        padding = "x" * 6000
        sparse_html = f"<html><head><title>App</title></head><body><!-- {padding} --><div id='root'></div></body></html>"

        async def mock_get(url, **kwargs):
            return _make_response(200, sparse_html, str(url))

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        def mock_extract_content(html, url=""):
            return {"title": "App", "text": "tiny", "method": "mock"}

        mock_browser_instance = MagicMock()

        def mock_fetch_page(url):
            raise RuntimeError("Playwright crashed!")

        mock_browser_instance.fetch_page = mock_fetch_page
        mock_browser_instance.close = MagicMock()

        with patch("research_tool.wiki.is_safe_url", return_value=True), \
             patch("research_tool.wiki.extract_content", side_effect=mock_extract_content), \
             patch("research_tool.wiki.extract_links", return_value=[]), \
             patch("research_tool.wiki.chunk_wiki_content", return_value=[]), \
             patch("research_tool.wiki._make_embedding", return_value=None), \
             patch("research_tool.wiki.httpx.AsyncClient", return_value=mock_client), \
             patch("research_tool.wiki.ResearchStore") as MockStore, \
             patch("research_tool.web.Browser", return_value=mock_browser_instance):

            store_inst = MockStore.return_value
            store_inst.get_content_hash.return_value = None
            store_inst.store_page = MagicMock()
            store_inst.store_chunks = MagicMock()
            store_inst.store_links = MagicMock()
            store_inst.mark_domain_pages_stale = MagicMock()
            store_inst.get_stale_pages = MagicMock(return_value=[])
            store_inst.mark_page_active = MagicMock()

            crawler = WikiCrawler()
            crawler._pw_last_fetch = 0.0

            stats = await crawler.crawl(
                seed_url=SEED_URL,
                db_path=":memory:",
                concurrency=1,
                progress_callback=lambda s: None,
            )

        # Crawl should complete without crashing
        # The page with "tiny" text (< 200 chars, > 5KB HTML) triggers fallback,
        # which fails, but the httpx content ("tiny") is used as best-effort.
        # "tiny" still counts as extracted text, so it should be visited
        assert stats["visited"] == 1


# ── Test: no fallback when content sufficient ────────────────────────────────


class TestNoFallbackWhenContentSufficient:
    """Page with exactly 200 chars extracted text does NOT trigger fallback."""

    def test_no_fallback_when_content_sufficient(self):
        asyncio.run(self._run())

    async def _run(self):
        text_200 = "a" * 200
        html = _make_html("Test", text_200)

        async def mock_get(url, **kwargs):
            return _make_response(200, html, str(url))

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        def mock_extract_content(html_arg, url=""):
            return {"title": "Test", "text": text_200, "method": "mock"}

        call_log = {"playwright_called": False}
        mock_browser_instance = MagicMock()

        def mock_fetch_page(url):
            call_log["playwright_called"] = True
            return (html, url)

        mock_browser_instance.fetch_page = mock_fetch_page
        mock_browser_instance.close = MagicMock()

        with patch("research_tool.wiki.is_safe_url", return_value=True), \
             patch("research_tool.wiki.extract_content", side_effect=mock_extract_content), \
             patch("research_tool.wiki.extract_links", return_value=[]), \
             patch("research_tool.wiki.chunk_wiki_content", return_value=[]), \
             patch("research_tool.wiki._make_embedding", return_value=None), \
             patch("research_tool.wiki.httpx.AsyncClient", return_value=mock_client), \
             patch("research_tool.wiki.ResearchStore") as MockStore, \
             patch("research_tool.web.Browser", return_value=mock_browser_instance):

            store_inst = MockStore.return_value
            store_inst.get_content_hash.return_value = None
            store_inst.store_page = MagicMock()
            store_inst.store_chunks = MagicMock()
            store_inst.store_links = MagicMock()
            store_inst.mark_domain_pages_stale = MagicMock()
            store_inst.get_stale_pages = MagicMock(return_value=[])
            store_inst.mark_page_active = MagicMock()

            crawler = WikiCrawler()
            stats = await crawler.crawl(
                seed_url=SEED_URL,
                db_path=":memory:",
                concurrency=1,
                progress_callback=lambda s: None,
            )

        assert not call_log["playwright_called"], "Playwright should NOT trigger when text >= 200 chars"
        assert stats["visited"] == 1

        # Also verify the detection function directly
        assert not _needs_playwright_fallback(
            "https://docs.example.com/page",
            "a" * 200,
            "x" * 10000,
        )


# ── Test: Browser launch failure ────────────────────────────────────────────


class TestPlaywrightBrowserLaunchFailure:
    """Browser fails to launch -> JS-heavy pages skipped with warning,
    httpx content used as best-effort."""

    def test_playwright_browser_launch_failure(self):
        asyncio.run(self._run())

    async def _run(self):
        # JS-heavy page that will trigger Playwright fallback
        padding = "x" * 6000
        sparse_html = f"<html><head><title>App</title></head><body><!-- {padding} --><div id='root'></div></body></html>"
        httpx_text = "Best effort httpx content here"

        async def mock_get(url, **kwargs):
            return _make_response(200, sparse_html, str(url))

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        def mock_extract_content(html_arg, url=""):
            return {"title": "App", "text": httpx_text, "method": "mock"}

        with patch("research_tool.wiki.is_safe_url", return_value=True), \
             patch("research_tool.wiki.extract_content", side_effect=mock_extract_content), \
             patch("research_tool.wiki.extract_links", return_value=[]), \
             patch("research_tool.wiki.chunk_wiki_content", return_value=[]), \
             patch("research_tool.wiki._make_embedding", return_value=None), \
             patch("research_tool.wiki.httpx.AsyncClient", return_value=mock_client), \
             patch("research_tool.wiki.ResearchStore") as MockStore, \
             patch("research_tool.web.Browser", side_effect=RuntimeError("Failed to launch browser")):

            store_inst = MockStore.return_value
            store_inst.get_content_hash.return_value = None
            store_inst.store_page = MagicMock()
            store_inst.store_chunks = MagicMock()
            store_inst.store_links = MagicMock()
            store_inst.mark_domain_pages_stale = MagicMock()
            store_inst.get_stale_pages = MagicMock(return_value=[])
            store_inst.mark_page_active = MagicMock()

            crawler = WikiCrawler()
            crawler._pw_last_fetch = 0.0

            stats = await crawler.crawl(
                seed_url=SEED_URL,
                db_path=":memory:",
                concurrency=1,
                progress_callback=lambda s: None,
            )

        # Browser launch failed, but httpx content was used as fallback
        # The page should still be visited (httpx_text has enough content)
        assert stats["visited"] == 1
        # store_page should have been called with the httpx content
        store_inst.store_page.assert_called_once()
        call_kwargs = store_inst.store_page.call_args
        assert call_kwargs[1]["extracted_text"] == httpx_text


# ── Test: load_auth_config ──────────────────────────────────────────────────


class TestLoadAuthConfigValid:
    """YAML config with bearer token loads correctly."""

    def test_load_auth_config_valid(self, tmp_path):
        config_file = tmp_path / "wiki_auth.yaml"
        config_file.write_text(
            'credentials:\n'
            '  "*.atlassian.net":\n'
            '    type: bearer\n'
            '    token: "ATATT3xFake"\n'
            '  "github.com":\n'
            '    type: token\n'
            '    token: "ghp_fake123"\n'
        )
        os.chmod(config_file, 0o600)

        result = load_auth_config(path=config_file)

        assert "*.atlassian.net" in result
        assert result["*.atlassian.net"]["type"] == "bearer"
        assert result["*.atlassian.net"]["token"] == "ATATT3xFake"
        assert "github.com" in result
        assert result["github.com"]["type"] == "token"
        assert result["github.com"]["token"] == "ghp_fake123"


class TestLoadAuthConfigMissingDefault:
    """No file at default path returns empty dict."""

    def test_load_auth_config_missing_default(self):
        with patch("research_tool.wiki._DEFAULT_AUTH_CONFIG_PATH") as mock_path:
            mock_path.exists.return_value = False
            result = load_auth_config(path=None)
        assert result == {}


class TestLoadAuthConfigMissingSpecified:
    """Specified path doesn't exist raises FileNotFoundError."""

    def test_load_auth_config_missing_specified(self, tmp_path):
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="Auth config file not found"):
            load_auth_config(path=missing)


class TestLoadAuthConfigMalformed:
    """Malformed YAML raises ValueError."""

    def test_load_auth_config_malformed(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("credentials:\n  invalid: yaml: content: [")
        os.chmod(config_file, 0o600)

        with pytest.raises(ValueError, match="Malformed YAML"):
            load_auth_config(path=config_file)


# ── Test: match_credentials ─────────────────────────────────────────────────


class TestMatchCredentialsExact:
    """Exact domain match returns credentials."""

    def test_match_credentials_exact(self):
        config = {
            "github.com": {"type": "token", "token": "ghp_abc"},
            "*.example.com": {"type": "bearer", "token": "tok123"},
        }
        result = match_credentials("https://github.com/org/repo", config)
        assert result is not None
        assert result["type"] == "token"
        assert result["token"] == "ghp_abc"


class TestMatchCredentialsGlob:
    """Wildcard *.example.com matches wiki.example.com."""

    def test_match_credentials_glob(self):
        config = {
            "*.example.com": {"type": "bearer", "token": "tok_glob"},
        }
        result = match_credentials("https://wiki.example.com/page", config)
        assert result is not None
        assert result["token"] == "tok_glob"


class TestMatchCredentialsExactOverGlob:
    """Exact match takes precedence over glob."""

    def test_match_credentials_exact_over_glob(self):
        config = {
            "*.example.com": {"type": "bearer", "token": "glob_token"},
            "wiki.example.com": {"type": "token", "token": "exact_token"},
        }
        result = match_credentials("https://wiki.example.com/page", config)
        assert result is not None
        assert result["type"] == "token"
        assert result["token"] == "exact_token"


class TestMatchCredentialsNoMatch:
    """Unknown domain returns None."""

    def test_match_credentials_no_match(self):
        config = {
            "github.com": {"type": "token", "token": "ghp_abc"},
        }
        result = match_credentials("https://unknown.org/page", config)
        assert result is None

    def test_match_credentials_empty_config(self):
        result = match_credentials("https://example.com/page", {})
        assert result is None

    def test_match_credentials_none_config(self):
        result = match_credentials("https://example.com/page", None)
        assert result is None


# ── Test: build_auth_headers ────────────────────────────────────────────────


class TestBuildAuthHeadersBasic:
    """Basic type produces correct base64-encoded Authorization header."""

    def test_build_auth_headers_basic(self):
        import base64
        cred = {"type": "basic", "username": "user@example.com", "token": "secret123"}
        headers = build_auth_headers(cred)
        expected = base64.b64encode(b"user@example.com:secret123").decode()
        assert headers == {"Authorization": f"Basic {expected}"}


class TestBuildAuthHeadersBearer:
    """Bearer type produces correct Authorization header."""

    def test_build_auth_headers_bearer(self):
        cred = {"type": "bearer", "token": "ATATT3xFake"}
        headers = build_auth_headers(cred)
        assert headers == {"Authorization": "Bearer ATATT3xFake"}


class TestBuildAuthHeadersToken:
    """Token type produces correct Authorization header."""

    def test_build_auth_headers_token(self):
        cred = {"type": "token", "token": "ghp_fake123"}
        headers = build_auth_headers(cred)
        assert headers == {"Authorization": "token ghp_fake123"}


class TestBuildAuthHeadersCookie:
    """Cookie type produces correct Cookie header."""

    def test_build_auth_headers_cookie(self):
        cred = {"type": "cookie", "value": "session_id=abc123"}
        headers = build_auth_headers(cred)
        assert headers == {"Cookie": "session_id=abc123"}


class TestBuildAuthHeadersCustom:
    """Header type produces correct custom header."""

    def test_build_auth_headers_header(self):
        cred = {"type": "header", "name": "X-Api-Key", "value": "secret123"}
        headers = build_auth_headers(cred)
        assert headers == {"X-Api-Key": "secret123"}


# ── Test: merge_cli_auth ────────────────────────────────────────────────────


class TestMergeCliAuthGithubToken:
    """Github token added to config."""

    def test_merge_cli_auth_github_token(self):
        config = {"*.atlassian.net": {"type": "bearer", "token": "old"}}
        result = merge_cli_auth(config, github_token="ghp_new_token")

        # github.com entry should be added
        assert "github.com" in result
        assert result["github.com"]["type"] == "token"
        assert result["github.com"]["token"] == "ghp_new_token"
        # Existing entries preserved
        assert "*.atlassian.net" in result

    def test_merge_cli_auth_github_token_overrides(self):
        config = {"github.com": {"type": "token", "token": "old_token"}}
        result = merge_cli_auth(config, github_token="ghp_new_token")
        assert result["github.com"]["token"] == "ghp_new_token"

    def test_merge_cli_auth_empty_config(self):
        result = merge_cli_auth({}, github_token="ghp_test")
        assert result["github.com"]["token"] == "ghp_test"

    def test_merge_cli_auth_none_config(self):
        result = merge_cli_auth(None, github_token="ghp_test")
        assert result["github.com"]["token"] == "ghp_test"


class TestMergeCliAuthHeaderSeedOnly:
    """Auth headers only applied to seed domain."""

    def test_merge_cli_auth_header_seed_only(self):
        config = {}
        result = merge_cli_auth(
            config,
            auth_headers=["X-Api-Key: secret123"],
            seed_domain="wiki.internal.com",
        )

        # Seed domain should have the header
        assert "wiki.internal.com" in result
        assert result["wiki.internal.com"]["type"] == "header"
        assert result["wiki.internal.com"]["name"] == "X-Api-Key"
        assert result["wiki.internal.com"]["value"] == "secret123"

    def test_merge_cli_auth_header_no_seed_domain(self):
        """auth_headers without seed_domain should be a no-op."""
        config = {}
        result = merge_cli_auth(config, auth_headers=["X-Api-Key: secret123"])
        assert result == {}


# ── Test: auth header not leaked to external ────────────────────────────────


class TestAuthHeaderNotLeakedToExternal:
    """Auth for seed domain is NOT sent to external domain requests."""

    def test_auth_header_not_leaked_to_external(self):
        auth_config = {
            "docs.example.com": {"type": "bearer", "token": "secret_token"},
        }

        # Seed domain should match
        seed_cred = match_credentials("https://docs.example.com/page", auth_config)
        assert seed_cred is not None
        assert seed_cred["token"] == "secret_token"

        # External domain should NOT match
        external_cred = match_credentials("https://evil.com/steal", auth_config)
        assert external_cred is None

        # Another external domain should NOT match
        other_cred = match_credentials("https://other.example.com/page", auth_config)
        assert other_cred is None


# ── Test: auth config file permission warning ──────────────────────────────


class TestAuthConfigPermissionWarning:
    """Warn at load time if config file permissions are too permissive."""

    def test_permission_warning(self, tmp_path):
        config_file = tmp_path / "wiki_auth.yaml"
        config_file.write_text(
            'credentials:\n'
            '  "github.com":\n'
            '    type: token\n'
            '    token: "ghp_test"\n'
        )
        # Make group-readable (overly permissive)
        os.chmod(config_file, 0o644)

        import logging
        with patch("research_tool.wiki.logger") as mock_logger:
            result = load_auth_config(path=config_file)
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "readable by group/others" in warning_msg

        # Config should still load successfully (warn, don't block)
        assert "github.com" in result


# ── Test: stale pages reported ─────────────────────────────────────────────


class TestStalePagesReported:
    """First crawl stores 3 pages. Re-crawl only visits 2 -> 1 page reported as stale."""

    def test_stale_pages_reported(self):
        asyncio.run(self._run())

    async def _run(self):
        # Seed links to p1 only (not p2 -- p2 becomes stale)
        seed_html = _make_html(
            "Start", "Welcome",
            ["https://docs.example.com/p1"],
        )
        p1_html = _make_html("Page 1", "Content 1")

        def mock_extract_links(html, page_url):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                links.append({"target_url": a["href"], "anchor_text": a.get_text()})
            return links

        def mock_extract_content(html, url=""):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.string if soup.title else ""
            text = soup.get_text()
            return {"title": title, "text": text, "method": "mock"}

        async def mock_get(url, **kwargs):
            url_str = str(url)
            if "p1" in url_str:
                return _make_response(200, p1_html, url_str)
            return _make_response(200, seed_html, url_str)

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        # Simulate that p2 was previously stored and is now stale
        stale_page = {"url": "https://docs.example.com/p2", "title": "Page 2"}

        with patch("research_tool.wiki.is_safe_url", return_value=True), \
             patch("research_tool.wiki.extract_content", side_effect=mock_extract_content), \
             patch("research_tool.wiki.extract_links", side_effect=mock_extract_links), \
             patch("research_tool.wiki.chunk_wiki_content", return_value=[]), \
             patch("research_tool.wiki._make_embedding", return_value=None), \
             patch("research_tool.wiki.httpx.AsyncClient", return_value=mock_client), \
             patch("research_tool.wiki.ResearchStore") as MockStore:

            store_inst = MockStore.return_value
            store_inst.get_content_hash.return_value = None
            store_inst.store_page = MagicMock()
            store_inst.store_chunks = MagicMock()
            store_inst.store_links = MagicMock()
            store_inst.mark_domain_pages_stale = MagicMock()
            store_inst.get_stale_pages = MagicMock(return_value=[stale_page])
            store_inst.mark_page_active = MagicMock()

            crawler = WikiCrawler()
            stats = await crawler.crawl(
                seed_url=SEED_URL,
                db_path=":memory:",
                concurrency=1,
                progress_callback=lambda s: None,
            )

        assert stats["stale_pages"] == 1
        store_inst.get_stale_pages.assert_called_once_with(SEED_NETLOC)


# ── Test: no stale pages ──────────────────────────────────────────────────


class TestNoStalePages:
    """Re-crawl visits all pages -> stats show 0 stale."""

    def test_no_stale_pages(self):
        asyncio.run(self._run())

    async def _run(self):
        html = _make_html("Test", "Content here")

        async def mock_get(url, **kwargs):
            return _make_response(200, html, str(url))

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("research_tool.wiki.is_safe_url", return_value=True), \
             patch("research_tool.wiki.extract_content", return_value={
                 "title": "Test", "text": "Content here", "method": "mock",
             }), \
             patch("research_tool.wiki.extract_links", return_value=[]), \
             patch("research_tool.wiki.chunk_wiki_content", return_value=[]), \
             patch("research_tool.wiki._make_embedding", return_value=None), \
             patch("research_tool.wiki.httpx.AsyncClient", return_value=mock_client), \
             patch("research_tool.wiki.ResearchStore") as MockStore:

            store_inst = MockStore.return_value
            store_inst.get_content_hash.return_value = None
            store_inst.store_page = MagicMock()
            store_inst.store_chunks = MagicMock()
            store_inst.store_links = MagicMock()
            store_inst.mark_domain_pages_stale = MagicMock()
            store_inst.get_stale_pages = MagicMock(return_value=[])
            store_inst.mark_page_active = MagicMock()

            crawler = WikiCrawler()
            stats = await crawler.crawl(
                seed_url=SEED_URL,
                db_path=":memory:",
                concurrency=1,
                progress_callback=lambda s: None,
            )

        assert stats["stale_pages"] == 0
        store_inst.get_stale_pages.assert_called_once_with(SEED_NETLOC)


# ── Test: new vs updated distinction ──────────────────────────────────────


class TestNewVsUpdatedDistinction:
    """First crawl -> pages counted as new_pages. Re-crawl with changed content
    -> counted as updated. Same content -> counted as skipped_unchanged."""

    def test_new_vs_updated_distinction(self):
        asyncio.run(self._run())

    async def _run(self):
        # Three pages: seed links to p1 and p2
        seed_html = _make_html(
            "Start", "Welcome",
            ["https://docs.example.com/p1", "https://docs.example.com/p2"],
        )
        p1_html = _make_html("Page 1", "New content for page 1")
        p2_html = _make_html("Page 2", "Same old content for page 2")

        # Use fixed text strings for extract_content so hashes are predictable
        page_texts = {
            "start": "Welcome to the start page",
            "p1": "New content for page 1",
            "p2": "Same old content for page 2",
        }
        p2_hash = hashlib.sha256(page_texts["p2"].encode()).hexdigest()

        def mock_extract_links(html, page_url):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                links.append({"target_url": a["href"], "anchor_text": a.get_text()})
            return links

        def mock_extract_content(html, url=""):
            # Return predictable text based on URL to control hashing
            if "p1" in url:
                return {"title": "Page 1", "text": page_texts["p1"], "method": "mock"}
            elif "p2" in url:
                return {"title": "Page 2", "text": page_texts["p2"], "method": "mock"}
            return {"title": "Start", "text": page_texts["start"], "method": "mock"}

        async def mock_get(url, **kwargs):
            url_str = str(url)
            if "p1" in url_str:
                return _make_response(200, p1_html, url_str)
            elif "p2" in url_str:
                return _make_response(200, p2_html, url_str)
            return _make_response(200, seed_html, url_str)

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        def hash_side_effect(url):
            # seed: never seen before (new)
            if "start" in url:
                return None
            # p1: previously seen with different hash (updated)
            if "p1" in url:
                return "old_different_hash"
            # p2: same hash as current content (unchanged)
            if "p2" in url:
                return p2_hash
            return None

        with patch("research_tool.wiki.is_safe_url", return_value=True), \
             patch("research_tool.wiki.extract_content", side_effect=mock_extract_content), \
             patch("research_tool.wiki.extract_links", side_effect=mock_extract_links), \
             patch("research_tool.wiki.chunk_wiki_content", return_value=[]), \
             patch("research_tool.wiki._make_embedding", return_value=None), \
             patch("research_tool.wiki.httpx.AsyncClient", return_value=mock_client), \
             patch("research_tool.wiki.ResearchStore") as MockStore:

            store_inst = MockStore.return_value
            store_inst.get_content_hash.side_effect = hash_side_effect
            store_inst.store_page = MagicMock()
            store_inst.store_chunks = MagicMock()
            store_inst.store_links = MagicMock()
            store_inst.mark_domain_pages_stale = MagicMock()
            store_inst.get_stale_pages = MagicMock(return_value=[])
            store_inst.mark_page_active = MagicMock()

            crawler = WikiCrawler()
            stats = await crawler.crawl(
                seed_url=SEED_URL,
                db_path=":memory:",
                concurrency=1,
                progress_callback=lambda s: None,
            )

        # seed is new (stored_hash=None), p1 is updated (hash differs), p2 is unchanged
        assert stats["new_pages"] == 1, f"Expected 1 new page, got {stats['new_pages']}"
        assert stats["updated"] == 1, f"Expected 1 updated page, got {stats['updated']}"
        assert stats["skipped_unchanged"] == 1, f"Expected 1 skipped, got {stats['skipped_unchanged']}"


# ── Test: stats include stale_pages key ────────────────────────────────────


class TestStatsIncludeStalePageKey:
    """Stats dict always includes stale_pages key, even when 0."""

    def test_stats_include_stale_pages_key(self):
        asyncio.run(self._run())

    async def _run(self):
        html = _make_html("Test", "Content here")

        async def mock_get(url, **kwargs):
            return _make_response(200, html, str(url))

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("research_tool.wiki.is_safe_url", return_value=True), \
             patch("research_tool.wiki.extract_content", return_value={
                 "title": "Test", "text": "Content here", "method": "mock",
             }), \
             patch("research_tool.wiki.extract_links", return_value=[]), \
             patch("research_tool.wiki.chunk_wiki_content", return_value=[]), \
             patch("research_tool.wiki._make_embedding", return_value=None), \
             patch("research_tool.wiki.httpx.AsyncClient", return_value=mock_client), \
             patch("research_tool.wiki.ResearchStore") as MockStore:

            store_inst = MockStore.return_value
            store_inst.get_content_hash.return_value = None
            store_inst.store_page = MagicMock()
            store_inst.store_chunks = MagicMock()
            store_inst.store_links = MagicMock()
            store_inst.mark_domain_pages_stale = MagicMock()
            store_inst.get_stale_pages = MagicMock(return_value=[])
            store_inst.mark_page_active = MagicMock()

            crawler = WikiCrawler()
            stats = await crawler.crawl(
                seed_url=SEED_URL,
                db_path=":memory:",
                concurrency=1,
                progress_callback=lambda s: None,
            )

        # All expected keys should be present
        assert "stale_pages" in stats
        assert "new_pages" in stats
        assert "updated" in stats
        assert stats["stale_pages"] == 0
        assert stats["new_pages"] >= 0
        assert stats["updated"] >= 0
