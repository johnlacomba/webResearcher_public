"""
BFS wiki crawler that exhaustively ingests a documentation site.

Enumerates all same-domain pages from a seed URL using async httpx,
classifies links as same-domain/external/repo, and orchestrates the
extract -> chunk -> embed -> store pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import re
import stat
import sys
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse, urlunparse, urlencode, parse_qs, quote, unquote

import httpx
import yaml

from research_tool.store import ResearchStore, _make_embedding, DocumentChunk
from research_tool.web import is_safe_url, extract_content, extract_links
from research_tool.wiki_chunker import chunk_wiki_content

logger = logging.getLogger(__name__)

# ── Default auth config path ────────────────────────────────────────────────

_DEFAULT_AUTH_CONFIG_PATH = Path("~/.config/research_tool/wiki_auth.yaml").expanduser()


# ── Auth helpers ─────────────────────────────────────────────────────────────


def load_auth_config(path: str | Path | None = None) -> dict:
    """Load per-source authentication config from a YAML file.

    Args:
        path: Path to the YAML config file.  When *None*, uses the default
              ``~/.config/research_tool/wiki_auth.yaml``.

    Returns:
        A dict mapping domain globs to credential dicts, e.g.::

            {"*.atlassian.net": {"type": "bearer", "token": "..."}}

        Returns an empty dict if no path was specified and the default
        file does not exist.

    Raises:
        FileNotFoundError: If an explicit *path* was given but does not exist.
        ValueError: If the YAML content is malformed.
    """
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Auth config file not found: {config_path}"
            )
    else:
        config_path = _DEFAULT_AUTH_CONFIG_PATH
        if not config_path.exists():
            return {}

    # Warn about overly permissive file permissions
    try:
        file_stat = os.stat(config_path)
        if file_stat.st_mode & (stat.S_IRGRP | stat.S_IROTH):
            logger.warning(
                "Auth config file %s is readable by group/others. "
                "Consider running: chmod 600 %s",
                config_path,
                config_path,
            )
    except OSError:
        pass  # stat failed — not critical, move on

    try:
        raw = config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Malformed YAML in auth config {config_path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        return {}

    return data.get("credentials", {}) or {}


def match_credentials(url: str, auth_config: dict) -> dict | None:
    """Return the credential dict for *url* from *auth_config*, or ``None``.

    Exact domain matches take precedence over glob patterns.
    """
    if not auth_config:
        return None

    netloc = urlparse(url).netloc.lower()

    # First pass: exact (non-glob) matches
    for pattern, cred in auth_config.items():
        if "*" not in pattern and "?" not in pattern and "[" not in pattern:
            if netloc == pattern.lower():
                return cred

    # Second pass: glob matches
    for pattern, cred in auth_config.items():
        if fnmatch(netloc, pattern.lower()):
            return cred

    return None


def build_auth_headers(credential: dict) -> dict[str, str]:
    """Convert a credential dict to HTTP headers.

    Supported types:
        basic   → ``{"Authorization": "Basic base64(username:token)"}``
        bearer  → ``{"Authorization": "Bearer <token>"}``
        token   → ``{"Authorization": "token <token>"}``
        cookie  → ``{"Cookie": "<value>"}``
        header  → ``{credential["name"]: credential["value"]}``
    """
    import base64

    cred_type = credential.get("type", "").lower()

    if cred_type == "basic":
        username = credential["username"]
        token = credential["token"]
        encoded = base64.b64encode(f"{username}:{token}".encode()).decode()
        return {"Authorization": f"Basic {encoded}"}
    if cred_type == "bearer":
        return {"Authorization": f"Bearer {credential['token']}"}
    if cred_type == "token":
        return {"Authorization": f"token {credential['token']}"}
    if cred_type == "cookie":
        return {"Cookie": credential["value"]}
    if cred_type == "header":
        return {credential["name"]: credential["value"]}

    logger.warning("Unknown auth credential type: %s", cred_type)
    return {}


def merge_cli_auth(
    auth_config: dict,
    github_token: str | None = None,
    auth_headers: list[str] | None = None,
    seed_domain: str | None = None,
) -> dict:
    """Merge CLI-provided credentials into an existing auth config.

    CLI overrides take precedence over values already in *auth_config*.

    Args:
        auth_config: Existing auth config (modified in-place and returned).
        github_token: If provided, adds ``github.com`` bearer-style entry.
        auth_headers: List of ``"Header: Value"`` strings.  Applied to
                      *seed_domain* only (NOT external domains).
        seed_domain: The seed URL's domain (netloc).  Required when
                     *auth_headers* is non-empty.

    Returns:
        The (possibly mutated) *auth_config* dict.
    """
    merged = dict(auth_config) if auth_config else {}

    if github_token:
        merged["github.com"] = {"type": "token", "token": github_token}

    if auth_headers and seed_domain:
        # Parse "Header: Value" strings and create a compound header entry
        # for the seed domain.
        headers_dict: dict[str, str] = {}
        for header_str in auth_headers:
            if ":" not in header_str:
                logger.warning("Ignoring malformed auth header: %s", header_str)
                continue
            name, _, value = header_str.partition(":")
            headers_dict[name.strip()] = value.strip()

        if headers_dict:
            # Store as a list of header entries keyed by seed domain
            merged[seed_domain] = {
                "type": "header",
                "name": list(headers_dict.keys())[0],
                "value": list(headers_dict.values())[0],
            }
            # For multiple headers, store them all as _extra_headers
            if len(headers_dict) > 1:
                merged[seed_domain]["_extra_headers"] = headers_dict

    return merged


# ── JS-heavy wiki domains (need Playwright rendering) ────────────────────────

_JS_HEAVY_DOMAINS = {"*.atlassian.net", "*.notion.site"}


def _is_js_heavy_domain(url: str) -> bool:
    """Return True if the URL's domain is a known JS-heavy wiki platform."""
    netloc = urlparse(url).netloc.lower()
    return any(fnmatch(netloc, p) for p in _JS_HEAVY_DOMAINS)


def _needs_playwright_fallback(url: str, extracted_text: str, raw_html: str) -> bool:
    """Return True if the page likely needs JavaScript rendering.

    Triggers on:
    (a) extracted text < 200 chars AND raw HTML > 5 KB — indicates JS-rendered content
    (b) URL domain matches a known JS-heavy wiki platform
    """
    if _is_js_heavy_domain(url):
        return True
    if len(extracted_text) < 200 and len(raw_html) > 5120:
        return True
    return False


# ── Repo URL patterns ────────────────────────────────────────────────────────

_REPO_PATTERNS = [
    re.compile(r"^https?://github\.com/([^/]+)/([^/]+)(/|$)", re.IGNORECASE),
    re.compile(r"^https?://bitbucket\.org/([^/]+)/([^/]+)(/|$)", re.IGNORECASE),
]


def _is_repo_url(url: str) -> bool:
    """Return True if the URL matches a known code-hosting repo pattern."""
    return any(pat.search(url) for pat in _REPO_PATTERNS)


def _repo_prefix(url: str) -> str | None:
    """Extract the repo URL prefix (e.g. 'github.com/org/repo/') for BFS exclusion.

    Returns None if the URL is not a recognized repo URL.
    """
    for pat in _REPO_PATTERNS:
        m = pat.search(url)
        if m:
            parsed = urlparse(url)
            return f"{parsed.netloc}/{m.group(1)}/{m.group(2)}/"
    return None


# ── Confluence Cloud detection ────────────────────────────────────────────────


def _is_confluence_cloud(url: str) -> bool:
    """Return True if the URL points to a Confluence Cloud wiki."""
    parsed = urlparse(url)
    return (
        fnmatch(parsed.netloc.lower(), "*.atlassian.net")
        and parsed.path.startswith("/wiki")
    )


def _extract_confluence_space_key(url: str) -> str | None:
    """Extract space key from a Confluence URL, or None for all spaces."""
    m = re.match(r"/wiki/spaces/([^/]+)", urlparse(url).path)
    return m.group(1) if m else None


# ── URL Canonicalization ──────────────────────────────────────────────────────


def canonicalize_url(url: str) -> str:
    """Canonicalize a URL for deduplication.

    - Lowercase scheme and host
    - Strip fragments (#section)
    - Sort query parameters alphabetically
    - Normalize percent-encoding (uppercase hex digits)
    - Strip trailing slashes for non-root paths
    """
    parsed = urlparse(url)

    # Lowercase scheme and host
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Strip fragment
    # Sort query params
    query_params = parse_qs(parsed.query, keep_blank_values=True)
    sorted_query = urlencode(
        sorted(
            ((k, v[0]) for k, v in query_params.items()),
            key=lambda x: x[0],
        )
    ) if query_params else ""

    # Normalize percent-encoding in path
    path = _normalize_percent_encoding(parsed.path)

    # Strip trailing slash for non-root paths
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    return urlunparse((scheme, netloc, path, parsed.params, sorted_query, ""))


def _normalize_percent_encoding(s: str) -> str:
    """Decode then re-encode with uppercase hex digits."""
    decoded = unquote(s)
    # Re-encode only characters that need it, using uppercase hex
    result = []
    for ch in decoded:
        if ch in (' ', '#', '%', '?', '&', '=', '+'):
            result.append(quote(ch))
        else:
            result.append(ch)
    return "".join(result)


# ── Domain classification ────────────────────────────────────────────────────


def classify_url(url: str, seed_netloc: str) -> str:
    """Classify a URL relative to the seed domain.

    Returns:
        "same-domain" — same netloc as the seed, enqueue for full crawl
        "repo" — matches a known code-hosting repo pattern
        "external" — different domain, not a repo
    """
    parsed = urlparse(url)
    if parsed.netloc.lower() == seed_netloc.lower():
        return "same-domain"
    if _is_repo_url(url):
        return "repo"
    return "external"


# ── Async Rate Limiter ────────────────────────────────────────────────────────


class AsyncRateLimiter:
    """Per-domain async rate limiter with random jitter.

    Uses asyncio.Lock per domain and asyncio.sleep for delays.
    """

    DEFAULT_DELAY = (0.5, 1.5)

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}

    def _get_lock(self, domain: str) -> asyncio.Lock:
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()
        return self._locks[domain]

    async def wait(self, domain: str) -> None:
        """Wait until it's safe to make a request to the given domain."""
        lock = self._get_lock(domain)
        async with lock:
            now = time.monotonic()
            last = self._last_request.get(domain, 0.0)
            min_delay, max_delay = self.DEFAULT_DELAY
            target = last + random.uniform(min_delay, max_delay)
            if now < target:
                await asyncio.sleep(target - now)
            self._last_request[domain] = time.monotonic()


# ── WikiCrawler ───────────────────────────────────────────────────────────────


_PLAYWRIGHT_BASE_DELAY = 3.0  # seconds between Playwright fetches


class WikiCrawler:
    """BFS wiki crawler that exhaustively ingests a documentation site."""

    def __init__(self) -> None:
        self._browser = None
        self._pw_last_fetch: float = 0.0

    def _init_browser(self) -> None:
        """Lazily create a Browser instance on first Playwright fallback."""
        if self._browser is not None:
            return
        from research_tool.web import Browser

        self._browser = Browser()

    async def _playwright_fetch(
        self,
        url: str,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """Fetch a page via Playwright (sync Browser wrapped in to_thread).

        Returns (html, final_url). Raises on failure.
        Rate-limits at _PLAYWRIGHT_BASE_DELAY between requests.
        """
        # Rate-limit Playwright requests
        now = time.monotonic()
        elapsed = now - self._pw_last_fetch
        if elapsed < _PLAYWRIGHT_BASE_DELAY:
            await asyncio.sleep(_PLAYWRIGHT_BASE_DELAY - elapsed)

        def _sync_fetch():
            self._init_browser()
            if extra_headers:
                return self._browser.fetch_page(
                    url, extra_headers=extra_headers
                )
            return self._browser.fetch_page(url)

        result = await asyncio.to_thread(_sync_fetch)
        self._pw_last_fetch = time.monotonic()
        return result

    async def crawl(
        self,
        seed_url: str,
        db_path: str,
        auth_config: dict | None = None,
        concurrency: int = 10,
        branch: str | None = None,
        progress_callback: Callable | None = None,
    ) -> dict:
        """Main entry point. BFS-crawl from seed_url and ingest all pages.

        Args:
            seed_url: Starting URL for the crawl.
            db_path: Path to the SQLite database file.
            auth_config: Optional dict mapping domain globs to credential
                dicts (as returned by :func:`load_auth_config`).
            concurrency: Number of concurrent worker coroutines.
            branch: Optional branch identifier for versioned docs.
            progress_callback: Optional callback(stats_dict) called after each page.

        Returns:
            Final stats dict with discovered/visited/skipped/failed counts.
        """
        # Confluence Cloud: use REST API instead of BFS HTML scraping.
        # Confluence web pages reject Basic auth; the REST API accepts it.
        if _is_confluence_cloud(seed_url):
            cred = match_credentials(seed_url, auth_config) if auth_config else None
            if cred:
                logger.info(
                    "Detected Confluence Cloud — using REST API for %s",
                    seed_url,
                )
                return await self._crawl_confluence(
                    seed_url, db_path, auth_config, concurrency,
                    progress_callback,
                )
            logger.warning(
                "Confluence Cloud detected but no credentials found; "
                "falling back to BFS crawl (likely to fail with 401)"
            )

        store = ResearchStore(db_path)
        seed_parsed = urlparse(seed_url)
        seed_netloc = seed_parsed.netloc.lower()

        # Mark all existing pages for this domain as stale
        store.mark_domain_pages_stale(seed_netloc)

        # BFS state
        queue: asyncio.Queue[str] = asyncio.Queue()
        visited: set[str] = set()
        repo_prefixes: set[str] = set()
        repos_found: list[str] = []
        rate_limiter = AsyncRateLimiter()

        stats = {
            "discovered": 0,
            "visited": 0,
            "skipped_unchanged": 0,
            "new_pages": 0,
            "updated": 0,
            "failed": 0,
            "repos_found": 0,
            "stale_pages": 0,
            "elapsed_s": 0.0,
        }
        stats_lock = asyncio.Lock()

        start_time = time.monotonic()

        # Seed the queue
        canon_seed = canonicalize_url(seed_url)
        await queue.put(canon_seed)
        visited.add(canon_seed)
        stats["discovered"] = 1

        async def _worker() -> None:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=30.0,
                headers={"User-Agent": "WebResearcher/1.0"},
            ) as client:
                while True:
                    try:
                        url = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        # Wait a moment, then check again; break if still empty
                        await asyncio.sleep(0.1)
                        try:
                            url = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                    try:
                        await self._process_page(
                            url=url,
                            client=client,
                            store=store,
                            seed_netloc=seed_netloc,
                            queue=queue,
                            visited=visited,
                            repo_prefixes=repo_prefixes,
                            repos_found=repos_found,
                            rate_limiter=rate_limiter,
                            stats=stats,
                            stats_lock=stats_lock,
                            auth_config=auth_config,
                        )
                    except Exception:
                        logger.exception("Unhandled error processing %s", url)
                        async with stats_lock:
                            stats["failed"] += 1
                    finally:
                        queue.task_done()

                    # Report progress
                    async with stats_lock:
                        stats["elapsed_s"] = time.monotonic() - start_time
                        current_stats = dict(stats)
                    if progress_callback:
                        progress_callback(current_stats)
                    else:
                        self._default_progress(current_stats)

        # Run workers until the queue is drained
        # We use a loop: launch workers, wait for queue to drain, repeat
        # to handle pages that enqueue new URLs
        try:
            while not queue.empty():
                workers = [asyncio.create_task(_worker()) for _ in range(concurrency)]
                await queue.join()
                # Cancel any still-running workers
                for w in workers:
                    w.cancel()
                # Wait for all workers to finish (suppressing CancelledError)
                await asyncio.gather(*workers, return_exceptions=True)
        finally:
            # Close the Playwright browser if it was created
            if self._browser is not None:
                try:
                    await asyncio.to_thread(self._browser.close)
                except Exception:
                    logger.debug("Error closing Playwright browser", exc_info=True)
                self._browser = None

        stats["elapsed_s"] = time.monotonic() - start_time
        stats["repos_found"] = len(repos_found)
        stats["repos_found_urls"] = list(repos_found)

        # Report stale pages (unreachable after re-crawl)
        stale_pages = store.get_stale_pages(seed_netloc)
        stats["stale_pages"] = len(stale_pages)
        if stale_pages:
            logger.warning(
                "%d pages no longer reachable: %s",
                len(stale_pages),
                [p["url"] for p in stale_pages[:10]],  # Log first 10
            )

        logger.info(
            "Wiki crawl complete: %d pages visited (%d new, %d updated), "
            "%d skipped unchanged, %d stale, "
            "%d failed, %d repos found, %.1fs elapsed",
            stats["visited"],
            stats["new_pages"],
            stats["updated"],
            stats["skipped_unchanged"],
            stats["stale_pages"],
            stats["failed"],
            stats["repos_found"],
            stats["elapsed_s"],
        )

        return stats

    async def _process_page(
        self,
        url: str,
        client: httpx.AsyncClient,
        store: ResearchStore,
        seed_netloc: str,
        queue: asyncio.Queue,
        visited: set[str],
        repo_prefixes: set[str],
        repos_found: list[str],
        rate_limiter: AsyncRateLimiter,
        stats: dict,
        stats_lock: asyncio.Lock,
        auth_config: dict | None = None,
    ) -> None:
        """Fetch, extract, chunk, embed, and store a single page."""

        # SSRF check before fetch
        safe = await asyncio.to_thread(is_safe_url, url)
        if not safe:
            logger.warning("SSRF check failed for URL: %s", url)
            async with stats_lock:
                stats["failed"] += 1
            return

        # Rate limit
        domain = urlparse(url).netloc
        await rate_limiter.wait(domain)

        # Resolve per-request auth headers
        request_headers: dict[str, str] = {}
        if auth_config:
            cred = match_credentials(url, auth_config)
            if cred:
                request_headers = build_auth_headers(cred)

        # JS-heavy domains (e.g. *.atlassian.net): go straight to Playwright.
        # These platforms reject Basic auth on HTML pages and require a
        # full browser render for JS-driven content.
        if _is_js_heavy_domain(url):
            logger.info("JS-heavy domain detected, using Playwright for %s", url)
            try:
                pw_html, pw_final_url = await self._playwright_fetch(
                    url, extra_headers=request_headers or None,
                )
                html = pw_html
                extracted = await asyncio.to_thread(extract_content, html, url)
                title = extracted.get("title", "")
                extracted_text = extracted.get("text", "")
            except Exception:
                logger.warning(
                    "Playwright fetch failed for JS-heavy domain %s",
                    url,
                    exc_info=True,
                )
                async with stats_lock:
                    stats["failed"] += 1
                return
        else:
            # Standard httpx fetch for non-JS-heavy domains
            try:
                response = await client.get(url, headers=request_headers)
            except httpx.TimeoutException:
                logger.warning("Timeout fetching %s", url)
                async with stats_lock:
                    stats["failed"] += 1
                return
            except httpx.RequestError as exc:
                logger.warning("Request error fetching %s: %s", url, exc)
                async with stats_lock:
                    stats["failed"] += 1
                return

            # SSRF check on final URL after redirects
            final_url = str(response.url)
            if final_url != url:
                final_safe = await asyncio.to_thread(is_safe_url, final_url)
                if not final_safe:
                    logger.warning("SSRF check failed for redirected URL: %s", final_url)
                    async with stats_lock:
                        stats["failed"] += 1
                    return

            # Check HTTP status
            if response.status_code >= 400:
                logger.warning("HTTP %d for %s", response.status_code, url)
                async with stats_lock:
                    stats["failed"] += 1
                return

            html = response.text

            # Extract content
            extracted = await asyncio.to_thread(extract_content, html, url)
            title = extracted.get("title", "")
            extracted_text = extracted.get("text", "")

            # Playwright fallback for thin content (JS-rendered pages)
            if _needs_playwright_fallback(url, extracted_text, html):
                logger.info("Playwright fallback triggered for %s", url)
                try:
                    pw_html, pw_final_url = await self._playwright_fetch(
                        url, extra_headers=request_headers or None,
                    )
                    html = pw_html
                    extracted = await asyncio.to_thread(extract_content, html, url)
                    title = extracted.get("title", "") or title
                    extracted_text = extracted.get("text", "")
                except Exception:
                    logger.warning(
                        "Playwright fallback failed for %s; using httpx content",
                        url,
                        exc_info=True,
                    )

        if not extracted_text.strip():
            logger.debug("No content extracted from %s", url)
            async with stats_lock:
                stats["visited"] += 1
            store.mark_page_active(url)
            return

        # Content hash for change detection
        content_hash = hashlib.sha256(extracted_text.encode()).hexdigest()

        # Check if content is unchanged
        stored_hash = store.get_content_hash(url)
        if stored_hash == content_hash:
            logger.debug("Content unchanged for %s, skipping re-chunk", url)
            store.mark_page_active(url)
            async with stats_lock:
                stats["skipped_unchanged"] += 1
            # Still extract links to continue BFS
            links = await asyncio.to_thread(extract_links, html, url)
            await self._enqueue_links(
                links, seed_netloc, queue, visited,
                repo_prefixes, repos_found, stats, stats_lock,
            )
            return

        # Track new vs updated pages
        async with stats_lock:
            if stored_hash is None:
                stats["new_pages"] += 1
            else:
                stats["updated"] += 1

        # Chunk the content
        chunks = await asyncio.to_thread(chunk_wiki_content, html, url)

        # Embed each chunk (offload CPU-bound work)
        for chunk in chunks:
            embedding = await asyncio.to_thread(_make_embedding, chunk.text)
            chunk.embedding = embedding

        # Store page and chunks (DB writes on main asyncio thread)
        store.store_page(
            url=url,
            title=title,
            html=html,
            extracted_text=extracted_text,
            content_hash=content_hash,
            crawl_status="active",
        )
        if chunks:
            store.store_chunks(chunks)

        # Extract and store links
        links = await asyncio.to_thread(extract_links, html, url)
        store.store_links(url, links)

        # Enqueue discovered links
        await self._enqueue_links(
            links, seed_netloc, queue, visited,
            repo_prefixes, repos_found, stats, stats_lock,
        )

        # Mark page active
        store.mark_page_active(url)

        async with stats_lock:
            stats["visited"] += 1

    # ── Confluence REST API crawl ──────────────────────────────────────────

    async def _crawl_confluence(
        self,
        seed_url: str,
        db_path: str,
        auth_config: dict | None,
        concurrency: int,
        progress_callback: Callable | None,
    ) -> dict:
        """Crawl a Confluence Cloud wiki via the REST API.

        Uses ``/wiki/rest/api/content`` with Basic auth to enumerate and
        fetch all pages.  Falls back to all-spaces enumeration when the
        seed URL does not target a specific space.
        """
        from bs4 import BeautifulSoup

        parsed = urlparse(seed_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        seed_netloc = parsed.netloc.lower()

        cred = match_credentials(seed_url, auth_config) if auth_config else None
        if not cred:
            logger.error("No credentials for Confluence at %s", seed_netloc)
            return {
                "discovered": 0, "visited": 0, "skipped_unchanged": 0,
                "new_pages": 0, "updated": 0, "failed": 0,
                "repos_found": 0, "stale_pages": 0, "elapsed_s": 0.0,
            }

        headers = {**build_auth_headers(cred), "Accept": "application/json"}

        store = ResearchStore(db_path)
        store.mark_domain_pages_stale(seed_netloc)

        stats = {
            "discovered": 0, "visited": 0, "skipped_unchanged": 0,
            "new_pages": 0, "updated": 0, "failed": 0,
            "repos_found": 0, "stale_pages": 0, "elapsed_s": 0.0,
        }
        stats_lock = asyncio.Lock()
        start_time = time.monotonic()
        rate_limiter = AsyncRateLimiter()

        space_key = _extract_confluence_space_key(seed_url)

        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "WebResearcher/1.0"},
        ) as client:
            # Enumerate pages
            all_pages: list[dict] = []
            if space_key:
                pages = await self._confluence_list_pages(
                    client, base_url, headers, rate_limiter, space_key,
                )
                all_pages.extend(pages)
            else:
                spaces = await self._confluence_list_spaces(
                    client, base_url, headers, rate_limiter,
                )
                logger.info("Found %d Confluence spaces", len(spaces))
                for space in spaces:
                    pages = await self._confluence_list_pages(
                        client, base_url, headers, rate_limiter,
                        space["key"],
                    )
                    all_pages.extend(pages)

            stats["discovered"] = len(all_pages)
            logger.info(
                "Discovered %d Confluence pages to process", len(all_pages),
            )

            # Process pages with concurrency control
            sem = asyncio.Semaphore(concurrency)

            async def _process(page_data: dict) -> None:
                async with sem:
                    await self._process_confluence_page(
                        page_data, base_url, store, stats, stats_lock,
                    )
                    async with stats_lock:
                        current = dict(stats)
                        current["elapsed_s"] = time.monotonic() - start_time
                    if progress_callback:
                        progress_callback(current)
                    else:
                        self._default_progress(current)

            tasks = [asyncio.create_task(_process(p)) for p in all_pages]
            await asyncio.gather(*tasks, return_exceptions=True)

        # Stale page reporting
        stale_pages = store.get_stale_pages(seed_netloc)
        stats["stale_pages"] = len(stale_pages)
        stats["elapsed_s"] = time.monotonic() - start_time

        if stale_pages:
            logger.warning(
                "%d Confluence pages no longer reachable: %s",
                len(stale_pages),
                [p["url"] for p in stale_pages[:10]],
            )

        logger.info(
            "Confluence crawl complete: %d pages (%d new, %d updated), "
            "%d skipped unchanged, %d stale, %d failed, %.1fs elapsed",
            stats["visited"], stats["new_pages"], stats["updated"],
            stats["skipped_unchanged"], stats["stale_pages"],
            stats["failed"], stats["elapsed_s"],
        )

        return stats

    async def _confluence_list_spaces(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        headers: dict[str, str],
        rate_limiter: AsyncRateLimiter,
    ) -> list[dict]:
        """List all global spaces via the Confluence REST API."""
        spaces: list[dict] = []
        start = 0
        limit = 100
        netloc = urlparse(base_url).netloc
        while True:
            await rate_limiter.wait(netloc)
            url = (
                f"{base_url}/wiki/rest/api/space"
                f"?start={start}&limit={limit}&type=global"
            )
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                logger.error(
                    "Confluence spaces API returned HTTP %d: %s",
                    resp.status_code, resp.text[:200],
                )
                break
            data = resp.json()
            results = data.get("results", [])
            spaces.extend(results)
            if len(results) < limit:
                break
            start += limit
        return spaces

    async def _confluence_list_pages(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        headers: dict[str, str],
        rate_limiter: AsyncRateLimiter,
        space_key: str,
    ) -> list[dict]:
        """List all pages in a Confluence space via the REST API."""
        pages: list[dict] = []
        start = 0
        limit = 100
        netloc = urlparse(base_url).netloc
        while True:
            await rate_limiter.wait(netloc)
            url = (
                f"{base_url}/wiki/rest/api/content"
                f"?type=page&spaceKey={space_key}"
                f"&start={start}&limit={limit}"
                f"&expand=body.storage,ancestors,space,version"
            )
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                logger.error(
                    "Confluence pages API (space=%s) returned HTTP %d: %s",
                    space_key, resp.status_code, resp.text[:200],
                )
                break
            data = resp.json()
            results = data.get("results", [])
            pages.extend(results)
            if len(results) < limit:
                break
            start += limit
        return pages

    async def _process_confluence_page(
        self,
        page_data: dict,
        base_url: str,
        store: ResearchStore,
        stats: dict,
        stats_lock: asyncio.Lock,
    ) -> None:
        """Process a single Confluence page from an API response."""
        from bs4 import BeautifulSoup

        try:
            page_id = page_data["id"]
            title = page_data.get("title", "")
            space = page_data.get("space", {})
            space_key = space.get("key", "")
            space_name = space.get("name", "")

            # Build the canonical web URL
            webui = page_data.get("_links", {}).get("webui", "")
            page_url = (
                f"{base_url}/wiki{webui}"
                if webui
                else f"{base_url}/wiki/spaces/{space_key}/pages/{page_id}"
            )

            body_html = (
                page_data.get("body", {})
                .get("storage", {})
                .get("value", "")
            )
            if not body_html.strip():
                logger.debug(
                    "Empty body for Confluence page %s (%s)", page_id, title,
                )
                store.mark_page_active(page_url)
                async with stats_lock:
                    stats["visited"] += 1
                return

            # Breadcrumb from ancestors
            ancestors = page_data.get("ancestors", [])
            breadcrumb_parts = [a["title"] for a in ancestors]
            if space_name:
                breadcrumb_parts.insert(0, space_name)
            breadcrumb = " > ".join(breadcrumb_parts)

            # Wrap in full HTML for the wiki chunker
            header_html = f"<h1>{title}</h1>"
            if breadcrumb:
                header_html = f"<nav>{breadcrumb}</nav>{header_html}"
            full_html = (
                f"<html><head><title>{title}</title></head>"
                f"<body>{header_html}{body_html}</body></html>"
            )

            # Plain text for content hash
            soup = BeautifulSoup(body_html, "html.parser")
            extracted_text = soup.get_text(separator=" ", strip=True)

            if not extracted_text.strip():
                logger.debug(
                    "No text in Confluence page %s (%s)", page_id, title,
                )
                store.mark_page_active(page_url)
                async with stats_lock:
                    stats["visited"] += 1
                return

            content_hash = hashlib.sha256(extracted_text.encode()).hexdigest()
            stored_hash = store.get_content_hash(page_url)

            if stored_hash == content_hash:
                logger.debug("Content unchanged for %s, skipping", page_url)
                store.mark_page_active(page_url)
                async with stats_lock:
                    stats["skipped_unchanged"] += 1
                return

            async with stats_lock:
                if stored_hash is None:
                    stats["new_pages"] += 1
                else:
                    stats["updated"] += 1

            chunks = await asyncio.to_thread(
                chunk_wiki_content, full_html, page_url,
            )

            for chunk in chunks:
                embedding = await asyncio.to_thread(_make_embedding, chunk.text)
                chunk.embedding = embedding

            store.store_page(
                url=page_url,
                title=title,
                html=full_html,
                extracted_text=extracted_text,
                content_hash=content_hash,
                crawl_status="active",
            )
            if chunks:
                store.store_chunks(chunks)

            store.mark_page_active(page_url)

            async with stats_lock:
                stats["visited"] += 1

        except Exception:
            logger.warning(
                "Error processing Confluence page %s",
                page_data.get("id", "?"),
                exc_info=True,
            )
            async with stats_lock:
                stats["failed"] += 1

    # ── BFS link processing ────────────────────────────────────────────────

    async def _enqueue_links(
        self,
        links: list[dict],
        seed_netloc: str,
        queue: asyncio.Queue,
        visited: set[str],
        repo_prefixes: set[str],
        repos_found: list[str],
        stats: dict,
        stats_lock: asyncio.Lock,
    ) -> None:
        """Classify discovered links and enqueue same-domain ones for BFS."""
        for link in links:
            target = link.get("target_url", "")
            if not target:
                continue

            canon = canonicalize_url(target)

            # Skip if already visited
            if canon in visited:
                continue

            # Check if this URL falls under a known repo prefix
            parsed = urlparse(canon)
            url_path = f"{parsed.netloc}{parsed.path}"
            if any(url_path.startswith(prefix) for prefix in repo_prefixes):
                continue

            classification = classify_url(canon, seed_netloc)

            if classification == "repo":
                repos_found.append(canon)
                # Add repo prefix to prevent crawling subordinate pages
                prefix = _repo_prefix(canon)
                if prefix:
                    repo_prefixes.add(prefix)
                visited.add(canon)
            elif classification == "same-domain":
                visited.add(canon)
                await queue.put(canon)
                async with stats_lock:
                    stats["discovered"] += 1
            else:
                # External: mark as visited so we don't re-process
                visited.add(canon)

    @staticmethod
    def _default_progress(stats: dict) -> None:
        """Print progress to stderr (mirrors ResearchLoop._progress)."""
        print(
            f"\r[wiki-crawl] visited={stats['visited']} "
            f"new={stats.get('new_pages', 0)} "
            f"updated={stats.get('updated', 0)} "
            f"discovered={stats['discovered']} "
            f"skipped={stats['skipped_unchanged']} "
            f"failed={stats['failed']}",
            end="",
            file=sys.stderr,
            flush=True,
        )
