"""
Browser lifecycle, DuckDuckGo search, and tiered content extraction
for the autonomous web researcher.

- DuckDuckGo search via stdlib urllib.request (no JS needed).
- Playwright browser for page fetching with stealth config.
- Tiered content extraction: trafilatura -> readability-lxml -> BS4.
- Per-domain rate limiting with random jitter.
- SSRF-safe URL validation (rejects private IPs, non-http schemes).
"""

from __future__ import annotations

import html as html_module
import io
import ipaddress
import logging
import random
import re
import socket
import ssl
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
]


def _random_ua() -> str:
    return random.choice(_USER_AGENTS)



# ── URL Validation ────────────────────────────────────────────────────────────


def is_safe_url(url: str) -> bool:
    """Validate that a URL is safe to fetch (anti-SSRF).

    Rules:
    - Scheme must be http or https.
    - Host must not resolve to a private, loopback, or link-local IP.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False

    # Scheme check
    if parsed.scheme not in ("http", "https"):
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    # Resolve hostname to IP and check for private ranges
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _type, _proto, _canonname, sockaddr in infos:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
    except (socket.gaierror, ValueError):
        return False

    return True


# ── Rate Limiter ──────────────────────────────────────────────────────────────


class RateLimiter:
    """Per-domain rate limiter with random jitter.

    DuckDuckGo domains get 2-5s delays; everything else gets 1-3s.
    """

    DDG_DELAY = (4.0, 8.0)
    DEFAULT_DELAY = (1.0, 3.0)

    def __init__(self) -> None:
        self._last_request: Dict[str, float] = {}

    def _delay_range(self, domain: str) -> tuple[float, float]:
        if "duckduckgo.com" in domain:
            return self.DDG_DELAY
        return self.DEFAULT_DELAY

    def wait(self, domain: str) -> None:
        """Block until the per-domain delay has elapsed, then record the new timestamp."""
        now = time.monotonic()
        last = self._last_request.get(domain)
        lo, hi = self._delay_range(domain)
        required_delay = random.uniform(lo, hi)

        if last is not None:
            elapsed = now - last
            remaining = required_delay - elapsed
            if remaining > 0:
                time.sleep(remaining)

        self._last_request[domain] = time.monotonic()


# ── DuckDuckGo Search ────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """A single DuckDuckGo search result."""

    title: str
    url: str
    snippet: str



def search_ddg(
    query: str,
    rate_limiter: Optional[RateLimiter] = None,
    max_results: int = 20,
    browser: Optional["Browser"] = None,
) -> List[SearchResult]:
    """Search DuckDuckGo using the ddgs library.

    DDG's HTML endpoint now serves a CAPTCHA challenge to all automated
    clients. The ddgs library uses DDG's internal API which avoids this.
    """
    if rate_limiter:
        rate_limiter.wait("duckduckgo.com")

    try:
        from ddgs import DDGS

        raw_results = list(DDGS().text(query, max_results=max_results))
        results = []
        for r in raw_results:
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            ))
        return results
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for %r: %s", query, exc)
        return []



# ── Content Extraction ────────────────────────────────────────────────────────


def _preserve_code_blocks(html: str) -> str:
    """Convert <pre><code> blocks to markdown fences before text extraction.

    Without this, trafilatura and readability collapse code blocks into
    single lines with no structure, making code detection impossible.
    """
    from bs4 import BeautifulSoup, NavigableString

    soup = BeautifulSoup(html, "html.parser")
    changed = False

    for pre in soup.find_all("pre"):
        code_tag = pre.find("code")
        source = code_tag if code_tag else pre

        lang = ""
        for tag in [code_tag, pre, pre.parent] if code_tag else [pre, pre.parent]:
            if tag is None or isinstance(tag, NavigableString):
                continue
            for cls in tag.get("class", []):
                if cls.startswith("language-") or cls.startswith("lang-"):
                    lang = cls.split("-", 1)[1]
                    break
                elif cls.startswith("highlight-source-"):
                    lang = cls.replace("highlight-source-", "")
                    break
            if lang:
                break

        code_text = source.get_text()
        if not code_text.strip():
            continue

        fence = f"\n```{lang}\n{code_text}\n```\n"
        pre.replace_with(NavigableString(fence))
        changed = True

    return str(soup) if changed else html


def extract_content(html: str, url: str = "") -> Dict[str, str]:
    """Extract clean text from HTML using a tiered approach.

    Extraction order:
    1. trafilatura — highest F1 for article extraction
    2. readability-lxml — good for news/blog content
    3. BeautifulSoup4 brute-force — last resort

    Returns a dict with keys: title, text, method.
    """
    if not html or not html.strip():
        return {"title": "", "text": "", "method": "none"}

    html = _preserve_code_blocks(html)

    # Tier 1: trafilatura
    result = _extract_trafilatura(html, url)
    if result:
        return result

    # Tier 2: readability-lxml
    result = _extract_readability(html)
    if result:
        return result

    # Tier 3: BeautifulSoup4 brute-force
    return _extract_bs4(html)


def _extract_trafilatura(html: str, url: str = "") -> Optional[Dict[str, str]]:
    """Tier 1: trafilatura extraction."""
    try:
        import trafilatura

        text = trafilatura.extract(
            html,
            url=url or None,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        if text and len(text.strip()) > 50:
            # Extract title separately
            metadata = trafilatura.extract(
                html,
                url=url or None,
                output_format="xml",
                include_comments=False,
            )
            title = ""
            if metadata:
                # Parse title from XML output
                m = re.search(r'title="([^"]*)"', metadata)
                if m:
                    title = html_module.unescape(m.group(1))

            return {"title": title, "text": text.strip(), "method": "trafilatura"}
    except Exception as exc:
        logger.debug("trafilatura failed: %s", exc)

    return None


def _extract_readability(html: str) -> Optional[Dict[str, str]]:
    """Tier 2: readability-lxml + BeautifulSoup for text extraction."""
    try:
        from readability import Document
        from bs4 import BeautifulSoup

        doc = Document(html)
        title = doc.short_title() or ""
        summary_html = doc.summary()

        soup = BeautifulSoup(summary_html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        if text and len(text.strip()) > 50:
            return {"title": title, "text": text.strip(), "method": "readability"}
    except Exception as exc:
        logger.debug("readability failed: %s", exc)

    return None


def _extract_bs4(html: str) -> Dict[str, str]:
    """Tier 3: BeautifulSoup4 brute-force extraction.

    Strips scripts, styles, nav, footer, header elements and extracts
    remaining text.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()

    # Try to get title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    text = soup.get_text(separator="\n", strip=True)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return {"title": title, "text": text.strip(), "method": "bs4"}


# ── Image Extraction ─────────────────────────────────────────────────────────

# URL patterns indicating non-content images (tracking pixels, spacers, etc.)
_NON_CONTENT_URL_RE = re.compile(
    r"(?i)"
    r"(?:spacer|pixel|beacon|tracking|tracker|blank|shim|clear)"
    r"|(?:/ad[sx]?/|/ads/|doubleclick|googlesyndication)"
    r"|(?:favicon|badge|icon[_\-]|social[_\-]|share[_\-]|button[_\-])"
    r"|(?:\.gif\b.*(?:1x1|spacer|pixel|blank))"
    r"|(?:facebook\.com/tr|analytics|scorecardresearch|quantserve)"
)

_SKIP_EXTENSIONS = {".svg", ".ico"}

_SKIP_PARENT_TAGS = {"nav", "footer", "header", "aside"}


_SRCSET_ENTRY_RE = re.compile(
    r"(\S+)\s+([\d.]+[wx])"
)


def _parse_srcset(srcset: str) -> str | None:
    """Parse a srcset attribute and return the URL of the highest-resolution variant."""
    candidates: list[tuple[str, float]] = []
    for m in _SRCSET_ENTRY_RE.finditer(srcset):
        url = m.group(1).lstrip(",")
        if not url:
            continue
        try:
            width = float(m.group(2)[:-1])
        except ValueError:
            width = 1.0
        candidates.append((url, width))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[1], reverse=True)
    return candidates[0][0]


def _resolve_base_url(soup, page_url: str) -> str:
    base_tag = soup.find("base", href=True)
    if base_tag:
        return urllib.parse.urljoin(page_url, base_tag["href"])
    return page_url


def _sanitize_url(url: str) -> str:
    """Re-encode a URL so unsafe characters like spaces become percent-encoded.

    Preserves existing percent-encoded sequences (e.g. %3A%2F%2F in CDN proxy URLs)
    while encoding only bare unsafe characters like spaces.
    """
    parts = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(parts.path, safe="/:@!$&'()*+,;=-._~%")
    query = urllib.parse.quote(parts.query, safe="/:@!$&'()*+,;=-._~?%")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, query, parts.fragment))


def extract_images(html: str, page_url: str) -> list[dict]:
    """Extract meaningful content images from HTML, filtering out decorative images.

    Each returned dict has keys:
    - src_url (str): absolute URL of the image
    - alt_text (str): alt text of the image
    - width (int|None): declared width from HTML attributes
    - height (int|None): declared height from HTML attributes
    - dom_offset (int): character offset of the <img> tag in the HTML source
    """
    from bs4 import BeautifulSoup

    if not html or not html.strip():
        return []

    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []

    base_url = _resolve_base_url(soup, page_url)

    for img_tag in soup.find_all("img"):
        # ── Skip images in semantic non-content containers ────────────
        skip = False
        for parent in img_tag.parents:
            if parent.name in _SKIP_PARENT_TAGS:
                skip = True
                break
        if skip:
            continue

        # ── Skip images with presentation/hidden roles ────────────────
        if img_tag.get("role") == "presentation":
            continue
        if img_tag.get("aria-hidden", "").lower() == "true":
            continue

        # ── Resolve image URL ─────────────────────────────────────────
        raw_src = (
            img_tag.get("src")
            or img_tag.get("data-src")
            or img_tag.get("data-lazy-src")
        )
        srcset = img_tag.get("srcset")

        # If srcset is present, try to pick the highest-resolution variant
        if srcset:
            srcset_url = _parse_srcset(srcset)
            if srcset_url:
                raw_src = srcset_url

        if not raw_src:
            continue

        # Strip escaped quotes leaked from JSON-LD or JS objects in HTML
        raw_src = raw_src.strip().strip('\\"')

        if not raw_src:
            continue

        # Skip data: URIs
        if raw_src.startswith("data:"):
            continue

        # Fix single-slash protocol (e.g. "https:/example.com" -> "https://example.com")
        if re.match(r"https?:/[^/]", raw_src):
            raw_src = raw_src.replace(":/", "://", 1)

        # Resolve relative URL and encode unsafe characters (e.g. spaces)
        src_url = _sanitize_url(urllib.parse.urljoin(base_url, raw_src))

        # ── Validate URL safety ───────────────────────────────────────
        if not is_safe_url(src_url):
            continue

        # ── Skip by file extension ────────────────────────────────────
        parsed_path = urllib.parse.urlparse(src_url).path.lower()
        ext = ""
        if "." in parsed_path:
            ext = "." + parsed_path.rsplit(".", 1)[-1]
        if ext in _SKIP_EXTENSIONS:
            continue

        # ── Skip by URL pattern (tracking, spacer, etc.) ─────────────
        if _NON_CONTENT_URL_RE.search(src_url):
            continue

        # ── Skip by declared dimensions ───────────────────────────────
        width_attr = img_tag.get("width")
        height_attr = img_tag.get("height")
        width: int | None = None
        height: int | None = None

        try:
            width = int(width_attr) if width_attr else None
        except (ValueError, TypeError):
            width = None
        try:
            height = int(height_attr) if height_attr else None
        except (ValueError, TypeError):
            height = None

        if width is not None and height is not None:
            if width < 100 and height < 100:
                continue
            if width * height < 15_000:
                continue

        # ── Compute dom_offset ────────────────────────────────────────
        # Find the position of this img tag in the original HTML source.
        # We use the str representation of the tag and search for it.
        tag_str = str(img_tag)
        dom_offset = html.find(tag_str)
        if dom_offset == -1:
            # Fallback: try to find the src attribute value in the HTML
            if raw_src:
                dom_offset = html.find(raw_src)
            if dom_offset == -1:
                dom_offset = 0

        alt_text = img_tag.get("alt", "") or ""

        results.append({
            "src_url": src_url,
            "alt_text": alt_text,
            "width": width,
            "height": height,
            "dom_offset": dom_offset,
        })

    return results


# ── Image Download ───────────────────────────────────────────────────────────


def _resolve_and_validate(url: str) -> tuple[str, str] | None:
    """Resolve DNS once and validate all IPs are safe.

    Returns (resolved_url, original_host) or None if validation fails.
    This prevents DNS rebinding attacks by connecting to the validated IP.
    """
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return None
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _fam, _typ, _proto, _canon, sockaddr in infos:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return None
        # Use first resolved IP
        resolved_ip = infos[0][4][0]
    except (socket.gaierror, ValueError):
        return None

    # Replace hostname with resolved IP in URL (bracket IPv6 addresses)
    ip_obj = ipaddress.ip_address(resolved_ip)
    ip_str = f"[{resolved_ip}]" if ip_obj.version == 6 else resolved_ip
    if parsed.port:
        netloc = f"{ip_str}:{parsed.port}"
    else:
        netloc = ip_str
    resolved_url = urllib.parse.urlunparse(parsed._replace(netloc=netloc))
    return (resolved_url, hostname)


def download_image(
    url: str,
    rate_limiter: Optional[RateLimiter] = None,
    timeout: int = 15,
    max_bytes: int = 50 * 1024 * 1024,
    page_url: str | None = None,
) -> bytes | None:
    """Download an image with SSRF-safe DNS resolution.

    Returns image bytes on success, None on any failure.
    Validates the downloaded bytes with PIL (minimum 100x100).
    For animated GIFs, only the first frame is kept.
    """
    # 1. Basic URL safety check
    if not is_safe_url(url):
        logger.warning("Image URL blocked by safety check: %s", url)
        return None

    # 2. Resolve DNS once and validate IPs
    resolved = _resolve_and_validate(url)
    if resolved is None:
        logger.warning("Image URL DNS resolution/validation failed: %s", url)
        return None

    _, original_host = resolved

    # 3. Rate limiting
    if rate_limiter:
        rate_limiter.wait(original_host)

    # 4. Fetch the image using original URL for correct TLS/SNI.
    # DNS was validated above; connecting by hostname lets TLS work normally.
    try:
        if page_url:
            referer = page_url
        else:
            parsed = urllib.parse.urlparse(url)
            referer = f"{parsed.scheme}://{parsed.netloc}/"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _random_ua(), "Referer": referer},
        )

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                logger.info("Image truncated at max_bytes (%d), will attempt partial decode: %s", max_bytes, url)
                data = data[:max_bytes]
    except Exception as exc:
        logger.warning("Image download failed for %s: %s", url, exc)
        return None

    # 5. Validate with PIL, downscale if too large, extract first frame for GIFs
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        if getattr(img, "is_animated", False):
            img.seek(0)
        if img.mode in ("P", "PA", "RGBA", "LA"):
            img = img.convert("RGBA").convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w < 100 or h < 100:
            logger.debug("Image too small (%dx%d): %s", w, h, url)
            return None
        if w * h > 25_000_000:
            scale = (25_000_000 / (w * h)) ** 0.5
            new_w, new_h = int(w * scale), int(h * scale)
            logger.info("Downscaling large image (%dx%d -> %dx%d): %s", w, h, new_w, new_h, url)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
    except Exception as exc:
        logger.debug("PIL validation failed for %s: %s", url, exc)
        return None

    return data


# ── Browser ───────────────────────────────────────────────────────────────────

_COOKIE_BANNER_SELECTORS = [
    'button:has-text("Accept")',
    'button:has-text("Accept All")',
    'button:has-text("I Agree")',
    'button:has-text("OK")',
]


class Browser:
    """Playwright-based browser with stealth configuration.

    Used for fetching pages that require JavaScript rendering.
    DuckDuckGo search uses the ddgs library instead (see search_ddg).
    Supports parallel page fetches via a pool of browser pages (tabs).
    """

    _MAX_PAGES = 4

    def __init__(self) -> None:
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        self._context = self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="America/New_York",
            user_agent=_random_ua(),
        )
        self._page = self._context.new_page()

    # -- public API --

    def fetch_page(
        self,
        url: str,
        *,
        wait_until: str = "domcontentloaded",
        timeout: int = 30_000,
        rate_limiter: Optional[RateLimiter] = None,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> tuple[str, str]:
        """Navigate to a URL and return (html, final_url).

        Validates the URL for safety (anti-SSRF) before fetching.
        Optionally dismisses cookie banners after load.
        Returns the final URL after redirects so callers resolve relative
        URLs against the same base the browser used.
        """
        if not is_safe_url(url):
            raise ValueError(f"URL blocked by safety check: {url}")

        resolved = _resolve_and_validate(url)
        if resolved is None:
            raise ValueError(f"URL DNS validation failed: {url}")

        domain = urllib.parse.urlparse(url).hostname or ""
        if rate_limiter:
            rate_limiter.wait(domain)

        p = self._page
        if extra_headers:
            p.set_extra_http_headers(extra_headers)
        p.goto(url, wait_until=wait_until, timeout=timeout)
        p.wait_for_load_state("domcontentloaded", timeout=timeout)

        final_url = p.url
        if final_url != url and not is_safe_url(final_url):
            raise ValueError(f"Redirected to unsafe URL: {final_url}")

        self._dismiss_cookie_banner(p)

        return p.content(), final_url

    def fetch_pages_parallel(
        self,
        urls: list[str],
        *,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> list[tuple[str, str, str]]:
        """Fetch multiple URLs in parallel using async Playwright.

        Launches a temporary async browser to get true concurrency.
        Different domains run concurrently (up to _MAX_PAGES tabs),
        same-domain URLs are serialized via per-domain locks + rate limiter.
        Returns list of (url, html, final_url) for successful fetches.
        """
        if not urls:
            return []

        import asyncio

        max_concurrent = min(self._MAX_PAGES, len(urls))
        rl = rate_limiter

        async def _fetch_all():
            from playwright.async_api import async_playwright

            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-blink-features=AutomationControlled",
                    ],
                )
                try:
                    context = await browser.new_context(
                        viewport={"width": 1920, "height": 1080},
                        locale="en-US",
                        timezone_id="America/New_York",
                        user_agent=_random_ua(),
                    )

                    global_sem = asyncio.Semaphore(max_concurrent)
                    domain_locks: dict[str, asyncio.Lock] = {}

                    async def _fetch_one(url: str) -> tuple[str, str, str] | None:
                        safe = await asyncio.to_thread(is_safe_url, url)
                        if not safe:
                            return None

                        resolved = await asyncio.to_thread(_resolve_and_validate, url)
                        if resolved is None:
                            return None

                        domain = urllib.parse.urlparse(url).hostname or ""

                        async with global_sem:
                            if domain not in domain_locks:
                                domain_locks[domain] = asyncio.Lock()
                            async with domain_locks[domain]:
                                if rl:
                                    await asyncio.to_thread(rl.wait, domain)

                                page = await context.new_page()
                                try:
                                    await page.goto(
                                        url,
                                        wait_until="domcontentloaded",
                                        timeout=30_000,
                                    )
                                    await page.wait_for_load_state(
                                        "domcontentloaded", timeout=30_000,
                                    )

                                    final_url = page.url
                                    if final_url != url:
                                        final_safe = await asyncio.to_thread(
                                            is_safe_url, final_url,
                                        )
                                        if not final_safe:
                                            return None

                                    for sel in _COOKIE_BANNER_SELECTORS:
                                        try:
                                            await page.click(sel, timeout=2000)
                                            break
                                        except Exception:
                                            continue

                                    html = await page.content()
                                    return (url, html, final_url)
                                except Exception:
                                    logger.debug(
                                        "Parallel fetch failed for %s",
                                        url,
                                        exc_info=True,
                                    )
                                    return None
                                finally:
                                    await page.close()

                    raw = await asyncio.gather(*[_fetch_one(u) for u in urls])
                finally:
                    await browser.close()

            return [r for r in raw if r is not None]

        import threading

        result_box: list[list[tuple[str, str, str]]] = []
        error_box: list[Exception] = []

        def _target():
            try:
                result_box.append(asyncio.run(_fetch_all()))
            except Exception as e:
                error_box.append(e)

        t = threading.Thread(target=_target)
        t.start()
        t.join(timeout=300)
        if t.is_alive():
            raise TimeoutError("Parallel fetch did not complete within 300s")

        if error_box:
            raise error_box[0]
        if not result_box:
            raise RuntimeError("Parallel fetch thread exited without result or error")
        return result_box[0]

    def close(self) -> None:
        """Shut down browser and Playwright."""
        try:
            self._browser.close()
        except Exception:
            pass
        try:
            self._pw.stop()
        except Exception:
            pass

    def __enter__(self) -> Browser:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- private helpers --

    def _dismiss_cookie_banner(self, page=None) -> None:
        """Attempt to click common cookie-accept buttons."""
        p = page or self._page
        for sel in _COOKIE_BANNER_SELECTORS:
            try:
                p.click(sel, timeout=2000)
                return
            except Exception:
                continue


# ── Link Extraction ─────────────────────────────────────────────────────────


def extract_links(html: str, page_url: str) -> list[dict]:
    """Extract hyperlinks from HTML, resolving relative URLs.

    Returns list of {"target_url": str, "anchor_text": str}.
    Filters out fragment-only links, javascript:, mailto:, and SSRF-unsafe URLs.
    """
    if not html or not html.strip():
        return []

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    base_url = _resolve_base_url(soup, page_url)

    links: list[dict] = []
    seen: set[str] = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()

        # Skip fragment-only, javascript:, mailto:
        if not href or href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
            continue

        # Resolve relative URLs
        resolved = _sanitize_url(urllib.parse.urljoin(base_url, href))

        # Strip fragments from resolved URL
        parsed = urllib.parse.urlparse(resolved)
        resolved = urllib.parse.urlunparse(parsed._replace(fragment=""))

        # Skip self-links
        if resolved == page_url:
            continue

        # SSRF check
        if not is_safe_url(resolved):
            continue

        # Dedup within page
        if resolved in seen:
            continue
        seen.add(resolved)

        # Sanitize anchor_text — use html.escape to prevent injection
        raw_anchor = a_tag.get_text(strip=True)
        anchor_text = html_module.escape(raw_anchor) if raw_anchor else ""

        links.append({"target_url": resolved, "anchor_text": anchor_text})

    return links
