"""
Wiki-aware HTML chunker for the web researcher.

Walks the DOM tree produced by BeautifulSoup, splitting at header tags
(h1-h6) to create section-bounded chunks that preserve wiki structure:
tables kept intact, header hierarchies used as section boundaries,
code blocks preserved as atomic units, and cross-page links recorded
as metadata.
"""

from __future__ import annotations

import hashlib
import logging
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag

from research_tool.store import DocumentChunk, chunk_web_content, classify_chunk_content_type

logger = logging.getLogger(__name__)

# Tags considered non-content (navigation/boilerplate)
_BOILERPLATE_TAGS = frozenset({"nav", "footer", "header", "aside", "script", "style", "noscript"})

# Header tags that define section boundaries
_HEADER_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

# Header tag to numeric level
_HEADER_LEVEL = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def _token_count(text: str) -> int:
    """Simple word-split token approximation."""
    return len(text.split())


def _make_chunk_id(page_url: str, section_path: str, index: int) -> str:
    """Generate a deterministic chunk ID from URL + section path + index."""
    key = f"{page_url}::{section_path}"
    prefix = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"{prefix}::chunk-{index}"


def _table_to_markdown(table_tag: Tag) -> str:
    """Convert an HTML <table> to a markdown table string.

    Handles <thead>/<tbody> structure as well as bare <tr> rows.
    If the table has <th> cells in the first row, those become the header row.
    """
    rows: list[list[str]] = []
    header_row_index: int | None = None

    for tr in table_tag.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        if not cell_texts:
            continue
        # Detect header row: first row with <th> cells
        if header_row_index is None and tr.find("th"):
            header_row_index = len(rows)
        rows.append(cell_texts)

    if not rows:
        return ""

    # Normalize column count across rows
    max_cols = max(len(r) for r in rows)
    for row in rows:
        while len(row) < max_cols:
            row.append("")

    lines: list[str] = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(row) + " |")
        # Add separator after header row
        if i == header_row_index:
            lines.append("| " + " | ".join("---" for _ in row) + " |")

    # If no header was detected, add a separator after the first row anyway
    if header_row_index is None and len(rows) > 1:
        lines.insert(1, "| " + " | ".join("---" for _ in rows[0]) + " |")

    return "\n".join(lines)


def _extract_links(html_fragment: str | Tag, page_url: str) -> list[str]:
    """Extract all <a href> links from an HTML fragment, resolving to absolute URLs."""
    if isinstance(html_fragment, str):
        soup = BeautifulSoup(html_fragment, "html.parser")
    else:
        soup = html_fragment

    links: list[str] = []
    seen: set[str] = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        absolute = urljoin(page_url, href)
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links


def _strip_boilerplate(soup: BeautifulSoup) -> None:
    """Remove navigation, footer, script, and other boilerplate tags in place."""
    for tag in soup.find_all(list(_BOILERPLATE_TAGS)):
        tag.decompose()


def _has_wiki_structure(soup: BeautifulSoup) -> bool:
    """Check whether the HTML has headers or tables (wiki-like structure)."""
    if soup.find(list(_HEADER_TAGS)):
        return True
    if soup.find("table"):
        return True
    return False


def _flatten_elements(container: Tag) -> list[Tag | NavigableString]:
    """Recursively flatten a container element, pulling out tables, pre blocks,
    and regular content nodes into a flat list.

    For container tags (div, section, article, span, etc.) that hold a mix
    of content, this ensures we can process tables and code blocks that are
    nested inside wrapper divs.
    """
    result: list[Tag | NavigableString] = []
    for child in container.children:
        if isinstance(child, NavigableString):
            result.append(child)
            continue
        if not isinstance(child, Tag):
            continue
        # Atomic elements: tables and pre blocks stay as-is
        if child.name in ("table", "pre"):
            result.append(child)
        # Headers should stay as-is (they were already processed for section splitting)
        elif child.name in _HEADER_TAGS:
            result.append(child)
        # Container elements: flatten recursively
        elif child.find("table") or child.find("pre"):
            result.extend(_flatten_elements(child))
        else:
            # Regular content element
            result.append(child)
    return result


def _collect_sections(soup: BeautifulSoup) -> list[tuple[list[str], list[Tag | NavigableString]]]:
    """Walk the cleaned DOM and split into sections bounded by header tags.

    Returns a list of (header_breadcrumb, content_elements) tuples.
    The breadcrumb is a list like ["Getting Started", "Installation", "Linux"].
    Content elements are the DOM nodes between this header and the next.
    """
    body = soup.find("body") or soup

    sections: list[tuple[list[str], list[Tag | NavigableString]]] = []
    header_stack: list[tuple[int, str]] = []  # (level, title)
    current_elements: list[Tag | NavigableString] = []
    current_breadcrumb: list[str] = []

    def _flush_section():
        nonlocal current_elements
        if current_elements:
            sections.append((list(current_breadcrumb), list(current_elements)))
            current_elements = []

    for element in body.children:
        if isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                current_elements.append(element)
            continue

        if not isinstance(element, Tag):
            continue

        tag_name = element.name

        if tag_name in _HEADER_TAGS:
            _flush_section()

            level = _HEADER_LEVEL[tag_name]
            title = element.get_text(strip=True)

            # Pop headers at same or deeper level to maintain hierarchy
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()

            header_stack.append((level, title))
            current_breadcrumb = [h[1] for h in header_stack]
        else:
            current_elements.append(element)

    _flush_section()
    return sections


def chunk_wiki_content(
    html: str,
    page_url: str,
    section_title: str | None = None,
    max_tokens: int = 500,
) -> list[DocumentChunk]:
    """Chunk raw HTML content preserving wiki structure.

    Splits at header tags (h1-h6) to create section-bounded chunks,
    keeps tables intact as markdown, preserves code blocks as atomic units,
    and records cross-page links as metadata.

    Args:
        html: Raw HTML content to chunk.
        page_url: The URL of the page (used for chunk IDs and resolving relative links).
        section_title: Optional override for the top-level section title.
        max_tokens: Maximum token count per chunk (tables/code blocks may exceed this).

    Returns:
        A list of DocumentChunk instances.
    """
    if not html or not html.strip():
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Strip boilerplate
    _strip_boilerplate(soup)

    # Check remaining content
    remaining_text = soup.get_text(strip=True)
    if not remaining_text:
        return []

    # If no wiki structure, fall back to chunk_web_content
    if not _has_wiki_structure(soup):
        plain_text = soup.get_text(separator="\n\n")
        return chunk_web_content(
            text=plain_text,
            page_url=page_url,
            section_title=section_title or "",
            max_tokens=max_tokens,
        )

    sections = _collect_sections(soup)

    if not sections:
        return []

    chunks: list[DocumentChunk] = []
    global_chunk_index = 0

    for breadcrumb, elements in sections:
        # Build the section title from breadcrumb
        if breadcrumb:
            sec_title = " > ".join(breadcrumb)
        elif section_title:
            sec_title = section_title
        else:
            sec_title = "(untitled)"

        # Prepend the outer section_title if provided and breadcrumb exists
        if section_title and breadcrumb:
            sec_title = section_title + " > " + sec_title

        # Accumulator state for this section
        current_parts: list[str] = []
        current_tokens = 0
        current_html_elements: list[Tag | NavigableString] = []

        def _flush_accumulator():
            """Flush accumulated text parts into a DocumentChunk."""
            nonlocal current_parts, current_tokens, current_html_elements, global_chunk_index
            if not current_parts:
                return
            merged = "\n\n".join(current_parts)
            if not merged.strip():
                current_parts = []
                current_tokens = 0
                current_html_elements = []
                return

            # Extract cross-links from accumulated HTML elements
            cross_links: list[str] = []
            seen_links: set[str] = set()
            for el in current_html_elements:
                for link in _extract_links(el, page_url):
                    if link not in seen_links:
                        seen_links.add(link)
                        cross_links.append(link)

            chunk_id = _make_chunk_id(page_url, sec_title, global_chunk_index)
            content_type = classify_chunk_content_type(merged)
            chunks.append(DocumentChunk(
                text=merged,
                page_url=page_url,
                chunk_id=chunk_id,
                section_title=sec_title,
                content_type=content_type,
                cross_links=cross_links if cross_links else None,
            ))
            global_chunk_index += 1
            current_parts = []
            current_tokens = 0
            current_html_elements = []

        def _add_atomic_chunk(text: str, content_type: str, source_element: Tag | None):
            """Add an atomic chunk (table or oversized code) directly."""
            nonlocal global_chunk_index
            links = _extract_links(source_element, page_url) if source_element else []
            chunk_id = _make_chunk_id(page_url, sec_title, global_chunk_index)
            chunks.append(DocumentChunk(
                text=text,
                page_url=page_url,
                chunk_id=chunk_id,
                section_title=sec_title,
                content_type=content_type,
                cross_links=links if links else None,
            ))
            global_chunk_index += 1

        # Flatten elements to handle nested containers
        flat_elements = []
        for el in elements:
            if isinstance(el, NavigableString):
                flat_elements.append(el)
            elif isinstance(el, Tag) and el.name not in ("table", "pre") and (el.find("table") or el.find("pre")):
                flat_elements.extend(_flatten_elements(el))
            else:
                flat_elements.append(el)

        for element in flat_elements:
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    tc = _token_count(text)
                    if current_tokens + tc > max_tokens and current_parts:
                        _flush_accumulator()
                    current_parts.append(text)
                    current_tokens += tc
                    current_html_elements.append(element)
                continue

            if not isinstance(element, Tag):
                continue

            # Handle tables: convert to markdown and keep atomic
            if element.name == "table":
                md_table = _table_to_markdown(element)
                if not md_table:
                    continue

                _flush_accumulator()
                ct = classify_chunk_content_type(md_table)
                _add_atomic_chunk(md_table, ct, element)
                continue

            # Handle code blocks: <pre> tags preserved intact
            if element.name == "pre":
                code_text = element.get_text()
                if not code_text.strip():
                    continue

                code_tc = _token_count(code_text)

                if current_tokens + code_tc <= max_tokens:
                    # Code block fits in current accumulator
                    current_parts.append(code_text)
                    current_tokens += code_tc
                    current_html_elements.append(element)
                else:
                    _flush_accumulator()
                    if code_tc <= max_tokens:
                        # Start a new accumulator with this code block
                        current_parts.append(code_text)
                        current_tokens += code_tc
                        current_html_elements.append(element)
                    else:
                        # Oversized code block: its own atomic chunk
                        _add_atomic_chunk(code_text, "code", element)
                continue

            # Regular content element
            text = element.get_text(separator=" ", strip=True)
            if not text:
                continue
            tc = _token_count(text)
            if current_tokens + tc > max_tokens and current_parts:
                _flush_accumulator()
            current_parts.append(text)
            current_tokens += tc
            current_html_elements.append(element)

        # Flush remaining content in this section
        _flush_accumulator()

    return chunks
