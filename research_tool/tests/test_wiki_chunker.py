"""Tests for research_tool.wiki_chunker — wiki-aware HTML chunking."""

import hashlib

import pytest

from research_tool.wiki_chunker import (
    chunk_wiki_content,
    _table_to_markdown,
    _extract_links,
    _token_count,
    _make_chunk_id,
)
from research_tool.store import DocumentChunk


# ── Helpers ──────────────────────────────────────────────────────────────────

PAGE_URL = "https://wiki.example.com/docs/getting-started"


def _find_chunk_by_section(chunks: list[DocumentChunk], substring: str) -> DocumentChunk | None:
    """Find the first chunk whose section_title contains the given substring."""
    for c in chunks:
        if substring in c.section_title:
            return c
    return None


def _all_section_titles(chunks: list[DocumentChunk]) -> list[str]:
    return [c.section_title for c in chunks]


# ── Happy Path: Header-bounded sections ──────────────────────────────────────


class TestHeaderBoundedSections:
    """HTML with h1/h2/h3 headers produces chunks bounded by header sections,
    with correct breadcrumb section_titles."""

    def test_simple_h1_h2_sections(self):
        html = """
        <h1>Getting Started</h1>
        <p>Welcome to the project. This guide will help you set up everything.</p>
        <h2>Installation</h2>
        <p>You can install with pip. Run the install command to proceed.</p>
        <h2>Configuration</h2>
        <p>Edit your config file to set up the project correctly.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        assert len(chunks) >= 3

        titles = _all_section_titles(chunks)
        assert any("Getting Started" in t for t in titles)
        assert any("Installation" in t for t in titles)
        assert any("Configuration" in t for t in titles)

    def test_breadcrumb_hierarchy(self):
        html = """
        <h1>Guide</h1>
        <p>Overview text for the guide section.</p>
        <h2>Setup</h2>
        <p>Setup instructions for the project configuration.</p>
        <h3>Linux</h3>
        <p>Linux-specific instructions for installation and running.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        linux_chunk = _find_chunk_by_section(chunks, "Linux")
        assert linux_chunk is not None
        assert "Guide" in linux_chunk.section_title
        assert "Setup" in linux_chunk.section_title
        assert "Linux" in linux_chunk.section_title
        # Should be breadcrumb format
        assert "Guide > Setup > Linux" in linux_chunk.section_title

    def test_h2_resets_when_new_h1_appears(self):
        html = """
        <h1>Chapter One</h1>
        <h2>Section A</h2>
        <p>Content for section A in chapter one text.</p>
        <h1>Chapter Two</h1>
        <h2>Section B</h2>
        <p>Content for section B in chapter two text.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        section_b = _find_chunk_by_section(chunks, "Section B")
        assert section_b is not None
        # Section B should be under Chapter Two, not Chapter One
        assert "Chapter Two" in section_b.section_title
        assert "Chapter One" not in section_b.section_title


# ── Happy Path: Table serialization ──────────────────────────────────────────


class TestTableSerialization:
    """HTML table is serialized as markdown table and kept as a single chunk."""

    def test_simple_table_to_markdown(self):
        html = """
        <h1>Data</h1>
        <table>
            <tr><th>Name</th><th>Value</th></tr>
            <tr><td>Alpha</td><td>1</td></tr>
            <tr><td>Beta</td><td>2</td></tr>
        </table>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        # Find the chunk containing the table
        table_chunk = None
        for c in chunks:
            if "Alpha" in c.text and "Beta" in c.text:
                table_chunk = c
                break
        assert table_chunk is not None
        # Should be markdown table format
        assert "| Name | Value |" in table_chunk.text
        assert "| --- | --- |" in table_chunk.text
        assert "| Alpha | 1 |" in table_chunk.text
        assert "| Beta | 2 |" in table_chunk.text

    def test_table_is_atomic_chunk(self):
        """A table should be in its own chunk, not merged with surrounding text."""
        html = """
        <h1>Report</h1>
        <p>Here is some introductory text before the table.</p>
        <table>
            <tr><th>Col1</th><th>Col2</th></tr>
            <tr><td>X</td><td>Y</td></tr>
        </table>
        <p>Here is some text after the table that follows.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        table_chunk = None
        for c in chunks:
            if "| Col1 | Col2 |" in c.text:
                table_chunk = c
                break
        assert table_chunk is not None
        # The table chunk should NOT contain the surrounding paragraph text
        assert "introductory" not in table_chunk.text
        assert "after the table" not in table_chunk.text


class TestTableToMarkdownHelper:
    """Unit tests for the _table_to_markdown helper."""

    def test_basic_conversion(self):
        from bs4 import BeautifulSoup

        table_html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        soup = BeautifulSoup(table_html, "html.parser")
        table = soup.find("table")
        result = _table_to_markdown(table)
        assert "| A | B |" in result
        assert "| --- | --- |" in result
        assert "| 1 | 2 |" in result

    def test_empty_table(self):
        from bs4 import BeautifulSoup

        table_html = "<table></table>"
        soup = BeautifulSoup(table_html, "html.parser")
        table = soup.find("table")
        result = _table_to_markdown(table)
        assert result == ""

    def test_no_header_row(self):
        from bs4 import BeautifulSoup

        table_html = "<table><tr><td>X</td><td>Y</td></tr><tr><td>1</td><td>2</td></tr></table>"
        soup = BeautifulSoup(table_html, "html.parser")
        table = soup.find("table")
        result = _table_to_markdown(table)
        # Should still produce a table with separator
        assert "| X | Y |" in result
        assert "---" in result


# ── Happy Path: Code block preservation ──────────────────────────────────────


class TestCodeBlockPreservation:
    """Code block (<pre><code>) is preserved intact within a chunk."""

    def test_code_block_preserved(self):
        html = """
        <h1>Example</h1>
        <p>Here is how to use the function properly:</p>
        <pre><code>def hello():
    print("Hello, world!")

hello()</code></pre>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        code_chunk = None
        for c in chunks:
            if 'def hello()' in c.text:
                code_chunk = c
                break
        assert code_chunk is not None
        # The code should be preserved with its formatting
        assert 'print("Hello, world!")' in code_chunk.text
        assert "hello()" in code_chunk.text

    def test_code_block_not_split(self):
        """A code block should never be split across chunk boundaries."""
        # Create a code block that's larger than max_tokens
        code_lines = [f"    line_{i} = compute(value_{i})" for i in range(60)]
        code_body = "\n".join(code_lines)
        html = f"""
        <h1>Code</h1>
        <p>Some text before.</p>
        <pre><code>{code_body}</code></pre>
        <p>Some text after.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL, max_tokens=50)
        # Find chunk with the code
        code_chunks = [c for c in chunks if "line_0" in c.text and "line_59" in c.text]
        # The entire code block should be in ONE chunk
        assert len(code_chunks) == 1


# ── Happy Path: Cross-page link extraction ───────────────────────────────────


class TestCrossPageLinks:
    """Cross-page links in chunk HTML are extracted and stored as metadata."""

    def test_links_extracted(self):
        html = """
        <h1>References</h1>
        <p>See the <a href="/docs/api">API docs</a> and
           <a href="https://external.com/guide">external guide</a>.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        assert len(chunks) >= 1
        ref_chunk = chunks[0]
        assert ref_chunk.cross_links is not None
        # Relative link should be resolved to absolute
        assert "https://wiki.example.com/docs/api" in ref_chunk.cross_links
        # Absolute link should be kept
        assert "https://external.com/guide" in ref_chunk.cross_links

    def test_fragment_only_links_excluded(self):
        html = """
        <h1>TOC</h1>
        <p>Jump to <a href="#section-2">Section 2</a> or
           visit <a href="/other-page">Other Page</a>.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        assert len(chunks) >= 1
        links = chunks[0].cross_links or []
        # Fragment-only links should be excluded
        fragment_links = [l for l in links if l.endswith("#section-2")]
        assert len(fragment_links) == 0
        # Regular link should be present
        assert any("/other-page" in l for l in links)

    def test_no_links_yields_none(self):
        html = """
        <h1>Plain</h1>
        <p>This paragraph has no links at all.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        assert len(chunks) >= 1
        assert chunks[0].cross_links is None


class TestExtractLinksHelper:
    """Unit tests for the _extract_links helper."""

    def test_relative_url_resolution(self):
        links = _extract_links(
            '<a href="/path/to/page">link</a>',
            "https://example.com/base/",
        )
        assert links == ["https://example.com/path/to/page"]

    def test_deduplication(self):
        links = _extract_links(
            '<a href="/page">A</a> <a href="/page">B</a>',
            "https://example.com/",
        )
        assert len(links) == 1

    def test_javascript_links_excluded(self):
        links = _extract_links(
            '<a href="javascript:void(0)">click</a>',
            "https://example.com/",
        )
        assert len(links) == 0


# ── Edge Case: Oversized table ───────────────────────────────────────────────


class TestOversizedTable:
    """Table exceeding max_tokens becomes its own oversized chunk rather than
    being split mid-row."""

    def test_large_table_not_split(self):
        # Create a table with many rows that exceeds max_tokens
        rows = "".join(
            f"<tr><td>Item {i}</td><td>Description of item number {i} with extra words</td><td>{i * 100}</td></tr>"
            for i in range(50)
        )
        html = f"""
        <h1>Inventory</h1>
        <table>
            <tr><th>Name</th><th>Description</th><th>Price</th></tr>
            {rows}
        </table>
        """
        chunks = chunk_wiki_content(html, PAGE_URL, max_tokens=20)
        # Find the table chunk
        table_chunks = [c for c in chunks if "Item 0" in c.text and "Item 49" in c.text]
        # The entire table must be in ONE chunk, even though it exceeds max_tokens
        assert len(table_chunks) == 1
        tc = _token_count(table_chunks[0].text)
        assert tc > 20  # Confirms it's oversized


# ── Edge Case: Nested header breadcrumbs ─────────────────────────────────────


class TestNestedHeaderBreadcrumbs:
    """Nested headers (h1 > h2 > h3) produce correct breadcrumb path."""

    def test_three_level_breadcrumb(self):
        html = """
        <h1>Getting Started</h1>
        <h2>Installation</h2>
        <h3>Linux</h3>
        <p>Use apt-get to install the package on Linux systems.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        linux_chunk = _find_chunk_by_section(chunks, "Linux")
        assert linux_chunk is not None
        assert linux_chunk.section_title == "Getting Started > Installation > Linux"

    def test_sibling_headers_share_parent(self):
        html = """
        <h1>Guide</h1>
        <h2>Part A</h2>
        <p>Content for part A of the guide section.</p>
        <h2>Part B</h2>
        <p>Content for part B of the guide section.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        part_a = _find_chunk_by_section(chunks, "Part A")
        part_b = _find_chunk_by_section(chunks, "Part B")
        assert part_a is not None
        assert part_b is not None
        assert part_a.section_title == "Guide > Part A"
        assert part_b.section_title == "Guide > Part B"

    def test_deep_nesting(self):
        html = """
        <h1>L1</h1>
        <h2>L2</h2>
        <h3>L3</h3>
        <h4>L4</h4>
        <p>Deep content at level four nesting depth.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        deep_chunk = _find_chunk_by_section(chunks, "L4")
        assert deep_chunk is not None
        assert deep_chunk.section_title == "L1 > L2 > L3 > L4"


# ── Edge Case: Fallback to chunk_web_content ─────────────────────────────────


class TestFallbackBehavior:
    """HTML with no headers and no tables falls back to chunk_web_content()."""

    def test_no_structure_falls_back(self):
        html = """
        <p>This is a plain page with no headers or tables at all.</p>
        <p>It has multiple paragraphs of plain content text.</p>
        <p>And more content in yet another paragraph here.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        # Should still produce chunks (via fallback)
        assert len(chunks) >= 1
        # Chunks should contain the text
        all_text = " ".join(c.text for c in chunks)
        assert "plain page" in all_text
        assert "multiple paragraphs" in all_text

    def test_fallback_uses_section_title(self):
        html = "<p>Simple content without any wiki structure.</p>"
        chunks = chunk_wiki_content(html, PAGE_URL, section_title="My Page")
        assert len(chunks) >= 1
        assert chunks[0].section_title == "My Page"


# ── Edge Case: Empty / boilerplate-only HTML ─────────────────────────────────


class TestEmptyAndBoilerplate:
    """Empty HTML or HTML with only nav/footer returns empty list."""

    def test_empty_string(self):
        assert chunk_wiki_content("", PAGE_URL) == []

    def test_whitespace_only(self):
        assert chunk_wiki_content("   \n\t  ", PAGE_URL) == []

    def test_none_html(self):
        assert chunk_wiki_content(None, PAGE_URL) == []

    def test_nav_and_footer_only(self):
        html = """
        <nav><ul><li><a href="/">Home</a></li></ul></nav>
        <footer><p>Copyright 2025</p></footer>
        """
        result = chunk_wiki_content(html, PAGE_URL)
        assert result == []

    def test_script_and_style_only(self):
        html = """
        <script>var x = 1;</script>
        <style>body { color: red; }</style>
        """
        result = chunk_wiki_content(html, PAGE_URL)
        assert result == []


# ── Error Path: Malformed HTML ───────────────────────────────────────────────


class TestMalformedHTML:
    """Malformed HTML (unclosed tags) still produces valid chunks via BS4's
    lenient parser."""

    def test_unclosed_tags(self):
        html = """
        <h1>Title</h1>
        <p>Some content with <b>unclosed bold
        <p>Another paragraph with unclosed tags here.
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        assert len(chunks) >= 1
        all_text = " ".join(c.text for c in chunks)
        assert "content" in all_text.lower()

    def test_mismatched_tags(self):
        html = """
        <h1>Section</h1>
        <div><p>Text inside mismatched div.</h2></p>
        <p>More text after broken tags content.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        assert len(chunks) >= 1

    def test_broken_table_still_parsed(self):
        html = """
        <h1>Data</h1>
        <table>
            <tr><th>A</th><th>B</th>
            <tr><td>1</td><td>2
        </table>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        assert len(chunks) >= 1
        # Should still have extracted some table content
        all_text = " ".join(c.text for c in chunks)
        assert "A" in all_text


# ── Chunk ID format ──────────────────────────────────────────────────────────


class TestChunkIDFormat:
    """Chunk IDs follow the pattern: sha256(page_url + '::' + section_path)[:12] + '::chunk-{i}'."""

    def test_chunk_id_format(self):
        html = """
        <h1>Intro</h1>
        <p>Some introductory content for the page.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        assert len(chunks) >= 1
        chunk = chunks[0]
        # Verify format: hex[:12]::chunk-N
        parts = chunk.chunk_id.split("::chunk-")
        assert len(parts) == 2
        prefix = parts[0]
        index = parts[1]
        assert len(prefix) == 12
        # Prefix should be valid hex
        int(prefix, 16)
        # Index should be a number
        int(index)

    def test_chunk_id_deterministic(self):
        html = """
        <h1>Test</h1>
        <p>Reproducible content for determinism test here.</p>
        """
        chunks1 = chunk_wiki_content(html, PAGE_URL)
        chunks2 = chunk_wiki_content(html, PAGE_URL)
        assert chunks1[0].chunk_id == chunks2[0].chunk_id

    def test_make_chunk_id_helper(self):
        cid = _make_chunk_id("https://example.com", "Section A", 0)
        expected_prefix = hashlib.sha256("https://example.com::Section A".encode()).hexdigest()[:12]
        assert cid == f"{expected_prefix}::chunk-0"


# ── Content type classification ──────────────────────────────────────────────


class TestContentTypeClassification:
    """Chunks get appropriate content_type via classify_chunk_content_type."""

    def test_text_content_type(self):
        html = """
        <h1>About</h1>
        <p>This is a paragraph of regular prose text content.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL)
        assert len(chunks) >= 1
        assert chunks[0].content_type == "text"


# ── Token counting ───────────────────────────────────────────────────────────


class TestTokenCounting:
    """Token counting uses simple word-split approximation."""

    def test_simple_count(self):
        assert _token_count("hello world foo bar") == 4

    def test_empty_string(self):
        assert _token_count("") == 0

    def test_respects_max_tokens(self):
        """Chunks should respect max_tokens boundary (excluding atomic elements)."""
        paragraphs = [f"<p>{'word ' * 50}</p>" for _ in range(5)]
        html = "<h1>Test</h1>" + "\n".join(paragraphs)
        chunks = chunk_wiki_content(html, PAGE_URL, max_tokens=60)
        # Each chunk's text should be roughly within the token limit
        for c in chunks:
            # Tables and code blocks can exceed, but regular text shouldn't by much
            if "| " not in c.text:
                tc = _token_count(c.text)
                # Allow some tolerance since we merge by paragraph
                assert tc <= 120, f"Chunk too large: {tc} tokens"


# ── Section title with outer section_title parameter ─────────────────────────


class TestSectionTitleOverride:
    """When section_title is passed, it prefixes the breadcrumb."""

    def test_section_title_prepended(self):
        html = """
        <h1>API</h1>
        <p>API documentation content for the reference.</p>
        """
        chunks = chunk_wiki_content(html, PAGE_URL, section_title="Docs")
        assert len(chunks) >= 1
        assert chunks[0].section_title.startswith("Docs")
        assert "API" in chunks[0].section_title
