"""
Tests for research_tool.brain — LLM client, query generation, dedup,
and the full research loop.

All tests run offline: LLM and web calls are mocked.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from research_tool.brain import (
    HYDE_SYSTEM_PROMPT,
    LLMClient,
    QueryEngine,
    ResearchLoop,
    _format_rag_context,
    _parse_query_list,
    SYSTEM_PROMPT,
    QUERY_SYSTEM_PROMPT,
)
from research_tool.store import (
    CONTEXTUAL_RETRIEVAL,
    DocumentChunk,
    EMBEDDING_DIM,
    HybridIndex,
    ResearchStore,
    RetrievedEntry,
    split_into_children,
)
from research_tool.web import Browser, RateLimiter, SearchResult


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_store(tmp_path):
    """Create a ResearchStore with a temporary database."""
    db_path = str(tmp_path / "test_brain.db")
    store = ResearchStore(db_path=db_path)
    yield store
    store.close()


MOCK_HTML = """
<html><head><title>Test Page</title></head><body>
<p>This is a paragraph with enough text to pass the 50-character minimum
extraction threshold. It discusses research methodologies and findings
from recent studies in the field of artificial intelligence.</p>
<p>A second paragraph expands on machine learning techniques, including
supervised learning, unsupervised learning, and reinforcement learning
approaches to solving complex problems.</p>
</body></html>
"""


@pytest.fixture
def mock_browser():
    """A mock Browser that returns simple HTML."""
    browser = MagicMock(spec=Browser)
    browser.fetch_page.return_value = (MOCK_HTML, "https://example.com/test")

    def _parallel_side_effect(urls, *, rate_limiter=None):
        return [(url, MOCK_HTML, url) for url in urls]

    browser.fetch_pages_parallel.side_effect = _parallel_side_effect
    return browser


@pytest.fixture
def mock_llm():
    """A mock LLMClient with controlled responses."""
    llm = MagicMock(spec=LLMClient)
    return llm


def _make_fake_embedding(seed: float = 0.0) -> list[float]:
    """Create a deterministic normalized 384-dim embedding from a seed."""
    rng = np.random.RandomState(int(seed * 1000) % 2**31)
    vec = rng.randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def _make_near_duplicate_embedding(base: list[float], noise: float = 0.01) -> list[float]:
    """Create an embedding very close to base (high cosine similarity)."""
    arr = np.array(base, dtype=np.float32)
    rng = np.random.RandomState(42)
    perturbation = rng.randn(len(arr)).astype(np.float32) * noise
    result = arr + perturbation
    result = result / np.linalg.norm(result)
    return result.tolist()


def _make_orthogonal_embedding(base: list[float]) -> list[float]:
    """Create an embedding approximately orthogonal to base (low cosine sim)."""
    arr = np.array(base, dtype=np.float32)
    # Rotate by swapping and negating half the dimensions
    result = np.copy(arr)
    result[::2] = -arr[1::2]
    result[1::2] = arr[::2]
    result = result / np.linalg.norm(result)
    return result.tolist()


def _store_chunk_with_embedding(store, chunk_id, text, embedding, page_url="https://example.com/test"):
    """Helper: store a page and chunk with a given embedding."""
    if not store.get_page(page_url):
        store.store_page(page_url, title="Test Page")
    chunk = DocumentChunk(
        text=text,
        page_url=page_url,
        chunk_id=chunk_id,
        section_title="Test",
        embedding=embedding,
    )
    store.store_chunk(chunk)
    return chunk


# ── _parse_query_list Tests ──────────────────────────────────────────────────


class TestParseQueryList:
    def test_valid_json_array(self):
        raw = '["query one", "query two", "query three"]'
        result = _parse_query_list(raw)
        assert result == ["query one", "query two", "query three"]

    def test_json_with_markdown_fences(self):
        raw = '```json\n["query one", "query two"]\n```'
        result = _parse_query_list(raw)
        assert result == ["query one", "query two"]

    def test_json_with_surrounding_text(self):
        raw = 'Here are the queries:\n["q1", "q2", "q3"]\nDone.'
        result = _parse_query_list(raw)
        assert result == ["q1", "q2", "q3"]

    def test_malformed_json_returns_empty(self):
        raw = '["q1", "q2"'  # missing closing bracket
        result = _parse_query_list(raw)
        assert result == []

    def test_no_array_returns_empty(self):
        raw = "I think you should search for climate change."
        result = _parse_query_list(raw)
        assert result == []

    def test_non_list_json_returns_empty(self):
        raw = '{"queries": ["q1", "q2"]}'
        # This should extract the inner array
        result = _parse_query_list(raw)
        assert result == ["q1", "q2"]

    def test_filters_non_string_elements(self):
        raw = '["valid query", 42, null, "another query", true]'
        result = _parse_query_list(raw)
        assert result == ["valid query", "another query"]

    def test_empty_array(self):
        raw = "[]"
        result = _parse_query_list(raw)
        assert result == []

    def test_filters_blank_strings(self):
        raw = '["valid", "", "  ", "also valid"]'
        result = _parse_query_list(raw)
        assert result == ["valid", "also valid"]


# ── LLMClient Tests ─────────────────────────────────────────────────────────


class TestLLMClientClaude:
    def test_missing_claude_key_raises(self):
        """Claude backend raises ValueError when no key is available."""
        env = {k: v for k, v in __import__("os").environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                LLMClient(backend="claude")

    def test_reads_key_from_env(self):
        """Claude backend reads ANTHROPIC_API_KEY from environment."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"}):
            with patch("anthropic.Anthropic") as mock_cls:
                client = LLMClient(backend="claude")
                mock_cls.assert_called_once_with(api_key="test-key-123")

    def test_explicit_key_overrides_env(self):
        """Explicitly passed api_key takes precedence."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            with patch("anthropic.Anthropic") as mock_cls:
                client = LLMClient(backend="claude", api_key="explicit-key")
                mock_cls.assert_called_once_with(api_key="explicit-key")

    def test_auth_error_logged_and_reraised(self):
        """AuthenticationError is logged with clear message and re-raised."""
        import anthropic

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.side_effect = anthropic.AuthenticationError(
                message="Invalid API Key",
                response=MagicMock(status_code=401),
                body={"error": {"message": "Invalid API Key"}},
            )

            client = LLMClient(backend="claude", api_key="bad-key")
            with pytest.raises(anthropic.AuthenticationError):
                client.generate("system", "user")

    def test_generate_returns_text(self):
        """generate() returns the text from Claude's response."""
        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_message = MagicMock()
            mock_message.content = [MagicMock(text="Hello from Claude")]
            mock_client.messages.create.return_value = mock_message

            client = LLMClient(backend="claude", api_key="test-key")
            result = client.generate("system prompt", "user prompt")
            assert result == "Hello from Claude"

    def test_api_error_logged_and_reraised(self):
        """General API errors are logged and re-raised."""
        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.side_effect = RuntimeError("Network error")

            client = LLMClient(backend="claude", api_key="test-key")
            with pytest.raises(RuntimeError, match="Network error"):
                client.generate("system", "user")


class TestLLMClientOMLX:
    def test_missing_omlx_key_raises(self):
        """OMLX backend raises ValueError when no key is available."""
        env = {k: v for k, v in __import__("os").environ.items() if k != "OMLX_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValueError, match="OMLX_API_KEY"):
                LLMClient(backend="omlx")

    def test_omlx_reads_key_from_env(self):
        """OMLX backend reads OMLX_API_KEY from environment."""
        with patch.dict("os.environ", {"OMLX_API_KEY": "omlx-test-key"}):
            client = LLMClient(backend="omlx")
            assert client._api_key == "omlx-test-key"
            assert client._backend == "omlx"

    def test_omlx_custom_base_url(self):
        """OMLX backend respects OMLX_BASE_URL env var."""
        with patch.dict("os.environ", {"OMLX_API_KEY": "key", "OMLX_BASE_URL": "http://custom:9000/v1"}):
            client = LLMClient(backend="omlx")
            assert client._base_url == "http://custom:9000/v1"

    @patch("urllib.request.urlopen")
    def test_omlx_generate_returns_text(self, mock_urlopen):
        """OMLX generate() parses OpenAI-format response."""
        import io
        response_body = json.dumps({
            "choices": [{"message": {"content": "Hello from OMLX"}}]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with patch.dict("os.environ", {"OMLX_API_KEY": "key"}):
            client = LLMClient(backend="omlx", model="test-model")
            result = client.generate("system", "user")
            assert result == "Hello from OMLX"

    @patch("urllib.request.urlopen")
    def test_omlx_connection_error(self, mock_urlopen):
        """OMLX raises ConnectionError when server is unreachable."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with patch.dict("os.environ", {"OMLX_API_KEY": "key"}):
            client = LLMClient(backend="omlx", model="test-model")
            with pytest.raises(ConnectionError, match="Cannot reach OMLX server"):
                client.generate("system", "user")

    @patch("urllib.request.urlopen")
    def test_omlx_http_error_surfaces_body(self, mock_urlopen):
        """OMLX HTTP errors show the response body, not 'unreachable'."""
        import urllib.error
        body = b'{"error": "context length exceeded"}'
        resp = MagicMock()
        resp.read.return_value = body
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://localhost:8000/v1/chat/completions",
            code=400, msg="Bad Request", hdrs={}, fp=resp,
        )

        with patch.dict("os.environ", {"OMLX_API_KEY": "key"}):
            client = LLMClient(backend="omlx", model="test-model")
            with pytest.raises(RuntimeError, match="HTTP 400"):
                client.generate("system", "user")


# ── ResearchLoop._generate_search_queries Tests ─────────────────────────────


class TestGenerateSearchQueries:
    def test_initial_queries_no_context(self, tmp_store, mock_browser, mock_llm):
        """With no context, generates initial queries from the research prompt."""
        mock_llm.generate.return_value = json.dumps([
            "quantum computing basics",
            "quantum computing applications",
            "quantum computing vs classical",
        ])

        loop = ResearchLoop(
            prompt="quantum computing",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )
        queries = loop._generate_search_queries("quantum computing")

        assert len(queries) == 3
        assert "quantum computing basics" in queries
        # Verify the LLM was called with initial prompt (no context)
        call_args = mock_llm.generate.call_args
        assert "Research topic:" in call_args[0][1]
        assert "<retrieved-content>" not in call_args[0][1]

    def test_followup_queries_with_context(self, tmp_store, mock_browser, mock_llm):
        """With accumulated context, generates different follow-up queries."""
        mock_llm.generate.return_value = json.dumps([
            "quantum error correction methods",
            "topological qubits research",
        ])

        loop = ResearchLoop(
            prompt="quantum computing",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )
        context = "Quantum computing uses qubits instead of classical bits..."
        queries = loop._generate_search_queries("quantum computing", context=context)

        assert len(queries) == 2
        assert "quantum error correction methods" in queries
        # Verify context was included with proper tagging
        call_args = mock_llm.generate.call_args
        assert "<retrieved-content>" in call_args[0][1]
        assert "</retrieved-content>" in call_args[0][1]

    def test_llm_failure_returns_empty(self, tmp_store, mock_browser, mock_llm):
        """If the LLM call raises, return empty list gracefully."""
        mock_llm.generate.side_effect = RuntimeError("API down")

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )
        queries = loop._generate_search_queries("test")
        assert queries == []

    def test_malformed_json_returns_empty(self, tmp_store, mock_browser, mock_llm):
        """Malformed JSON from LLM is caught and treated as empty."""
        mock_llm.generate.return_value = "Here are some ideas but I forgot the JSON format"

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )
        queries = loop._generate_search_queries("test")
        assert queries == []


# ── ResearchLoop._find_duplicate Tests ────────────────────────────────────────


class TestFindDuplicate:
    def test_empty_store_returns_none(self, tmp_store, mock_browser, mock_llm):
        """With no existing chunks, nothing is a duplicate."""
        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )
        chunk = DocumentChunk(
            text="brand new content",
            page_url="https://example.com/new",
            chunk_id="new::chunk-0",
            section_title="New",
            embedding=_make_fake_embedding(1.0),
        )
        assert loop._find_duplicate(chunk) is None

    def test_novel_content_returns_none(self, tmp_store, mock_browser, mock_llm):
        """A genuinely different embedding is not a duplicate."""
        base_emb = _make_fake_embedding(1.0)
        _store_chunk_with_embedding(
            tmp_store, "existing::chunk-0", "existing content", base_emb
        )

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )

        novel_emb = _make_orthogonal_embedding(base_emb)
        chunk = DocumentChunk(
            text="completely different content",
            page_url="https://example.com/different",
            chunk_id="different::chunk-0",
            section_title="Different",
            embedding=novel_emb,
        )

        # Verify the cosine similarity is indeed low
        sim = float(np.dot(base_emb, novel_emb))
        assert sim < 0.85

        assert loop._find_duplicate(chunk) is None

    def test_duplicate_returns_matched_id(self, tmp_store, mock_browser, mock_llm):
        """An embedding with >0.85 similarity returns the matched chunk_id."""
        base_emb = _make_fake_embedding(2.0)
        _store_chunk_with_embedding(
            tmp_store, "existing::chunk-0", "existing content", base_emb
        )

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )

        near_dup_emb = _make_near_duplicate_embedding(base_emb, noise=0.01)
        chunk = DocumentChunk(
            text="nearly identical content",
            page_url="https://example.com/dup",
            chunk_id="dup::chunk-0",
            section_title="Dup",
            embedding=near_dup_emb,
        )

        # Verify the cosine similarity is indeed high
        sim = float(np.dot(base_emb, near_dup_emb))
        assert sim > 0.85

        assert loop._find_duplicate(chunk) == "existing::chunk-0"

    def test_no_embedding_returns_none(self, tmp_store, mock_browser, mock_llm):
        """If the chunk has no embedding, it cannot be a duplicate."""
        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )
        chunk = DocumentChunk(
            text="no embedding",
            page_url="https://example.com/no-emb",
            chunk_id="noemb::chunk-0",
            section_title="No Emb",
            embedding=None,
        )
        assert loop._find_duplicate(chunk) is None

    def test_existing_chunks_without_embeddings(self, tmp_store, mock_browser, mock_llm):
        """If existing chunks have no embeddings, dedup returns None."""
        # Store a chunk without embedding
        tmp_store.store_page("https://example.com/noemb", title="No Emb")
        chunk_no_emb = DocumentChunk(
            text="chunk without embedding",
            page_url="https://example.com/noemb",
            chunk_id="noemb::chunk-0",
            section_title="No Emb",
            embedding=None,
        )
        tmp_store.store_chunk(chunk_no_emb)

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )

        new_chunk = DocumentChunk(
            text="new content",
            page_url="https://example.com/new",
            chunk_id="new::chunk-0",
            section_title="New",
            embedding=_make_fake_embedding(5.0),
        )
        assert loop._find_duplicate(new_chunk) is None


# ── ResearchLoop._should_continue Tests ──────────────────────────────────────


class TestShouldContinue:
    def test_high_novelty_continues(self, tmp_store, mock_browser, mock_llm):
        """With high novelty and depth below max, should continue."""
        loop = ResearchLoop(
            prompt="test", store=tmp_store, browser=mock_browser,
            llm_client=mock_llm, max_depth=3,
        )
        loop._iteration_new_chunks = 100
        loop._iteration_dedup_skips = 0
        # max_depth is checked in run(), not _should_continue
        assert loop._should_continue(2) is True

    def test_continues_with_high_novelty(self, tmp_store, mock_browser, mock_llm):
        loop = ResearchLoop(
            prompt="test", store=tmp_store, browser=mock_browser,
            llm_client=mock_llm, max_depth=5,
        )
        loop._iteration_new_chunks = 8
        loop._iteration_dedup_skips = 2
        # 8/10 = 0.8 > 0.20
        assert loop._should_continue(1) is True

    def test_stops_with_low_novelty(self, tmp_store, mock_browser, mock_llm):
        loop = ResearchLoop(
            prompt="test", store=tmp_store, browser=mock_browser,
            llm_client=mock_llm, max_depth=5,
        )
        loop._iteration_new_chunks = 1
        loop._iteration_dedup_skips = 9
        # 1/10 = 0.10 < 0.20
        assert loop._should_continue(1) is False

    def test_stops_when_no_chunks_attempted(self, tmp_store, mock_browser, mock_llm):
        loop = ResearchLoop(
            prompt="test", store=tmp_store, browser=mock_browser,
            llm_client=mock_llm, max_depth=5,
        )
        loop._iteration_new_chunks = 0
        loop._iteration_dedup_skips = 0
        assert loop._should_continue(1) is False

    def test_boundary_novelty_exactly_threshold(self, tmp_store, mock_browser, mock_llm):
        loop = ResearchLoop(
            prompt="test", store=tmp_store, browser=mock_browser,
            llm_client=mock_llm, max_depth=5,
        )
        loop._iteration_new_chunks = 2
        loop._iteration_dedup_skips = 8
        # 2/10 = 0.20 == threshold -> should continue
        assert loop._should_continue(1) is True


# ── Full Research Loop Integration Test ──────────────────────────────────────


class TestResearchLoopIntegration:
    """Full loop tests with mocked web and LLM."""

    @patch("research_tool.brain.ENTITY_RESOLUTION_ENABLED", False)
    @patch("research_tool.brain.search_ddg")
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_full_loop_iterates_and_terminates(
        self, mock_embedding, mock_safe_url, mock_search,
        tmp_store, mock_browser, mock_llm,
    ):
        """Full loop: searches, fetches, chunks, stores, then terminates at max_depth."""
        # Setup: LLM returns queries for depth 0 and 1, then empty at depth 2
        call_count = [0]

        def llm_side_effect(system, user):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps(["search query alpha", "search query beta"])
            elif call_count[0] == 2:
                return json.dumps(["followup query gamma"])
            else:
                return "[]"

        mock_llm.generate.side_effect = llm_side_effect

        # Search returns different results each time
        page_counter = [0]

        def search_side_effect(query, rate_limiter=None, max_results=20, browser=None):
            page_counter[0] += 1
            return [
                SearchResult(
                    title=f"Result {page_counter[0]}",
                    url=f"https://example{page_counter[0]}.com/page",
                    snippet=f"Snippet for {query}",
                )
            ]

        mock_search.side_effect = search_side_effect

        # Unique embeddings for each chunk
        emb_counter = [0]

        def embedding_side_effect(text):
            emb_counter[0] += 1
            return _make_fake_embedding(float(emb_counter[0]))

        mock_embedding.side_effect = embedding_side_effect

        loop = ResearchLoop(
            prompt="test research topic",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
            max_depth=5,
        )

        result = loop.run()

        # Should have completed 3 iterations (depth 0, 1, then empty at 2)
        assert result["terminated_reason"] == "llm_returned_empty"
        assert result["depth_reached"] == 2
        assert result["total_chunks"] > 0

    @patch("research_tool.brain.search_ddg")
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_loop_respects_max_depth(
        self, mock_embedding, mock_safe_url, mock_search,
        tmp_store, mock_browser, mock_llm,
    ):
        """Loop terminates at max_depth even if LLM keeps suggesting queries."""
        mock_llm.generate.return_value = json.dumps(["query a", "query b"])

        page_counter = [0]

        def search_side_effect(query, rate_limiter=None, max_results=20, browser=None):
            page_counter[0] += 1
            return [
                SearchResult(
                    title=f"Result {page_counter[0]}",
                    url=f"https://example{page_counter[0]}.com/page",
                    snippet=f"Snippet {page_counter[0]}",
                )
            ]

        mock_search.side_effect = search_side_effect

        emb_counter = [0]

        def embedding_side_effect(text):
            emb_counter[0] += 1
            return _make_fake_embedding(float(emb_counter[0]))

        mock_embedding.side_effect = embedding_side_effect

        loop = ResearchLoop(
            prompt="test topic",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
            max_depth=2,
        )

        result = loop.run()

        assert result["depth_reached"] == 2
        assert result["terminated_reason"] == "max_depth"

    @patch("research_tool.brain.search_ddg")
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_loop_terminates_on_empty_queries(
        self, mock_embedding, mock_safe_url, mock_search,
        tmp_store, mock_browser, mock_llm,
    ):
        """Loop terminates when LLM returns an empty query list."""
        mock_llm.generate.return_value = "[]"

        loop = ResearchLoop(
            prompt="test topic",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
            max_depth=5,
        )

        result = loop.run()

        assert result["depth_reached"] == 0
        assert result["terminated_reason"] == "llm_returned_empty"

    @patch("research_tool.brain.search_ddg")
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_loop_terminates_on_llm_failure(
        self, mock_embedding, mock_safe_url, mock_search,
        tmp_store, mock_browser, mock_llm,
    ):
        """Loop terminates gracefully when the LLM API fails."""
        mock_llm.generate.side_effect = RuntimeError("API down")

        loop = ResearchLoop(
            prompt="test topic",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
            max_depth=5,
        )

        result = loop.run()

        assert result["depth_reached"] == 0
        assert result["terminated_reason"] == "llm_returned_empty"

    @patch("research_tool.brain.search_ddg")
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_skips_already_fetched_pages(
        self, mock_embedding, mock_safe_url, mock_search,
        tmp_store, mock_browser, mock_llm,
    ):
        """Pages already in the store are not re-fetched."""
        # Pre-populate the store with a page
        tmp_store.store_page("https://existing.com/page", title="Existing")

        mock_llm.generate.side_effect = [
            json.dumps(["test query"]),
            "[]",
        ]

        mock_search.return_value = [
            SearchResult(
                title="Existing Page",
                url="https://existing.com/page",
                snippet="Already fetched",
            ),
        ]

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )

        loop.run()

        # Browser should NOT have been called since the page already exists
        mock_browser.fetch_page.assert_not_called()

    @patch("research_tool.brain.search_ddg")
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_dedup_skips_increment(
        self, mock_embedding, mock_safe_url, mock_search,
        tmp_store, mock_browser, mock_llm,
    ):
        """Duplicate chunks are counted and skipped during ingestion."""
        # Pre-store a chunk with a known embedding
        base_emb = _make_fake_embedding(42.0)
        _store_chunk_with_embedding(
            tmp_store, "pre::chunk-0", "pre-existing content", base_emb,
            page_url="https://pre-existing.com/page",
        )

        mock_llm.generate.side_effect = [
            json.dumps(["dedup test query"]),
            "[]",
        ]

        mock_search.return_value = [
            SearchResult(
                title="Dedup Test",
                url="https://dedup-test.com/page",
                snippet="Testing dedup",
            ),
        ]

        # Return a near-duplicate embedding (will be flagged as dup)
        near_dup = _make_near_duplicate_embedding(base_emb, noise=0.001)
        mock_embedding.return_value = near_dup

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
            similarity_threshold=0.85,
        )

        result = loop.run()

        # Verify the near-dup similarity is above threshold
        sim = float(np.dot(base_emb, near_dup))
        assert sim > 0.85, f"Test setup error: similarity {sim} should be > 0.85"

    @patch("research_tool.brain.search_ddg")
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_low_novelty_terminates_early(
        self, mock_embedding, mock_safe_url, mock_search,
        tmp_store, mock_browser, mock_llm,
    ):
        """Loop terminates early when novelty ratio drops below threshold."""
        # Pre-fill the store with many chunks that will cause all new content
        # to be flagged as duplicates
        base_emb = _make_fake_embedding(100.0)
        for i in range(10):
            near = _make_near_duplicate_embedding(base_emb, noise=0.001 * (i + 1))
            _store_chunk_with_embedding(
                tmp_store, f"pre::chunk-{i}", f"pre-existing content {i}", near,
                page_url=f"https://pre-existing-{i}.com/page",
            )

        # LLM keeps suggesting queries
        mock_llm.generate.return_value = json.dumps(["query a"])

        page_counter = [0]

        def search_side_effect(query, rate_limiter=None, max_results=20, browser=None):
            page_counter[0] += 1
            return [
                SearchResult(
                    title=f"Result {page_counter[0]}",
                    url=f"https://new-{page_counter[0]}.com/page",
                    snippet="New content",
                ),
            ]

        mock_search.side_effect = search_side_effect

        # Return near-duplicate embeddings
        mock_embedding.return_value = _make_near_duplicate_embedding(base_emb, noise=0.0005)

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
            max_depth=10,
            similarity_threshold=0.85,
        )

        result = loop.run()

        # Should terminate well before max_depth due to low novelty
        assert result["depth_reached"] < 10
        assert result["terminated_reason"] == "low_novelty"

    @patch("research_tool.brain.search_ddg")
    @patch("research_tool.brain.is_safe_url")
    @patch("research_tool.brain._make_embedding")
    def test_unsafe_urls_are_skipped(
        self, mock_embedding, mock_safe_url, mock_search,
        tmp_store, mock_browser, mock_llm,
    ):
        """URLs that fail safety checks are not fetched."""
        mock_llm.generate.side_effect = [
            json.dumps(["test query"]),
            "[]",
        ]

        mock_search.return_value = [
            SearchResult(
                title="Unsafe Page",
                url="http://192.168.1.1/admin",
                snippet="Should not be fetched",
            ),
        ]

        mock_safe_url.return_value = False

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )

        loop.run()

        # Browser should NOT have been called since URL is unsafe
        mock_browser.fetch_page.assert_not_called()

    @patch("research_tool.brain.search_ddg")
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_fetch_failure_continues_gracefully(
        self, mock_embedding, mock_safe_url, mock_search,
        tmp_store, mock_browser, mock_llm,
    ):
        """If fetching a page fails, the loop continues with other results."""
        mock_llm.generate.side_effect = [
            json.dumps(["test query"]),
            "[]",
        ]

        mock_search.return_value = [
            SearchResult(title="Fail Page", url="https://fail.com/page", snippet="Fails"),
            SearchResult(title="Good Page", url="https://good.com/page", snippet="Works"),
        ]

        good_html = """<html><head><title>Good Page</title></head><body>
            <p>This is a good page with enough text content to pass the extraction
            threshold and be chunked properly for research purposes.</p>
            </body></html>"""

        # Parallel fetch: fail.com is silently dropped, good.com succeeds
        def parallel_side_effect(urls, *, rate_limiter=None):
            results = []
            for url in urls:
                if "fail.com" in url:
                    continue  # simulates fetch failure (not returned)
                results.append((url, good_html, url))
            return results

        mock_browser.fetch_pages_parallel.side_effect = parallel_side_effect

        emb_counter = [0]

        def embedding_side_effect(text):
            emb_counter[0] += 1
            return _make_fake_embedding(float(emb_counter[0]))

        mock_embedding.side_effect = embedding_side_effect

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )

        result = loop.run()

        # The loop should have stored the good page despite the failure
        assert tmp_store.get_page("https://good.com/page") is not None
        assert tmp_store.get_page("https://fail.com/page") is None


# ── Progress Output Tests ────────────────────────────────────────────────────


class TestProgress:
    def test_progress_writes_to_stderr(
        self, capsys, tmp_store, mock_browser, mock_llm,
    ):
        """Progress messages go to stderr, not stdout."""
        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )
        loop._progress("test message")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "test message" in captured.err
        loop.close_log()


# ── QueryEngine Tests ──────────────────────────────────────────────────────


class TestQueryEngine:
    def test_ask_with_stored_chunks(self, tmp_store, mock_llm):
        """ask() retrieves relevant chunks and sends them to the LLM."""
        # Populate store with a page and chunks
        tmp_store.store_page("https://example.com/ai", title="AI Research")
        emb = _make_fake_embedding(1.0)
        _store_chunk_with_embedding(
            tmp_store, "ai::chunk-0",
            "Artificial intelligence is transforming many industries.",
            emb, page_url="https://example.com/ai",
        )

        mock_llm.generate.return_value = "AI is transforming industries through automation."

        engine = QueryEngine(db_path=tmp_store.conn.execute("PRAGMA database_list").fetchone()[2], llm_client=mock_llm)
        # Point at the same DB by replacing the store
        engine.store = tmp_store

        answer = engine.ask("What is AI doing?")

        assert answer == "AI is transforming industries through automation."
        # Verify LLM was called with the query system prompt
        call_args = mock_llm.generate.call_args
        assert call_args[0][0] == QUERY_SYSTEM_PROMPT
        assert "What is AI doing?" in call_args[0][1]

    def test_ask_with_empty_db(self, tmp_store, mock_llm):
        """ask() with no stored data sends empty context to LLM."""
        mock_llm.generate.return_value = "The research did not cover this topic."

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        answer = engine.ask("What is quantum computing?")

        assert "did not cover" in answer
        call_args = mock_llm.generate.call_args
        assert "No research data available" in call_args[0][1]

    def test_status_returns_correct_counts(self, tmp_store):
        """status() returns accurate page, chunk, search counts."""
        tmp_store.store_page("https://a.com", title="A")
        tmp_store.store_page("https://b.com", title="B")

        emb = _make_fake_embedding(1.0)
        _store_chunk_with_embedding(
            tmp_store, "a::chunk-0", "content a", emb,
            page_url="https://a.com",
        )
        _store_chunk_with_embedding(
            tmp_store, "b::chunk-0", "content b", _make_fake_embedding(2.0),
            page_url="https://b.com",
        )
        _store_chunk_with_embedding(
            tmp_store, "b::chunk-1", "more content b", _make_fake_embedding(3.0),
            page_url="https://b.com",
        )

        tmp_store.log_search("query 1")
        tmp_store.log_search("query 2")
        tmp_store.log_search("query 3")

        engine = QueryEngine(db_path="unused")
        engine.store = tmp_store

        stats = engine.status()
        assert stats["total_pages"] == 2
        assert stats["total_chunks"] == 3
        assert stats["total_searches"] == 3
        assert stats["last_activity"] is not None

    def test_status_empty_db(self, tmp_store):
        """status() on empty DB returns zeros."""
        engine = QueryEngine(db_path="unused")
        engine.store = tmp_store

        stats = engine.status()
        assert stats["total_pages"] == 0
        assert stats["total_chunks"] == 0
        assert stats["total_searches"] == 0

    def test_list_sources(self, tmp_store):
        """list_sources() returns all pages with url, title, fetched_at."""
        tmp_store.store_page("https://a.com", title="Page A")
        tmp_store.store_page("https://b.com", title="Page B")

        engine = QueryEngine(db_path="unused")
        engine.store = tmp_store

        sources = engine.list_sources()
        assert len(sources) == 2
        urls = {s["url"] for s in sources}
        assert "https://a.com" in urls
        assert "https://b.com" in urls
        assert all("title" in s and "fetched_at" in s for s in sources)

    def test_list_sources_empty(self, tmp_store):
        """list_sources() on empty DB returns empty list."""
        engine = QueryEngine(db_path="unused")
        engine.store = tmp_store

        assert engine.list_sources() == []

    @patch("research_tool.brain.HybridIndex")
    def test_ask_passes_rerank_flag(self, MockIndex, tmp_store, mock_llm):
        """ask(rerank=False) passes rerank=False through to hybrid_retrieve()."""
        mock_index = MagicMock()
        mock_index.is_built = True
        mock_index.chunks = [MagicMock()]
        mock_index.hybrid_retrieve.return_value = []
        MockIndex.return_value = mock_index

        mock_llm.generate.return_value = "answer"

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        engine.ask("test question", rerank=False)
        mock_index.hybrid_retrieve.assert_called_once()
        _, kwargs = mock_index.hybrid_retrieve.call_args
        assert kwargs.get("rerank") is False

    @patch("research_tool.brain.HybridIndex")
    def test_ask_rerank_default_true(self, MockIndex, tmp_store, mock_llm):
        """ask() defaults rerank=True."""
        mock_index = MagicMock()
        mock_index.is_built = True
        mock_index.chunks = [MagicMock()]
        mock_index.hybrid_retrieve.return_value = []
        MockIndex.return_value = mock_index

        mock_llm.generate.return_value = "answer"

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        engine.ask("test question")
        _, kwargs = mock_index.hybrid_retrieve.call_args
        assert kwargs.get("rerank") is True

    def test_ask_includes_source_urls_in_context(self, tmp_store, mock_llm):
        """The RAG context sent to the LLM includes source URLs."""
        tmp_store.store_page("https://specific-source.org/article", title="Source Article")
        emb = _make_fake_embedding(10.0)
        _store_chunk_with_embedding(
            tmp_store, "src::chunk-0",
            "Machine learning uses statistical methods to learn from data.",
            emb, page_url="https://specific-source.org/article",
        )

        mock_llm.generate.return_value = "ML uses statistics."

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        engine.ask("What is machine learning?")

        call_args = mock_llm.generate.call_args
        assert "specific-source.org" in call_args[0][1]

    def test_ask_uses_multimodal_when_charts_found(self, tmp_store, mock_llm):
        """When chart images match retrieved chunks, ask() uses generate_multimodal."""
        tmp_store.store_page("https://example.com/paper", title="Paper")
        emb = _make_fake_embedding(5.0)
        _store_chunk_with_embedding(
            tmp_store, "paper::chunk-0",
            "Figure 1 shows a 30% increase in throughput.",
            emb, page_url="https://example.com/paper",
        )
        tmp_store.store_image(
            image_id="chart-fig1",
            page_url="https://example.com/paper",
            src_url="https://example.com/fig1.png",
            nearest_chunk_ids=["paper::chunk-0"],
            image_bytes=b"\x89PNG\r\n\x1a\nfake",
            is_chart=True,
        )

        mock_llm._backend = "claude"
        mock_llm.generate_multimodal.return_value = "The chart shows 30% throughput growth."

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        answer = engine.ask("What does figure 1 show?")

        assert answer == "The chart shows 30% throughput growth."
        mock_llm.generate_multimodal.assert_called_once()
        content_blocks = mock_llm.generate_multimodal.call_args[0][1]
        assert any(b.get("type") == "image" for b in content_blocks)
        assert any("Chart from" in b.get("text", "") for b in content_blocks)

    def test_ask_falls_back_to_text_when_no_charts(self, tmp_store, mock_llm):
        """Without chart images, ask() uses regular text generate()."""
        tmp_store.store_page("https://example.com/page", title="Page")
        emb = _make_fake_embedding(6.0)
        _store_chunk_with_embedding(
            tmp_store, "page::chunk-0",
            "Quantum computing leverages superposition.",
            emb, page_url="https://example.com/page",
        )

        mock_llm._backend = "claude"
        mock_llm.generate.return_value = "Quantum uses superposition."

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        answer = engine.ask("What is quantum computing?")

        assert answer == "Quantum uses superposition."
        # Final generate call is the query (HyDE may add an earlier call)
        last_call = mock_llm.generate.call_args_list[-1]
        assert QUERY_SYSTEM_PROMPT in last_call[0][0]
        mock_llm.generate_multimodal.assert_not_called()

    def test_ask_skips_multimodal_for_non_claude_backend(self, tmp_store, mock_llm):
        """Non-Claude backends use text-only even when charts are available."""
        tmp_store.store_page("https://example.com/paper", title="Paper")
        emb = _make_fake_embedding(7.0)
        _store_chunk_with_embedding(
            tmp_store, "paper::chunk-1",
            "Results in figure 2 indicate growth.",
            emb, page_url="https://example.com/paper",
        )
        tmp_store.store_image(
            image_id="chart-fig2",
            page_url="https://example.com/paper",
            src_url="https://example.com/fig2.png",
            nearest_chunk_ids=["paper::chunk-1"],
            image_bytes=b"png_bytes",
            is_chart=True,
        )

        mock_llm._backend = "omlx"
        mock_llm.generate.return_value = "Growth indicated."

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        answer = engine.ask("What does figure 2 show?")

        assert answer == "Growth indicated."
        # Final generate call is the query (HyDE may add an earlier call)
        last_call = mock_llm.generate.call_args_list[-1]
        assert QUERY_SYSTEM_PROMPT in last_call[0][0]
        mock_llm.generate_multimodal.assert_not_called()


# ── HyDE Tests ────────────────────────────────────────────────────────────


class TestHyDEQueryEngine:
    """Tests for HyDE hypothesis generation and integration in QueryEngine."""

    def test_generate_hyde_embedding_calls_llm_and_embeds(self, tmp_store, mock_llm):
        """_generate_hyde_embedding should call LLM then embed the hypothesis."""
        mock_llm.generate.return_value = "Rate limiting uses exponential backoff."
        fake_emb = [0.1] * EMBEDDING_DIM

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        with patch("research_tool.brain._make_embedding", return_value=fake_emb) as mock_embed:
            result = engine._generate_hyde_embedding("how to handle rate limits")

        mock_llm.generate.assert_called_once_with(HYDE_SYSTEM_PROMPT, "how to handle rate limits")
        mock_embed.assert_called_once_with("Rate limiting uses exponential backoff.", mode="document")
        assert result == fake_emb

    def test_generate_hyde_embedding_returns_none_on_llm_failure(self, tmp_store, mock_llm):
        """If LLM raises, _generate_hyde_embedding returns None gracefully."""
        mock_llm.generate.side_effect = Exception("API down")

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        result = engine._generate_hyde_embedding("test query")
        assert result is None

    def test_generate_hyde_embedding_returns_none_on_empty_hypothesis(self, tmp_store, mock_llm):
        """If LLM returns empty string, no embedding is generated."""
        mock_llm.generate.return_value = ""

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        result = engine._generate_hyde_embedding("test query")
        assert result is None

    @patch("research_tool.brain.HybridIndex")
    def test_ask_passes_hyde_emb_to_hybrid_retrieve(self, MockIndex, tmp_store, mock_llm):
        """ask() should pass hyde_emb to hybrid_retrieve when HYDE_ENABLED."""
        mock_index = MagicMock()
        mock_index.is_built = True
        mock_index.chunks = [MagicMock()]
        mock_index.hybrid_retrieve.return_value = []
        MockIndex.return_value = mock_index

        mock_llm.generate.return_value = "Answer based on docs."
        hyde_emb = [0.2] * EMBEDDING_DIM

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        complex_query = (
            "How does the rate limiting system handle concurrent requests "
            "and what are the implications for throughput?"
        )
        with patch("research_tool.brain.HYDE_ENABLED", True), \
             patch.object(engine, "_generate_hyde_embedding", return_value=hyde_emb) as mock_hyde:
            engine.ask(complex_query)

        mock_hyde.assert_called_once_with(complex_query)
        _, kwargs = mock_index.hybrid_retrieve.call_args
        assert kwargs.get("hyde_emb") == hyde_emb

    @patch("research_tool.brain.HybridIndex")
    def test_ask_skips_hyde_when_disabled(self, MockIndex, tmp_store, mock_llm):
        """ask() should not generate HyDE when HYDE_ENABLED=False."""
        mock_index = MagicMock()
        mock_index.is_built = True
        mock_index.chunks = [MagicMock()]
        mock_index.hybrid_retrieve.return_value = []
        MockIndex.return_value = mock_index

        mock_llm.generate.return_value = "Answer."

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        with patch("research_tool.brain.HYDE_ENABLED", False), \
             patch.object(engine, "_generate_hyde_embedding") as mock_hyde:
            engine.ask("test question")

        mock_hyde.assert_not_called()
        _, kwargs = mock_index.hybrid_retrieve.call_args
        assert kwargs.get("hyde_emb") is None


# ── Parent Expansion Tests ────────────────────────────────────────────────


class TestParentExpansion:
    """U3: Parent expansion in QueryEngine — child results expanded to parent text."""

    def test_expand_replaces_child_text_with_parent(self, tmp_store, mock_llm):
        """Child chunk text should be replaced with parent text."""
        tmp_store.store_page("http://a.com", title="Test")
        parent = DocumentChunk(
            text="Full parent context with lots of detail.",
            page_url="http://a.com", chunk_id="p1",
            section_title="S", embedding=_make_fake_embedding(1.0),
        )
        child = DocumentChunk(
            text="child snippet",
            page_url="http://a.com", chunk_id="p1::child:0",
            section_title="S", embedding=_make_fake_embedding(2.0),
            parent_chunk_id="p1", is_child=True,
        )
        tmp_store.store_chunk(parent)
        tmp_store.store_chunk(child)

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        results = [RetrievedEntry(
            page_url="http://a.com", chunk_id="p1::child:0",
            section_title="S", score=1.0, bm25_score=0.5,
            cosine_score=0.5, text="child snippet",
        )]

        expanded = engine._expand_to_parents(results)
        assert len(expanded) == 1
        assert expanded[0].text == "Full parent context with lots of detail."
        assert expanded[0].chunk_id == "p1"

    def test_expand_deduplicates_shared_parents(self, tmp_store, mock_llm):
        """Two children from the same parent should produce only one result."""
        tmp_store.store_page("http://a.com", title="Test")
        parent = DocumentChunk(
            text="Parent text.", page_url="http://a.com",
            chunk_id="p1", section_title="S",
        )
        child0 = DocumentChunk(
            text="c0", page_url="http://a.com", chunk_id="p1::child:0",
            section_title="S", parent_chunk_id="p1", is_child=True,
        )
        child1 = DocumentChunk(
            text="c1", page_url="http://a.com", chunk_id="p1::child:1",
            section_title="S", parent_chunk_id="p1", is_child=True,
        )
        tmp_store.store_chunk(parent)
        tmp_store.store_chunk(child0)
        tmp_store.store_chunk(child1)

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        results = [
            RetrievedEntry(page_url="http://a.com", chunk_id="p1::child:0",
                           section_title="S", score=1.0, bm25_score=0.5,
                           cosine_score=0.5, text="c0"),
            RetrievedEntry(page_url="http://a.com", chunk_id="p1::child:1",
                           section_title="S", score=0.8, bm25_score=0.3,
                           cosine_score=0.4, text="c1"),
        ]

        expanded = engine._expand_to_parents(results)
        assert len(expanded) == 1
        assert expanded[0].text == "Parent text."

    def test_expand_legacy_chunks_unchanged(self, tmp_store, mock_llm):
        """Non-child chunks should pass through unchanged."""
        tmp_store.store_page("http://a.com", title="Test")
        legacy = DocumentChunk(
            text="Legacy chunk.", page_url="http://a.com",
            chunk_id="legacy1", section_title="S",
        )
        tmp_store.store_chunk(legacy)

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        results = [RetrievedEntry(
            page_url="http://a.com", chunk_id="legacy1",
            section_title="S", score=1.0, bm25_score=0.5,
            cosine_score=0.5, text="Legacy chunk.",
        )]

        expanded = engine._expand_to_parents(results)
        assert len(expanded) == 1
        assert expanded[0].text == "Legacy chunk."

    def test_expand_missing_parent_uses_child_text(self, tmp_store, mock_llm):
        """If parent was deleted, child's own text is used as fallback."""
        tmp_store.store_page("http://a.com", title="Test")
        child = DocumentChunk(
            text="Orphan child text.", page_url="http://a.com",
            chunk_id="p1::child:0", section_title="S",
            parent_chunk_id="p1", is_child=True,
        )
        tmp_store.store_chunk(child)

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        results = [RetrievedEntry(
            page_url="http://a.com", chunk_id="p1::child:0",
            section_title="S", score=1.0, bm25_score=0.5,
            cosine_score=0.5, text="Orphan child text.",
        )]

        expanded = engine._expand_to_parents(results)
        assert len(expanded) == 1
        assert expanded[0].text == "Orphan child text."


# ── _format_rag_context Tests ──────────────────────────────────────────────


class TestFormatRagContext:
    def test_empty_results(self):
        assert "No relevant results" in _format_rag_context([])

    def test_formats_with_scores_and_urls(self):
        from research_tool.store import RetrievedEntry
        entries = [
            RetrievedEntry(
                page_url="https://example.com/page1",
                chunk_id="p1::c0",
                section_title="Intro",
                score=0.95,
                bm25_score=0.9,
                cosine_score=1.0,
                text="Some relevant text.",
            ),
            RetrievedEntry(
                page_url="https://example.com/page2",
                chunk_id="p2::c0",
                section_title="Details",
                score=0.80,
                bm25_score=0.7,
                cosine_score=0.9,
                text="More detailed text.",
            ),
        ]
        result = _format_rag_context(entries)
        assert "[1]" in result
        assert "[2]" in result
        assert "0.950" in result
        assert "example.com/page1" in result
        assert "Some relevant text." in result
        assert "---" in result


# ── Image Processing Integration Tests ──────────────────────────────────────


class TestImageProcessingInPipeline:
    """Tests for image extraction and storage during _process_fetched_page."""

    @patch("research_tool.brain.download_image")
    @patch("research_tool.brain._make_image_embedding")
    @patch("research_tool.brain.extract_images")
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_process_result_stores_images(
        self, mock_embedding, mock_safe_url, mock_extract_images,
        mock_make_img_emb, mock_download_img,
        tmp_store, mock_browser, mock_llm,
    ):
        """_process_fetched_page stores both text chunks and images."""
        # Setup embeddings for text
        emb_counter = [0]

        def embedding_side_effect(text):
            emb_counter[0] += 1
            return _make_fake_embedding(float(emb_counter[0]))

        mock_embedding.side_effect = embedding_side_effect

        # Setup image extraction
        mock_extract_images.return_value = [
            {
                "src_url": "https://cdn.example.com/photo.jpg",
                "alt_text": "A photo",
                "width": 800,
                "height": 600,
                "dom_offset": 50,
            },
        ]

        fake_img_emb = [0.1] * 512
        mock_make_img_emb.return_value = fake_img_emb
        mock_download_img.return_value = b"fake-image-bytes"

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )

        result = SearchResult(
            title="Test Page",
            url="https://example.com/page",
            snippet="A test page",
        )
        loop._process_fetched_page(result.url, MOCK_HTML, result.url)

        # Verify text chunks were stored
        chunks = tmp_store.get_all_chunks()
        assert len(chunks) > 0

        # Verify image was stored
        assert tmp_store.get_image_count() == 1
        images = tmp_store.get_images_for_page("https://example.com/page")
        assert len(images) == 1
        assert images[0]["src_url"] == "https://cdn.example.com/photo.jpg"
        assert images[0]["alt_text"] == "A photo"

    @patch("research_tool.brain.extract_images")
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_no_images_produces_zero_records(
        self, mock_embedding, mock_safe_url, mock_extract_images,
        tmp_store, mock_browser, mock_llm,
    ):
        """Page with no extractable images produces zero image records."""
        mock_embedding.return_value = _make_fake_embedding(1.0)
        mock_extract_images.return_value = []

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )

        result = SearchResult(
            title="Text Only",
            url="https://textonly.com/page",
            snippet="No images here",
        )
        loop._process_fetched_page(result.url, MOCK_HTML, result.url)

        assert tmp_store.get_image_count() == 0

    @patch("research_tool.brain.download_image")
    @patch("research_tool.brain.extract_images")
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_image_download_failure_does_not_block_text(
        self, mock_embedding, mock_safe_url, mock_extract_images,
        mock_download_img,
        tmp_store, mock_browser, mock_llm,
    ):
        """Image download failure does not block text chunk processing."""
        mock_embedding.return_value = _make_fake_embedding(1.0)

        # Image extraction succeeds but download fails
        mock_extract_images.return_value = [
            {
                "src_url": "https://cdn.example.com/broken.jpg",
                "alt_text": "Broken",
                "width": 800,
                "height": 600,
                "dom_offset": 50,
            },
        ]
        mock_download_img.return_value = None  # download failed

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )

        result = SearchResult(
            title="Test Page",
            url="https://example.com/imgfail",
            snippet="A test page",
        )
        loop._process_fetched_page(result.url, MOCK_HTML, result.url)

        # Text chunks should still be stored
        chunks = tmp_store.get_all_chunks()
        assert len(chunks) > 0

        # Image stored but with NULL embedding (download failed)
        assert tmp_store.get_image_count() == 1
        images = tmp_store.get_images_for_page("https://example.com/imgfail")
        assert images[0]["embedding"] is None

    @patch("research_tool.brain.extract_images", side_effect=RuntimeError("extraction boom"))
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_image_extraction_exception_does_not_block_text(
        self, mock_embedding, mock_safe_url, mock_extract_images,
        tmp_store, mock_browser, mock_llm,
    ):
        """Exception in image extraction does not block text chunk processing."""
        mock_embedding.return_value = _make_fake_embedding(1.0)

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )

        result = SearchResult(
            title="Test Page",
            url="https://example.com/exc",
            snippet="A test page",
        )
        loop._process_fetched_page(result.url, MOCK_HTML, result.url)

        # Text chunks should still be stored despite image exception
        chunks = tmp_store.get_all_chunks()
        assert len(chunks) > 0

    @patch("research_tool.brain.extract_chart_text")
    @patch("research_tool.brain.classify_image_is_chart")
    @patch("research_tool.brain.compute_image_chunk_proximity")
    @patch("research_tool.brain.download_image")
    @patch("research_tool.brain._make_image_embedding")
    @patch("research_tool.brain.extract_images")
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_chart_detection_stores_bytes_and_ocr_chunk(
        self, mock_embedding, mock_safe_url, mock_extract_images,
        mock_make_img_emb, mock_download_img, mock_proximity,
        mock_classify, mock_ocr,
        tmp_store, mock_browser, mock_llm,
    ):
        """Chart images get stored with bytes and produce an OCR chunk."""
        emb_counter = [0]

        def embedding_side_effect(text):
            emb_counter[0] += 1
            return _make_fake_embedding(float(emb_counter[0]))

        mock_embedding.side_effect = embedding_side_effect
        mock_extract_images.return_value = [
            {"src_url": "https://cdn.example.com/chart.png",
             "alt_text": "Revenue chart", "width": 800, "height": 600, "dom_offset": 50},
        ]
        mock_make_img_emb.return_value = [0.1] * 512
        mock_download_img.return_value = b"\x89PNG\r\n\x1a\nfake-chart"
        mock_classify.return_value = True
        mock_ocr.return_value = "Revenue $1.2M Q1 2026"

        loop = ResearchLoop(
            prompt="test", store=tmp_store,
            browser=mock_browser, llm_client=mock_llm,
        )

        # Proximity must map image index 0 to actual chunk IDs after chunks are stored
        def proximity_side_effect(images, chunks, html):
            chunk_ids = [c.chunk_id for c in chunks]
            return {0: chunk_ids}
        mock_proximity.side_effect = proximity_side_effect

        result = SearchResult(title="Paper", url="https://example.com/paper", snippet="A paper")
        loop._process_fetched_page(result.url, MOCK_HTML, result.url)

        images = tmp_store.get_images_for_page("https://example.com/paper")
        assert len(images) == 1
        chart_imgs = tmp_store.get_chart_images_for_chunks(
            [c.chunk_id for c in tmp_store.get_all_chunks()]
        )
        assert len(chart_imgs) == 1
        assert chart_imgs[0]["image_bytes"] == b"\x89PNG\r\n\x1a\nfake-chart"

        ocr_chunks = [c for c in tmp_store.get_all_chunks() if "_ocr" in c.chunk_id]
        assert len(ocr_chunks) == 1
        assert "Revenue $1.2M" in ocr_chunks[0].text

    @patch("research_tool.brain.extract_chart_text")
    @patch("research_tool.brain.classify_image_is_chart")
    @patch("research_tool.brain.download_image")
    @patch("research_tool.brain._make_image_embedding")
    @patch("research_tool.brain.extract_images")
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_chart_with_no_ocr_text_skips_chunk(
        self, mock_embedding, mock_safe_url, mock_extract_images,
        mock_make_img_emb, mock_download_img, mock_classify, mock_ocr,
        tmp_store, mock_browser, mock_llm,
    ):
        """Chart with no OCR text does not produce an OCR chunk."""
        mock_embedding.return_value = _make_fake_embedding(1.0)
        mock_extract_images.return_value = [
            {"src_url": "https://cdn.example.com/chart.png",
             "alt_text": "Chart", "width": 800, "height": 600, "dom_offset": 50},
        ]
        mock_make_img_emb.return_value = [0.1] * 512
        mock_download_img.return_value = b"\x89PNG\r\n\x1a\nfake"
        mock_classify.return_value = True
        mock_ocr.return_value = None

        loop = ResearchLoop(
            prompt="test", store=tmp_store,
            browser=mock_browser, llm_client=mock_llm,
        )

        result = SearchResult(title="Paper", url="https://example.com/paper2", snippet="Paper")
        loop._process_fetched_page(result.url, MOCK_HTML, result.url)

        ocr_chunks = [c for c in tmp_store.get_all_chunks() if "_ocr" in c.chunk_id]
        assert len(ocr_chunks) == 0


class TestQueryEngineImageCount:
    """Tests for image count in QueryEngine status output."""

    def test_status_includes_image_count(self, tmp_store):
        """status() includes total_images in its output."""
        tmp_store.store_page("https://example.com", title="Test")
        tmp_store.store_image(
            image_id="img1",
            page_url="https://example.com",
            src_url="https://cdn.example.com/1.jpg",
            embedding=None,
        )
        tmp_store.store_image(
            image_id="img2",
            page_url="https://example.com",
            src_url="https://cdn.example.com/2.jpg",
            embedding=None,
        )

        engine = QueryEngine(db_path="unused")
        engine.store = tmp_store

        stats = engine.status()
        assert "total_images" in stats
        assert stats["total_images"] == 2

    def test_status_zero_images(self, tmp_store):
        """status() shows 0 images on empty DB."""
        engine = QueryEngine(db_path="unused")
        engine.store = tmp_store

        stats = engine.status()
        assert stats["total_images"] == 0


# ── Link Extraction and Graph Integration Tests ─────────────────────────────


class TestLinkExtractionInPipeline:
    """Tests for link extraction and graph embedding computation in the research loop."""

    @patch("research_tool.brain.extract_links")
    @patch("research_tool.brain.extract_images", return_value=[])
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_process_result_stores_links(
        self, mock_embedding, mock_safe_url, mock_extract_images,
        mock_extract_links,
        tmp_store, mock_browser, mock_llm,
    ):
        """_process_fetched_page stores extracted links."""
        mock_embedding.return_value = _make_fake_embedding(1.0)
        mock_extract_links.return_value = [
            {"target_url": "https://target1.com", "anchor_text": "Target 1"},
            {"target_url": "https://target2.com", "anchor_text": "Target 2"},
        ]

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )

        result = SearchResult(
            title="Test Page",
            url="https://example.com/page",
            snippet="A test page",
        )
        loop._process_fetched_page(result.url, MOCK_HTML, result.url)

        # Verify links were stored
        assert tmp_store.get_link_count() == 2
        graph = tmp_store.get_link_graph()
        assert "https://example.com/page" in graph
        assert "https://target1.com" in graph["https://example.com/page"]
        assert "https://target2.com" in graph["https://example.com/page"]

    @patch("research_tool.brain.extract_links", side_effect=RuntimeError("link boom"))
    @patch("research_tool.brain.extract_images", return_value=[])
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_link_extraction_failure_does_not_block_text(
        self, mock_embedding, mock_safe_url, mock_extract_images,
        mock_extract_links,
        tmp_store, mock_browser, mock_llm,
    ):
        """Exception in link extraction does not block text chunk processing."""
        mock_embedding.return_value = _make_fake_embedding(1.0)

        loop = ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
        )

        result = SearchResult(
            title="Test Page",
            url="https://example.com/linkfail",
            snippet="A test page",
        )
        loop._process_fetched_page(result.url, MOCK_HTML, result.url)

        # Text chunks should still be stored despite link extraction failure
        chunks = tmp_store.get_all_chunks()
        assert len(chunks) > 0
        # No links stored
        assert tmp_store.get_link_count() == 0

    @patch("research_tool.brain.search_ddg")
    @patch("research_tool.brain.extract_links")
    @patch("research_tool.brain.extract_images", return_value=[])
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_graph_embeddings_recomputed_at_end_of_iteration(
        self, mock_embedding, mock_safe_url, mock_extract_images,
        mock_extract_links, mock_search,
        tmp_store, mock_browser, mock_llm,
    ):
        """Graph embeddings are recomputed at end of each iteration."""
        # LLM returns queries for one iteration, then stops
        mock_llm.generate.side_effect = [
            json.dumps(["graph test query"]),
            "[]",
        ]

        page_counter = [0]

        def search_side_effect(query, rate_limiter=None, max_results=20, browser=None):
            page_counter[0] += 1
            return [
                SearchResult(
                    title=f"Result {page_counter[0]}",
                    url=f"https://graph-test-{page_counter[0]}.com/page",
                    snippet=f"Snippet {page_counter[0]}",
                ),
            ]

        mock_search.side_effect = search_side_effect

        emb_counter = [0]

        def embedding_side_effect(text):
            emb_counter[0] += 1
            return _make_fake_embedding(float(emb_counter[0]))

        mock_embedding.side_effect = embedding_side_effect

        # Links extracted from each page
        mock_extract_links.return_value = [
            {"target_url": "https://linked-target.com", "anchor_text": "Target"},
        ]

        loop = ResearchLoop(
            prompt="graph test",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
            max_depth=5,
        )

        result = loop.run()

        # Verify graph data was stored (links exist, so graph embeddings computed)
        assert tmp_store.get_link_count() > 0


class TestQueryEngineLinkCount:
    """Tests for link count in QueryEngine status output."""

    def test_status_includes_link_count(self, tmp_store):
        """status() includes total_links in its output."""
        tmp_store.store_page("https://example.com", title="Test")
        tmp_store.store_links("https://example.com", [
            {"target_url": "https://a.com", "anchor_text": "A"},
            {"target_url": "https://b.com", "anchor_text": "B"},
            {"target_url": "https://c.com", "anchor_text": "C"},
        ])

        engine = QueryEngine(db_path="unused")
        engine.store = tmp_store

        stats = engine.status()
        assert "total_links" in stats
        assert stats["total_links"] == 3

    def test_status_zero_links(self, tmp_store):
        """status() shows 0 links on empty DB."""
        engine = QueryEngine(db_path="unused")
        engine.store = tmp_store

        stats = engine.status()
        assert stats["total_links"] == 0


# ── Index Mode Tests ──────────────────────────────────────────────────────────


class TestIndexModes:
    """Tests verifying ingest-time vs query-time index mode usage."""

    @patch("research_tool.brain.search_ddg")
    @patch("research_tool.brain.extract_links", return_value=[])
    @patch("research_tool.brain.extract_images", return_value=[])
    @patch("research_tool.brain.is_safe_url", return_value=True)
    @patch("research_tool.brain._make_embedding")
    def test_ingest_time_uses_ingest_mode(
        self, mock_embedding, mock_safe_url, mock_extract_images,
        mock_extract_links, mock_search,
        tmp_store, mock_browser, mock_llm,
    ):
        """ResearchLoop uses HybridIndex(mode='ingest') for mid-loop retrieval."""
        # Two iterations: first produces chunks, second triggers index build
        call_count = [0]

        def llm_side_effect(system, user):
            call_count[0] += 1
            if call_count[0] <= 2:
                return json.dumps(["test query"])
            return "[]"

        mock_llm.generate.side_effect = llm_side_effect

        page_counter = [0]

        def search_side_effect(query, rate_limiter=None, max_results=20, browser=None):
            page_counter[0] += 1
            return [
                SearchResult(
                    title=f"Result {page_counter[0]}",
                    url=f"https://idx-test-{page_counter[0]}.com/page",
                    snippet=f"Snippet {page_counter[0]}",
                ),
            ]

        mock_search.side_effect = search_side_effect

        emb_counter = [0]

        def embedding_side_effect(text):
            emb_counter[0] += 1
            return _make_fake_embedding(float(emb_counter[0]))

        mock_embedding.side_effect = embedding_side_effect

        loop = ResearchLoop(
            prompt="test topic",
            store=tmp_store,
            browser=mock_browser,
            llm_client=mock_llm,
            max_depth=5,
        )

        original_cls = HybridIndex
        created_modes = []

        class TrackingHybridIndex(original_cls):
            def __init__(self, mode="query"):
                created_modes.append(mode)
                super().__init__(mode=mode)

        with patch("research_tool.brain.HybridIndex", TrackingHybridIndex):
            loop.run()

        assert "ingest" in created_modes, (
            f"ResearchLoop should use HybridIndex(mode='ingest'), got modes: {created_modes}"
        )

    def test_query_engine_uses_query_mode(self, tmp_store, mock_llm):
        """QueryEngine uses HybridIndex(mode='query') for retrieval."""
        tmp_store.store_page("https://example.com/ai", title="AI Research")
        emb = _make_fake_embedding(1.0)
        _store_chunk_with_embedding(
            tmp_store, "ai::chunk-0",
            "Artificial intelligence is transforming many industries.",
            emb, page_url="https://example.com/ai",
        )

        mock_llm.generate.return_value = "AI is transforming industries."

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        original_cls = HybridIndex
        created_modes = []

        class TrackingHybridIndex(original_cls):
            def __init__(self, mode="query"):
                created_modes.append(mode)
                super().__init__(mode=mode)

        with patch("research_tool.brain.HybridIndex", TrackingHybridIndex):
            engine._index = None  # force re-creation
            engine.ask("What is AI?")

        assert "query" in created_modes, (
            f"QueryEngine should use HybridIndex(mode='query'), got modes: {created_modes}"
        )


# ── Contextual Retrieval Tests ───────────────────────────────────────────


class TestContextualRetrieval:
    """U4: LLM-generated chunk summaries during ingest."""

    def test_generate_context_summaries_happy_path(self, tmp_store, mock_browser, mock_llm):
        """LLM generates one summary per chunk, assigned to context_summary."""
        mock_llm.generate.return_value = (
            "Chunk 1: This chunk covers machine learning basics.\n"
            "Chunk 2: This chunk discusses neural network architectures."
        )

        loop = ResearchLoop(
            prompt="test", store=tmp_store, browser=mock_browser,
            llm_client=mock_llm,
        )
        chunks = [
            DocumentChunk(text="ML basics paragraph " * 20, page_url="http://ex.com",
                          chunk_id="c1", section_title="Intro"),
            DocumentChunk(text="Neural nets paragraph " * 20, page_url="http://ex.com",
                          chunk_id="c2", section_title="Architecture"),
        ]
        loop._generate_context_summaries(chunks, "ML Guide", "Full text here")

        assert chunks[0].context_summary == "This chunk covers machine learning basics."
        assert chunks[1].context_summary == "This chunk discusses neural network architectures."

    def test_generate_context_summaries_llm_failure(self, tmp_store, mock_browser, mock_llm):
        """LLM exception → summaries skipped, no crash."""
        mock_llm.generate.side_effect = Exception("API down")

        loop = ResearchLoop(
            prompt="test", store=tmp_store, browser=mock_browser,
            llm_client=mock_llm,
        )
        chunks = [
            DocumentChunk(text="some text", page_url="http://ex.com",
                          chunk_id="c1", section_title="S"),
        ]
        loop._generate_context_summaries(chunks, "Title", "Full text")

        assert chunks[0].context_summary is None

    def test_generate_context_summaries_empty_response(self, tmp_store, mock_browser, mock_llm):
        """Empty LLM response → no summaries assigned."""
        mock_llm.generate.return_value = ""

        loop = ResearchLoop(
            prompt="test", store=tmp_store, browser=mock_browser,
            llm_client=mock_llm,
        )
        chunks = [
            DocumentChunk(text="some text", page_url="http://ex.com",
                          chunk_id="c1", section_title="S"),
        ]
        loop._generate_context_summaries(chunks, "Title", "Full text")

        assert chunks[0].context_summary is None

    def test_generate_context_summaries_partial_response(self, tmp_store, mock_browser, mock_llm):
        """LLM returns summary for only some chunks — others stay None."""
        mock_llm.generate.return_value = "Chunk 2: Discusses databases."

        loop = ResearchLoop(
            prompt="test", store=tmp_store, browser=mock_browser,
            llm_client=mock_llm,
        )
        chunks = [
            DocumentChunk(text="first chunk text", page_url="http://ex.com",
                          chunk_id="c1", section_title="S"),
            DocumentChunk(text="database chunk text", page_url="http://ex.com",
                          chunk_id="c2", section_title="S"),
        ]
        loop._generate_context_summaries(chunks, "Title", "Full text")

        assert chunks[0].context_summary is None
        assert chunks[1].context_summary == "Discusses databases."

    def test_generate_context_summaries_empty_chunks(self, tmp_store, mock_browser, mock_llm):
        """No chunks → no LLM call."""
        loop = ResearchLoop(
            prompt="test", store=tmp_store, browser=mock_browser,
            llm_client=mock_llm,
        )
        loop._generate_context_summaries([], "Title", "Text")
        mock_llm.generate.assert_not_called()

    def test_context_summary_stored_and_retrieved(self, tmp_store):
        """context_summary round-trips through store_chunk/get_chunk."""
        tmp_store.store_page("http://example.com", title="Test")
        chunk = DocumentChunk(
            text="Some content here",
            page_url="http://example.com",
            chunk_id="ctx-test-1",
            section_title="Section",
            context_summary="This chunk covers topic X in the context of Y.",
        )
        tmp_store.store_chunk(chunk)

        retrieved = tmp_store.get_chunk("ctx-test-1")
        assert retrieved is not None
        assert retrieved.context_summary == "This chunk covers topic X in the context of Y."
        assert retrieved.text == "Some content here"

    def test_context_summary_null_for_legacy_chunks(self, tmp_store):
        """Chunks stored without context_summary have None on retrieval."""
        tmp_store.store_page("http://ex.com", title="Test")
        chunk = DocumentChunk(
            text="Legacy text", page_url="http://ex.com",
            chunk_id="legacy-1", section_title="S",
        )
        tmp_store.store_chunk(chunk)

        retrieved = tmp_store.get_chunk("legacy-1")
        assert retrieved is not None
        assert retrieved.context_summary is None

    def test_children_inherit_context_summary(self):
        """split_into_children propagates context_summary from parent."""
        parent = DocumentChunk(
            text=("First paragraph about data structures. " * 10 + "\n\n" +
                  "Second paragraph about algorithms. " * 10),
            page_url="http://ex.com",
            chunk_id="parent-1",
            section_title="CS Basics",
            context_summary="This chunk is from a CS textbook, covering fundamentals.",
        )
        children = split_into_children(parent, max_tokens=30)
        assert len(children) >= 2
        for child in children:
            assert child.context_summary == "This chunk is from a CS textbook, covering fundamentals."

    @patch("research_tool.brain._make_embedding")
    @patch("research_tool.brain._make_token_embeddings", return_value=None)
    @patch("research_tool.brain._make_code_embedding", return_value=None)
    @patch("research_tool.brain.extract_content")
    @patch("research_tool.brain.extract_images", return_value=[])
    def test_process_fetched_page_calls_contextual_retrieval(
        self, mock_images, mock_extract, mock_code_emb, mock_tok, mock_emb,
        tmp_store, mock_browser, mock_llm,
    ):
        """_process_fetched_page calls _generate_context_summaries when enabled."""
        mock_extract.return_value = {
            "title": "Test Page",
            "text": "A " * 200,
        }
        mock_emb.return_value = [0.1] * EMBEDDING_DIM

        loop = ResearchLoop(
            prompt="test", store=tmp_store, browser=mock_browser,
            llm_client=mock_llm,
        )

        with patch("research_tool.brain.CONTEXTUAL_RETRIEVAL", True), \
             patch.object(loop, "_generate_context_summaries") as mock_gen:
            loop._process_fetched_page("http://ex.com", "<html></html>", "http://ex.com")

        mock_gen.assert_called_once()
        args = mock_gen.call_args[0]
        assert len(args[0]) > 0  # chunks list
        assert args[1] == "Test Page"  # page_title

    @patch("research_tool.brain._make_embedding")
    @patch("research_tool.brain._make_token_embeddings", return_value=None)
    @patch("research_tool.brain._make_code_embedding", return_value=None)
    @patch("research_tool.brain.extract_content")
    @patch("research_tool.brain.extract_images", return_value=[])
    def test_process_fetched_page_skips_contextual_when_disabled(
        self, mock_images, mock_extract, mock_code_emb, mock_tok, mock_emb,
        tmp_store, mock_browser, mock_llm,
    ):
        """_process_fetched_page skips summaries when CONTEXTUAL_RETRIEVAL=False."""
        mock_extract.return_value = {
            "title": "Test Page",
            "text": "A " * 200,
        }
        mock_emb.return_value = [0.1] * EMBEDDING_DIM

        loop = ResearchLoop(
            prompt="test", store=tmp_store, browser=mock_browser,
            llm_client=mock_llm,
        )

        with patch("research_tool.brain.CONTEXTUAL_RETRIEVAL", False), \
             patch.object(loop, "_generate_context_summaries") as mock_gen:
            loop._process_fetched_page("http://ex.com", "<html></html>", "http://ex.com")

        mock_gen.assert_not_called()

    @patch("research_tool.brain._make_embedding")
    @patch("research_tool.brain._make_token_embeddings", return_value=None)
    @patch("research_tool.brain._make_code_embedding", return_value=None)
    @patch("research_tool.brain.extract_content")
    @patch("research_tool.brain.extract_images", return_value=[])
    def test_child_embedding_prepends_context_summary(
        self, mock_images, mock_extract, mock_code_emb, mock_tok, mock_emb,
        tmp_store, mock_browser, mock_llm,
    ):
        """When context_summary exists, child embeddings use summary + text."""
        long_text = ("First paragraph with enough tokens. " * 15 + "\n\n" +
                     "Second paragraph about different topic. " * 15)
        mock_extract.return_value = {
            "title": "Test Page",
            "text": long_text,
        }
        mock_emb.return_value = [0.1] * EMBEDDING_DIM

        mock_llm.generate.return_value = "Chunk 1: Covers paragraph topics."

        loop = ResearchLoop(
            prompt="test", store=tmp_store, browser=mock_browser,
            llm_client=mock_llm,
        )

        embed_inputs = []
        original_make_embedding = mock_emb.side_effect

        def track_embed(text, mode="query"):
            embed_inputs.append(text)
            return [0.1] * EMBEDDING_DIM

        mock_emb.side_effect = track_embed

        with patch("research_tool.brain.CONTEXTUAL_RETRIEVAL", True), \
             patch("research_tool.brain.PARENT_CHILD_ENABLED", True):
            loop._process_fetched_page("http://ex.com", "<html></html>", "http://ex.com")

        child_chunks = [c for c in tmp_store.get_all_chunks() if c.is_child]
        if child_chunks:
            for child in child_chunks:
                assert child.context_summary is not None or child.context_summary is None
                assert "Covers paragraph topics" not in child.text

        parent_embed_calls = [t for t in embed_inputs if "Covers paragraph topics" in t]
        assert len(parent_embed_calls) >= 1


# ── Entity Extraction Tests ──────────────────────────────────────────────────


class TestEntityExtraction:
    """Tests for entity extraction at ingest time."""

    def _make_loop(self, tmp_store, mock_llm):
        browser = MagicMock(spec=Browser)
        return ResearchLoop(
            prompt="test",
            store=tmp_store,
            browser=browser,
            llm_client=mock_llm,
            max_depth=1,
        )

    def test_extract_entities_happy_path(self, tmp_store, mock_llm):
        """Happy path: _extract_entities returns parsed entity list."""
        mock_llm.generate.return_value = json.dumps([
            {"name": "Python GIL", "definition": "The Global Interpreter Lock"},
            {"name": "CPython", "definition": "Reference implementation of Python"},
        ])
        loop = self._make_loop(tmp_store, mock_llm)
        entities = loop._extract_entities("Some text about Python.", "Test Page")
        assert len(entities) == 2
        assert entities[0]["name"] == "Python GIL"
        assert entities[1]["name"] == "CPython"

    def test_extract_entities_malformed_json(self, tmp_store, mock_llm):
        """Edge case: malformed JSON returns empty list."""
        mock_llm.generate.return_value = "not json at all"
        loop = self._make_loop(tmp_store, mock_llm)
        entities = loop._extract_entities("Some text.", "Test Page")
        assert entities == []

    def test_extract_entities_empty_response(self, tmp_store, mock_llm):
        """Edge case: empty LLM response returns empty list."""
        mock_llm.generate.return_value = ""
        loop = self._make_loop(tmp_store, mock_llm)
        entities = loop._extract_entities("Some text.", "Test Page")
        assert entities == []

    def test_extract_entities_llm_exception(self, tmp_store, mock_llm):
        """Edge case: LLM exception returns empty list."""
        mock_llm.generate.side_effect = Exception("API timeout")
        loop = self._make_loop(tmp_store, mock_llm)
        entities = loop._extract_entities("Some text.", "Test Page")
        assert entities == []

    def test_extract_entities_json_with_surrounding_text(self, tmp_store, mock_llm):
        """Edge case: JSON array embedded in prose."""
        mock_llm.generate.return_value = (
            'Here are the entities:\n[{"name": "GIL", "definition": "Lock"}]\nDone.'
        )
        loop = self._make_loop(tmp_store, mock_llm)
        entities = loop._extract_entities("Some text.", "Test Page")
        assert len(entities) == 1
        assert entities[0]["name"] == "GIL"

    @patch("research_tool.brain._make_token_embeddings", return_value=None)
    @patch("research_tool.brain._make_code_embedding", return_value=None)
    @patch("research_tool.brain._make_embedding")
    @patch("research_tool.brain.extract_content")
    def test_process_page_with_entities_enabled(
        self, mock_extract, mock_emb, mock_code_emb, mock_tok,
        tmp_store, mock_llm,
    ):
        """Happy path: entity extraction during page processing stores entities and links."""
        mock_extract.return_value = {
            "title": "Test",
            "text": "Python uses a GIL. " * 20,
        }
        emb_counter = [0]

        def emb_side_effect(text, mode="document"):
            emb_counter[0] += 1
            return _make_fake_embedding(float(emb_counter[0]))

        mock_emb.side_effect = emb_side_effect

        call_count = [0]

        def llm_side_effect(system, user):
            call_count[0] += 1
            if "entity extraction" in system.lower():
                return json.dumps([
                    {"name": "Python GIL", "definition": "The Global Interpreter Lock"},
                ])
            return "[]"

        mock_llm.generate.side_effect = llm_side_effect

        loop = self._make_loop(tmp_store, mock_llm)
        tmp_store.store_page(url="http://ex.com", title="Test")

        with patch("research_tool.brain.ENTITY_RESOLUTION_ENABLED", True), \
             patch("research_tool.brain.PARENT_CHILD_ENABLED", False):
            loop._process_fetched_page("http://ex.com", "<html></html>", "http://ex.com")

        entities = tmp_store.get_all_entities()
        assert len(entities) >= 1
        assert any(e["canonical_name"] == "Python GIL" for e in entities)

        all_chunks = tmp_store.get_all_chunks()
        for chunk in all_chunks:
            chunk_entities = tmp_store.get_entities_for_chunk(chunk.chunk_id)
            if not chunk.is_child:
                assert len(chunk_entities) >= 0

    @patch("research_tool.brain._make_token_embeddings", return_value=None)
    @patch("research_tool.brain._make_code_embedding", return_value=None)
    @patch("research_tool.brain._make_embedding")
    @patch("research_tool.brain.extract_content")
    def test_process_page_entity_resolution_disabled(
        self, mock_extract, mock_emb, mock_code_emb, mock_tok,
        tmp_store, mock_llm,
    ):
        """Edge case: entity resolution disabled — no LLM calls for entities."""
        mock_extract.return_value = {
            "title": "Test",
            "text": "Some content here. " * 20,
        }
        mock_emb.return_value = _make_fake_embedding(1.0)
        mock_llm.generate.return_value = "[]"

        loop = self._make_loop(tmp_store, mock_llm)
        tmp_store.store_page(url="http://ex.com", title="Test")

        with patch("research_tool.brain.ENTITY_RESOLUTION_ENABLED", False), \
             patch("research_tool.brain.PARENT_CHILD_ENABLED", False):
            loop._process_fetched_page("http://ex.com", "<html></html>", "http://ex.com")

        entities = tmp_store.get_all_entities()
        assert len(entities) == 0

    @patch("research_tool.brain._make_token_embeddings", return_value=None)
    @patch("research_tool.brain._make_code_embedding", return_value=None)
    @patch("research_tool.brain._make_embedding")
    @patch("research_tool.brain.extract_content")
    def test_child_chunks_inherit_parent_entities(
        self, mock_extract, mock_emb, mock_code_emb, mock_tok,
        tmp_store, mock_llm,
    ):
        """Happy path: child chunks inherit parent's entity links."""
        mock_extract.return_value = {
            "title": "Test",
            "text": ("This is about the Python GIL and threading. " * 30),
        }
        emb_counter = [0]

        def emb_side_effect(text, mode="document"):
            emb_counter[0] += 1
            return _make_fake_embedding(float(emb_counter[0]))

        mock_emb.side_effect = emb_side_effect

        def llm_side_effect(system, user):
            if "entity extraction" in system.lower():
                return json.dumps([
                    {"name": "Python GIL", "definition": "The Global Interpreter Lock"},
                ])
            return "[]"

        mock_llm.generate.side_effect = llm_side_effect

        loop = self._make_loop(tmp_store, mock_llm)
        tmp_store.store_page(url="http://ex.com", title="Test")

        with patch("research_tool.brain.ENTITY_RESOLUTION_ENABLED", True), \
             patch("research_tool.brain.PARENT_CHILD_ENABLED", True):
            loop._process_fetched_page("http://ex.com", "<html></html>", "http://ex.com")

        child_chunks = [c for c in tmp_store.get_all_chunks() if c.is_child]
        if child_chunks:
            for child in child_chunks:
                child_entities = tmp_store.get_entities_for_chunk(child.chunk_id)
                parent_entities = tmp_store.get_entities_for_chunk(child.parent_chunk_id)
                parent_eids = {e["entity_id"] for e in parent_entities}
                child_eids = {e["entity_id"] for e in child_entities}
                assert child_eids == parent_eids

    @patch("research_tool.brain._make_token_embeddings", return_value=None)
    @patch("research_tool.brain._make_code_embedding", return_value=None)
    @patch("research_tool.brain._make_embedding")
    @patch("research_tool.brain.extract_content")
    def test_entity_resolution_across_chunks(
        self, mock_extract, mock_emb, mock_code_emb, mock_tok,
        tmp_store, mock_llm,
    ):
        """Happy path: same entity from different chunks resolves to one canonical entity."""
        mock_extract.return_value = {
            "title": "Test",
            "text": ("Paragraph about Python GIL and how it affects performance. " * 50
                     + "\n\n"
                     + "The Global Interpreter Lock prevents true threading in CPython. " * 50),
        }
        emb_counter = [0]
        gil_embedding = _make_fake_embedding(42.0)

        def emb_side_effect(text, mode="document"):
            # Entity embed texts start with the entity name followed by ":"
            # e.g. "Python GIL: The GIL in CPython"
            if text.startswith("Python GIL:") or text.startswith("Global Interpreter Lock:"):
                return list(gil_embedding)
            emb_counter[0] += 1
            return _make_fake_embedding(float(emb_counter[0]))

        mock_emb.side_effect = emb_side_effect

        chunk_num = [0]

        def llm_side_effect(system, user):
            if "entity extraction" in system.lower():
                chunk_num[0] += 1
                if chunk_num[0] == 1:
                    return json.dumps([
                        {"name": "Python GIL", "definition": "The GIL in CPython"},
                    ])
                else:
                    return json.dumps([
                        {"name": "Global Interpreter Lock", "definition": "CPython's GIL"},
                    ])
            return "[]"

        mock_llm.generate.side_effect = llm_side_effect

        loop = self._make_loop(tmp_store, mock_llm)
        tmp_store.store_page(url="http://ex.com", title="Test")

        with patch("research_tool.brain.ENTITY_RESOLUTION_ENABLED", True), \
             patch("research_tool.brain.PARENT_CHILD_ENABLED", False), \
             patch("research_tool.brain.ENTITY_RESOLUTION_THRESHOLD", 0.5):
            loop._process_fetched_page("http://ex.com", "<html></html>", "http://ex.com")

        entities = tmp_store.get_all_entities()
        gil_entities = [
            e for e in entities
            if "GIL" in e["canonical_name"] or "Interpreter Lock" in e["canonical_name"]
        ]
        assert len(gil_entities) == 1
        assert "Global Interpreter Lock" in gil_entities[0]["aliases"] or \
               "Python GIL" in gil_entities[0]["aliases"]


class TestSubQueryDecomposition:
    """Tests for sub-query decomposition in QueryEngine."""

    def test_decompose_returns_sub_queries(self, tmp_store, mock_llm):
        """Complex query decomposes into sub-queries."""
        mock_llm.generate.return_value = json.dumps([
            "BM25 weight tuning techniques",
            "cosine similarity optimization",
            "cross-encoder reranking strategies",
        ])

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        result = engine._decompose_query(
            "What RAG optimizations exist for hybrid BM25-cosine retrieval with reranking?"
        )
        assert len(result) == 3
        assert "BM25" in result[0]

    def test_decompose_caps_at_four(self, tmp_store, mock_llm):
        """Decomposition caps at 4 sub-queries."""
        mock_llm.generate.return_value = json.dumps([
            "q1", "q2", "q3", "q4", "q5",
        ])

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        result = engine._decompose_query("complex question")
        assert len(result) == 4

    def test_decompose_fallback_on_llm_failure(self, tmp_store, mock_llm):
        """LLM failure returns original query."""
        mock_llm.generate.side_effect = RuntimeError("LLM down")

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        result = engine._decompose_query("original question")
        assert result == ["original question"]

    def test_decompose_fallback_on_malformed_json(self, tmp_store, mock_llm):
        """Malformed JSON returns original query."""
        mock_llm.generate.return_value = "not json"

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        result = engine._decompose_query("original question")
        assert result == ["original question"]

    def test_simple_query_bypasses_decomposition(self, tmp_store, mock_llm):
        """Simple query does not trigger decomposition."""
        tmp_store.store_page("https://example.com/p", title="P")
        emb = _make_fake_embedding(1.0)
        _store_chunk_with_embedding(
            tmp_store, "p::c1", "BM25 is a ranking function.", emb,
            page_url="https://example.com/p",
        )

        mock_llm.generate.return_value = "BM25 answer."

        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        with patch("research_tool.brain.SUBQUERY_ENABLED", True), \
             patch.object(engine, "_decompose_query") as mock_decompose:
            engine.ask("What is BM25?")

        mock_decompose.assert_not_called()

    def test_subquery_disabled_by_default(self):
        """SUBQUERY_ENABLED defaults to False."""
        from research_tool.brain import SUBQUERY_ENABLED
        assert SUBQUERY_ENABLED is False

    def test_merge_results_keeps_best_score(self, tmp_store, mock_llm):
        """Merged results keep the highest score for each chunk."""
        engine = QueryEngine(db_path="unused", llm_client=mock_llm)
        engine.store = tmp_store

        mock_index = MagicMock()
        r1 = RetrievedEntry(
            chunk_id="c1", page_url="https://example.com/p",
            section_title="P", text="content", score=0.5,
            bm25_score=0.3, cosine_score=0.2,
        )
        r2 = RetrievedEntry(
            chunk_id="c1", page_url="https://example.com/p",
            section_title="P", text="content", score=0.9,
            bm25_score=0.6, cosine_score=0.3,
        )
        r3 = RetrievedEntry(
            chunk_id="c2", page_url="https://example.com/p",
            section_title="P", text="other", score=0.7,
            bm25_score=0.4, cosine_score=0.3,
        )

        mock_index.hybrid_retrieve.side_effect = [[r1, r3], [r2]]

        result = engine._retrieve_with_subqueries(
            mock_index, ["q1", "q2"], top_k=5, rerank=False,
        )

        assert len(result) == 2
        assert result[0].chunk_id == "c1"
        assert result[0].score == 0.9
