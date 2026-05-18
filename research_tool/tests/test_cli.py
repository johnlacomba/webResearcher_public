"""Tests for research_tool.__main__ CLI entry point."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

import pytest

from research_tool.__main__ import _ensure_db_dir, _make_llm, main


class TestEnsureDbDir:
    def test_creates_parent_directory(self, tmp_path):
        """Creates parent directories for the DB path."""
        db_path = str(tmp_path / "sub" / "dir" / "test.db")
        _ensure_db_dir(db_path)
        assert (tmp_path / "sub" / "dir").is_dir()

    def test_no_op_for_cwd_relative(self):
        """No error when db_path has no parent directory."""
        _ensure_db_dir("research.db")  # should not raise


class TestMakeLlm:
    def test_claude_backend_requires_key(self):
        """Claude backend raises without ANTHROPIC_API_KEY."""
        env = {k: v for k, v in __import__("os").environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            args = type("Args", (), {"llm": "claude", "model": None})()
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                _make_llm(args)

    def test_omlx_backend_requires_key(self):
        """OMLX backend raises without OMLX_API_KEY."""
        env = {k: v for k, v in __import__("os").environ.items() if k != "OMLX_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            args = type("Args", (), {"llm": "omlx", "model": None})()
            with pytest.raises(ValueError, match="OMLX_API_KEY"):
                _make_llm(args)

    def test_omlx_backend_creates_client(self):
        """OMLX backend creates LLMClient when key is set."""
        with patch.dict("os.environ", {"OMLX_API_KEY": "omlx-test-123"}):
            args = type("Args", (), {"llm": "omlx", "model": "test-model"})()
            client = _make_llm(args)
            assert client._backend == "omlx"
            assert client._model == "test-model"


class TestHelpOutput:
    def test_main_help(self):
        """--help shows subcommands."""
        result = subprocess.run(
            [sys.executable, "-m", "research_tool", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "research" in result.stdout
        assert "query" in result.stdout
        assert "status" in result.stdout
        assert "benchmark" in result.stdout

    def test_research_help_shows_llm_flag(self):
        """research --help shows --llm and --model flags."""
        result = subprocess.run(
            [sys.executable, "-m", "research_tool", "research", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--llm" in result.stdout
        assert "--model" in result.stdout
        assert "--max-depth" in result.stdout
        assert "omlx" in result.stdout

    def test_query_help_shows_llm_flag(self):
        """query --help shows --llm flag."""
        result = subprocess.run(
            [sys.executable, "-m", "research_tool", "query", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--llm" in result.stdout

    def test_query_help_shows_no_rerank_flag(self):
        """query --help shows --no-rerank flag."""
        result = subprocess.run(
            [sys.executable, "-m", "research_tool", "query", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--no-rerank" in result.stdout

    def test_status_help_no_llm_flag(self):
        """status --help does NOT show --llm flag (no LLM needed)."""
        result = subprocess.run(
            [sys.executable, "-m", "research_tool", "status", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--llm" not in result.stdout


class TestStatusCommand:
    def test_status_on_fresh_db_no_api_key(self, tmp_path):
        """status command works without any API key set."""
        db_path = str(tmp_path / "test.db")
        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("ANTHROPIC_API_KEY", "OMLX_API_KEY")}
        with patch.dict("os.environ", env, clear=True):
            with patch("sys.argv", ["research_tool", "status", "--db", db_path]):
                main()

    def test_status_shows_counts(self, tmp_path, capsys):
        """status command prints page/chunk/search counts."""
        db_path = str(tmp_path / "test.db")

        from research_tool.store import ResearchStore
        store = ResearchStore(db_path=db_path)
        store.store_page("https://a.com", title="Page A")
        store.log_search("test query")
        store.close()

        with patch("sys.argv", ["research_tool", "status", "--db", db_path]):
            main()

        captured = capsys.readouterr()
        assert "Total pages:    1" in captured.out
        assert "Total searches: 1" in captured.out
        assert "https://a.com" in captured.out

    def test_status_shows_image_and_link_counts(self, tmp_path, capsys):
        """status command prints image and link counts."""
        db_path = str(tmp_path / "test_imglink.db")

        from research_tool.store import ResearchStore
        store = ResearchStore(db_path=db_path)
        store.store_page("https://a.com", title="Page A")
        store.store_image(
            image_id="img1",
            page_url="https://a.com",
            src_url="https://cdn.example.com/1.jpg",
            embedding=None,
        )
        store.store_links("https://a.com", [
            {"target_url": "https://b.com", "anchor_text": "B"},
            {"target_url": "https://c.com", "anchor_text": "C"},
        ])
        store.close()

        with patch("sys.argv", ["research_tool", "status", "--db", db_path]):
            main()

        captured = capsys.readouterr()
        assert "Total images:   1" in captured.out
        assert "Total links:    2" in captured.out


class TestMissingApiKey:
    def test_research_without_any_key_raises(self, tmp_path):
        """research command raises ValueError when no API key is set (default=claude)."""
        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("ANTHROPIC_API_KEY", "OMLX_API_KEY")}
        db_path = str(tmp_path / "test.db")
        with patch.dict("os.environ", env, clear=True):
            with patch("sys.argv", ["research_tool", "research", "test", "--db", db_path]):
                with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                    main()

    def test_query_without_any_key_raises(self, tmp_path):
        """query command raises ValueError when no API key is set."""
        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("ANTHROPIC_API_KEY", "OMLX_API_KEY")}
        db_path = str(tmp_path / "test.db")
        with patch.dict("os.environ", env, clear=True):
            with patch("sys.argv", ["research_tool", "query", "test question", "--db", db_path]):
                with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                    main()


class TestNoRerankFlag:
    def test_no_rerank_passes_false_to_engine(self, tmp_path):
        """--no-rerank flag passes rerank=False through to engine.ask."""
        db_path = str(tmp_path / "test.db")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("research_tool.brain.QueryEngine.ask", return_value="answer") as mock_ask:
                with patch("research_tool.brain.QueryEngine.close"):
                    with patch("sys.argv", ["research_tool", "query", "test q", "--no-rerank", "--db", db_path]):
                        main()
                mock_ask.assert_called_once_with("test q", rerank=False)

    def test_default_passes_rerank_true(self, tmp_path):
        """Without --no-rerank, rerank=True is passed to engine.ask."""
        db_path = str(tmp_path / "test.db")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("research_tool.brain.QueryEngine.ask", return_value="answer") as mock_ask:
                with patch("research_tool.brain.QueryEngine.close"):
                    with patch("sys.argv", ["research_tool", "query", "test q", "--db", db_path]):
                        main()
                mock_ask.assert_called_once_with("test q", rerank=True)

    def test_query_json_flag_outputs_json(self, tmp_path, capsys):
        """--json flag produces JSON output."""
        db_path = str(tmp_path / "test.db")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("research_tool.brain.QueryEngine.ask", return_value="test answer"):
                with patch("research_tool.brain.QueryEngine.close"):
                    with patch("sys.argv", ["research_tool", "query", "test q", "--json", "--db", db_path]):
                        main()
        import json
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["question"] == "test q"
        assert output["answer"] == "test answer"

    def test_status_json_flag_outputs_json(self, tmp_path, capsys):
        """status --json produces JSON output."""
        db_path = str(tmp_path / "test.db")
        from research_tool.store import ResearchStore
        store = ResearchStore(db_path=db_path)
        store.store_page("https://a.com", title="A")
        store.close()

        with patch("sys.argv", ["research_tool", "status", "--json", "--db", db_path]):
            main()
        import json
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["total_pages"] == 1
        assert "sources" in output


class TestWikiSubcommand:
    def test_wiki_basic_args(self):
        """wiki subcommand parses URL correctly."""
        with patch("sys.argv", ["research_tool", "wiki", "https://docs.example.com"]):
            with patch("research_tool.__main__.cmd_wiki") as mock_cmd:
                main()
                args = mock_cmd.call_args[0][0]
                assert args.url == "https://docs.example.com"
                assert args.concurrency == 10
                assert args.branch is None
                assert args.db == "research.db"
                assert args.auth_config is None
                assert args.github_token is None
                assert args.auth_header is None

    def test_wiki_all_flags(self):
        """wiki subcommand parses all optional flags."""
        with patch("sys.argv", [
            "research_tool", "wiki", "https://wiki.example.com",
            "--branch", "develop",
            "--concurrency", "5",
            "--db", "/tmp/wiki.db",
            "--github-token", "ghp_abc123",
            "--auth-header", "X-Token: abc",
            "--auth-header", "X-Other: def",
            "--auth-config", "/tmp/auth.yaml",
        ]):
            with patch("research_tool.__main__.cmd_wiki") as mock_cmd:
                main()
                args = mock_cmd.call_args[0][0]
                assert args.branch == "develop"
                assert args.concurrency == 5
                assert args.db == "/tmp/wiki.db"
                assert args.github_token == "ghp_abc123"
                assert args.auth_header == ["X-Token: abc", "X-Other: def"]
                assert args.auth_config == "/tmp/auth.yaml"

    def test_wiki_missing_url_exits(self):
        """wiki subcommand without URL exits with error."""
        with patch("sys.argv", ["research_tool", "wiki"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_wiki_in_help_output(self, capsys):
        """wiki subcommand appears in --help."""
        with patch("sys.argv", ["research_tool", "--help"]):
            with pytest.raises(SystemExit):
                main()
        captured = capsys.readouterr()
        assert "wiki" in captured.out

    def test_cmd_wiki_invokes_crawler(self, tmp_path):
        """cmd_wiki calls WikiCrawler.crawl with correct arguments."""
        import asyncio
        from research_tool.__main__ import cmd_wiki

        mock_stats = {"visited": 3, "failed": 0, "elapsed_s": 1.0}
        args = type("Args", (), {
            "url": "https://docs.example.com",
            "db": str(tmp_path / "test.db"),
            "branch": "main",
            "concurrency": 5,
            "auth_config": None,
            "github_token": None,
            "auth_header": None,
        })()

        async def mock_crawl(**kwargs):
            return mock_stats

        with patch("research_tool.wiki.WikiCrawler") as MockCrawler:
            instance = MockCrawler.return_value
            instance.crawl = mock_crawl
            cmd_wiki(args)


class TestNoSubcommand:
    def test_no_args_exits_with_error(self):
        """Running with no subcommand exits with error."""
        with patch("sys.argv", ["research_tool"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2
