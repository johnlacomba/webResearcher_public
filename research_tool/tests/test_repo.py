"""Tests for research_tool.repo — repository cloning, indexing, and ingestion."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch, call

import pytest

from research_tool.repo import (
    RepoIndexer,
    _validate_branch,
    parse_repo_url,
)
from research_tool.store import ResearchStore


# ── Test: parse_repo_url ──────────────────────────────────────────────────────


class TestParseRepoUrlGithub:
    """Various GitHub URL formats correctly parsed."""

    def test_simple_github_url(self):
        result = parse_repo_url("https://github.com/org/repo")
        assert result == ("github.com", "org", "repo")

    def test_github_with_trailing_slash(self):
        result = parse_repo_url("https://github.com/org/repo/")
        assert result == ("github.com", "org", "repo")

    def test_github_tree_branch_path(self):
        result = parse_repo_url("https://github.com/org/repo/tree/main/src/lib")
        assert result == ("github.com", "org", "repo")

    def test_github_blob_path(self):
        result = parse_repo_url("https://github.com/org/repo/blob/develop/README.md")
        assert result == ("github.com", "org", "repo")

    def test_github_dot_git_suffix(self):
        result = parse_repo_url("https://github.com/org/repo.git")
        assert result == ("github.com", "org", "repo")

    def test_github_http_scheme(self):
        result = parse_repo_url("http://github.com/org/repo")
        assert result == ("github.com", "org", "repo")

    def test_github_case_insensitive_domain(self):
        result = parse_repo_url("https://GitHub.COM/MyOrg/MyRepo")
        assert result == ("github.com", "MyOrg", "MyRepo")


class TestParseRepoUrlBitbucket:
    """Bitbucket URL parsed."""

    def test_simple_bitbucket_url(self):
        result = parse_repo_url("https://bitbucket.org/org/repo")
        assert result == ("bitbucket.org", "org", "repo")

    def test_bitbucket_with_trailing_slash(self):
        result = parse_repo_url("https://bitbucket.org/team/project/")
        assert result == ("bitbucket.org", "team", "project")

    def test_bitbucket_with_subpath(self):
        result = parse_repo_url("https://bitbucket.org/team/project/src/main/README.md")
        assert result == ("bitbucket.org", "team", "project")


class TestParseRepoUrlNonRepo:
    """Non-repo URLs return None."""

    def test_generic_url(self):
        assert parse_repo_url("https://example.com/some/page") is None

    def test_gitlab_not_matched(self):
        assert parse_repo_url("https://gitlab.com/org/repo") is None

    def test_empty_string(self):
        assert parse_repo_url("") is None

    def test_plain_domain(self):
        assert parse_repo_url("https://github.com/") is None

    def test_github_user_only(self):
        assert parse_repo_url("https://github.com/org") is None

    def test_docs_domain(self):
        assert parse_repo_url("https://docs.python.org/3/library/os.html") is None


# ── Test: branch validation ──────────────────────────────────────────────────


class TestBranchValidation:
    """Valid branch names pass, invalid ones rejected."""

    def test_simple_branch(self):
        assert _validate_branch("main") is True

    def test_branch_with_slash(self):
        assert _validate_branch("feature/my-feature") is True

    def test_branch_with_dots(self):
        assert _validate_branch("release-1.2.3") is True

    def test_branch_with_underscore(self):
        assert _validate_branch("my_branch") is True

    def test_branch_with_spaces(self):
        assert _validate_branch("my branch") is False

    def test_branch_with_semicolon(self):
        assert _validate_branch("main;rm -rf /") is False

    def test_branch_with_backtick(self):
        assert _validate_branch("main`whoami`") is False

    def test_branch_with_dollar(self):
        assert _validate_branch("$HOME") is False

    def test_branch_with_pipe(self):
        assert _validate_branch("main|cat /etc/passwd") is False

    def test_empty_branch(self):
        assert _validate_branch("") is False


# ── Test: clone command construction ─────────────────────────────────────────


class TestCloneCommand:
    """git clone commands are constructed correctly."""

    @patch("research_tool.repo.subprocess.run")
    def test_clone_command_with_branch(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        RepoIndexer._clone_repo(
            "https://github.com/org/repo.git", "/tmp/dest", branch="develop",
        )
        args = mock_run.call_args[0][0]
        assert "--branch" in args
        idx = args.index("--branch")
        assert args[idx + 1] == "develop"

    @patch("research_tool.repo.subprocess.run")
    def test_clone_command_default(self, mock_run):
        """git clone without --branch when not specified."""
        mock_run.return_value = MagicMock(returncode=0)
        RepoIndexer._clone_repo(
            "https://github.com/org/repo.git", "/tmp/dest",
        )
        args = mock_run.call_args[0][0]
        assert "--branch" not in args

    @patch("research_tool.repo.subprocess.run")
    def test_clone_uses_separator(self, mock_run):
        """git clone command includes -- before positional args."""
        mock_run.return_value = MagicMock(returncode=0)
        RepoIndexer._clone_repo(
            "https://github.com/org/repo.git", "/tmp/dest",
        )
        args = mock_run.call_args[0][0]
        separator_idx = args.index("--")
        # URL and dest must come after the separator
        assert args[separator_idx + 1] == "https://github.com/org/repo.git"
        assert args[separator_idx + 2] == "/tmp/dest"

    @patch("research_tool.repo.subprocess.run")
    def test_clone_uses_depth_1(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        RepoIndexer._clone_repo(
            "https://github.com/org/repo.git", "/tmp/dest",
        )
        args = mock_run.call_args[0][0]
        assert "--depth" in args
        idx = args.index("--depth")
        assert args[idx + 1] == "1"

    @patch("research_tool.repo.subprocess.run")
    def test_clone_uses_single_branch(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        RepoIndexer._clone_repo(
            "https://github.com/org/repo.git", "/tmp/dest",
        )
        args = mock_run.call_args[0][0]
        assert "--single-branch" in args

    @patch("research_tool.repo.subprocess.run")
    def test_clone_timeout_300(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        RepoIndexer._clone_repo(
            "https://github.com/org/repo.git", "/tmp/dest",
        )
        assert mock_run.call_args[1]["timeout"] == 300

    def test_clone_rejects_invalid_branch(self):
        with pytest.raises(ValueError, match="Invalid branch name"):
            RepoIndexer._clone_repo(
                "https://github.com/org/repo.git",
                "/tmp/dest",
                branch="main;rm -rf /",
            )


# ── Test: file discovery ─────────────────────────────────────────────────────


class TestFileDiscovery:
    """Mock file tree filters correctly."""

    def test_skips_git_dir(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("gitconfig")
        (tmp_path / "main.py").write_text("print('hello')")

        files = RepoIndexer._discover_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in files]
        assert "main.py" in basenames
        assert "config" not in basenames

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "package.js").write_text("module.exports = {}")
        (tmp_path / "app.js").write_text("console.log('hi')")

        files = RepoIndexer._discover_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in files]
        assert "app.js" in basenames
        assert "package.js" not in basenames

    def test_skips_pycache(self, tmp_path):
        pc = tmp_path / "__pycache__"
        pc.mkdir()
        (pc / "mod.cpython-312.pyc").write_text("")
        (tmp_path / "mod.py").write_text("pass")

        files = RepoIndexer._discover_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in files]
        assert "mod.py" in basenames
        # __pycache__ dir should be pruned — no files from it should appear
        relative_files = [os.path.relpath(f, tmp_path) for f in files]
        assert not any("__pycache__" in r for r in relative_files)

    def test_skips_venv(self, tmp_path):
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "activate.sh").write_text("#!/bin/bash")
        (tmp_path / "script.sh").write_text("#!/bin/bash")

        files = RepoIndexer._discover_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in files]
        assert "script.sh" in basenames
        assert "activate.sh" not in basenames

    def test_skips_vendor_build_dist_tox(self, tmp_path):
        for d in ("vendor", "build", "dist", ".tox"):
            skip_dir = tmp_path / d
            skip_dir.mkdir()
            (skip_dir / "file.py").write_text("pass")

        (tmp_path / "main.py").write_text("pass")

        files = RepoIndexer._discover_files(str(tmp_path))
        assert len(files) == 1
        assert files[0].endswith("main.py")

    def test_includes_indexable_extensions(self, tmp_path):
        for ext in (".py", ".js", ".ts", ".go", ".rs", ".md", ".yaml"):
            (tmp_path / f"file{ext}").write_text("content")
        (tmp_path / "file.png").write_text("binary")
        (tmp_path / "file.exe").write_text("binary")

        files = RepoIndexer._discover_files(str(tmp_path))
        extensions = {os.path.splitext(f)[1] for f in files}
        assert ".py" in extensions
        assert ".js" in extensions
        assert ".md" in extensions
        assert ".png" not in extensions
        assert ".exe" not in extensions

    def test_nested_directories(self, tmp_path):
        src = tmp_path / "src" / "lib"
        src.mkdir(parents=True)
        (src / "util.py").write_text("pass")
        (tmp_path / "README.md").write_text("# Readme")

        files = RepoIndexer._discover_files(str(tmp_path))
        assert len(files) == 2


# ── Test: wiki clone failure ─────────────────────────────────────────────────


class TestWikiCloneFailure:
    """404 on wiki clone does not crash."""

    @patch("research_tool.repo.subprocess.run")
    def test_wiki_clone_failure_continues(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=128, stderr="fatal: repository not found",
        )
        store = MagicMock(spec=ResearchStore)

        result = RepoIndexer._scrape_wiki("org", "repo", store)
        assert result == 0
        # store should NOT have been called since wiki clone failed
        store.store_page.assert_not_called()

    @patch("research_tool.repo.subprocess.run")
    def test_wiki_clone_timeout_handled(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=120)
        store = MagicMock(spec=ResearchStore)

        result = RepoIndexer._scrape_wiki("org", "repo", store)
        assert result == 0


# ── Test: cleanup on error ───────────────────────────────────────────────────


class TestCleanupOnError:
    """Temp directory cleaned up even when error occurs."""

    @patch("research_tool.repo.shutil.rmtree")
    @patch("research_tool.repo.subprocess.run")
    @patch("research_tool.repo.tempfile.mkdtemp", return_value="/tmp/repo_index_abc")
    def test_cleanup_on_clone_failure(self, mock_mkdtemp, mock_run, mock_rmtree):
        mock_run.return_value = MagicMock(
            returncode=128, stderr="fatal: not found",
        )
        indexer = RepoIndexer()

        with pytest.raises(RuntimeError):
            indexer.index_repo("github.com", "org", "repo", MagicMock(spec=ResearchStore))

        mock_rmtree.assert_called_once_with("/tmp/repo_index_abc", ignore_errors=True)

    @patch("research_tool.repo.shutil.rmtree")
    @patch("research_tool.repo.RepoIndexer._scrape_issues", return_value=0)
    @patch("research_tool.repo.RepoIndexer._scrape_wiki", return_value=0)
    @patch("research_tool.repo.RepoIndexer._discover_files", side_effect=OSError("disk error"))
    @patch("research_tool.repo.subprocess.run")
    @patch("research_tool.repo.tempfile.mkdtemp", return_value="/tmp/repo_index_xyz")
    def test_cleanup_on_discover_failure(
        self, mock_mkdtemp, mock_run, mock_discover, mock_wiki, mock_issues, mock_rmtree,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        indexer = RepoIndexer()

        with pytest.raises(OSError):
            indexer.index_repo("github.com", "org", "repo", MagicMock(spec=ResearchStore))

        mock_rmtree.assert_called_once_with("/tmp/repo_index_xyz", ignore_errors=True)

    @patch("research_tool.repo.shutil.rmtree")
    @patch("research_tool.repo.RepoIndexer._scrape_issues", return_value=0)
    @patch("research_tool.repo.RepoIndexer._scrape_wiki", return_value=0)
    @patch("research_tool.repo.RepoIndexer._discover_files", return_value=[])
    @patch("research_tool.repo.subprocess.run")
    @patch("research_tool.repo.tempfile.mkdtemp", return_value="/tmp/repo_index_ok")
    def test_cleanup_on_success(
        self, mock_mkdtemp, mock_run, mock_discover, mock_wiki, mock_issues, mock_rmtree,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        store = MagicMock(spec=ResearchStore)
        indexer = RepoIndexer()

        indexer.index_repo("github.com", "org", "repo", store)

        mock_rmtree.assert_called_once_with("/tmp/repo_index_ok", ignore_errors=True)


# ── Test: repo URL pattern ───────────────────────────────────────────────────


class TestRepoUrlPattern:
    """Stored pages use repo:// URL prefix."""

    @patch("research_tool.repo._make_embedding", return_value=[0.1] * 768)
    @patch("research_tool.repo.chunk_web_content")
    def test_doc_file_uses_repo_prefix(self, mock_chunk, mock_embed, tmp_path):
        (tmp_path / "README.md").write_text("# Hello World")

        mock_chunk_obj = MagicMock()
        mock_chunk_obj.text = "Hello World"
        mock_chunk.return_value = [mock_chunk_obj]

        store = MagicMock(spec=ResearchStore)

        RepoIndexer._index_file(
            str(tmp_path / "README.md"),
            str(tmp_path),
            "github.com", "org", "repo",
            store,
        )

        # Verify store_page was called with a repo:// URL
        store.store_page.assert_called_once()
        call_kwargs = store.store_page.call_args
        url_arg = call_kwargs[1]["url"] if "url" in (call_kwargs[1] or {}) else call_kwargs[0][0]
        assert url_arg.startswith("repo://")
        assert "github.com/org/repo" in url_arg

    @patch("research_tool.repo._make_embedding", return_value=[0.1] * 768)
    @patch("research_tool.repo.chunk_code_file")
    def test_code_file_uses_repo_prefix(self, mock_chunk, mock_embed, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')")

        from research_tool.code_chunker import CodeChunk
        mock_chunk.return_value = [
            CodeChunk(
                text="print('hello')",
                file_path=str(tmp_path / "main.py"),
                language="python",
                metadata={"function": "main"},
            ),
        ]

        store = MagicMock(spec=ResearchStore)

        RepoIndexer._index_file(
            str(tmp_path / "main.py"),
            str(tmp_path),
            "github.com", "org", "repo",
            store,
        )

        store.store_page.assert_called_once()
        call_args = store.store_page.call_args
        url_arg = call_args[1]["url"]
        assert url_arg == "repo://github.com/org/repo/main.py"

    @patch("research_tool.repo._make_embedding", return_value=[0.1] * 768)
    @patch("research_tool.repo.chunk_web_content")
    def test_nested_file_repo_url(self, mock_chunk, mock_embed, tmp_path):
        src = tmp_path / "src" / "lib"
        src.mkdir(parents=True)
        (src / "util.js").write_text("export default {}")

        # .js is not a doc extension, so it goes through code_chunker path
        # Patch code_chunker instead
        from research_tool.code_chunker import CodeChunk

        with patch("research_tool.repo.chunk_code_file") as mock_code:
            mock_code.return_value = [
                CodeChunk(
                    text="export default {}",
                    file_path=str(src / "util.js"),
                    language="javascript",
                    metadata={"function": ""},
                ),
            ]
            store = MagicMock(spec=ResearchStore)

            RepoIndexer._index_file(
                str(src / "util.js"),
                str(tmp_path),
                "github.com", "org", "repo",
                store,
            )

            url_arg = store.store_page.call_args[1]["url"]
            assert url_arg == "repo://github.com/org/repo/src/lib/util.js"


# ── Test: full index_repo integration (mocked) ──────────────────────────────


class TestIndexRepoIntegration:
    """End-to-end index_repo with all externals mocked."""

    @patch("research_tool.repo.RepoIndexer._scrape_issues", return_value=5)
    @patch("research_tool.repo.RepoIndexer._scrape_wiki", return_value=2)
    @patch("research_tool.repo.RepoIndexer._index_file", return_value=3)
    @patch("research_tool.repo.RepoIndexer._discover_files", return_value=["/tmp/a.py", "/tmp/b.py"])
    @patch("research_tool.repo.subprocess.run")
    @patch("research_tool.repo.shutil.rmtree")
    @patch("research_tool.repo.tempfile.mkdtemp", return_value="/tmp/repo_index_test")
    def test_returns_correct_stats(
        self, mock_mkdtemp, mock_rmtree, mock_run, mock_discover,
        mock_index, mock_wiki, mock_issues,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        store = MagicMock(spec=ResearchStore)
        indexer = RepoIndexer()

        stats = indexer.index_repo("github.com", "org", "repo", store)

        assert stats["files_indexed"] == 2
        assert stats["chunks_stored"] == 6  # 2 files * 3 chunks each
        assert stats["wiki_pages"] == 2
        assert stats["issues"] == 5

    @patch("research_tool.repo.RepoIndexer._scrape_issues", return_value=0)
    @patch("research_tool.repo.RepoIndexer._scrape_wiki", return_value=0)
    @patch("research_tool.repo.RepoIndexer._discover_files", return_value=[])
    @patch("research_tool.repo.subprocess.run")
    @patch("research_tool.repo.shutil.rmtree")
    @patch("research_tool.repo.tempfile.mkdtemp", return_value="/tmp/repo_index_bb")
    def test_bitbucket_skips_wiki_and_issues(
        self, mock_mkdtemp, mock_rmtree, mock_run, mock_discover,
        mock_wiki, mock_issues,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        store = MagicMock(spec=ResearchStore)
        indexer = RepoIndexer()

        stats = indexer.index_repo("bitbucket.org", "org", "repo", store)

        # Wiki/issues scraping only for github.com
        mock_wiki.assert_not_called()
        mock_issues.assert_not_called()
        assert stats["wiki_pages"] == 0
        assert stats["issues"] == 0
