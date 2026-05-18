"""Repository indexing: clone, parse, chunk, and store source code from Git repos.

Detects GitHub/Bitbucket repo URLs, shallow-clones the target branch,
indexes source files via tree-sitter, and scrapes repo wiki pages and
issues via the GitHub API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from research_tool.code_chunker import chunk_code_file, detect_language
from research_tool.store import (
    DocumentChunk,
    ResearchStore,
    _make_embedding,
    chunk_web_content,
)

logger = logging.getLogger(__name__)

# ── Repo URL detection ───────────────────────────────────────────────────────

_REPO_URL_RE = re.compile(
    r"^https?://"
    r"(?P<platform>github\.com|bitbucket\.org)"
    r"/(?P<org>[^/]+)"
    r"/(?P<repo>[^/?#]+)"
    r"(?:/.*)?$",
    re.IGNORECASE,
)

_BRANCH_RE = re.compile(r"^[a-zA-Z0-9._/\-]+$")

# ── Indexable file extensions ────────────────────────────────────────────────

_INDEXABLE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".cs", ".java", ".go", ".rb",
    ".rs", ".c", ".cpp", ".h", ".hpp", ".swift", ".kt", ".scala",
    ".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml",
    ".html", ".css", ".scss", ".sql", ".sh", ".bash", ".dockerfile",
}

_DOC_EXTENSIONS = {
    ".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml",
}

# ── Directories to skip during file discovery ───────────────────────────────

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "vendor",
    "build", "dist", ".tox",
}


# ── Workspace URL detection ─────────────────────────────────────────────────

_BB_WORKSPACE_RE = re.compile(
    r"^https?://bitbucket\.org/(?P<workspace>[^/]+)"
    r"(?:/workspace(?:/.*)?)?/*$",
    re.IGNORECASE,
)


def parse_workspace_url(url: str) -> tuple[str, str] | None:
    """Detect Bitbucket workspace URLs (not individual repos).

    Returns ``("bitbucket.org", workspace_slug)`` or ``None``.
    Matches URLs like:
        https://bitbucket.org/myteam/
        https://bitbucket.org/myteam/workspace/overview/
    Does NOT match repo URLs (those have a second path component
    that isn't "workspace").
    """
    parsed = parse_repo_url(url)
    if parsed:
        _, _, repo = parsed
        if repo.lower() != "workspace":
            return None
    m = _BB_WORKSPACE_RE.match(url)
    if not m:
        return None
    return ("bitbucket.org", m.group("workspace"))


def list_workspace_repos(
    platform: str,
    workspace: str,
    auth_config: dict | None = None,
) -> list[dict]:
    """List repositories in a Bitbucket workspace via the REST API.

    Returns a list of dicts with ``platform``, ``org``, ``repo_name``,
    and ``full_name`` keys.
    """
    if platform != "bitbucket.org":
        logger.warning("Workspace listing only supported for Bitbucket")
        return []

    headers: dict[str, str] = {"Accept": "application/json"}
    if auth_config:
        from research_tool.wiki import match_credentials, build_auth_headers
        cred = match_credentials(
            f"https://bitbucket.org/{workspace}", auth_config,
        )
        if cred:
            headers.update(build_auth_headers(cred))

    repos: list[dict] = []
    url: str | None = (
        f"https://api.bitbucket.org/2.0/repositories/{workspace}"
        f"?pagelen=100"
    )

    while url:
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.error("Bitbucket API error listing repos: %s", exc)
            break

        for repo in data.get("values", []):
            slug = repo.get("slug", "")
            repos.append({
                "platform": "bitbucket.org",
                "org": workspace,
                "repo_name": slug,
                "full_name": repo.get("full_name", f"{workspace}/{slug}"),
            })

        url = data.get("next")

    return repos


# ── Public helpers ───────────────────────────────────────────────────────────


def parse_repo_url(url: str) -> tuple[str, str, str] | None:
    """Detect GitHub/Bitbucket repo URLs.

    Returns:
        ``(platform, org, repo_name)`` tuple, or ``None`` if the URL is
        not a recognised repository URL.  The *repo_name* component has
        any ``.git`` suffix stripped.
    """
    m = _REPO_URL_RE.match(url)
    if not m:
        return None
    platform = m.group("platform").lower()
    org = m.group("org")
    repo_name = m.group("repo")
    # Strip trailing .git
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    return (platform, org, repo_name)


def _validate_branch(branch: str) -> bool:
    """Return True if *branch* is safe to pass to ``git clone --branch``."""
    return bool(_BRANCH_RE.match(branch))


# ── RepoIndexer ──────────────────────────────────────────────────────────────


class RepoIndexer:
    """Clone, index, and store source code from a Git repository."""

    def index_repo(
        self,
        platform: str,
        org: str,
        repo_name: str,
        store: ResearchStore,
        auth_config: dict | None = None,
        branch: str | None = None,
    ) -> dict:
        """Clone, index source files, and return stats.

        Args:
            platform: Hosting platform (``github.com`` or ``bitbucket.org``).
            org: Organisation or user name.
            repo_name: Repository name.
            store: :class:`ResearchStore` instance for persisting chunks.
            auth_config: Optional auth credentials dict (see ``wiki.py``).
            branch: Optional branch to clone.  When *None*, the repo's
                    default branch is used.

        Returns:
            Dict with ``files_indexed``, ``chunks_stored``,
            ``wiki_pages``, ``issues``, and ``skipped`` counts.
            When ``skipped`` is ``True`` the repo was unchanged.
        """
        stats = {
            "files_indexed": 0,
            "chunks_stored": 0,
            "wiki_pages": 0,
            "issues": 0,
            "skipped": False,
        }

        branch_suffix = f"@{branch}" if branch else ""
        meta_url = f"repo-meta://{platform}/{org}/{repo_name}{branch_suffix}"

        # ── 0. Check if the repo has changed since last index ───────
        current_commit = self._get_latest_commit(
            platform, org, repo_name, auth_config, branch,
        )
        if current_commit:
            stored_commit = store.get_content_hash(meta_url)
            if stored_commit == current_commit:
                logger.debug(
                    "Repo %s/%s unchanged at %s, skipping",
                    org, repo_name, current_commit[:12],
                )
                stats["skipped"] = True
                return stats

        tmpdir = tempfile.mkdtemp(prefix="repo_index_")

        try:
            # ── 1. Get source files ─────────────────────────────────────
            clone_url = self._build_clone_url(
                platform, org, repo_name, auth_config,
            )
            try:
                self._clone_repo(clone_url, tmpdir, branch)
            except RuntimeError:
                if platform == "bitbucket.org" and auth_config:
                    logger.info(
                        "git clone failed for %s/%s; falling back to "
                        "Bitbucket zip download",
                        org, repo_name,
                    )
                    self._download_via_api(
                        platform, org, repo_name, tmpdir, auth_config,
                        branch=branch,
                    )
                else:
                    raise

            # ── 2. Discover & index files ────────────────────────────────
            files = self._discover_files(tmpdir)
            for filepath in files:
                n = self._index_file(
                    filepath, tmpdir, platform, org, repo_name, store,
                    branch=branch,
                )
                stats["chunks_stored"] += n
                stats["files_indexed"] += 1

            # ── 3. GitHub wiki ───────────────────────────────────────────
            if platform == "github.com":
                stats["wiki_pages"] = self._scrape_wiki(
                    org, repo_name, store,
                )

            # ── 4. GitHub issues ─────────────────────────────────────────
            if platform == "github.com":
                stats["issues"] = self._scrape_issues(
                    org, repo_name, store, auth_config,
                )

            # ── 5. Record the commit hash ───────────────────────────────
            if current_commit:
                store.store_page(
                    url=meta_url,
                    title=f"{org}/{repo_name}",
                    content_hash=current_commit,
                )

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return stats

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _get_latest_commit(
        platform: str,
        org: str,
        repo_name: str,
        auth_config: dict | None,
        branch: str | None = None,
    ) -> str | None:
        """Fetch the latest commit hash from the remote API.

        Returns the full SHA hex string, or ``None`` if the lookup fails.
        """
        try:
            if platform == "bitbucket.org":
                from research_tool.wiki import build_auth_headers, match_credentials
                cred = match_credentials(
                    f"https://{platform}/{org}/{repo_name}",
                    auth_config,
                ) if auth_config else None
                headers = {
                    **(build_auth_headers(cred) if cred else {}),
                    "Accept": "application/json",
                }
                api = f"https://api.bitbucket.org/2.0/repositories/{org}/{repo_name}"
                if not branch:
                    req = urllib.request.Request(
                        f"{api}?fields=mainbranch.name", headers=headers,
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        branch = (
                            json.loads(resp.read().decode())
                            .get("mainbranch", {})
                            .get("name", "main")
                        )
                req2 = urllib.request.Request(
                    f"{api}/refs/branches/{branch}?fields=target.hash",
                    headers=headers,
                )
                with urllib.request.urlopen(req2, timeout=15) as resp2:
                    return (
                        json.loads(resp2.read().decode())
                        .get("target", {})
                        .get("hash")
                    )
            elif platform == "github.com":
                ref = branch or "HEAD"
                api = f"https://api.github.com/repos/{org}/{repo_name}/commits/{ref}"
                headers: dict[str, str] = {"Accept": "application/json"}
                if auth_config:
                    from research_tool.wiki import build_auth_headers, match_credentials
                    cred = match_credentials(
                        f"https://{platform}/{org}/{repo_name}",
                        auth_config,
                    )
                    if cred:
                        headers.update(build_auth_headers(cred))
                req = urllib.request.Request(api, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read().decode()).get("sha")
        except Exception as exc:
            logger.debug(
                "Could not fetch latest commit for %s/%s: %s",
                org, repo_name, exc,
            )
        return None

    @staticmethod
    def _build_clone_url(
        platform: str,
        org: str,
        repo_name: str,
        auth_config: dict | None,
    ) -> str:
        """Build clone URL, embedding credentials for private repos.

        Bitbucket Cloud git operations require ``x-token-auth:{token}``
        regardless of whether the credential type is "basic" or "token".
        The email:token format only works for REST API calls.
        """
        if auth_config:
            from research_tool.wiki import match_credentials
            from urllib.parse import quote as urlquote
            repo_url = f"https://{platform}/{org}/{repo_name}"
            cred = match_credentials(repo_url, auth_config)
            if cred:
                token = urlquote(cred.get("token", ""), safe="")
                if platform == "bitbucket.org":
                    return f"https://x-token-auth:{token}@{platform}/{org}/{repo_name}.git"
                if cred.get("type") == "basic":
                    user = urlquote(cred["username"], safe="")
                    return f"https://{user}:{token}@{platform}/{org}/{repo_name}.git"
                if cred.get("type") == "token":
                    return f"https://x-token-auth:{token}@{platform}/{org}/{repo_name}.git"
        return f"https://{platform}/{org}/{repo_name}.git"

    @staticmethod
    def _download_via_api(
        platform: str,
        org: str,
        repo_name: str,
        dest: str,
        auth_config: dict,
        branch: str | None = None,
    ) -> None:
        """Download repo source via Bitbucket's zip archive endpoint.

        Downloads the entire repo as a zip in one request, then extracts
        it to *dest*.  Much faster than fetching files individually.
        """
        import zipfile
        from io import BytesIO
        from research_tool.wiki import build_auth_headers, match_credentials

        cred = match_credentials(f"https://{platform}/{org}/{repo_name}", auth_config)
        if not cred:
            raise RuntimeError(f"No credentials for {platform}/{org}/{repo_name}")
        headers = build_auth_headers(cred)

        ref = branch or "HEAD"
        zip_url = f"https://bitbucket.org/{org}/{repo_name}/get/{ref}.zip"
        req = urllib.request.Request(zip_url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                archive_bytes = resp.read()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download zip for {org}/{repo_name}: {exc}"
            ) from exc

        with zipfile.ZipFile(BytesIO(archive_bytes)) as zf:
            # Bitbucket zips have a top-level directory like
            # "org-repo-commithash/".  Strip it so files land in dest.
            prefix = ""
            names = zf.namelist()
            if names:
                first = names[0]
                if "/" in first:
                    prefix = first.split("/", 1)[0] + "/"

            for member in zf.infolist():
                if member.is_dir():
                    continue
                inner_path = member.filename
                if prefix and inner_path.startswith(prefix):
                    inner_path = inner_path[len(prefix):]
                if not inner_path:
                    continue
                out_path = os.path.join(dest, inner_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with zf.open(member) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())

        logger.info(
            "Extracted %s/%s from zip archive (%d bytes)",
            org, repo_name, len(archive_bytes),
        )

    @staticmethod
    def _clone_repo(url: str, dest: str, branch: str | None = None) -> None:
        """Shallow-clone a repository into *dest*."""
        cmd = ["git", "clone", "--depth", "1", "--single-branch"]
        if branch is not None:
            if not _validate_branch(branch):
                raise ValueError(
                    f"Invalid branch name: {branch!r}"
                )
            cmd.extend(["--branch", branch])
        cmd.append("--")
        cmd.extend([url, dest])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone failed (rc={result.returncode}): {result.stderr}"
            )

    @staticmethod
    def _discover_files(root: str) -> list[str]:
        """Walk *root* and return paths of indexable files."""
        files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skipped directories in-place
            dirnames[:] = [
                d for d in dirnames if d not in _SKIP_DIRS
            ]
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in _INDEXABLE_EXTENSIONS:
                    files.append(os.path.join(dirpath, fname))
        return files

    @staticmethod
    def _index_file(
        filepath: str,
        clone_root: str,
        platform: str,
        org: str,
        repo_name: str,
        store: ResearchStore,
        branch: str | None = None,
    ) -> int:
        """Read, chunk, embed, and store a single file.  Returns chunk count."""
        rel_path = os.path.relpath(filepath, clone_root)
        branch_suffix = f"@{branch}" if branch else ""
        page_url = f"repo://{platform}/{org}/{repo_name}/{rel_path}{branch_suffix}"

        try:
            source_text = Path(filepath).read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.warning("Could not read %s", filepath)
            return 0

        if not source_text.strip():
            return 0

        ext = os.path.splitext(filepath)[1].lower()

        if ext in _DOC_EXTENSIONS:
            # Documentation / config files — use paragraph chunker
            doc_chunks = chunk_web_content(
                source_text, page_url, section_title=rel_path,
            )
            for chunk in doc_chunks:
                chunk.embedding = _make_embedding(chunk.text)
            if doc_chunks:
                store.store_page(url=page_url, title=rel_path, extracted_text=source_text)
                store.store_chunks(doc_chunks)
            return len(doc_chunks)
        else:
            # Source code files — use AST-aware chunker
            language = detect_language(filepath)
            code_chunks = chunk_code_file(filepath, source_text, language, branch=branch or "")
            if not code_chunks:
                return 0

            doc_chunk_objs: list[DocumentChunk] = []
            for i, cc in enumerate(code_chunks):
                chunk_id = hashlib.sha256(
                    f"{page_url}::{i}::{cc.text[:64]}".encode()
                ).hexdigest()[:16]
                embedding = _make_embedding(cc.text)
                doc_chunk_objs.append(DocumentChunk(
                    text=cc.text,
                    page_url=page_url,
                    chunk_id=chunk_id,
                    section_title=cc.metadata.get("function", "")
                    or cc.metadata.get("class", "")
                    or rel_path,
                    embedding=embedding,
                    content_type="code",
                ))

            store.store_page(url=page_url, title=rel_path, extracted_text=source_text)
            store.store_chunks(doc_chunk_objs)
            return len(doc_chunk_objs)

    @staticmethod
    def _scrape_wiki(org: str, repo_name: str, store: ResearchStore) -> int:
        """Clone and index the GitHub wiki (if it exists).  Returns page count."""
        wiki_url = f"https://github.com/{org}/{repo_name}.wiki.git"
        tmpdir = tempfile.mkdtemp(prefix="repo_wiki_")
        pages = 0

        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--", wiki_url, tmpdir],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.info(
                    "Wiki clone failed for %s/%s (likely no wiki): %s",
                    org, repo_name, result.stderr.strip(),
                )
                return 0

            for dirpath, _, filenames in os.walk(tmpdir):
                # Skip .git directory
                if ".git" in dirpath.split(os.sep):
                    continue
                for fname in filenames:
                    if not fname.lower().endswith((".md", ".mediawiki", ".rst", ".txt")):
                        continue
                    fpath = os.path.join(dirpath, fname)
                    rel = os.path.relpath(fpath, tmpdir)
                    page_url = f"repo://github.com/{org}/{repo_name}/wiki/{rel}"
                    try:
                        text = Path(fpath).read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    if not text.strip():
                        continue

                    chunks = chunk_web_content(text, page_url, section_title=rel)
                    for chunk in chunks:
                        chunk.embedding = _make_embedding(chunk.text)
                    if chunks:
                        store.store_page(
                            url=page_url, title=rel, extracted_text=text,
                        )
                        store.store_chunks(chunks)
                    pages += 1
        except subprocess.TimeoutExpired:
            logger.warning("Wiki clone timed out for %s/%s", org, repo_name)
        except Exception:
            logger.exception("Unexpected error scraping wiki for %s/%s", org, repo_name)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return pages

    @staticmethod
    def _scrape_issues(
        org: str,
        repo_name: str,
        store: ResearchStore,
        auth_config: dict | None = None,
    ) -> int:
        """Fetch GitHub issues via the API and index them.  Returns issue count."""
        api_url = f"https://api.github.com/repos/{org}/{repo_name}/issues?state=all&per_page=100"

        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "WebResearcher/1.0",
        }

        # Add auth if available
        if auth_config:
            from research_tool.wiki import match_credentials, build_auth_headers

            cred = match_credentials(f"https://github.com/{org}/{repo_name}", auth_config)
            if cred:
                headers.update(build_auth_headers(cred))

        count = 0
        try:
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            logger.warning(
                "Failed to fetch issues for %s/%s (rate-limited or private repo)",
                org, repo_name,
                exc_info=True,
            )
            return 0

        if not isinstance(data, list):
            return 0

        for issue in data:
            number = issue.get("number", 0)
            title = issue.get("title", "")
            body = issue.get("body") or ""
            text = f"# {title}\n\n{body}" if body else f"# {title}"
            page_url = f"repo://github.com/{org}/{repo_name}/issues/{number}"

            chunks = chunk_web_content(text, page_url, section_title=title)
            for chunk in chunks:
                chunk.embedding = _make_embedding(chunk.text)
            if chunks:
                store.store_page(url=page_url, title=title, extracted_text=text)
                store.store_chunks(chunks)
            count += 1

        return count
