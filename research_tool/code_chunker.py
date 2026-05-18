"""AST-aware code chunking using tree-sitter.

Splits source code at syntactic boundaries (function/method/class level)
and enriches each chunk with contextual metadata (file path, namespace,
signature, docstrings).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_TREE_SITTER_AVAILABLE = False
try:
    import tree_sitter as _ts

    _TREE_SITTER_AVAILABLE = True
except ImportError:
    pass

EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    # Python
    ".py": "python",
    ".pyi": "python",
    ".pyw": "python",
    # Go
    ".go": "go",
    # Java / JVM
    ".java": "java",
    ".kt": "java",
    ".kts": "java",
    ".groovy": "java",
    ".gradle": "java",
    # JavaScript / derivatives
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # TypeScript / derivatives
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    # HTML / templates
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".svelte": "html",
    ".vue": "html",
    # Bash / shell
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".fish": "bash",
    ".ksh": "bash",
    # C# / .NET
    ".cs": "c_sharp",
    ".csproj": "xml",
    ".fsproj": "xml",
    ".vbproj": "xml",
    ".sln": "xml",
    ".props": "xml",
    ".targets": "xml",
    ".nuspec": "xml",
    # C / C++
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".hh": "cpp",
    # Rust
    ".rs": "rust",
    # Ruby
    ".rb": "ruby",
    ".rake": "ruby",
    ".gemspec": "ruby",
    # HCL / Terraform
    ".tf": "hcl",
    ".tfvars": "hcl",
    ".hcl": "hcl",
    # Dockerfile
    ".dockerfile": "dockerfile",
    # YAML / config
    ".yaml": "yaml",
    ".yml": "yaml",
    # TOML
    ".toml": "toml",
    # JSON
    ".json": "json",
    ".jsonc": "json",
    # XML / project files
    ".xml": "xml",
    ".xsl": "xml",
    ".xsd": "xml",
    ".svg": "xml",
    ".plist": "xml",
    ".xaml": "xml",
    ".resx": "xml",
    # CSS / stylesheets
    ".css": "css",
    ".scss": "css",
    ".less": "css",
    # SQL
    ".sql": "sql",
    # Protobuf / gRPC
    ".proto": "protobuf",
    # Markdown / docs
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "restructuredtext",
}

_TOKEN_ESTIMATE_RATIO = 4

_SMALL_CLASS_TOKENS = 500
_MIN_CHUNK_CHARS = 250

_FUNCTION_NODE_TYPES: dict[str, list[str]] = {
    "python": ["function_definition"],
    "go": ["function_declaration", "method_declaration"],
    "java": ["method_declaration", "constructor_declaration"],
    "javascript": ["function_declaration", "method_definition", "arrow_function"],
    "typescript": ["function_declaration", "method_definition", "arrow_function"],
    "bash": ["function_definition"],
    "html": [],
    "c_sharp": ["method_declaration", "constructor_declaration", "property_declaration"],
    "c": ["function_definition"],
    "cpp": ["function_definition"],
    "rust": ["function_item"],
    "ruby": ["method"],
    "hcl": [],
    "dockerfile": [],
    "yaml": [],
    "toml": [],
    "json": [],
    "xml": [],
    "css": [],
    "sql": [],
}

_CLASS_NODE_TYPES: dict[str, list[str]] = {
    "python": ["class_definition"],
    "go": ["type_declaration"],
    "java": ["class_declaration", "interface_declaration", "enum_declaration"],
    "javascript": ["class_declaration"],
    "typescript": ["class_declaration", "enum_declaration"],
    "bash": [],
    "html": ["element"],
    "c_sharp": ["class_declaration", "struct_declaration", "interface_declaration", "record_declaration", "enum_declaration"],
    "c": [],
    "cpp": ["class_specifier"],
    "rust": ["impl_item", "enum_item"],
    "ruby": ["class", "module"],
    "hcl": ["block"],
    "dockerfile": [],
    "yaml": [],
    "toml": [],
    "json": [],
    "xml": [],
    "css": [],
    "sql": [],
}

_loaded_languages: dict[str, object] = {}


@dataclass
class CodeChunk:
    text: str
    file_path: str
    language: str
    content_type: str = "code"
    metadata: dict[str, str] = field(default_factory=dict)


_GRAMMAR_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "go": ("tree_sitter_go", "language"),
    "java": ("tree_sitter_java", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "bash": ("tree_sitter_bash", "language"),
    "html": ("tree_sitter_html", "language"),
    "c_sharp": ("tree_sitter_c_sharp", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "hcl": ("tree_sitter_hcl", "language"),
    "dockerfile": ("tree_sitter_dockerfile", "language"),
    "yaml": ("tree_sitter_yaml", "language"),
    "toml": ("tree_sitter_toml", "language"),
    "json": ("tree_sitter_json", "language"),
    "xml": ("tree_sitter_xml", "language_xml"),
    "css": ("tree_sitter_css", "language"),
    "sql": ("tree_sitter_sql", "language"),
}


def _get_language(lang_name: str):
    """Get tree-sitter Language object for a language name."""
    if not _TREE_SITTER_AVAILABLE:
        return None
    if lang_name in _loaded_languages:
        return _loaded_languages[lang_name]

    spec = _GRAMMAR_IMPORT_MAP.get(lang_name)
    if spec is None:
        return None

    module_name, func_name = spec
    try:
        import importlib
        mod = importlib.import_module(module_name)
        language = _ts.Language(getattr(mod, func_name)())
        _loaded_languages[lang_name] = language
        return language
    except (ImportError, Exception) as e:
        logger.warning("Failed to load tree-sitter grammar for %s: %s", lang_name, e)
        return None


def _estimate_tokens(text: str) -> int:
    return len(text) // _TOKEN_ESTIMATE_RATIO


def _get_node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_parent_context(node, source_bytes: bytes, language: str) -> list[str]:
    """Walk up the tree to find parent class/namespace names."""
    parents = []
    current = node.parent
    class_types = _CLASS_NODE_TYPES.get(language, [])
    namespace_types = ["namespace_declaration"]
    while current is not None:
        if current.type in class_types or current.type in namespace_types:
            name_node = current.child_by_field_name("name")
            if name_node:
                parents.append(_get_node_text(name_node, source_bytes))
        current = current.parent
    parents.reverse()
    return parents


def _extract_signature(node, source_bytes: bytes) -> str:
    """Extract the function/method signature (first line up to the body)."""
    text = _get_node_text(node, source_bytes)
    lines = text.split("\n")
    sig_lines = []
    for line in lines:
        sig_lines.append(line)
        stripped = line.strip()
        if stripped.endswith("{") or stripped.endswith(":") or stripped.endswith(")"):
            break
    return "\n".join(sig_lines)


def _extract_docstring(node, source_bytes: bytes, language: str) -> str:
    """Extract docstring/comment immediately preceding or inside the node."""
    if language == "python":
        body = node.child_by_field_name("body")
        if body and body.child_count > 0:
            first = body.children[0]
            if first.type == "expression_statement":
                expr = first.children[0] if first.child_count > 0 else None
                if expr and expr.type == "string":
                    return _get_node_text(expr, source_bytes).strip('"\' \n')
    # Walk backward through ALL consecutive comment siblings
    comments: list[str] = []
    prev = node.prev_sibling
    while prev and prev.type in ("comment", "block_comment", "doc_comment"):
        comments.append(_get_node_text(prev, source_bytes))
        prev = prev.prev_sibling
    if comments:
        comments.reverse()
        return "\n".join(comments)
    return ""


def _enrich_chunk(code_text: str, file_path: str, parents: list[str],
                  signature: str, docstring: str,
                  class_docstring: str = "",
                  module_docstring: str = "",
                  branch: str = "") -> str:
    """Prepend contextual metadata to a code chunk for embedding."""
    file_label = f"{file_path} (branch {branch})" if branch else file_path
    parts = [f"// File: {file_label}"]
    if module_docstring:
        for line in module_docstring.split("\n"):
            stripped = line.lstrip("/ ").rstrip()
            if stripped:
                parts.append(f"// {stripped}")
    if parents:
        parts.append(f"// {'.'.join(parents)}")
    if class_docstring:
        for line in class_docstring.split("\n"):
            stripped = line.lstrip("/ ").rstrip()
            if stripped:
                parts.append(f"// {stripped}")
    if signature and signature != code_text.split("\n")[0]:
        parts.append(f"// {signature.split(chr(10))[0].strip()}")
    if docstring:
        for line in docstring.split("\n"):
            stripped = line.lstrip("/ ").rstrip()
            if stripped:
                parts.append(f"// {stripped}")
    parts.append(code_text)
    return "\n".join(parts)


_FILENAME_LANGUAGE_MAP: dict[str, str] = {
    "Dockerfile": "dockerfile",
    "Makefile": "bash",
    "Jenkinsfile": "java",
    "Gemfile": "ruby",
    "Rakefile": "ruby",
    "Vagrantfile": "ruby",
    "Pipfile": "toml",
}


def detect_language(file_path: str) -> str | None:
    """Detect language from file extension or filename."""
    p = Path(file_path)
    ext = p.suffix.lower()
    if ext:
        lang = EXTENSION_LANGUAGE_MAP.get(ext)
        if lang:
            return lang
    name = p.name
    if name.startswith("Dockerfile"):
        return "dockerfile"
    return _FILENAME_LANGUAGE_MAP.get(name)


def _merge_small_chunks(chunks: list[CodeChunk], min_chars: int = _MIN_CHUNK_CHARS) -> list[CodeChunk]:
    """Merge chunks below min_chars with adjacent same-namespace siblings.

    For function-level chunks, only merges when namespace is non-empty.
    For type-level chunks (enums, records, structs), also merges adjacent
    top-level declarations sharing the same (possibly empty) namespace.
    """
    if not chunks:
        return []
    def _is_type_level(c: CodeChunk) -> bool:
        return "class" in c.metadata and "function" not in c.metadata

    result: list[CodeChunk] = []
    for chunk in chunks:
        ns = chunk.metadata.get("namespace", "")
        prev_ns = result[-1].metadata.get("namespace", "") if result else ""
        both_type_level = result and _is_type_level(chunk) and _is_type_level(result[-1])
        same_ns = ns == prev_ns and (ns or both_type_level)
        should_merge = same_ns and both_type_level and (
            len(chunk.text) < min_chars or len(result[-1].text) < min_chars
        )
        if should_merge:
            prev = result[-1]
            merged_text = prev.text + "\n\n" + chunk.text
            merged_meta = dict(prev.metadata)
            for key in ("class", "function"):
                prev_val = prev.metadata.get(key, "")
                cur_val = chunk.metadata.get(key, "")
                if cur_val and prev_val:
                    merged_meta[key] = f"{prev_val}, {cur_val}"
                elif cur_val:
                    merged_meta[key] = cur_val
            result[-1] = CodeChunk(
                text=merged_text,
                file_path=prev.file_path,
                language=prev.language,
                metadata=merged_meta,
            )
        else:
            result.append(chunk)
    return result


def chunk_code_file(file_path: str, source_text: str, language: str | None = None,
                    branch: str = "") -> list[CodeChunk]:
    """Split a source code file into semantically complete chunks at AST boundaries.

    Args:
        file_path: Path to the source file (for metadata enrichment)
        source_text: Source code content
        language: Language name (auto-detected from extension if None)
        branch: Git branch name to embed in chunk metadata headers

    Returns:
        List of CodeChunk objects with enriched text and metadata
    """
    if not source_text.strip():
        return []

    if language is None:
        language = detect_language(file_path)

    if language is None:
        return _fallback_chunk(file_path, source_text, branch=branch)

    if not _TREE_SITTER_AVAILABLE:
        logger.warning("tree-sitter not available, falling back to paragraph chunking")
        return _fallback_chunk(file_path, source_text, language, branch=branch)

    ts_language = _get_language(language)
    if ts_language is None:
        logger.warning("No tree-sitter grammar for %s, falling back", language)
        return _fallback_chunk(file_path, source_text, language, branch=branch)

    try:
        parser = _ts.Parser(ts_language)
        source_bytes = source_text.encode("utf-8")
        tree = parser.parse(source_bytes)
        chunks = _extract_chunks_from_tree(tree, source_bytes, file_path, language, branch=branch)
        if not chunks:
            return _fallback_chunk(file_path, source_text, language, branch=branch)
        return _merge_small_chunks(chunks)
    except Exception as e:
        logger.warning("Tree-sitter parse failed for %s: %s", file_path, e)
        return _fallback_chunk(file_path, source_text, language, branch=branch)


def _hcl_block_label(node, source_bytes: bytes) -> str:
    """Extract HCL block type and labels, e.g. 'resource.aws_instance.web'."""
    parts = []
    for child in node.children:
        if child.type == "identifier":
            parts.append(_get_node_text(child, source_bytes))
        elif child.type == "string_lit":
            raw = _get_node_text(child, source_bytes).strip('"')
            parts.append(raw)
        elif child.type in ("block_start", "body", "block_end"):
            break
    return ".".join(parts)


def _extract_go_receiver_type(node, source_bytes: bytes) -> str:
    """Extract the receiver type name from a Go method_declaration node."""
    # method_declaration has a parameter_list as the first child after 'func'
    # containing the receiver, e.g. (f *Foo) or (f Foo)
    for child in node.children:
        if child.type == "parameter_list":
            # First parameter_list is the receiver
            for param in child.children:
                if param.type == "parameter_declaration":
                    # Look for a type_identifier (or pointer_type containing one)
                    for tnode in param.children:
                        if tnode.type == "type_identifier":
                            return _get_node_text(tnode, source_bytes)
                        if tnode.type == "pointer_type":
                            for inner in tnode.children:
                                if inner.type == "type_identifier":
                                    return _get_node_text(inner, source_bytes)
            break  # Only check the first parameter_list (receiver)
    return ""


def _collect_go_type_docstrings(root_node, source_bytes: bytes) -> dict[str, str]:
    """Pre-scan Go AST for type_declaration nodes and collect their doc comments."""
    type_docs: dict[str, str] = {}
    for child in root_node.children:
        if child.type == "type_declaration":
            # Extract the type name from the type_spec child
            for spec in child.children:
                if spec.type == "type_spec":
                    name_node = spec.child_by_field_name("name")
                    if not name_node:
                        # Fallback: first type_identifier child
                        for inner in spec.children:
                            if inner.type == "type_identifier":
                                name_node = inner
                                break
                    if name_node:
                        type_name = _get_node_text(name_node, source_bytes)
                        doc = _extract_docstring(child, source_bytes, "go")
                        if doc:
                            type_docs[type_name] = doc
    return type_docs


def _collect_go_package_comment(root_node, source_bytes: bytes) -> str:
    """Collect the package-level doc comment block from a Go AST root."""
    comments: list[str] = []
    for child in root_node.children:
        if child.type == "comment":
            comments.append(_get_node_text(child, source_bytes))
        elif child.type == "package_clause":
            break
        else:
            break
    return "\n".join(comments) if comments else ""


def _first_sentence(text: str) -> str:
    """Return the first sentence of text (up to the first period followed by whitespace/end)."""
    for i, ch in enumerate(text):
        if ch == "." and (i + 1 >= len(text) or text[i + 1] in " \n\r\t"):
            return text[: i + 1]
    return text


def _extract_module_docstring(root_node, source_bytes: bytes, language: str) -> str:
    """Extract the first sentence of the module-level docstring or leading comment."""
    if language == "python":
        for child in root_node.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        raw = _get_node_text(sub, source_bytes)
                        for q in ('"""', "'''"):
                            if raw.startswith(q) and raw.endswith(q):
                                return _first_sentence(raw[3:-3].strip())
                        return _first_sentence(raw.strip("\"'").strip())
            elif child.type in ("comment",):
                continue
            else:
                break
        return ""
    if language == "go":
        full = _collect_go_package_comment(root_node, source_bytes)
        if full:
            cleaned = " ".join(
                line.lstrip("/ ").rstrip() for line in full.split("\n") if line.strip()
            )
            return _first_sentence(cleaned)
        return ""
    comments: list[str] = []
    for child in root_node.children:
        if child.type in ("comment", "block_comment", "doc_comment"):
            comments.append(_get_node_text(child, source_bytes))
        elif child.type in (
            "package_declaration", "using_directive", "import_declaration",
            "preproc_include", "preproc_ifdef", "preproc_ifndef",
        ):
            continue
        else:
            break
    if comments:
        cleaned = " ".join(
            line.lstrip("/ *").rstrip() for line in "\n".join(comments).split("\n") if line.strip()
        )
        return _first_sentence(cleaned)
    return ""


def _extract_full_module_preamble(root_node, source_bytes: bytes, language: str) -> str:
    """Extract the full module-level docstring/comment text (not truncated)."""
    if language == "python":
        for child in root_node.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        raw = _get_node_text(sub, source_bytes)
                        for q in ('"""', "'''"):
                            if raw.startswith(q) and raw.endswith(q):
                                return raw[3:-3].strip()
                        return raw.strip("\"'").strip()
            elif child.type in ("comment",):
                continue
            else:
                break
        return ""
    if language == "go":
        full = _collect_go_package_comment(root_node, source_bytes)
        if full:
            return " ".join(
                line.lstrip("/ ").rstrip() for line in full.split("\n") if line.strip()
            )
        return ""
    comments: list[str] = []
    for child in root_node.children:
        if child.type in ("comment", "block_comment", "doc_comment"):
            comments.append(_get_node_text(child, source_bytes))
        elif child.type in (
            "package_declaration", "using_directive", "import_declaration",
            "preproc_include", "preproc_ifdef", "preproc_ifndef",
        ):
            continue
        else:
            break
    if comments:
        return " ".join(
            line.lstrip("/ *").rstrip() for line in "\n".join(comments).split("\n") if line.strip()
        )
    return ""


def _extract_chunks_from_tree(tree, source_bytes: bytes, file_path: str,
                              language: str, branch: str = "") -> list[CodeChunk]:
    """Walk the AST and extract chunks at function/class boundaries."""
    func_types = _FUNCTION_NODE_TYPES.get(language, [])
    class_types = _CLASS_NODE_TYPES.get(language, [])
    chunks: list[CodeChunk] = []

    # Map class node.id -> docstring for propagation to method chunks
    _class_docstrings: dict[int, str] = {}

    # Extract module-level docstring (first sentence) for all chunk headers
    _module_doc = _extract_module_docstring(tree.root_node, source_bytes, language)

    # Emit full module preamble as a standalone chunk when it has more content
    _full_preamble = _extract_full_module_preamble(tree.root_node, source_bytes, language)
    if _full_preamble and _full_preamble != _module_doc:
        _file_label = f"{file_path} (branch {branch})" if branch else file_path
        preamble_text = f"// File: {_file_label}\n// {_full_preamble}"
        chunks.append(CodeChunk(
            text=preamble_text,
            file_path=file_path,
            language=language,
            metadata={"preamble": "true", "namespace": ""},
        ))

    # Pre-collect Go type declaration docstrings
    _go_type_docs: dict[str, str] = {}
    if language == "go":
        _go_type_docs = _collect_go_type_docstrings(tree.root_node, source_bytes)

    def _get_class_docstring(node) -> str:
        """Look up the docstring of the enclosing class/type, if any."""
        # For Go: receiver type docstring (package comment is in module_docstring)
        if language == "go":
            if node.type == "method_declaration":
                recv_type = _extract_go_receiver_type(node, source_bytes)
                if recv_type and recv_type in _go_type_docs:
                    return _go_type_docs[recv_type]
            return ""
        # For other languages, walk up the tree to find the enclosing class
        current = node.parent
        while current is not None:
            if current.type in class_types:
                doc = _class_docstrings.get(current.id, "")
                if doc:
                    return doc
            current = current.parent
        return ""

    def walk(node):
        if node.type in class_types:
            class_text = _get_node_text(node, source_bytes)
            token_est = _estimate_tokens(class_text)
            # Extract and cache the class docstring for method-level propagation
            cls_doc = _extract_docstring(node, source_bytes, language)
            _class_docstrings[node.id] = cls_doc
            if token_est < _SMALL_CLASS_TOKENS:
                # Small class: chunk as a single unit
                parents = _find_parent_context(node, source_bytes, language)
                name_node = node.child_by_field_name("name")
                if name_node is None:
                    for child in node.children:
                        if child.type == "type_spec":
                            name_node = child.child_by_field_name("name")
                            break
                name = _get_node_text(name_node, source_bytes) if name_node else ""
                if not name and language == "hcl":
                    name = _hcl_block_label(node, source_bytes)
                enriched = _enrich_chunk(class_text, file_path, parents, "", cls_doc,
                                         module_docstring=_module_doc,
                                         branch=branch)
                chunks.append(CodeChunk(
                    text=enriched,
                    file_path=file_path,
                    language=language,
                    metadata={"class": name, "namespace": ".".join(parents)},
                ))
                return  # Don't recurse into small classes
            # Large class: recurse to chunk at method level
            for child in node.children:
                walk(child)
            return

        if node.type in func_types:
            code_text = _get_node_text(node, source_bytes)
            parents = _find_parent_context(node, source_bytes, language)
            signature = _extract_signature(node, source_bytes)
            docstring = _extract_docstring(node, source_bytes, language)
            class_doc = _get_class_docstring(node)
            name_node = node.child_by_field_name("name")
            name = _get_node_text(name_node, source_bytes) if name_node else ""
            enriched = _enrich_chunk(code_text, file_path, parents, signature, docstring,
                                     class_docstring=class_doc,
                                     module_docstring=_module_doc,
                                     branch=branch)
            chunks.append(CodeChunk(
                text=enriched,
                file_path=file_path,
                language=language,
                metadata={
                    "function": name,
                    "namespace": ".".join(parents),
                    "signature": signature.split("\n")[0].strip(),
                },
            ))
            return

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return chunks


def _fallback_chunk(file_path: str, source_text: str, language: str = "unknown",
                    branch: str = "") -> list[CodeChunk]:
    """Fallback: chunk the entire file as a single code block."""
    file_label = f"{file_path} (branch {branch})" if branch else file_path
    enriched = f"// File: {file_label}\n{source_text}"
    return [CodeChunk(
        text=enriched,
        file_path=file_path,
        language=language,
        metadata={"fallback": "true"},
    )]
