"""Tests for the benchmark corpus and ground-truth annotations."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
CORPUS_DIR = BENCHMARKS_DIR / "corpus"
GROUND_TRUTH_PATH = BENCHMARKS_DIR / "ground_truth.json"
HYDE_CACHE_PATH = BENCHMARKS_DIR / "hyde_cache.json"


@pytest.fixture(scope="module")
def ground_truth():
    return json.loads(GROUND_TRUTH_PATH.read_text())


@pytest.fixture(scope="module")
def corpus_texts(ground_truth):
    texts = {}
    for doc in ground_truth["documents"]:
        path = BENCHMARKS_DIR / doc["path"]
        texts[path.name] = path.read_text()
    return texts


class TestCorpusFiles:
    def test_all_corpus_files_exist(self, ground_truth):
        for doc in ground_truth["documents"]:
            path = BENCHMARKS_DIR / doc["path"]
            assert path.exists(), f"Missing corpus file: {doc['path']}"
            assert path.stat().st_size > 0, f"Empty corpus file: {doc['path']}"

    def test_python_file_is_valid_syntax(self):
        py_path = CORPUS_DIR / "sample_python.py"
        source = py_path.read_text()
        ast.parse(source, filename=str(py_path))

    def test_go_file_has_package_declaration(self):
        go_path = CORPUS_DIR / "sample_go.go"
        source = go_path.read_text()
        assert source.startswith("// Package") or "package " in source.split("\n", 20)[0:20]

    def test_wiki_html_is_parseable(self):
        html_path = CORPUS_DIR / "wiki_page.html"
        source = html_path.read_text()
        assert "<h1>" in source
        assert "</body>" in source
        assert "<table>" in source

    def test_web_article_has_multiple_paragraphs(self, corpus_texts):
        text = corpus_texts["web_article.txt"]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        assert len(paragraphs) >= 5, "web_article.txt should have 5+ paragraphs"

    def test_mixed_article_has_code_blocks(self, corpus_texts):
        text = corpus_texts["web_article_mixed.txt"]
        code_blocks = text.count("```")
        assert code_blocks >= 6, "web_article_mixed.txt should have 3+ fenced code blocks"

    def test_multi_section_has_many_paragraphs(self, corpus_texts):
        text = corpus_texts["multi_section_doc.txt"]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        assert len(paragraphs) >= 15, "multi_section_doc.txt should have 15+ paragraphs"

    def test_python_file_has_multiple_classes(self, corpus_texts):
        text = corpus_texts["sample_python.py"]
        classes = re.findall(r"^class \w+", text, re.MULTILINE)
        assert len(classes) >= 3, f"sample_python.py should have 3+ classes, found {classes}"

    def test_go_file_has_multiple_structs(self, corpus_texts):
        text = corpus_texts["sample_go.go"]
        structs = re.findall(r"type \w+ struct", text)
        assert len(structs) >= 3, f"sample_go.go should have 3+ structs, found {structs}"


class TestGroundTruth:
    def test_loads_valid_json(self, ground_truth):
        assert "documents" in ground_truth
        assert "queries" in ground_truth

    def test_has_sufficient_queries(self, ground_truth):
        assert len(ground_truth["queries"]) >= 100

    def test_every_query_has_required_fields(self, ground_truth):
        for i, q in enumerate(ground_truth["queries"]):
            assert "query" in q, f"Query {i} missing 'query'"
            assert "expected_passages" in q, f"Query {i} missing 'expected_passages'"
            assert "source_doc" in q, f"Query {i} missing 'source_doc'"
            assert "expected_content_types" in q, f"Query {i} missing 'expected_content_types'"
            assert "difficulty" in q, f"Query {i} missing 'difficulty'"

    def test_difficulty_levels_are_valid(self, ground_truth):
        valid = {"easy", "medium", "hard"}
        for q in ground_truth["queries"]:
            assert q["difficulty"] in valid, f"Invalid difficulty: {q['difficulty']}"

    def test_content_types_are_valid(self, ground_truth):
        valid = {"text", "code"}
        for q in ground_truth["queries"]:
            for ct in q["expected_content_types"]:
                assert ct in valid, f"Invalid content_type: {ct}"

    def test_every_source_doc_references_existing_file(self, ground_truth):
        doc_names = {d["path"].split("/")[-1] for d in ground_truth["documents"]}
        for q in ground_truth["queries"]:
            assert q["source_doc"] in doc_names, f"Unknown source_doc: {q['source_doc']}"

    def test_every_passage_appears_in_source(self, ground_truth, corpus_texts):
        for i, q in enumerate(ground_truth["queries"]):
            doc_text = corpus_texts[q["source_doc"]]
            for passage in q["expected_passages"]:
                assert passage in doc_text, (
                    f"Q{i} passage not found in {q['source_doc']}: '{passage[:60]}...'"
                )

    def test_all_documents_have_queries(self, ground_truth):
        doc_names = {d["path"].split("/")[-1] for d in ground_truth["documents"]}
        queried_docs = {q["source_doc"] for q in ground_truth["queries"]}
        missing = doc_names - queried_docs
        assert not missing, f"Documents with no queries: {missing}"

    def test_difficulty_distribution(self, ground_truth):
        counts = {}
        for q in ground_truth["queries"]:
            counts[q["difficulty"]] = counts.get(q["difficulty"], 0) + 1
        assert counts.get("easy", 0) >= 20, "Need at least 20 easy queries"
        assert counts.get("medium", 0) >= 20, "Need at least 20 medium queries"
        assert counts.get("hard", 0) >= 5, "Need at least 5 hard queries"


class TestHydeCache:
    def test_hyde_cache_exists_and_loads(self):
        assert HYDE_CACHE_PATH.exists()
        data = json.loads(HYDE_CACHE_PATH.read_text())
        assert "queries" in data
        assert "model" in data
        assert "embedding_dim" in data
