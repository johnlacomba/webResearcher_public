"""
SQLite storage, ONNX embeddings, BM25, hybrid retrieval, and chunking
for the autonomous web researcher.

Embedding layer ported from omlx_agent/omlx_agent.py (lines 450-566)
with SHA-256 verification, atomic downloads, and error logging.
HybridIndex and BM25 ported from omlx_agent/omlx_agent.py (lines 243-381).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import struct
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_MAX_SPECTRAL_NODES = 400_000

# ── RRF Configuration ─────────────────────────────────────────────────────────
RRF_K = int(os.environ.get("RRF_K", "60"))
RRF_WEIGHT_BM25 = float(os.environ.get("RRF_WEIGHT_BM25", "2.0"))
RRF_WEIGHT_TEXT_COSINE = float(os.environ.get("RRF_WEIGHT_TEXT_COSINE", "1.0"))
RRF_WEIGHT_IMAGE_COSINE = float(os.environ.get("RRF_WEIGHT_IMAGE_COSINE", "0.5"))
RRF_WEIGHT_GRAPH = float(os.environ.get("RRF_WEIGHT_GRAPH", "0.3"))
RRF_WEIGHT_MAXSIM = float(os.environ.get("RRF_WEIGHT_MAXSIM", "1.0"))
HYDE_ENABLED = os.environ.get("HYDE_ENABLED", "1") == "1"
RRF_WEIGHT_HYDE = float(os.environ.get("RRF_WEIGHT_HYDE", "0.8"))
RRF_WEIGHT_RECENCY = float(os.environ.get("RRF_WEIGHT_RECENCY", "0.4"))
RECENCY_HALF_LIFE_DAYS = float(os.environ.get("RECENCY_HALF_LIFE_DAYS", "30"))
CORROBORATION_STRENGTH_CAP = float(os.environ.get("CORROBORATION_STRENGTH_CAP", "5.0"))
CORROBORATION_DEDUP_INCREMENT = float(os.environ.get("CORROBORATION_DEDUP_INCREMENT", "0.3"))
CORROBORATION_GRAPH_INCREMENT = float(os.environ.get("CORROBORATION_GRAPH_INCREMENT", "1.0"))
PARENT_CHILD_ENABLED = os.environ.get("PARENT_CHILD_ENABLED", "1") == "1"
CHILD_CHUNK_MAX_TOKENS = int(os.environ.get("CHILD_CHUNK_MAX_TOKENS", "200"))
PARAGRAPH_MAX_TOKENS = int(os.environ.get("PARAGRAPH_MAX_TOKENS", "400"))
CONTEXTUAL_RETRIEVAL = os.environ.get("CONTEXTUAL_RETRIEVAL", "0") == "1"
ENTITY_RESOLUTION_ENABLED = os.environ.get("ENTITY_RESOLUTION_ENABLED", "1") == "1"
ENTITY_RESOLUTION_THRESHOLD = float(os.environ.get("ENTITY_RESOLUTION_THRESHOLD", "0.75"))
RRF_WEIGHT_ENTITY = float(os.environ.get("RRF_WEIGHT_ENTITY", "0.6"))

# ── Embedding model toggles (benchmark use) ────────────────────────────────
# True = production default (all models active). Benchmarks override to
# isolate embedding combinations.
JINA_CODE_EMBEDDING_ENABLED = True
TOKEN_EMBEDDING_ENABLED = True

# ── Chunk-mode routing (benchmark use) ───────────────────────────────────────
# "auto" = route by doc_type (production default); benchmarks override to
# "paragraph", "wiki", or "code" to force a specific chunker.
CHUNK_MODE = "auto"

QUERY_COMPLEXITY_SIMPLE_MAX_WORDS = int(os.environ.get("QUERY_COMPLEXITY_SIMPLE_MAX_WORDS", "8"))
QUERY_COMPLEXITY_COMPLEX_MIN_WORDS = int(os.environ.get("QUERY_COMPLEXITY_COMPLEX_MIN_WORDS", "15"))

MODEL_CACHE_DIR = os.path.expanduser("~/.web_researcher/models")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ── ONNX Embedding Layer (ported from omlx_agent/omlx_agent.py lines 454-566) ─

_ONNX_AVAILABLE = False
try:
    import onnxruntime  # type: ignore
    from tokenizers import Tokenizer as HFTokenizer  # type: ignore
    import numpy as np  # type: ignore

    _ONNX_AVAILABLE = True
except ImportError:
    pass

_OCR_AVAILABLE = False
try:
    from rapidocr_onnxruntime import RapidOCR as _RapidOCR  # type: ignore

    _OCR_AVAILABLE = True
except ImportError:
    pass

# Legacy MiniLM model (384-dim) — kept for migration (U3)
_LEGACY_ONNX_MODEL_COMMIT = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
_LEGACY_ONNX_MODEL_URL = (
    f"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/"
    f"resolve/{_LEGACY_ONNX_MODEL_COMMIT}/onnx/model_qint8_arm64.onnx"
)
_LEGACY_TOKENIZER_URL = (
    f"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/"
    f"resolve/{_LEGACY_ONNX_MODEL_COMMIT}/tokenizer.json"
)
_LEGACY_ONNX_MODEL_SHA256 = "4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474"
_LEGACY_TOKENIZER_SHA256 = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"

# nomic-embed-text-v1.5 MRL model (768-dim, Matryoshka)
_NOMIC_MODEL_COMMIT = "main"
_NOMIC_MODEL_URL = (
    f"https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/"
    f"resolve/{_NOMIC_MODEL_COMMIT}/onnx/model_quantized.onnx"
)
_NOMIC_TOKENIZER_URL = (
    f"https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/"
    f"resolve/{_NOMIC_MODEL_COMMIT}/tokenizer.json"
)
# SHAs populated after first verified download
_NOMIC_MODEL_SHA256 = ""
_NOMIC_TOKENIZER_SHA256 = ""
EMBEDDING_DIM = 768

_embedding_session = None
_embedding_tokenizer = None
_embedding_input_names: list[str] = []

# ── CLIP ViT-B/32 ONNX (Xenova/clip-vit-base-patch32) ────────────────────────
CLIP_EMBEDDING_DIM = 512
_CLIP_MODEL_COMMIT = "d15189d7028b43f1d3e65039190477f6af591c2a"
_CLIP_VISION_URL = (
    f"https://huggingface.co/Xenova/clip-vit-base-patch32/"
    f"resolve/{_CLIP_MODEL_COMMIT}/onnx/vision_model_int8.onnx"
)
_CLIP_TEXT_URL = (
    f"https://huggingface.co/Xenova/clip-vit-base-patch32/"
    f"resolve/{_CLIP_MODEL_COMMIT}/onnx/text_model_int8.onnx"
)
_CLIP_TOKENIZER_URL = (
    f"https://huggingface.co/Xenova/clip-vit-base-patch32/"
    f"resolve/{_CLIP_MODEL_COMMIT}/tokenizer.json"
)
_CLIP_VISION_SHA256 = "0ab0c1b3ace708e539633af1744d5a95247fe4e14d3e08ff197ef82a6cb9bd93"
_CLIP_TEXT_SHA256 = "18845f2ccc35223bb7fec403383a131154b11ac0918df25cf51986df5efd3a21"
_CLIP_TOKENIZER_SHA256 = "f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd"

_clip_vision_session = None
_clip_text_session = None
_clip_tokenizer = None

# ── Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2) ────────────────────────
_RERANKER_MODEL_COMMIT = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
_RERANKER_MODEL_URL = (
    f"https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/"
    f"resolve/{_RERANKER_MODEL_COMMIT}/onnx/model_qint8_arm64.onnx"
)
_RERANKER_TOKENIZER_URL = (
    f"https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/"
    f"resolve/{_RERANKER_MODEL_COMMIT}/tokenizer.json"
)
_RERANKER_SHA256 = "3573b6b9593cb2f75987a31815d409ca3dd8808629118fd20451bb1a5d90cec7"
_RERANKER_TOKENIZER_SHA256 = "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"

_reranker_session = None
_reranker_tokenizer = None

# ── ColBERT Token Embedding Model ────────────────────────────────────────────
COLBERT_DIM = 128
_COLBERT_MODEL_COMMIT = "main"
_COLBERT_MODEL_URL = (
    f"https://huggingface.co/colbert-ir/colbertv2.0/"
    f"resolve/{_COLBERT_MODEL_COMMIT}/model.onnx"
)
_COLBERT_TOKENIZER_URL = (
    f"https://huggingface.co/colbert-ir/colbertv2.0/"
    f"resolve/{_COLBERT_MODEL_COMMIT}/tokenizer.json"
)
_COLBERT_MODEL_SHA256 = ""
_COLBERT_TOKENIZER_SHA256 = ""

_colbert_session = None
_colbert_tokenizer = None
_colbert_input_names: list[str] = []

# ── Code Embedding Model (content-type-aware routing) ─────────────────────────
# jinaai/jina-embeddings-v2-base-code: 768-dim, ALiBi attention, 30+ languages
# Falls back to nomic-embed-text-v1.5 when the code model is not installed.
_CODE_MODEL_COMMIT = "main"
_CODE_MODEL_URL = os.environ.get(
    "CODE_EMBEDDING_MODEL_URL",
    f"https://huggingface.co/jinaai/jina-embeddings-v2-base-code/"
    f"resolve/{_CODE_MODEL_COMMIT}/onnx/model_quantized.onnx",
)
_CODE_TOKENIZER_URL = os.environ.get(
    "CODE_EMBEDDING_TOKENIZER_URL",
    f"https://huggingface.co/jinaai/jina-embeddings-v2-base-code/"
    f"resolve/{_CODE_MODEL_COMMIT}/tokenizer.json",
)
# SHAs populated after first verified download
_CODE_MODEL_SHA256 = os.environ.get("CODE_EMBEDDING_MODEL_SHA256", "")
_CODE_TOKENIZER_SHA256 = os.environ.get("CODE_EMBEDDING_TOKENIZER_SHA256", "")

_code_embedding_session = None
_code_embedding_tokenizer = None
_code_embedding_input_names: list[str] = []

RERANK_CANDIDATES = int(os.environ.get("RERANK_CANDIDATES", "50"))
RERANK_DIVERSITY_THRESHOLD = float(os.environ.get("RERANK_DIVERSITY_THRESHOLD", "0.2"))
RERANK_EXPANSION_MULTIPLIER = float(os.environ.get("RERANK_EXPANSION_MULTIPLIER", "2.0"))
RERANK_MAX_CANDIDATES = int(os.environ.get("RERANK_MAX_CANDIDATES", "200"))

# CLIP image preprocessing constants (ImageNet mean/std used by CLIP)
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
_CLIP_IMAGE_SIZE = 224


_QUESTION_WORDS = {"what", "how", "why", "when", "where", "which", "who", "explain", "describe", "compare"}
_CONJUNCTION_MARKERS = {"and", "or"}
_COMPLEX_MARKERS = {"versus", "vs", "between", "relationship", "implications", "analyze", "evaluate"}


def classify_query_complexity(query: str) -> str:
    """Classify a query as 'simple', 'moderate', or 'complex' using heuristics.

    Returns one of: 'simple', 'moderate', 'complex'.
    """
    if not query or not query.strip():
        return "simple"

    words = query.split()
    word_count = len(words)
    lower_words = {w.lower().rstrip("?.,!:;") for w in words}

    has_question_word = bool(lower_words & _QUESTION_WORDS)
    has_complex_marker = bool(lower_words & _COMPLEX_MARKERS)
    has_conjunction_only = bool(lower_words & _CONJUNCTION_MARKERS) and not has_complex_marker
    has_multiple_clauses = any(sep in query for sep in (",", ";", " - ", " — "))

    if word_count <= QUERY_COMPLEXITY_SIMPLE_MAX_WORDS and not has_complex_marker and not has_conjunction_only:
        return "simple"

    if (word_count >= QUERY_COMPLEXITY_COMPLEX_MIN_WORDS
            or has_complex_marker
            or (has_conjunction_only and word_count > QUERY_COMPLEXITY_SIMPLE_MAX_WORDS)
            or (has_question_word and has_multiple_clauses)):
        return "complex"

    return "moderate"


def _verify_file_sha256(path: str, expected_hash: str) -> bool:
    """Verify file integrity via SHA-256. Ported from omlx_agent."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest() == expected_hash


def _download_verified(url: str, dest_path: str, expected_hash: str, timeout: int = 120) -> None:
    """Atomic download with SHA-256 verification. Ported from omlx_agent."""
    tmp_path = dest_path + ".tmp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
        if expected_hash and not _verify_file_sha256(tmp_path, expected_hash):
            raise ValueError(f"SHA-256 mismatch for {os.path.basename(dest_path)}")
        os.replace(tmp_path, dest_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _ensure_embedding_model() -> bool:
    """Download and initialize the nomic-embed-text-v1.5 ONNX model if needed."""
    global _embedding_session, _embedding_tokenizer, _embedding_input_names
    if _embedding_session is not None and _embedding_tokenizer is not None:
        return True
    if not _ONNX_AVAILABLE:
        return False
    os.makedirs(MODEL_CACHE_DIR, mode=0o700, exist_ok=True)
    model_path = os.path.join(MODEL_CACHE_DIR, "nomic_embed_v1.5.onnx")
    tokenizer_path = os.path.join(MODEL_CACHE_DIR, "nomic_tokenizer.json")
    try:
        if not os.path.isfile(model_path) or (
            _NOMIC_MODEL_SHA256 and not _verify_file_sha256(model_path, _NOMIC_MODEL_SHA256)
        ):
            logger.info("Downloading nomic-embed-text-v1.5 ONNX model...")
            _download_verified(_NOMIC_MODEL_URL, model_path, _NOMIC_MODEL_SHA256, timeout=180)
        if not os.path.isfile(tokenizer_path) or (
            _NOMIC_TOKENIZER_SHA256 and not _verify_file_sha256(tokenizer_path, _NOMIC_TOKENIZER_SHA256)
        ):
            logger.info("Downloading nomic tokenizer...")
            _download_verified(_NOMIC_TOKENIZER_URL, tokenizer_path, _NOMIC_TOKENIZER_SHA256, timeout=30)
        _embedding_session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        _embedding_input_names = [inp.name for inp in _embedding_session.get_inputs()]
        _embedding_tokenizer = HFTokenizer.from_file(tokenizer_path)
        _embedding_tokenizer.enable_truncation(max_length=8192)
        _embedding_tokenizer.enable_padding(length=512)
        return True
    except Exception:
        logger.exception("Failed to initialize embedding model")
        _embedding_session = None
        _embedding_tokenizer = None
        _embedding_input_names = []
        for p in (model_path, tokenizer_path):
            try:
                expected = _NOMIC_MODEL_SHA256 if "model" in os.path.basename(p) else _NOMIC_TOKENIZER_SHA256
                if expected and os.path.isfile(p) and not _verify_file_sha256(p, expected):
                    os.remove(p)
            except OSError:
                pass
        return False


_TASK_PREFIXES = {
    "document": "search_document: ",
    "query": "search_query: ",
}


def _make_embedding(text: str, mode: str = "document") -> list[float] | None:
    """Generate a normalized 768-dim embedding using nomic-embed-text-v1.5.

    Args:
        text: Input text to embed.
        mode: "document" for ingest, "query" for retrieval. Controls task prefix.
    """
    if not _ensure_embedding_model():
        return None
    try:
        prefix = _TASK_PREFIXES.get(mode, _TASK_PREFIXES["document"])
        prefixed_text = prefix + text

        encoded = _embedding_tokenizer.encode(prefixed_text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in _embedding_input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = _embedding_session.run(None, inputs)
        hidden_states = outputs[0]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(hidden_states * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        sentence_embedding = (sum_embeddings / sum_mask)[0]
        norm = np.linalg.norm(sentence_embedding)
        if norm > 0:
            sentence_embedding = sentence_embedding / norm
        return sentence_embedding.tolist()
    except Exception:
        logger.exception("Failed to generate embedding")
        return None


def _make_embedding_truncated(text: str, dims: int, mode: str = "document") -> list[float] | None:
    """Generate a truncated MRL embedding at the specified dimension.

    MRL models produce embeddings valid at any prefix length. Truncate to `dims`
    dimensions and re-normalize to unit length.
    """
    full = _make_embedding(text, mode=mode)
    if full is None:
        return None
    truncated = full[:dims]
    norm = math.sqrt(sum(x * x for x in truncated))
    if norm > 0:
        truncated = [x / norm for x in truncated]
    return truncated


# ── Code Embedding Model Lifecycle ────────────────────────────────────────────


def _ensure_code_model() -> bool:
    """Download and initialize jina-embeddings-v2-base-code ONNX model."""
    global _code_embedding_session, _code_embedding_tokenizer, _code_embedding_input_names
    if _code_embedding_session is not None and _code_embedding_tokenizer is not None:
        return True
    if not _ONNX_AVAILABLE:
        return False
    os.makedirs(MODEL_CACHE_DIR, mode=0o700, exist_ok=True)
    model_path = os.path.join(MODEL_CACHE_DIR, "jina_code_model_quantized.onnx")
    tokenizer_path = os.path.join(MODEL_CACHE_DIR, "jina_code_tokenizer.json")
    try:
        if not os.path.exists(model_path) or (
            _CODE_MODEL_SHA256 and not _verify_file_sha256(model_path, _CODE_MODEL_SHA256)
        ):
            _download_verified(_CODE_MODEL_URL, model_path, _CODE_MODEL_SHA256, timeout=180)
        if not os.path.exists(tokenizer_path) or (
            _CODE_TOKENIZER_SHA256 and not _verify_file_sha256(tokenizer_path, _CODE_TOKENIZER_SHA256)
        ):
            _download_verified(_CODE_TOKENIZER_URL, tokenizer_path, _CODE_TOKENIZER_SHA256, timeout=30)
        _code_embedding_session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        _code_embedding_input_names = [inp.name for inp in _code_embedding_session.get_inputs()]
        _code_embedding_tokenizer = HFTokenizer.from_file(tokenizer_path)
        _code_embedding_tokenizer.enable_truncation(max_length=512)
        _code_embedding_tokenizer.enable_padding(length=512)
        return True
    except Exception:
        logger.warning("Code embedding model unavailable, will fall back to nomic")
        return False


def _make_code_embedding(text: str) -> list[float] | None:
    """Generate embedding using the code-specific model."""
    if not _ensure_code_model():
        return None
    try:
        encoded = _code_embedding_tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in _code_embedding_input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = _code_embedding_session.run(None, inputs)
        hidden_states = outputs[0]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(hidden_states * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        sentence_embedding = (sum_embeddings / sum_mask)[0]

        # Project to EMBEDDING_DIM if needed
        if len(sentence_embedding) != EMBEDDING_DIM:
            if len(sentence_embedding) > EMBEDDING_DIM:
                sentence_embedding = sentence_embedding[:EMBEDDING_DIM]
            else:
                padded = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                padded[:len(sentence_embedding)] = sentence_embedding
                sentence_embedding = padded

        norm = np.linalg.norm(sentence_embedding)
        if norm > 0:
            sentence_embedding = sentence_embedding / norm
        return sentence_embedding.tolist()
    except Exception:
        logger.exception("Failed to generate code embedding")
        return None


def make_content_aware_embedding(text: str, content_type: str = "text", mode: str = "document") -> list[float] | None:
    """Generate the primary (nomic) embedding for any content type.

    Always uses nomic-embed-text-v1.5 for the primary embedding so NL queries
    produce meaningful cosine similarity against all chunk types. Code-specific
    jina embeddings are stored separately in the code_embedding column.
    """
    return _make_embedding(text, mode=mode)


_CODE_DECL_RE = re.compile(
    r"\b(?:class|struct|enum|record|interface)\s+(\w+)"
    r"|\btype\s+(\w+)\s+(?:struct|interface)\b"
    r"|\b(?:def|func|fn|function)\s+(?:\([^)]*\)\s*)?(\w+)"
    r"|\bpublic\s+\w+(?:<[^>]+>)?\s+([A-Z]\w*)\s*[{(;]"
)
_GO_FIELD_RE = re.compile(r"^\s+([A-Z]\w+)\s+\S", re.MULTILINE)


def _extract_code_identifiers(text: str) -> list[str]:
    """Extract declaration-level identifiers from code text."""
    seen: set[str] = set()
    result: list[str] = []
    for m in _CODE_DECL_RE.finditer(text):
        name = m.group(1) or m.group(2) or m.group(3) or m.group(4)
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    for m in _GO_FIELD_RE.finditer(text):
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


_QUERY_STOP_WORDS = frozenset({
    "what", "how", "does", "which", "when", "where", "who", "that", "this",
    "these", "those", "the", "a", "an", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "for", "to", "of",
    "in", "on", "at", "by", "with", "from", "and", "or", "not", "if",
    "it", "its", "they", "their", "there", "than",
})

def _simple_stems(word: str) -> list[str]:
    """Return the word plus simple stem variants for identifier matching."""
    stems = [word]
    if word.endswith("ies") and len(word) > 4:
        stems.append(word[:-3] + "y")
    elif word.endswith("es") and len(word) > 3:
        stems.append(word[:-2])
    elif word.endswith("s") and len(word) > 3:
        stems.append(word[:-1])
    return stems


_QUERY_IDENT_RE = re.compile(
    r"[A-Z][a-z]+(?:[A-Z][a-z]+)+"  # CamelCase: KeyPrefix, TaskScheduler
    r"|[a-z]+(?:_[a-z0-9]+)+"       # snake_case: max_concurrent, limit_per_host
    r"|[A-Z]{2,}(?:_[A-Z0-9]+)*"    # UPPER_CASE: CRITICAL, PENDING
)


def _extract_query_key_terms(query: str) -> str:
    """Extract identifier-like and technical terms from a query.

    Returns a short string of key terms suitable for creating a focused
    embedding, or empty string if no distinctive terms found.
    """
    terms: list[str] = []
    seen: set[str] = set()

    for m in _QUERY_IDENT_RE.finditer(query):
        t = m.group()
        if t.lower() not in seen:
            seen.add(t.lower())
            terms.append(t)

    for word in query.split():
        clean = word.strip("?.,!;:\"'()")
        low = clean.lower()
        if low in _QUERY_STOP_WORDS or low in seen or len(clean) < 4:
            continue
        seen.add(low)
        terms.append(clean)

    return " ".join(terms) if terms else ""


def make_code_embedding_prefix(
    text: str,
    language: str | None = None,
    class_name: str | None = None,
    namespace: str | None = None,
) -> str:
    """Build a contextual prefix for code chunk embedding."""
    parts: list[str] = []
    if language:
        parts.append(language.replace("_", " "))
    if namespace and class_name:
        parts.append(f"{namespace}.{class_name}")
    elif class_name:
        parts.append(class_name)

    identifiers = _extract_code_identifiers(text)
    if identifiers:
        parts.append(", ".join(identifiers))

    if not parts:
        return text
    return " — ".join(parts) + "\n" + text


# ── CLIP Model Lifecycle ──────────────────────────────────────────────────────


def _ensure_clip_model() -> bool:
    """Download and initialize the CLIP ONNX models if needed."""
    global _clip_vision_session, _clip_text_session, _clip_tokenizer
    if (
        _clip_vision_session is not None
        and _clip_text_session is not None
        and _clip_tokenizer is not None
    ):
        return True
    if not _ONNX_AVAILABLE:
        return False
    os.makedirs(MODEL_CACHE_DIR, mode=0o700, exist_ok=True)
    vision_path = os.path.join(MODEL_CACHE_DIR, "clip_vision_model_int8.onnx")
    text_path = os.path.join(MODEL_CACHE_DIR, "clip_text_model_int8.onnx")
    tokenizer_path = os.path.join(MODEL_CACHE_DIR, "clip_tokenizer.json")
    try:
        # Download vision model
        if not os.path.isfile(vision_path) or (
            _CLIP_VISION_SHA256
            and not _verify_file_sha256(vision_path, _CLIP_VISION_SHA256)
        ):
            logger.info("Downloading CLIP vision model...")
            _download_verified(
                _CLIP_VISION_URL, vision_path, _CLIP_VISION_SHA256, timeout=120
            )
        # Download text model
        if not os.path.isfile(text_path) or (
            _CLIP_TEXT_SHA256
            and not _verify_file_sha256(text_path, _CLIP_TEXT_SHA256)
        ):
            logger.info("Downloading CLIP text model...")
            _download_verified(
                _CLIP_TEXT_URL, text_path, _CLIP_TEXT_SHA256, timeout=120
            )
        # Download tokenizer
        if not os.path.isfile(tokenizer_path) or (
            _CLIP_TOKENIZER_SHA256
            and not _verify_file_sha256(tokenizer_path, _CLIP_TOKENIZER_SHA256)
        ):
            logger.info("Downloading CLIP tokenizer...")
            _download_verified(
                _CLIP_TOKENIZER_URL, tokenizer_path, _CLIP_TOKENIZER_SHA256, timeout=30
            )
        _clip_vision_session = onnxruntime.InferenceSession(
            vision_path, providers=["CPUExecutionProvider"]
        )
        _clip_text_session = onnxruntime.InferenceSession(
            text_path, providers=["CPUExecutionProvider"]
        )
        _clip_tokenizer = HFTokenizer.from_file(tokenizer_path)
        _clip_tokenizer.enable_truncation(max_length=77)
        _clip_tokenizer.enable_padding(length=77)
        return True
    except Exception:
        logger.exception("Failed to initialize CLIP model")
        _clip_vision_session = None
        _clip_text_session = None
        _clip_tokenizer = None
        for p, sha in (
            (vision_path, _CLIP_VISION_SHA256),
            (text_path, _CLIP_TEXT_SHA256),
            (tokenizer_path, _CLIP_TOKENIZER_SHA256),
        ):
            try:
                if sha and os.path.isfile(p) and not _verify_file_sha256(p, sha):
                    os.remove(p)
            except OSError:
                pass
        return False


def _make_image_embedding(image_bytes: bytes) -> list[float] | None:
    """Generate a normalized CLIP image embedding from raw image bytes."""
    if not image_bytes or not _ensure_clip_model():
        return None
    try:
        from PIL import Image
        import io
        import numpy as _np

        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        if w * h > 25_000_000:
            scale = (25_000_000 / (w * h)) ** 0.5
            new_w, new_h = int(w * scale), int(h * scale)
            logger.info("Downscaling large image for CLIP (%dx%d -> %dx%d)", w, h, new_w, new_h)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        if img.mode in ("P", "PA", "RGBA", "LA"):
            img = img.convert("RGBA").convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Resize: shortest edge to 224, then center crop to 224x224
        scale = _CLIP_IMAGE_SIZE / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        # Center crop
        left = (new_w - _CLIP_IMAGE_SIZE) // 2
        top = (new_h - _CLIP_IMAGE_SIZE) // 2
        img = img.crop((left, top, left + _CLIP_IMAGE_SIZE, top + _CLIP_IMAGE_SIZE))

        # Convert to numpy, normalize
        pixel_values = _np.array(img, dtype=_np.float32) / 255.0
        mean = _np.array(_CLIP_MEAN, dtype=_np.float32)
        std = _np.array(_CLIP_STD, dtype=_np.float32)
        pixel_values = (pixel_values - mean) / std

        # HWC -> CHW -> NCHW
        pixel_values = pixel_values.transpose(2, 0, 1)
        pixel_values = _np.expand_dims(pixel_values, axis=0)

        # Run inference
        outputs = _clip_vision_session.run(None, {"pixel_values": pixel_values})
        embedding = outputs[0][0]  # First output, first batch item

        # L2 normalize
        norm = _np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()
    except Exception:
        logger.exception("Failed to generate CLIP image embedding")
        return None


def _make_clip_text_embedding(text: str) -> list[float] | None:
    """Generate a normalized CLIP text embedding for text-to-image queries."""
    if not text or not _ensure_clip_model():
        return None
    try:
        import numpy as _np

        encoded = _clip_tokenizer.encode(text)
        input_ids = _np.array([encoded.ids], dtype=_np.int64)

        outputs = _clip_text_session.run(
            None,
            {"input_ids": input_ids},
        )
        embedding = outputs[0][0]

        # L2 normalize
        norm = _np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()
    except Exception:
        logger.exception("Failed to generate CLIP text embedding")
        return None


# ── Chart/Figure Detection via CLIP Zero-Shot ────────────────────────────────

_CHART_LABELS = [
    "a scientific chart or graph",
    "a data visualization with axes",
    "a bar chart or line graph",
]
_NON_CHART_LABELS = [
    "a photograph of people or objects",
    "a logo or icon",
    "a screenshot of a user interface",
]
_CHART_DETECTION_MARGIN = 0.0

_chart_label_embeddings = None  # np.ndarray or None
_non_chart_label_embeddings = None  # np.ndarray or None


def _ensure_chart_labels() -> bool:
    """Embed chart/non-chart label texts via CLIP text encoder. Cached as numpy arrays."""
    global _chart_label_embeddings, _non_chart_label_embeddings
    if _chart_label_embeddings is not None and _non_chart_label_embeddings is not None:
        return True
    import numpy as _np

    chart_embs = [_make_clip_text_embedding(label) for label in _CHART_LABELS]
    non_chart_embs = [_make_clip_text_embedding(label) for label in _NON_CHART_LABELS]
    if any(e is None for e in chart_embs) or any(e is None for e in non_chart_embs):
        return False
    _chart_label_embeddings = _np.array(chart_embs, dtype=_np.float32)
    _non_chart_label_embeddings = _np.array(non_chart_embs, dtype=_np.float32)
    return True


def classify_image_is_chart(clip_embedding: list[float] | None) -> bool:
    """Classify an image as chart/figure vs photo/decorative using CLIP zero-shot."""
    if not clip_embedding or not _ensure_chart_labels():
        return False
    try:
        import numpy as _np

        img_vec = _np.array(clip_embedding, dtype=_np.float32)
        best_chart = float(_np.dot(_chart_label_embeddings, img_vec).max())
        best_non_chart = float(_np.dot(_non_chart_label_embeddings, img_vec).max())
        return best_chart > best_non_chart + _CHART_DETECTION_MARGIN
    except Exception:
        logger.debug("Chart classification failed", exc_info=True)
        return False


# ── ColBERT Model Lifecycle ───────────────────────────────────────────────────


def _ensure_colbert_model() -> bool:
    """Download and initialize the ColBERT ONNX model if needed."""
    global _colbert_session, _colbert_tokenizer, _colbert_input_names
    if _colbert_session is not None and _colbert_tokenizer is not None:
        return True
    if not _ONNX_AVAILABLE:
        return False
    os.makedirs(MODEL_CACHE_DIR, mode=0o700, exist_ok=True)
    model_path = os.path.join(MODEL_CACHE_DIR, "colbert_v2.onnx")
    tokenizer_path = os.path.join(MODEL_CACHE_DIR, "colbert_tokenizer.json")
    try:
        if not os.path.isfile(model_path) or (
            _COLBERT_MODEL_SHA256 and not _verify_file_sha256(model_path, _COLBERT_MODEL_SHA256)
        ):
            logger.info("Downloading ColBERT ONNX model...")
            _download_verified(_COLBERT_MODEL_URL, model_path, _COLBERT_MODEL_SHA256, timeout=180)
        if not os.path.isfile(tokenizer_path) or (
            _COLBERT_TOKENIZER_SHA256 and not _verify_file_sha256(tokenizer_path, _COLBERT_TOKENIZER_SHA256)
        ):
            logger.info("Downloading ColBERT tokenizer...")
            _download_verified(_COLBERT_TOKENIZER_URL, tokenizer_path, _COLBERT_TOKENIZER_SHA256, timeout=30)
        _colbert_session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        _colbert_input_names = [inp.name for inp in _colbert_session.get_inputs()]
        _colbert_tokenizer = HFTokenizer.from_file(tokenizer_path)
        _colbert_tokenizer.enable_truncation(max_length=512)
        _colbert_tokenizer.enable_padding(length=512)
        return True
    except Exception:
        logger.exception("Failed to initialize ColBERT model")
        _colbert_session = None
        _colbert_tokenizer = None
        _colbert_input_names = []
        for p, sha in (
            (model_path, _COLBERT_MODEL_SHA256),
            (tokenizer_path, _COLBERT_TOKENIZER_SHA256),
        ):
            try:
                if sha and os.path.isfile(p) and not _verify_file_sha256(p, sha):
                    os.remove(p)
            except OSError:
                pass
        return False


def _make_token_embeddings(text: str) -> "np.ndarray | None":
    """Generate per-token L2-normalized embeddings using ColBERT.

    Returns ndarray of shape (num_tokens, 128) or None if model unavailable.
    """
    if not text or not _ensure_colbert_model():
        return None
    try:
        encoded = _colbert_tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in _colbert_input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = _colbert_session.run(None, inputs)
        hidden_states = outputs[0][0]  # (seq_len, hidden_dim)

        # Only keep non-padding tokens
        mask = encoded.attention_mask
        num_tokens = sum(mask)
        token_embeddings = hidden_states[:num_tokens]

        # Project to COLBERT_DIM if needed (model may output 768-dim)
        if token_embeddings.shape[1] > COLBERT_DIM:
            token_embeddings = token_embeddings[:, :COLBERT_DIM]

        # L2-normalize each token vector
        norms = np.linalg.norm(token_embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        token_embeddings = token_embeddings / norms

        return token_embeddings.astype(np.float32)
    except Exception:
        logger.exception("Failed to generate token embeddings")
        return None


def _token_embeddings_to_blob(token_embeddings: "np.ndarray") -> bytes:
    """Serialize token embeddings array to blob: [num_tokens(2B), dim(2B), float32 data]."""
    num_tokens, dim = token_embeddings.shape
    header = struct.pack("<HH", num_tokens, dim)
    return header + token_embeddings.tobytes()


def _blob_to_token_embeddings(blob: bytes) -> "np.ndarray | None":
    """Deserialize token embeddings blob back to ndarray."""
    if len(blob) < 4:
        return None
    try:
        num_tokens, dim = struct.unpack_from("<HH", blob, 0)
        expected_size = 4 + num_tokens * dim * 4
        if len(blob) != expected_size:
            logger.warning("Token embedding blob size mismatch: expected %d, got %d", expected_size, len(blob))
            return None
        data = np.frombuffer(blob[4:], dtype=np.float32).reshape(num_tokens, dim)
        return data.copy()
    except Exception:
        logger.warning("Failed to deserialize token embeddings blob")
        return None


# ── MUVERA Fixed Dimensional Encoding ─────────────────────────���───────────────

MUVERA_BUCKETS = int(os.environ.get("MUVERA_BUCKETS", "20"))
MUVERA_FDE_DIM = MUVERA_BUCKETS * COLBERT_DIM
_MUVERA_SEED = 42
_muvera_hyperplanes: "np.ndarray | None" = None


def _get_muvera_hyperplanes(n_buckets: int = MUVERA_BUCKETS) -> "np.ndarray":
    """Get or create the SimHash hyperplane matrix (deterministic via fixed seed)."""
    global _muvera_hyperplanes
    if _muvera_hyperplanes is not None and _muvera_hyperplanes.shape[0] == n_buckets:
        return _muvera_hyperplanes
    rng = np.random.RandomState(_MUVERA_SEED)
    _muvera_hyperplanes = rng.randn(n_buckets, COLBERT_DIM).astype(np.float32)
    return _muvera_hyperplanes


def _compute_fde_vector(token_embeddings: "np.ndarray", n_buckets: int = MUVERA_BUCKETS) -> "np.ndarray":
    """Compute MUVERA Fixed Dimensional Encoding via SimHash bucketing.

    Partitions the token embedding space into n_buckets using random hyperplanes,
    averages the tokens in each bucket, and concatenates into a single vector.

    Args:
        token_embeddings: (N, 128) array of L2-normalized token vectors
        n_buckets: number of SimHash buckets (default 20)

    Returns:
        (n_buckets * 128,) FDE vector, L2-normalized
    """
    n_tokens, dim = token_embeddings.shape
    hyperplanes = _get_muvera_hyperplanes(n_buckets)

    # SimHash: assign each token to a bucket based on max dot product with hyperplanes
    # Shape: (n_tokens, n_buckets)
    projections = token_embeddings @ hyperplanes.T
    bucket_assignments = projections.argmax(axis=1)

    # Average tokens in each bucket
    fde = np.zeros((n_buckets, dim), dtype=np.float32)
    for b in range(n_buckets):
        mask = bucket_assignments == b
        if mask.any():
            fde[b] = token_embeddings[mask].mean(axis=0)

    fde_flat = fde.reshape(-1)
    norm = np.linalg.norm(fde_flat)
    if norm > 0:
        fde_flat /= norm
    return fde_flat


# ── OCR Text Extraction ──────────────────────────────────────────────────────

_MAX_OCR_TEXT_LENGTH = 2000
_ocr_engine = None
_ocr_init_failed = False


def _ensure_ocr_engine() -> "_RapidOCR | None":
    """Lazily initialize OCR engine on first use. Returns engine or None."""
    global _ocr_engine, _ocr_init_failed
    if _ocr_engine is not None:
        return _ocr_engine
    if _ocr_init_failed or not _OCR_AVAILABLE:
        return None
    try:
        _ocr_engine = _RapidOCR()
        return _ocr_engine
    except Exception:
        _ocr_init_failed = True
        logger.warning("Failed to initialize OCR engine", exc_info=True)
        return None


def extract_chart_text(image_bytes: bytes) -> str | None:
    """Extract visible text from a chart image via OCR. Returns None on failure."""
    engine = _ensure_ocr_engine()
    if engine is None:
        return None
    try:
        result, _ = engine(image_bytes)
        if not result:
            return None

        # Filter by confidence and sort top-to-bottom, left-to-right
        boxes = [(box, text, conf) for box, text, conf in result if conf > 0.5]
        if not boxes:
            return None

        # Sort by vertical position (top of bounding box), then horizontal
        boxes.sort(key=lambda b: (min(p[1] for p in b[0]), min(p[0] for p in b[0])))

        text = " ".join(text for _, text, _ in boxes)
        if len(text) > _MAX_OCR_TEXT_LENGTH:
            text = text[:_MAX_OCR_TEXT_LENGTH]
        return text if text.strip() else None
    except Exception:
        logger.debug("OCR extraction failed", exc_info=True)
        return None


# ── Cross-Encoder Reranker Lifecycle ─────────────────────────────────────────


def _ensure_reranker_model() -> bool:
    """Download and initialize the cross-encoder reranker model if needed."""
    global _reranker_session, _reranker_tokenizer
    if _reranker_session is not None and _reranker_tokenizer is not None:
        return True
    if not _ONNX_AVAILABLE:
        return False
    os.makedirs(MODEL_CACHE_DIR, mode=0o700, exist_ok=True)
    model_path = os.path.join(MODEL_CACHE_DIR, "reranker_qint8_arm64.onnx")
    tokenizer_path = os.path.join(MODEL_CACHE_DIR, "reranker_tokenizer.json")
    try:
        if not os.path.isfile(model_path) or not _verify_file_sha256(model_path, _RERANKER_SHA256):
            logger.info("Downloading cross-encoder reranker model...")
            _download_verified(_RERANKER_MODEL_URL, model_path, _RERANKER_SHA256, timeout=120)
        if not os.path.isfile(tokenizer_path) or not _verify_file_sha256(tokenizer_path, _RERANKER_TOKENIZER_SHA256):
            logger.info("Downloading reranker tokenizer...")
            _download_verified(_RERANKER_TOKENIZER_URL, tokenizer_path, _RERANKER_TOKENIZER_SHA256, timeout=30)
        _reranker_session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        _reranker_tokenizer = HFTokenizer.from_file(tokenizer_path)
        _reranker_tokenizer.enable_truncation(max_length=512)
        return True
    except Exception:
        logger.exception("Failed to initialize reranker model")
        _reranker_session = None
        _reranker_tokenizer = None
        for p, sha in (
            (model_path, _RERANKER_SHA256),
            (tokenizer_path, _RERANKER_TOKENIZER_SHA256),
        ):
            try:
                if sha and os.path.isfile(p) and not _verify_file_sha256(p, sha):
                    os.remove(p)
            except OSError:
                pass
        return False


def _rerank_pairs(query: str, passages: list[str]) -> list[float] | None:
    """Score (query, passage) pairs using the cross-encoder reranker.

    Returns a list of relevance scores (one per passage), or None on failure.
    """
    if not passages:
        return []
    if not _ensure_reranker_model():
        return None
    try:
        import numpy as _np

        encodings = [_reranker_tokenizer.encode(query, passage) for passage in passages]

        max_len = min(max(len(enc.ids) for enc in encodings), 512)
        batch_ids = _np.zeros((len(encodings), max_len), dtype=_np.int64)
        batch_mask = _np.zeros((len(encodings), max_len), dtype=_np.int64)
        batch_type_ids = _np.zeros((len(encodings), max_len), dtype=_np.int64)
        for i, enc in enumerate(encodings):
            length = min(len(enc.ids), max_len)
            batch_ids[i, :length] = enc.ids[:length]
            batch_mask[i, :length] = enc.attention_mask[:length]
            batch_type_ids[i, :length] = enc.type_ids[:length]

        outputs = _reranker_session.run(
            None,
            {
                "input_ids": batch_ids,
                "attention_mask": batch_mask,
                "token_type_ids": batch_type_ids,
            },
        )
        logits = outputs[0]
        if logits.ndim == 2:
            scores = logits[:, 0]
        else:
            scores = logits.flatten()
        return scores.tolist()
    except Exception:
        logger.exception("Failed to rerank pairs")
        return None


# ── Tokenization (ported from omlx_agent/omlx_agent.py line 450) ──────────────

_COMPOUND_RE = re.compile(
    r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]+|[0-9]+"
)


def _split_compound(token: str) -> list[str]:
    """Split camelCase, PascalCase, and snake_case into sub-tokens."""
    parts = re.split(r"[_-]", token)
    result: list[str] = []
    for part in parts:
        if not part:
            continue
        subs = _COMPOUND_RE.findall(part)
        if subs:
            result.extend(s.lower() for s in subs)
        else:
            result.append(part.lower())
    return result


def _rag_tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 scoring.

    Extracts word-like tokens, then decomposes camelCase/PascalCase/snake_case
    into sub-tokens so BM25 can match partial identifiers (e.g. query "status"
    matches code token "TaskStatus").
    """
    raw_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text)
    result: list[str] = []
    for t in raw_tokens:
        result.append(t.lower())
        subs = _split_compound(t)
        if len(subs) > 1:
            result.extend(subs)
    return result


def _rag_token_count(text: str) -> int:
    """Fast token count without compound decomposition (for chunk sizing)."""
    return len(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text))


# ── Data Classes ───────────────────────────────────────────────────────────────
# Adapted from omlx_agent/omlx_agent.py lines 42-70.
# Changed 'path' to 'page_url' for web content context.

@dataclass
class DocumentChunk:
    """A chunk of web page content with metadata."""
    text: str
    page_url: str
    chunk_id: str
    section_title: str
    embedding: list[float] | None = None
    code_embedding: list[float] | None = None
    summary_embedding: list[float] | None = None
    content_type: str = "text"
    parent_chunk_id: str | None = None
    is_child: bool = False
    context_summary: str | None = None
    cross_links: list[str] | None = None


@dataclass
class RetrievedEntry:
    """A retrieval result with scoring information."""
    page_url: str
    chunk_id: str
    section_title: str
    score: float
    bm25_score: float
    cosine_score: float
    text: str
    image_cosine_score: float = 0.0
    graph_score: float = 0.0
    maxsim_score: float = 0.0
    hyde_cosine_score: float = 0.0
    recency_score: float = 0.0
    entity_score: float = 0.0
    rerank_score: float | None = None


_CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)
_CODE_KEYWORDS_RE = re.compile(
    r"\b(?:def |function |class |import |from |const |let |var |return |if \(|for \(|while \(|#include|package |public |private |func )\b"
)
_CODE_SYNTAX_RE = re.compile(r"[{};=\[\]()]+")


def classify_chunk_content_type(text: str) -> str:
    """Heuristic: classify a text chunk as 'code' or 'text'.

    Looks for fenced code blocks, code keywords, and syntactic density.
    """
    lines = text.split("\n")
    total_lines = max(len(lines), 1)

    fence_count = len(_CODE_FENCE_RE.findall(text))
    if fence_count >= 2:
        fenced_lines = 0
        inside = False
        for line in lines:
            if line.strip().startswith("```"):
                inside = not inside
            elif inside:
                fenced_lines += 1
        if fenced_lines / total_lines > 0.3:
            return "code"

    keyword_hits = len(_CODE_KEYWORDS_RE.findall(text))
    syntax_chars = sum(len(m) for m in _CODE_SYNTAX_RE.findall(text))
    text_len = max(len(text), 1)
    syntax_density = syntax_chars / text_len

    if keyword_hits >= 3 and syntax_density > 0.02:
        return "code"

    if syntax_density > 0.04 and keyword_hits >= 1:
        return "code"

    indented_lines = sum(1 for l in lines if l.startswith("    ") or l.startswith("\t"))
    if indented_lines / total_lines > 0.5 and keyword_hits >= 2:
        return "code"

    return "text"


# ── Web Content Chunking ──────────────────────────────────────────────────────
# Adapted from omlx_agent _chunk_markdown (lines 139-170).
# Instead of splitting by ## headers, splits by paragraphs with ~500 token max.

def chunk_web_content(
    text: str,
    page_url: str,
    section_title: str = "",
    max_tokens: int | None = None,
) -> list[DocumentChunk]:
    """Split web content into paragraph-based chunks of up to ~max_tokens each.

    Adapted from omlx_agent's _chunk_markdown for web content.
    Splits on double-newlines (paragraph breaks), then merges small paragraphs
    up to the token budget. Strips empty chunks.
    """
    if max_tokens is None:
        max_tokens = PARAGRAPH_MAX_TOKENS
    if not text or not text.strip():
        return []

    # Split into paragraphs on double-newline boundaries
    paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    chunks: list[DocumentChunk] = []
    current_parts: list[str] = []
    current_token_count = 0

    for para in paragraphs:
        para_tokens = _rag_token_count(para)

        # If a single paragraph exceeds max_tokens, sub-split on sentence
        # boundaries so all content is preserved in properly sized chunks
        if para_tokens > max_tokens:
            if current_parts:
                _flush_chunk(chunks, current_parts, page_url, section_title, len(chunks))
                current_parts = []
                current_token_count = 0
            for sub in _split_oversized(para, max_tokens):
                _flush_chunk(chunks, [sub], page_url, section_title, len(chunks))
            continue

        # If adding this paragraph would exceed the budget, flush first
        if current_token_count + para_tokens > max_tokens and current_parts:
            _flush_chunk(chunks, current_parts, page_url, section_title, len(chunks))
            current_parts = []
            current_token_count = 0

        current_parts.append(para)
        current_token_count += para_tokens

    # Flush remaining
    if current_parts:
        _flush_chunk(chunks, current_parts, page_url, section_title, len(chunks))

    return chunks


def split_into_children(
    parent: DocumentChunk,
    max_tokens: int = 200,
) -> list[DocumentChunk]:
    """Split a parent chunk into smaller child chunks for retrieval precision.

    Each child inherits the parent's page_url, section_title, and content_type.
    Child chunk_ids are deterministic: "{parent_chunk_id}::child:{i}".
    If the parent is already <= max_tokens, returns a single child identical to the parent text.
    """
    text = parent.text
    if not text or not text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if not paragraphs:
        return []

    children: list[DocumentChunk] = []
    current_parts: list[str] = []
    current_token_count = 0

    def _flush_child() -> None:
        if not current_parts:
            return
        merged = "\n\n".join(current_parts)
        if not merged.strip():
            return
        child_id = f"{parent.chunk_id}::child:{len(children)}"
        children.append(DocumentChunk(
            text=merged,
            page_url=parent.page_url,
            chunk_id=child_id,
            section_title=parent.section_title,
            content_type=parent.content_type,
            parent_chunk_id=parent.chunk_id,
            is_child=True,
            context_summary=parent.context_summary,
        ))

    for para in paragraphs:
        para_tokens = _rag_token_count(para)

        if para_tokens > max_tokens:
            if current_parts:
                _flush_child()
                current_parts = []
                current_token_count = 0
            for sub in _split_oversized(para, max_tokens):
                current_parts = [sub]
                _flush_child()
                current_parts = []
                current_token_count = 0
            continue

        if current_token_count + para_tokens > max_tokens and current_parts:
            _flush_child()
            current_parts = []
            current_token_count = 0

        current_parts.append(para)
        current_token_count += para_tokens

    _flush_child()
    return children


_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<![A-Z])"       # not after single uppercase (U.S.)
    r"(?<!\d)"           # not after digit (1. 2. 3.)
    r"(?<!Dr)"           # not after Dr.
    r"(?<!Mr)"           # not after Mr.
    r"(?<!Ms)"           # not after Ms.
    r"(?<!Jr)"           # not after Jr.
    r"(?<!Sr)"           # not after Sr.
    r"(?<!St)"           # not after St.
    r"(?<!vs)"           # not after vs.
    r"(?<!No)"           # not after No.
    r"(?<!Eq)"           # not after Eq.
    r"(?<!Mrs)"          # not after Mrs.
    r"(?<!Prof)"         # not after Prof.
    r"(?<!Fig)"          # not after Fig.
    r"(?<!Vol)"          # not after Vol.
    r"(?<!Ave)"          # not after Ave.
    r"(?<!etc)"          # not after etc.
    r"(?<!e\.g\.)"       # not after e.g.
    r"(?<!i\.e\.)"       # not after i.e.
    r"(?<=[.!?])\s+"
)

_PUNCT_FALLBACK_RE = re.compile(r"(?<=[,;:])\s+")


def _split_oversized(text: str, max_tokens: int) -> list[str]:
    """Split text that exceeds max_tokens into smaller pieces.

    Tries sentence boundaries first, avoiding splits after common
    abbreviations. Falls back to punctuation boundaries (commas,
    semicolons, colons), then word boundaries.
    """
    sentences = _SENTENCE_BOUNDARY_RE.split(text)
    if len(sentences) > 1:
        return _merge_splits(sentences, max_tokens)

    # Fallback: try splitting on commas, semicolons, colons
    clauses = _PUNCT_FALLBACK_RE.split(text)
    if len(clauses) > 1:
        return _merge_splits(clauses, max_tokens)

    # Last resort: word-boundary split
    words = text.split()
    pieces = []
    current: list[str] = []
    current_count = 0
    for word in words:
        wt = _rag_token_count(word)
        if current_count + wt > max_tokens and current:
            pieces.append(" ".join(current))
            current = []
            current_count = 0
        current.append(word)
        current_count += wt
    if current:
        pieces.append(" ".join(current))
    return pieces


def _merge_splits(segments: list[str], max_tokens: int) -> list[str]:
    """Merge pre-split segments into chunks that fit within max_tokens."""
    pieces = []
    current: list[str] = []
    current_count = 0
    for seg in segments:
        st = _rag_token_count(seg)
        if st > max_tokens:
            if current:
                pieces.append(" ".join(current))
                current = []
                current_count = 0
            pieces.extend(_split_oversized(seg, max_tokens))
            continue
        if current_count + st > max_tokens and current:
            pieces.append(" ".join(current))
            current = []
            current_count = 0
        current.append(seg)
        current_count += st
    if current:
        pieces.append(" ".join(current))
    return pieces


def _flush_chunk(
    chunks: list[DocumentChunk],
    parts: list[str],
    page_url: str,
    section_title: str,
    index: int,
) -> None:
    """Create a DocumentChunk from accumulated paragraph parts."""
    merged = "\n\n".join(parts)
    if not merged.strip():
        return
    # Generate a deterministic chunk_id from URL and index
    url_hash = hashlib.sha256(page_url.encode()).hexdigest()[:12]
    chunk_id = f"{url_hash}::chunk-{index}"
    chunks.append(DocumentChunk(
        text=merged,
        page_url=page_url,
        chunk_id=chunk_id,
        section_title=section_title or "(untitled)",
    ))


# ── Embedding Serialization ───────────────────────────────────────────────────

_BLOB_MAGIC = 0x4D  # 'M' — distinguishes new-format from legacy headerless float32


def _embedding_to_blob(embedding: list[float], quantize: bool = False) -> bytes:
    """Serialize an embedding vector to a binary blob.

    New format: [magic(1B), version(1B), dim(2B LE), payload...]
      version 1: float32 payload
      version 2: min(4B float32) + max(4B float32) + int8 payload

    When quantize=False (default): uses version 1 (float32 with header).
    When quantize=True: uses version 2 (int8 scalar quantization).
    """
    dim = len(embedding)
    if quantize:
        vec_min = min(embedding)
        vec_max = max(embedding)
        val_range = vec_max - vec_min
        if val_range < 1e-10:
            int8_vals = bytes(dim)
        else:
            int8_vals = bytes(
                max(0, min(255, int(((v - vec_min) / val_range) * 255.0 + 0.5)))
                for v in embedding
            )
        header = struct.pack("<BBH", _BLOB_MAGIC, 2, dim)
        minmax = struct.pack("<ff", vec_min, vec_max)
        return header + minmax + int8_vals
    else:
        header = struct.pack("<BBH", _BLOB_MAGIC, 1, dim)
        payload = struct.pack(f"<{dim}f", *embedding)
        return header + payload


def _blob_to_embedding(blob: bytes) -> list[float]:
    """Deserialize a binary blob to an embedding vector.

    Auto-detects format: magic byte 0x4D at offset 0 → new format,
    otherwise legacy headerless float32.
    """
    if len(blob) >= 4 and blob[0] == _BLOB_MAGIC:
        _, version, dim = struct.unpack_from("<BBH", blob, 0)
        expected_len_v1 = 4 + dim * 4
        expected_len_v2 = 12 + dim
        if version == 1 and len(blob) == expected_len_v1:
            return list(struct.unpack_from(f"<{dim}f", blob, 4))
        elif version == 2 and len(blob) == expected_len_v2:
            vec_min, vec_max = struct.unpack_from("<ff", blob, 4)
            int8_data = blob[12 : 12 + dim]
            val_range = vec_max - vec_min
            if val_range < 1e-10:
                return [vec_min] * dim
            return [(b / 255.0) * val_range + vec_min for b in int8_data]
        # Header looks like magic but size doesn't match — treat as legacy
    # Legacy format: raw float32 without header
    n_floats = len(blob) // 4
    return list(struct.unpack(f"<{n_floats}f", blob))


# ── Image-Chunk Proximity Mapping ────────────────────────────────────────────


def compute_image_chunk_proximity(
    image_dicts: list[dict],
    chunks: list[DocumentChunk],
    html: str,
) -> dict[int, list[str]]:
    """Map each image to its nearest chunk(s) by DOM position in the HTML.

    For each image (identified by index in *image_dicts*), finds the 1-2
    closest chunks (immediately before and/or after the image in the DOM)
    by searching for each chunk's text prefix in the HTML source.

    Returns a dict mapping image index -> list of chunk_ids.
    """
    if not image_dicts or not chunks or not html:
        return {}

    # Compute approximate DOM offsets for each chunk by searching for
    # a prefix of its text in the HTML source.
    chunk_positions: list[tuple[int, str]] = []  # (offset, chunk_id)
    for chunk in chunks:
        prefix = chunk.text[:100]
        pos = html.find(prefix)
        if pos != -1:
            chunk_positions.append((pos, chunk.chunk_id))

    if not chunk_positions:
        return {}

    # Sort chunk positions by offset
    chunk_positions.sort(key=lambda x: x[0])

    proximity: dict[int, list[str]] = {}
    for idx, img_info in enumerate(image_dicts):
        img_offset = img_info.get("dom_offset", 0)
        nearest: list[str] = []

        # Find the chunk just before and just after the image
        before_id: str | None = None
        after_id: str | None = None
        for cp_offset, cp_id in chunk_positions:
            if cp_offset <= img_offset:
                before_id = cp_id
            elif after_id is None:
                after_id = cp_id

        if before_id is not None:
            nearest.append(before_id)
        if after_id is not None:
            nearest.append(after_id)

        proximity[idx] = nearest

    return proximity


# ── Spectral Graph Embeddings ────────────────────────────────────────────────


def compute_spectral_embeddings(
    adjacency: dict[str, set[str]], k: int = 8
) -> dict[str, list[float]]:
    """Compute spectral graph embeddings via normalized Laplacian eigendecomposition.

    Uses shift-invert mode (sigma=1e-6) so ARPACK targets the largest eigenvalues
    of (L - σI)^{-1} instead of the smallest eigenvalues of L directly.  This
    avoids the convergence problems caused by clustered near-zero eigenvalues in
    web link graphs.

    Returns a dict mapping URL -> k-dim L2-normalized embedding vector.
    Cosine similarity between embeddings reflects structural similarity.
    """
    import numpy as _np

    if not adjacency:
        return {}

    nodes = sorted(
        set(adjacency.keys()) | {t for targets in adjacency.values() for t in targets}
    )
    n = len(nodes)
    if n < 2:
        return {nodes[0]: [0.0] * min(k, 1)} if n == 1 else {}

    if n > _MAX_SPECTRAL_NODES:
        logger.warning(
            "Page graph too large for spectral embeddings (%d nodes, max %d), skipping",
            n, _MAX_SPECTRAL_NODES,
        )
        return {}

    node_idx = {url: i for i, url in enumerate(nodes)}

    from scipy import sparse as _sp
    from scipy.sparse.linalg import eigsh as _eigsh

    rows: list[int] = []
    cols: list[int] = []
    for source, targets in adjacency.items():
        i = node_idx[source]
        for target in targets:
            if target in node_idx:
                j = node_idx[target]
                rows.extend([i, j])
                cols.extend([j, i])

    data = _np.ones(len(rows), dtype=_np.float64)
    A = _sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A.eliminate_zeros()

    # Degree vector and normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    degrees = _np.asarray(A.sum(axis=1)).ravel()
    d_inv_sqrt = _np.zeros(n, dtype=_np.float64)
    nonzero = degrees > 0
    d_inv_sqrt[nonzero] = 1.0 / _np.sqrt(degrees[nonzero])
    D_inv_sqrt = _sp.diags(d_inv_sqrt)
    L = _sp.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    num_eig = min(k + 1, n - 1)
    if num_eig < 2:
        return {url: [0.0] for url in nodes}

    import time as _time
    from scipy.sparse.linalg import ArpackNoConvergence

    t0 = _time.monotonic()
    try:
        eigenvalues, eigenvectors = _eigsh(L, k=num_eig, sigma=1e-6, which="LM")
    except (ArpackNoConvergence, RuntimeError):
        logger.warning(
            "Spectral decomposition did not converge (%d nodes) — returning zero embeddings", n,
        )
        return {url: [0.0] * k for url in nodes}
    elapsed = _time.monotonic() - t0
    logger.info("Spectral embeddings: %d nodes, %d edges, %.2fs", n, A.nnz // 2, elapsed)

    embeddings_matrix = eigenvectors[:, 1:]  # drop trivial eigenvector

    # L2 normalize each row
    norms = _np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_matrix = embeddings_matrix / norms

    return {url: embeddings_matrix[node_idx[url]].tolist() for url in nodes}


def compute_eigenvector_centrality(
    adjacency: dict[str, set[str]], max_iter: int = 100
) -> dict[str, float]:
    """Compute eigenvector centrality via power iteration on the adjacency matrix.

    Returns dict mapping URL -> centrality score (max normalized to 1.0).
    Uses sparse representation to handle large link graphs efficiently.
    """
    import numpy as _np
    from scipy import sparse as _sp

    if not adjacency:
        return {}

    nodes = sorted(
        set(adjacency.keys()) | {t for targets in adjacency.values() for t in targets}
    )
    n = len(nodes)
    if n == 0:
        return {}

    if n > _MAX_SPECTRAL_NODES:
        logger.warning(
            "Page graph too large for centrality (%d nodes, max %d), skipping",
            n, _MAX_SPECTRAL_NODES,
        )
        return {}

    node_idx = {url: i for i, url in enumerate(nodes)}

    # Build sparse adjacency matrix
    rows: list[int] = []
    cols: list[int] = []
    for source, targets in adjacency.items():
        i = node_idx[source]
        for target in targets:
            if target in node_idx:
                j = node_idx[target]
                rows.extend([i, j])
                cols.extend([j, i])

    data = _np.ones(len(rows), dtype=_np.float64)
    A = _sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Power iteration
    x = _np.ones(n, dtype=_np.float64) / _np.sqrt(n)
    for _ in range(max_iter):
        x_new = A @ x
        norm = _np.linalg.norm(x_new)
        if norm > 0:
            x_new = x_new / norm
        if _np.allclose(x, x_new, atol=1e-8):
            x = x_new
            break
        x = x_new

    # Normalize max to 1.0
    max_val = float(_np.max(_np.abs(x)))
    if max_val > 0:
        x = x / max_val

    return {url: float(x[node_idx[url]]) for url in nodes}


# ── SQLite Research Store ─────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pages (
    url TEXT PRIMARY KEY,
    title TEXT,
    html TEXT,
    extracted_text TEXT,
    fetched_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    page_url TEXT REFERENCES pages(url),
    section_title TEXT,
    text TEXT NOT NULL,
    embedding BLOB,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    result_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS images (
    image_id TEXT PRIMARY KEY,
    page_url TEXT REFERENCES pages(url),
    src_url TEXT NOT NULL,
    alt_text TEXT DEFAULT '',
    width INTEGER,
    height INTEGER,
    embedding BLOB,
    nearest_chunk_ids TEXT,
    image_bytes BLOB,
    is_chart INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS page_links (
    source_url TEXT NOT NULL,
    target_url TEXT NOT NULL,
    anchor_text TEXT DEFAULT '',
    PRIMARY KEY (source_url, target_url)
);

CREATE TABLE IF NOT EXISTS chunk_token_embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id),
    token_embeddings BLOB NOT NULL,
    token_count INTEGER NOT NULL,
    fde_vector BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    definition TEXT DEFAULT '',
    aliases TEXT DEFAULT '[]',
    entity_embedding BLOB,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id),
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    PRIMARY KEY (chunk_id, entity_id)
);
"""


def _ensure_new_columns(conn: sqlite3.Connection) -> None:
    """Add new columns to existing tables, ignoring if they already exist.

    Uses try/except per column because ALTER TABLE ADD COLUMN has no
    IF NOT EXISTS syntax in SQLite.
    """
    for col_sql in (
        "ALTER TABLE pages ADD COLUMN graph_embedding BLOB",
        "ALTER TABLE pages ADD COLUMN authority_score REAL",
        "ALTER TABLE images ADD COLUMN image_bytes BLOB",
        "ALTER TABLE images ADD COLUMN is_chart INTEGER DEFAULT 0",
        "ALTER TABLE chunks ADD COLUMN embedding_version INTEGER DEFAULT 0",
        "ALTER TABLE chunks ADD COLUMN content_type TEXT DEFAULT 'text'",
        "ALTER TABLE chunks ADD COLUMN parent_chunk_id TEXT",
        "ALTER TABLE chunks ADD COLUMN is_child INTEGER DEFAULT 0",
        "ALTER TABLE chunks ADD COLUMN context_summary TEXT",
        "ALTER TABLE chunks ADD COLUMN last_corroborated_at TEXT",
        "ALTER TABLE chunks ADD COLUMN corroboration_strength REAL DEFAULT 0.0",
        "ALTER TABLE chunk_token_embeddings ADD COLUMN fde_vector BLOB",
        "ALTER TABLE chunks ADD COLUMN code_embedding BLOB",
        "ALTER TABLE pages ADD COLUMN content_hash TEXT",
        "ALTER TABLE pages ADD COLUMN crawl_status TEXT DEFAULT 'active'",
        "ALTER TABLE chunks ADD COLUMN cross_links TEXT",
        "ALTER TABLE chunks ADD COLUMN summary_embedding BLOB",
    ):
        try:
            conn.execute(col_sql)
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_images_is_chart ON images (is_chart) WHERE is_chart = 1"
    )
    try:
        conn.execute(
            "UPDATE chunks SET last_corroborated_at = created_at "
            "WHERE last_corroborated_at IS NULL"
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass


class ResearchStore:
    """SQLite-backed storage for web research data with embedding support."""

    def __init__(self, db_path: str = "research.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database with correct PRAGMA ordering and schema.

        PRAGMA ordering per institutional learning:
        busy_timeout FIRST, then journal_mode, then foreign_keys.
        """
        # Create the file with owner-only permissions if it doesn't exist
        needs_create = not os.path.exists(self.db_path)
        self.conn = sqlite3.connect(self.db_path)

        if needs_create:
            try:
                os.chmod(self.db_path, 0o600)
            except OSError:
                logger.warning("Could not set database file permissions to 0600")

        self.conn.row_factory = sqlite3.Row

        # PRAGMA ordering: busy_timeout FIRST (institutional learning)
        self.conn.execute("PRAGMA busy_timeout = 5000")
        result = self.conn.execute("PRAGMA journal_mode = WAL").fetchone()
        if result and result[0].lower() != "wal":
            logger.warning("WAL mode not enabled; journal_mode is %s", result[0])
        self.conn.execute("PRAGMA foreign_keys = ON")

        self.conn.executescript(_SCHEMA_SQL)
        self.conn.commit()

        _ensure_new_columns(self.conn)

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    # ── Page Operations ────────────────────────────────────────────────────

    def store_page(
        self,
        url: str,
        title: str = "",
        html: str = "",
        extracted_text: str = "",
        content_hash: str | None = None,
        crawl_status: str | None = None,
    ) -> None:
        """Store or update a fetched web page.

        Uses COALESCE on content_hash and crawl_status so that callers
        passing None do not silently nullify wiki-set values.
        """
        self.conn.execute(
            """INSERT INTO pages (url, title, html, extracted_text,
                   content_hash, crawl_status)
               VALUES (?, ?, ?, ?, ?, COALESCE(?, 'active'))
               ON CONFLICT(url) DO UPDATE SET
                   title = excluded.title,
                   html = excluded.html,
                   extracted_text = excluded.extracted_text,
                   content_hash = COALESCE(excluded.content_hash, pages.content_hash),
                   crawl_status = COALESCE(excluded.crawl_status, pages.crawl_status),
                   fetched_at = datetime('now')""",
            (url, title, html, extracted_text, content_hash, crawl_status),
        )
        self.conn.commit()

    def get_page(self, url: str) -> dict | None:
        """Retrieve a stored page by URL."""
        row = self.conn.execute("SELECT * FROM pages WHERE url = ?", (url,)).fetchone()
        return dict(row) if row else None

    def get_all_pages_with_html(self) -> list[dict]:
        """Return all stored pages including HTML and extracted text."""
        rows = self.conn.execute(
            "SELECT url, title, html, extracted_text, fetched_at "
            "FROM pages ORDER BY fetched_at"
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_chunks_for_page(self, page_url: str) -> int:
        """Delete all chunks (and their token embeddings) for a page URL.

        Returns the number of chunks deleted.
        """
        chunk_ids = [
            row[0] for row in self.conn.execute(
                "SELECT chunk_id FROM chunks WHERE page_url = ?", (page_url,)
            ).fetchall()
        ]
        if not chunk_ids:
            return 0
        placeholders = ",".join("?" * len(chunk_ids))
        self.conn.execute(
            f"DELETE FROM chunk_entities WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        self.conn.execute(
            f"DELETE FROM chunk_token_embeddings WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        self.conn.execute(
            f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        self.conn.commit()
        return len(chunk_ids)

    # ── Wiki Crawl Status Operations ──────────────────────────────────────

    def mark_domain_pages_stale(self, domain: str) -> None:
        """Set crawl_status = 'stale' for all pages matching a domain."""
        self.conn.execute(
            "UPDATE pages SET crawl_status = 'stale' "
            "WHERE url LIKE 'https://' || ? || '%' "
            "OR url LIKE 'http://' || ? || '%'",
            (domain, domain),
        )
        self.conn.commit()

    def mark_page_active(self, url: str) -> None:
        """Set crawl_status = 'active' for a single page."""
        self.conn.execute(
            "UPDATE pages SET crawl_status = 'active' WHERE url = ?",
            (url,),
        )
        self.conn.commit()

    def get_stale_pages(self, domain: str) -> list[dict]:
        """Return pages still marked stale for a domain (unreachable after re-crawl)."""
        rows = self.conn.execute(
            "SELECT * FROM pages WHERE crawl_status = 'stale' "
            "AND (url LIKE 'https://' || ? || '%' "
            "OR url LIKE 'http://' || ? || '%')",
            (domain, domain),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_content_hash(self, url: str) -> str | None:
        """Return the stored content_hash for a URL, or None if not found."""
        row = self.conn.execute(
            "SELECT content_hash FROM pages WHERE url = ?", (url,)
        ).fetchone()
        if row is None:
            return None
        return row["content_hash"]

    # ── Chunk Operations ───────────────────────────────────────────────────

    def store_chunk(self, chunk: DocumentChunk) -> None:
        """Store a document chunk with optional embedding. Upserts on chunk_id."""
        embedding_blob = None
        emb_version = 0
        if chunk.embedding is not None:
            embedding_blob = _embedding_to_blob(chunk.embedding, quantize=True)
            emb_version = 1
        code_embedding_blob = None
        if chunk.code_embedding is not None:
            code_embedding_blob = _embedding_to_blob(chunk.code_embedding, quantize=True)
        summary_embedding_blob = None
        if chunk.summary_embedding is not None:
            summary_embedding_blob = _embedding_to_blob(chunk.summary_embedding, quantize=True)
        cross_links_json = json.dumps(chunk.cross_links) if chunk.cross_links is not None else None

        self.conn.execute(
            """INSERT INTO chunks (chunk_id, page_url, section_title, text, embedding,
                   embedding_version, content_type, parent_chunk_id, is_child,
                   context_summary, code_embedding, cross_links, summary_embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(chunk_id) DO UPDATE SET
                   page_url = excluded.page_url,
                   section_title = excluded.section_title,
                   text = excluded.text,
                   embedding = excluded.embedding,
                   embedding_version = excluded.embedding_version,
                   content_type = excluded.content_type,
                   parent_chunk_id = excluded.parent_chunk_id,
                   is_child = excluded.is_child,
                   context_summary = excluded.context_summary,
                   code_embedding = excluded.code_embedding,
                   cross_links = excluded.cross_links,
                   summary_embedding = excluded.summary_embedding,
                   created_at = datetime('now')""",
            (chunk.chunk_id, chunk.page_url, chunk.section_title, chunk.text,
             embedding_blob, emb_version, chunk.content_type,
             chunk.parent_chunk_id, 1 if chunk.is_child else 0,
             chunk.context_summary, code_embedding_blob, cross_links_json,
             summary_embedding_blob),
        )
        self.conn.commit()

    def store_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Store multiple chunks in a single transaction."""
        for chunk in chunks:
            embedding_blob = None
            emb_version = 0
            if chunk.embedding is not None:
                embedding_blob = _embedding_to_blob(chunk.embedding, quantize=True)
                emb_version = 1
            code_embedding_blob = None
            if chunk.code_embedding is not None:
                code_embedding_blob = _embedding_to_blob(chunk.code_embedding, quantize=True)
            summary_embedding_blob = None
            if chunk.summary_embedding is not None:
                summary_embedding_blob = _embedding_to_blob(chunk.summary_embedding, quantize=True)
            cross_links_json = json.dumps(chunk.cross_links) if chunk.cross_links is not None else None
            self.conn.execute(
                """INSERT INTO chunks (chunk_id, page_url, section_title, text, embedding,
                       embedding_version, content_type, parent_chunk_id, is_child,
                       context_summary, code_embedding, cross_links, summary_embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chunk_id) DO UPDATE SET
                       page_url = excluded.page_url,
                       section_title = excluded.section_title,
                       text = excluded.text,
                       embedding = excluded.embedding,
                       embedding_version = excluded.embedding_version,
                       content_type = excluded.content_type,
                       parent_chunk_id = excluded.parent_chunk_id,
                       is_child = excluded.is_child,
                       context_summary = excluded.context_summary,
                       code_embedding = excluded.code_embedding,
                       cross_links = excluded.cross_links,
                       summary_embedding = excluded.summary_embedding,
                       created_at = datetime('now')""",
                (chunk.chunk_id, chunk.page_url, chunk.section_title, chunk.text,
                 embedding_blob, emb_version, chunk.content_type,
                 chunk.parent_chunk_id, 1 if chunk.is_child else 0,
                 chunk.context_summary, code_embedding_blob, cross_links_json,
                 summary_embedding_blob),
            )
        self.conn.commit()

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Retrieve a chunk by ID, deserializing the embedding blob."""
        row = self.conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if not row:
            return None
        embedding = None
        if row["embedding"]:
            embedding = _blob_to_embedding(row["embedding"])
        keys = row.keys()
        code_embedding = None
        if "code_embedding" in keys and row["code_embedding"]:
            code_embedding = _blob_to_embedding(row["code_embedding"])
        summary_embedding = None
        if "summary_embedding" in keys and row["summary_embedding"]:
            summary_embedding = _blob_to_embedding(row["summary_embedding"])
        content_type = row["content_type"] if "content_type" in keys else "text"
        parent_chunk_id = row["parent_chunk_id"] if "parent_chunk_id" in keys else None
        is_child = bool(row["is_child"]) if "is_child" in keys else False
        context_summary = row["context_summary"] if "context_summary" in keys else None
        cross_links = None
        if "cross_links" in keys and row["cross_links"]:
            cross_links = json.loads(row["cross_links"])
        return DocumentChunk(
            text=row["text"],
            page_url=row["page_url"] or "",
            chunk_id=row["chunk_id"],
            section_title=row["section_title"] or "",
            embedding=embedding,
            code_embedding=code_embedding,
            summary_embedding=summary_embedding,
            content_type=content_type,
            parent_chunk_id=parent_chunk_id,
            is_child=is_child,
            context_summary=context_summary,
            cross_links=cross_links,
        )

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Retrieve all stored chunks with their embeddings."""
        rows = self.conn.execute("SELECT * FROM chunks").fetchall()
        chunks = []
        for row in rows:
            embedding = None
            if row["embedding"]:
                embedding = _blob_to_embedding(row["embedding"])
            keys = row.keys()
            code_embedding = None
            if "code_embedding" in keys and row["code_embedding"]:
                code_embedding = _blob_to_embedding(row["code_embedding"])
            summary_embedding = None
            if "summary_embedding" in keys and row["summary_embedding"]:
                summary_embedding = _blob_to_embedding(row["summary_embedding"])
            content_type = row["content_type"] if "content_type" in keys else "text"
            parent_chunk_id = row["parent_chunk_id"] if "parent_chunk_id" in keys else None
            is_child = bool(row["is_child"]) if "is_child" in keys else False
            context_summary = row["context_summary"] if "context_summary" in keys else None
            cross_links = None
            if "cross_links" in keys and row["cross_links"]:
                cross_links = json.loads(row["cross_links"])
            chunks.append(DocumentChunk(
                text=row["text"],
                page_url=row["page_url"] or "",
                chunk_id=row["chunk_id"],
                section_title=row["section_title"] or "",
                embedding=embedding,
                code_embedding=code_embedding,
                summary_embedding=summary_embedding,
                content_type=content_type,
                parent_chunk_id=parent_chunk_id,
                is_child=is_child,
                context_summary=context_summary,
                cross_links=cross_links,
            ))
        return chunks

    def get_parent_chunk(self, child_chunk_id: str) -> DocumentChunk | None:
        """Retrieve the parent chunk for a given child chunk ID."""
        child = self.get_chunk(child_chunk_id)
        if child is None or child.parent_chunk_id is None:
            return None
        return self.get_chunk(child.parent_chunk_id)

    def corroborate_chunk(self, chunk_id: str, strength_increment: float) -> None:
        """Touch a chunk's last_corroborated_at and accumulate strength."""
        self.conn.execute(
            "UPDATE chunks SET last_corroborated_at = datetime('now'), "
            "corroboration_strength = COALESCE(corroboration_strength, 0.0) + ? "
            "WHERE chunk_id = ?",
            (strength_increment, chunk_id),
        )
        self.conn.commit()

    def corroborate_chunks_for_page(self, page_url: str, strength_increment: float) -> int:
        """Touch all chunks for a page. Returns number of chunks updated."""
        cursor = self.conn.execute(
            "UPDATE chunks SET last_corroborated_at = datetime('now'), "
            "corroboration_strength = COALESCE(corroboration_strength, 0.0) + ? "
            "WHERE page_url = ?",
            (strength_increment, page_url),
        )
        self.conn.commit()
        return cursor.rowcount

    def get_chunk_corroboration_data(self) -> list[tuple[str, str, float]]:
        """Return (chunk_id, last_corroborated_at, corroboration_strength) for all chunks."""
        rows = self.conn.execute(
            "SELECT chunk_id, "
            "COALESCE(last_corroborated_at, created_at) AS last_corroborated_at, "
            "COALESCE(corroboration_strength, 0.0) AS corroboration_strength "
            "FROM chunks"
        ).fetchall()
        return [(row[0], row[1], row[2]) for row in rows]

    def has_unmigrated_chunks(self) -> bool:
        """Check if any chunks still have legacy (version 0) embeddings."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL AND embedding_version < 1"
        ).fetchone()
        return row[0] > 0

    # ── Entity Operations ──────────────────────────────────────────────────

    def store_entity(
        self,
        entity_id: str,
        canonical_name: str,
        definition: str = "",
        aliases: list[str] | None = None,
        entity_embedding: list[float] | None = None,
    ) -> None:
        """Store or update an entity. Upserts on entity_id."""
        import json as _json

        aliases_json = _json.dumps(aliases or [])
        emb_blob = None
        if entity_embedding is not None:
            emb_blob = _embedding_to_blob(entity_embedding, quantize=True)
        self.conn.execute(
            """INSERT INTO entities (entity_id, canonical_name, definition, aliases,
                   entity_embedding)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(entity_id) DO UPDATE SET
                   canonical_name = excluded.canonical_name,
                   definition = excluded.definition,
                   aliases = excluded.aliases,
                   entity_embedding = excluded.entity_embedding""",
            (entity_id, canonical_name, definition, aliases_json, emb_blob),
        )
        self.conn.commit()

    def store_chunk_entity(self, chunk_id: str, entity_id: str) -> None:
        """Link a chunk to an entity. Idempotent (INSERT OR IGNORE)."""
        self.conn.execute(
            "INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id) VALUES (?, ?)",
            (chunk_id, entity_id),
        )
        self.conn.commit()

    def get_entity_by_id(self, entity_id: str) -> dict | None:
        """Retrieve an entity by ID."""
        import json as _json

        row = self.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        if not row:
            return None
        embedding = None
        if row["entity_embedding"]:
            embedding = _blob_to_embedding(row["entity_embedding"])
        return {
            "entity_id": row["entity_id"],
            "canonical_name": row["canonical_name"],
            "definition": row["definition"],
            "aliases": _json.loads(row["aliases"]) if row["aliases"] else [],
            "entity_embedding": embedding,
            "created_at": row["created_at"],
        }

    def get_all_entities(self) -> list[dict]:
        """Retrieve all entities with their embeddings."""
        import json as _json

        rows = self.conn.execute("SELECT * FROM entities").fetchall()
        entities = []
        for row in rows:
            embedding = None
            if row["entity_embedding"]:
                embedding = _blob_to_embedding(row["entity_embedding"])
            entities.append({
                "entity_id": row["entity_id"],
                "canonical_name": row["canonical_name"],
                "definition": row["definition"],
                "aliases": _json.loads(row["aliases"]) if row["aliases"] else [],
                "entity_embedding": embedding,
                "created_at": row["created_at"],
            })
        return entities

    def get_entities_for_chunk(self, chunk_id: str) -> list[dict]:
        """Get all entities linked to a chunk."""
        import json as _json

        rows = self.conn.execute(
            """SELECT e.* FROM entities e
               JOIN chunk_entities ce ON e.entity_id = ce.entity_id
               WHERE ce.chunk_id = ?""",
            (chunk_id,),
        ).fetchall()
        results = []
        for row in rows:
            embedding = None
            if row["entity_embedding"]:
                embedding = _blob_to_embedding(row["entity_embedding"])
            results.append({
                "entity_id": row["entity_id"],
                "canonical_name": row["canonical_name"],
                "definition": row["definition"],
                "aliases": _json.loads(row["aliases"]) if row["aliases"] else [],
                "entity_embedding": embedding,
                "created_at": row["created_at"],
            })
        return results

    def get_chunks_for_entity(self, entity_id: str) -> list[DocumentChunk]:
        """Get all chunks linked to an entity."""
        rows = self.conn.execute(
            """SELECT c.* FROM chunks c
               JOIN chunk_entities ce ON c.chunk_id = ce.chunk_id
               WHERE ce.entity_id = ?""",
            (entity_id,),
        ).fetchall()
        chunks = []
        for row in rows:
            embedding = None
            if row["embedding"]:
                embedding = _blob_to_embedding(row["embedding"])
            keys = row.keys()
            code_embedding = None
            if "code_embedding" in keys and row["code_embedding"]:
                code_embedding = _blob_to_embedding(row["code_embedding"])
            summary_embedding = None
            if "summary_embedding" in keys and row["summary_embedding"]:
                summary_embedding = _blob_to_embedding(row["summary_embedding"])
            content_type = row["content_type"] if "content_type" in keys else "text"
            parent_chunk_id = row["parent_chunk_id"] if "parent_chunk_id" in keys else None
            is_child = bool(row["is_child"]) if "is_child" in keys else False
            context_summary = row["context_summary"] if "context_summary" in keys else None
            cross_links = None
            if "cross_links" in keys and row["cross_links"]:
                cross_links = json.loads(row["cross_links"])
            chunks.append(DocumentChunk(
                text=row["text"],
                page_url=row["page_url"] or "",
                chunk_id=row["chunk_id"],
                section_title=row["section_title"] or "",
                embedding=embedding,
                code_embedding=code_embedding,
                summary_embedding=summary_embedding,
                content_type=content_type,
                parent_chunk_id=parent_chunk_id,
                is_child=is_child,
                context_summary=context_summary,
                cross_links=cross_links,
            ))
        return chunks

    def get_chunk_entity_ids(self, chunk_id: str) -> list[str]:
        """Get entity IDs linked to a chunk (lightweight, no entity data)."""
        rows = self.conn.execute(
            "SELECT entity_id FROM chunk_entities WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchall()
        return [row[0] for row in rows]

    def find_similar_entity(
        self,
        embedding: list[float],
        threshold: float | None = None,
        matrix: "np.ndarray | None" = None,
        entity_ids: list[str] | None = None,
    ) -> str | None:
        """Find the most similar existing entity above the threshold.

        Returns the matched entity_id, or None if no match found.
        Accepts a pre-built embedding matrix and parallel entity_ids list
        to avoid O(N) DB loads per call.
        """
        if threshold is None:
            threshold = ENTITY_RESOLUTION_THRESHOLD

        if matrix is None:
            entities = self.get_all_entities()
            entities_with_emb = [e for e in entities if e["entity_embedding"] is not None]
            if not entities_with_emb:
                return None
            matrix = np.array(
                [e["entity_embedding"] for e in entities_with_emb], dtype=np.float32
            )
            entity_ids = [e["entity_id"] for e in entities_with_emb]

        if len(matrix) == 0 or entity_ids is None:
            return None

        query_vec = np.array(embedding, dtype=np.float32)
        similarities = matrix @ query_vec
        best_idx = int(np.argmax(similarities))
        max_sim = float(similarities[best_idx])

        if max_sim > threshold:
            return entity_ids[best_idx]
        return None

    def delete_entities_for_page(self, page_url: str) -> int:
        """Delete entity links for all chunks on a page, then orphaned entities."""
        chunk_ids = [
            row[0] for row in self.conn.execute(
                "SELECT chunk_id FROM chunks WHERE page_url = ?", (page_url,)
            ).fetchall()
        ]
        if not chunk_ids:
            return 0
        placeholders = ",".join("?" * len(chunk_ids))
        self.conn.execute(
            f"DELETE FROM chunk_entities WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        self.conn.execute(
            "DELETE FROM entities WHERE entity_id NOT IN "
            "(SELECT DISTINCT entity_id FROM chunk_entities)"
        )
        self.conn.commit()
        return len(chunk_ids)

    # ── Token Embedding Operations ────────────────────────────────────────

    def store_token_embeddings(self, chunk_id: str, token_embeddings: "np.ndarray") -> None:
        """Store ColBERT token embeddings and MUVERA FDE vector for a chunk."""
        blob = _token_embeddings_to_blob(token_embeddings)
        num_tokens = token_embeddings.shape[0]
        fde = _compute_fde_vector(token_embeddings)
        fde_blob = _embedding_to_blob(fde.tolist(), quantize=True)
        self.conn.execute(
            """INSERT INTO chunk_token_embeddings (chunk_id, token_embeddings, token_count, fde_vector)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(chunk_id) DO UPDATE SET
                   token_embeddings = excluded.token_embeddings,
                   token_count = excluded.token_count,
                   fde_vector = excluded.fde_vector""",
            (chunk_id, blob, num_tokens, fde_blob),
        )
        self.conn.commit()

    def get_token_embeddings(self, chunk_id: str) -> "np.ndarray | None":
        """Retrieve ColBERT token embeddings for a chunk."""
        row = self.conn.execute(
            "SELECT token_embeddings FROM chunk_token_embeddings WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return _blob_to_token_embeddings(row[0])

    def get_fde_vector(self, chunk_id: str) -> "np.ndarray | None":
        """Retrieve MUVERA FDE vector for a chunk."""
        row = self.conn.execute(
            "SELECT fde_vector FROM chunk_token_embeddings WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        emb = _blob_to_embedding(row[0])
        if emb is None:
            return None
        return np.array(emb, dtype=np.float32)

    def get_all_fde_vectors(self) -> dict[str, "np.ndarray"]:
        """Retrieve all FDE vectors keyed by chunk_id."""
        rows = self.conn.execute(
            "SELECT chunk_id, fde_vector FROM chunk_token_embeddings WHERE fde_vector IS NOT NULL"
        ).fetchall()
        result = {}
        for chunk_id, blob in rows:
            emb = _blob_to_embedding(blob)
            if emb is not None:
                result[chunk_id] = np.array(emb, dtype=np.float32)
        return result

    def get_all_code_embeddings(self) -> dict[str, "np.ndarray"]:
        """Retrieve all jina code embeddings keyed by chunk_id."""
        rows = self.conn.execute(
            "SELECT chunk_id, code_embedding FROM chunks WHERE code_embedding IS NOT NULL"
        ).fetchall()
        result = {}
        for chunk_id, blob in rows:
            emb = _blob_to_embedding(blob)
            if emb is not None:
                result[chunk_id] = np.array(emb, dtype=np.float32)
        return result

    # ── Search Log Operations ──────────────────────────────────────────────

    def get_all_pages(self) -> list[dict]:
        """Return all stored pages as dicts with url, title, fetched_at."""
        rows = self.conn.execute(
            "SELECT url, title, fetched_at FROM pages ORDER BY fetched_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_search_count(self) -> int:
        """Return total number of logged searches."""
        row = self.conn.execute("SELECT COUNT(*) AS cnt FROM searches").fetchone()
        return row["cnt"]

    def get_last_activity(self) -> str | None:
        """Return the most recent timestamp across pages, chunks, and searches."""
        row = self.conn.execute(
            """SELECT MAX(ts) AS last FROM (
                   SELECT MAX(fetched_at) AS ts FROM pages
                   UNION ALL
                   SELECT MAX(created_at) AS ts FROM chunks
                   UNION ALL
                   SELECT MAX(created_at) AS ts FROM searches
               )"""
        ).fetchone()
        return row["last"] if row else None

    def log_search(self, query: str, result_count: int = 0) -> int:
        """Log a search query and return the search ID."""
        cursor = self.conn.execute(
            "INSERT INTO searches (query, result_count) VALUES (?, ?)",
            (query, result_count),
        )
        self.conn.commit()
        return cursor.lastrowid

    # ── Image Operations ──────────────────────────────────────────────────

    def store_image(
        self,
        image_id: str,
        page_url: str,
        src_url: str,
        alt_text: str = "",
        width: int | None = None,
        height: int | None = None,
        embedding: list[float] | None = None,
        nearest_chunk_ids: list[str] | None = None,
        image_bytes: bytes | None = None,
        is_chart: bool = False,
    ) -> None:
        """Store or update an image record. Upserts on image_id."""
        embedding_blob = None
        if embedding is not None:
            embedding_blob = _embedding_to_blob(embedding)

        nearest_json = None
        if nearest_chunk_ids is not None:
            nearest_json = json.dumps(nearest_chunk_ids)

        self.conn.execute(
            """INSERT INTO images (image_id, page_url, src_url, alt_text, width, height,
                                   embedding, nearest_chunk_ids, image_bytes, is_chart)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(image_id) DO UPDATE SET
                   page_url = excluded.page_url,
                   src_url = excluded.src_url,
                   alt_text = excluded.alt_text,
                   width = excluded.width,
                   height = excluded.height,
                   embedding = excluded.embedding,
                   nearest_chunk_ids = excluded.nearest_chunk_ids,
                   image_bytes = excluded.image_bytes,
                   is_chart = excluded.is_chart,
                   created_at = datetime('now')""",
            (image_id, page_url, src_url, alt_text, width, height, embedding_blob,
             nearest_json, image_bytes, int(is_chart)),
        )
        self.conn.commit()

    def get_images_for_page(self, url: str) -> list[dict]:
        """Return all images for a page URL, deserializing stored blobs."""
        rows = self.conn.execute(
            "SELECT * FROM images WHERE page_url = ?", (url,)
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d["embedding"] is not None:
                d["embedding"] = _blob_to_embedding(d["embedding"])
            if d["nearest_chunk_ids"] is not None:
                d["nearest_chunk_ids"] = json.loads(d["nearest_chunk_ids"])
            results.append(d)
        return results

    def get_all_images(self) -> list[dict]:
        """Return all images with non-NULL embeddings, deserialized."""
        rows = self.conn.execute(
            "SELECT * FROM images WHERE embedding IS NOT NULL"
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["embedding"] = _blob_to_embedding(d["embedding"])
            if d["nearest_chunk_ids"] is not None:
                d["nearest_chunk_ids"] = json.loads(d["nearest_chunk_ids"])
            results.append(d)
        return results

    def get_image_count(self) -> int:
        """Return total count of images (including those with NULL embeddings)."""
        row = self.conn.execute("SELECT COUNT(*) AS cnt FROM images").fetchone()
        return row["cnt"]

    def get_chart_images_for_chunks(self, chunk_ids: list[str], max_results: int = 3) -> list[dict]:
        """Return chart images whose nearest_chunk_ids overlap with the given chunk_ids."""
        if not chunk_ids:
            return []
        chunk_set = set(chunk_ids)
        results: list[dict] = []
        cursor = self.conn.execute(
            "SELECT image_id, src_url, page_url, nearest_chunk_ids, image_bytes"
            " FROM images WHERE is_chart = 1"
        )
        for row in cursor:
            if row["nearest_chunk_ids"] and row["image_bytes"]:
                img_chunks = json.loads(row["nearest_chunk_ids"])
                if set(img_chunks) & chunk_set:
                    results.append({
                        "image_id": row["image_id"],
                        "src_url": row["src_url"],
                        "page_url": row["page_url"],
                        "image_bytes": row["image_bytes"],
                    })
                    if len(results) >= max_results:
                        break
        return results

    # ── Link Graph Operations ─────────────────────────────────────────────

    def store_links(self, source_url: str, links: list[dict]) -> None:
        """Batch insert page links with INSERT OR IGNORE (PK handles dedup)."""
        for link in links:
            self.conn.execute(
                "INSERT OR IGNORE INTO page_links (source_url, target_url, anchor_text) VALUES (?, ?, ?)",
                (source_url, link["target_url"], link.get("anchor_text", "")),
            )
        self.conn.commit()

    def get_link_graph(self) -> dict[str, set[str]]:
        """Query all rows from page_links and build adjacency dict."""
        rows = self.conn.execute("SELECT source_url, target_url FROM page_links").fetchall()
        graph: dict[str, set[str]] = {}
        for row in rows:
            src = row["source_url"]
            tgt = row["target_url"]
            if src not in graph:
                graph[src] = set()
            graph[src].add(tgt)
        return graph

    def get_authority_scores(self) -> dict[str, float]:
        """Return authority scores for all pages that have one."""
        rows = self.conn.execute(
            "SELECT url, authority_score FROM pages WHERE authority_score IS NOT NULL"
        ).fetchall()
        return {row["url"]: row["authority_score"] for row in rows}

    def get_link_count(self) -> int:
        """Return total number of stored page links."""
        row = self.conn.execute("SELECT COUNT(*) AS cnt FROM page_links").fetchone()
        return row["cnt"]

    def store_graph_data(
        self,
        embeddings: dict[str, list[float]],
        centrality: dict[str, float],
    ) -> None:
        """Batch update pages.graph_embedding and pages.authority_score."""
        for url, emb in embeddings.items():
            score = centrality.get(url, 0.0)
            self.conn.execute(
                "UPDATE pages SET graph_embedding = ?, authority_score = ? WHERE url = ?",
                (_embedding_to_blob(emb), score, url),
            )
        # Handle isolated pages -- give them zero-vector embeddings
        all_page_urls = [
            row["url"]
            for row in self.conn.execute("SELECT url FROM pages").fetchall()
        ]
        dim = len(next(iter(embeddings.values()))) if embeddings else 8
        for url in all_page_urls:
            if url not in embeddings:
                zero_emb = [0.0] * dim
                self.conn.execute(
                    "UPDATE pages SET graph_embedding = ?, authority_score = ? WHERE url = ?",
                    (_embedding_to_blob(zero_emb), 0.0, url),
                )
        self.conn.commit()


# ── Weighted Reciprocal Rank Fusion ──────────────────────────────────────────


def multi_signal_rrf(
    ranked_lists: dict[str, list[tuple[str, float]]],
    weights: dict[str, float],
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[str, float, dict[str, float]]]:
    """Weighted Reciprocal Rank Fusion across multiple signal ranked lists.

    Args:
        ranked_lists: signal_name -> [(chunk_id, raw_score), ...] sorted by score desc
        weights: signal_name -> weight multiplier
        k: RRF constant (default 60)
        top_k: number of results to return

    Returns:
        [(chunk_id, rrf_score, {signal_name: raw_score, ...}), ...] sorted by rrf_score desc
    """
    scores: dict[str, float] = {}
    signal_scores: dict[str, dict[str, float]] = {}

    for signal_name, ranked in ranked_lists.items():
        weight = weights.get(signal_name, 1.0)
        for rank, (chunk_id, raw_score) in enumerate(ranked):
            rrf_contribution = weight / (k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_contribution
            if chunk_id not in signal_scores:
                signal_scores[chunk_id] = {}
            signal_scores[chunk_id][signal_name] = raw_score

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(cid, score, signal_scores.get(cid, {})) for cid, score in sorted_results]


# ── Embedding Migration ───────────────────────────────────────────────────────


def migrate_embeddings(store: ResearchStore, batch_size: int = 100) -> dict:
    """Re-embed all legacy (version 0) chunks with the current MRL model.

    Returns a dict with migration statistics.
    """
    if not _ensure_embedding_model():
        raise RuntimeError(
            "Cannot migrate: embedding model unavailable. "
            "Ensure onnxruntime and tokenizers are installed and network is accessible."
        )

    rows = store.conn.execute(
        "SELECT chunk_id, text FROM chunks "
        "WHERE embedding IS NOT NULL AND embedding_version < 1"
    ).fetchall()

    total = len(rows)
    migrated = 0
    skipped = 0

    for i, row in enumerate(rows):
        text = row["text"]
        if not text:
            skipped += 1
            continue

        embedding = _make_embedding(text, mode="document")
        if embedding is None:
            logger.warning("Migration: failed to embed chunk %s, skipping", row["chunk_id"])
            skipped += 1
            continue

        blob = _embedding_to_blob(embedding, quantize=True)
        store.conn.execute(
            "UPDATE chunks SET embedding = ?, embedding_version = 1 WHERE chunk_id = ?",
            (blob, row["chunk_id"]),
        )
        migrated += 1

        if (i + 1) % batch_size == 0:
            store.conn.commit()
            logger.info("Migration progress: %d/%d chunks", i + 1, total)

    store.conn.commit()
    return {"total": total, "migrated": migrated, "skipped": skipped}


# ── Hybrid Retrieval Index ─────────────────────────────────────────────────────
# Ported from omlx_agent/omlx_agent.py lines 243-381.
# Adapted: uses page_url instead of path, embeddings stored on DocumentChunk.

class HybridIndex:
    """BM25 + cosine similarity hybrid retrieval index.

    Ported from omlx_agent HybridIndex with adaptations for web content.
    """

    def __init__(self, mode: str = "query") -> None:
        self.mode = mode  # "ingest" or "query"
        self.chunks: list[DocumentChunk] = []
        self._bm25_doc_freqs: dict[str, int] = {}
        self._bm25_term_freqs: dict[str, dict[str, int]] = {}
        self._bm25_doc_lens: dict[str, int] = {}
        self._bm25_avg_dl: float = 0.0
        self._built = False
        # Funnel search cache (MRL truncated views)
        self._funnel_128: dict[str, list[float]] = {}
        self._funnel_256: dict[str, list[float]] = {}
        # Query-mode only data
        self._images: list[dict] = []
        self._authority_scores: dict[str, float] = {}
        # Corroboration data for recency decay: chunk_id -> (timestamp, strength)
        self._corroboration: dict[str, tuple[str, float]] = {}
        # FDE vectors for MaxSim screening
        self._fde_vectors: dict[str, "np.ndarray"] = {}
        # Entity data for entity retrieval signal
        self._chunk_entity_ids: dict[str, set[str]] = {}
        self._entity_embeddings: dict[str, list[float]] = {}
        # Jina code embeddings for code-to-code search
        self._code_embeddings: dict[str, "np.ndarray"] = {}

    def add_chunk(self, chunk: DocumentChunk) -> None:
        """Add a chunk to the index (call build() after adding all chunks)."""
        self.chunks.append(chunk)

    def build(self) -> None:
        """Build the BM25 index from current chunks."""
        self._build_bm25()
        self._build_funnel_cache()
        self._built = True

    def build_from_store(self, store: ResearchStore) -> None:
        """Load all chunks from a ResearchStore and build the index.

        When child chunks exist, uses only children for retrieval (BM25,
        embeddings, funnel cache). Parents whose children are present are
        excluded. Legacy chunks without children participate directly.

        Raises RuntimeError if unmigrated (version 0) embeddings exist.
        """
        if store.has_unmigrated_chunks():
            raise RuntimeError(
                "Database contains unmigrated embeddings (version 0). "
                "Run `python3 -m research_tool migrate-embeddings --db <path>` "
                "to re-embed with the current model before querying."
            )
        all_chunks = store.get_all_chunks()
        parent_ids_with_children: set[str] = set()
        for c in all_chunks:
            if c.is_child and c.parent_chunk_id:
                parent_ids_with_children.add(c.parent_chunk_id)
        if parent_ids_with_children:
            self.chunks = [
                c for c in all_chunks
                if c.is_child or c.chunk_id not in parent_ids_with_children
            ]
        else:
            self.chunks = all_chunks
        self._build_bm25()
        self._build_funnel_cache()

        if self.mode == "query":
            self._images = store.get_all_images()
            self._authority_scores = store.get_authority_scores()
            corr_data = store.get_chunk_corroboration_data()
            self._corroboration = {
                cid: (ts, strength) for cid, ts, strength in corr_data
            }
            self._fde_vectors = store.get_all_fde_vectors()
            self._code_embeddings = store.get_all_code_embeddings()
            if ENTITY_RESOLUTION_ENABLED:
                chunk_ids_in_index = {c.chunk_id for c in self.chunks}
                for entity in store.get_all_entities():
                    if entity["entity_embedding"]:
                        self._entity_embeddings[entity["entity_id"]] = entity["entity_embedding"]
                for chunk in self.chunks:
                    eids = store.get_chunk_entity_ids(chunk.chunk_id)
                    if eids:
                        self._chunk_entity_ids[chunk.chunk_id] = set(eids)

        self._built = True

    def _build_funnel_cache(self) -> None:
        """Pre-compute truncated and re-normalized embedding views for funnel search."""
        self._funnel_128: dict[str, list[float]] = {}
        self._funnel_256: dict[str, list[float]] = {}
        for chunk in self.chunks:
            if chunk.embedding is None:
                continue
            emb = chunk.embedding
            # 128-dim prefix, re-normalized
            prefix_128 = emb[:128]
            norm_128 = sum(x * x for x in prefix_128) ** 0.5
            if norm_128 > 0:
                self._funnel_128[chunk.chunk_id] = [x / norm_128 for x in prefix_128]
            # 256-dim prefix, re-normalized
            prefix_256 = emb[:256]
            norm_256 = sum(x * x for x in prefix_256) ** 0.5
            if norm_256 > 0:
                self._funnel_256[chunk.chunk_id] = [x / norm_256 for x in prefix_256]

    def _build_bm25(self) -> None:
        """Build BM25 statistics from chunks. Ported from omlx_agent."""
        self._bm25_doc_freqs = {}
        self._bm25_term_freqs = {}
        self._bm25_doc_lens = {}
        total_len = 0
        for chunk in self.chunks:
            tokens = _rag_tokenize(chunk.text)
            freq: dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self._bm25_term_freqs[chunk.chunk_id] = freq
            self._bm25_doc_lens[chunk.chunk_id] = len(tokens)
            total_len += len(tokens)
            for term in set(tokens):
                self._bm25_doc_freqs[term] = self._bm25_doc_freqs.get(term, 0) + 1
        n = len(self.chunks)
        self._bm25_avg_dl = total_len / n if n else 1.0

    def bm25_retrieve(self, query: str, top_k: int = 5) -> list[RetrievedEntry]:
        """BM25 retrieval. Ported from omlx_agent."""
        if not self._built or not self.chunks:
            return []
        query_tokens = _rag_tokenize(query)
        if not query_tokens:
            return []
        n = len(self.chunks)
        scores: dict[str, float] = {}
        k1, b = 1.5, 0.75
        for chunk in self.chunks:
            cid = chunk.chunk_id
            tf = self._bm25_term_freqs.get(cid, {})
            dl = self._bm25_doc_lens.get(cid, 0)
            score = 0.0
            for term in query_tokens:
                df = self._bm25_doc_freqs.get(term, 0)
                if df == 0:
                    continue
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                term_freq = tf.get(term, 0)
                numerator = term_freq * (k1 + 1)
                denominator = term_freq + k1 * (1 - b + b * dl / self._bm25_avg_dl)
                score += idf * numerator / denominator
            if score > 0:
                scores[cid] = score
        if not scores:
            return []
        max_score = max(scores.values())
        if max_score > 0:
            scores = {cid: s / max_score for cid, s in scores.items()}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        chunk_map = {c.chunk_id: c for c in self.chunks}
        results = []
        for cid, score in ranked:
            chunk = chunk_map[cid]
            results.append(RetrievedEntry(
                page_url=chunk.page_url,
                chunk_id=cid,
                section_title=chunk.section_title,
                score=score,
                bm25_score=score,
                cosine_score=0.0,
                text=chunk.text,
            ))
        return results

    def _image_cosine_rank(self, query: str) -> list[tuple[str, float]]:
        """Rank chunks by image cosine similarity via proximity mapping."""
        if not self._images:
            return []

        query_emb = _make_clip_text_embedding(query)
        if query_emb is None:
            return []

        import json as _json

        # Compute cosine similarity for each image
        image_scores: list[tuple[str, list[str], float]] = []
        for img in self._images:
            if img.get("embedding") is None:
                continue
            cosine = sum(a * b for a, b in zip(query_emb, img["embedding"]))
            chunk_ids = img.get("nearest_chunk_ids") or []
            if isinstance(chunk_ids, str):
                try:
                    chunk_ids = _json.loads(chunk_ids)
                except (ValueError, TypeError):
                    chunk_ids = []
            image_scores.append((img.get("page_url", ""), chunk_ids, cosine))

        # Map image scores to chunks via proximity mapping
        chunk_best_score: dict[str, float] = {}
        for page_url, chunk_ids, cosine in image_scores:
            for cid in chunk_ids:
                if cid not in chunk_best_score or cosine > chunk_best_score[cid]:
                    chunk_best_score[cid] = cosine

        if not chunk_best_score:
            return []

        return sorted(chunk_best_score.items(), key=lambda x: x[1], reverse=True)

    def _graph_rank(self) -> list[tuple[str, float]]:
        """Rank chunks by page authority score (eigenvector centrality)."""
        if not self._authority_scores:
            return []

        # Broadcast authority to chunks
        chunk_scores: dict[str, float] = {}
        for chunk in self.chunks:
            authority = self._authority_scores.get(chunk.page_url, 0.0)
            chunk_scores[chunk.chunk_id] = authority

        # Filter out zero-authority chunks
        ranked = [(cid, score) for cid, score in chunk_scores.items() if score > 0]
        return sorted(ranked, key=lambda x: x[1], reverse=True)

    def _recency_rank(self) -> list[tuple[str, float]]:
        """Rank chunks by corroboration-aware time decay.

        Uses exponential decay from last_corroborated_at, boosted by
        accumulated corroboration_strength so that frequently-confirmed
        chunks decay more slowly.
        """
        if not self._corroboration:
            return []

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        decay_lambda = math.log(2) / RECENCY_HALF_LIFE_DAYS
        cap = CORROBORATION_STRENGTH_CAP

        scores: list[tuple[str, float]] = []
        for chunk_id, (ts, strength) in self._corroboration.items():
            try:
                created = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            age_days = max((now - created).total_seconds() / 86400, 0.0)
            base_decay = math.exp(-decay_lambda * age_days)
            strength_boost = min(strength / cap, 1.0)
            score = base_decay + (1.0 - base_decay) * strength_boost
            scores.append((chunk_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _entity_rank(self, query_emb: list[float], top_entities: int = 5) -> list[tuple[str, float]]:
        """Rank chunks by entity match against the query embedding."""
        if not self._entity_embeddings or not self._chunk_entity_ids:
            return []

        import numpy as np

        entity_ids = list(self._entity_embeddings.keys())
        matrix = np.array([self._entity_embeddings[eid] for eid in entity_ids], dtype=np.float32)
        q = np.array(query_emb, dtype=np.float32)
        sims = matrix @ q
        top_indices = np.argsort(sims)[::-1][:top_entities]
        matched_entities: dict[str, float] = {}
        for idx in top_indices:
            sim = float(sims[idx])
            if sim > ENTITY_RESOLUTION_THRESHOLD:
                matched_entities[entity_ids[idx]] = sim

        if not matched_entities:
            return []

        chunk_scores: dict[str, float] = {}
        for chunk_id, eids in self._chunk_entity_ids.items():
            total = 0.0
            for eid in eids:
                if eid in matched_entities:
                    total += matched_entities[eid]
            if total > 0:
                chunk_scores[chunk_id] = total

        return sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

    def _maxsim_score(self, query_tokens: "np.ndarray", doc_tokens: "np.ndarray") -> float:
        """ColBERT MaxSim: sum of per-query-token max similarities against doc tokens."""
        # query_tokens: (Q, 128), doc_tokens: (D, 128)
        sim_matrix = query_tokens @ doc_tokens.T  # (Q, D)
        return float(sim_matrix.max(axis=1).sum())

    def _maxsim_rank(self, query: str, store: "ResearchStore", top_k: int) -> list[tuple[str, float]]:
        """Two-stage MaxSim with IDF weighting.

        FDE broad screening → collect per-query-token max similarities across
        all candidates → compute soft IDF weights so common tokens ("what",
        "the") contribute less than distinctive tokens ("Keccak", "scheduler")
        → score each candidate with weighted MaxSim.
        """
        query_tokens = _make_token_embeddings(query)
        if query_tokens is None:
            return []

        # Stage 1: FDE-based broad screening
        fde_candidates = self._fde_screen(query_tokens, n_candidates=top_k * 3)

        if fde_candidates:
            candidate_ids = set(fde_candidates)
        else:
            candidate_ids = {c.chunk_id for c in self.chunks}

        # Stage 2: Collect per-token max similarities for all candidates
        all_max_sims: dict[str, "np.ndarray"] = {}
        for chunk_id in candidate_ids:
            doc_tokens = store.get_token_embeddings(chunk_id)
            if doc_tokens is None:
                continue
            sim_matrix = query_tokens @ doc_tokens.T  # (Q, D)
            all_max_sims[chunk_id] = sim_matrix.max(axis=1)  # (Q,)

        if not all_max_sims:
            return []

        # Compute soft IDF weights from corpus statistics
        n_docs = len(all_max_sims)
        stacked = np.stack(list(all_max_sims.values()))  # (N, Q)
        avg_max_sim = stacked.mean(axis=0)  # (Q,) — average match quality per query token
        idf_weights = 1.0 / (0.5 + avg_max_sim)
        idf_weights = idf_weights / idf_weights.sum()

        # Score with IDF-weighted MaxSim
        scores: list[tuple[str, float]] = []
        for chunk_id, max_sims in all_max_sims.items():
            score = float((max_sims * idf_weights).sum())
            scores.append((chunk_id, score))

        # Term overlap boost with IDF weighting: rare terms that subword
        # tokenization fragments get higher weight than shared-prefix terms.
        query_terms = [
            w.strip("?.,!;:\"'()").lower() for w in query.split()
            if len(w.strip("?.,!;:\"'()")) >= 4
            and w.strip("?.,!;:\"'()").lower() not in _QUERY_STOP_WORDS
        ]
        if query_terms:
            chunk_texts = {c.chunk_id: c.text.lower() for c in self.chunks}
            n_ch = len(chunk_texts)
            term_df = {t: sum(1 for txt in chunk_texts.values() if t in txt)
                       for t in query_terms}
            term_idf = {t: math.log(n_ch / (1 + df)) for t, df in term_df.items()}
            max_idf = sum(term_idf.values()) or 1.0

            boosted: list[tuple[str, float]] = []
            for chunk_id, score in scores:
                text_lower = chunk_texts.get(chunk_id, "")
                idf_sum = sum(term_idf[t] for t in query_terms if t in text_lower)
                if idf_sum > 0:
                    score += 0.09 * (idf_sum / max_idf)
                boosted.append((chunk_id, score))
            scores = boosted

            chunk_ids_map = {c.chunk_id: c for c in self.chunks}
            boosted2: list[tuple[str, float]] = []
            for chunk_id, score in scores:
                chunk_obj = chunk_ids_map.get(chunk_id)
                if chunk_obj:
                    ids_lower = [x.lower() for x in _extract_code_identifiers(chunk_obj.text)]
                    if ids_lower:
                        ids_lower_set = set(ids_lower)
                        exact = 0
                        substr = 0
                        for t in query_terms:
                            stems = _simple_stems(t)
                            if any(s in ids_lower_set for s in stems):
                                exact += 1
                            elif any(s in idl for s in stems for idl in ids_lower):
                                substr += 1
                        if exact + substr > 0:
                            score += (0.08 * exact + 0.03 * substr) / len(query_terms)
                            if exact + substr >= 2:
                                score += 0.025
                boosted2.append((chunk_id, score))
            scores = boosted2

            query_idents = [m.group() for m in _QUERY_IDENT_RE.finditer(query)]
            if query_idents:
                boosted3: list[tuple[str, float]] = []
                for chunk_id, score in scores:
                    chunk_obj = chunk_ids_map.get(chunk_id)
                    if chunk_obj:
                        hits = sum(1 for ident in query_idents if ident in chunk_obj.text)
                        if hits > 0:
                            score += 0.06 * (hits / len(query_idents))
                    boosted3.append((chunk_id, score))
                scores = boosted3

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _fde_screen(self, query_tokens: "np.ndarray", n_candidates: int) -> list[str]:
        """Screen chunks by FDE cosine similarity for initial candidate generation."""
        query_fde = _compute_fde_vector(query_tokens)
        if not self._fde_vectors:
            return []

        scores: list[tuple[str, float]] = []
        for chunk_id, fde in self._fde_vectors.items():
            cosine = float(np.dot(query_fde, fde))
            scores.append((chunk_id, cosine))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scores[:n_candidates]]

    def _funnel_cosine_rank(self, query_emb: list[float], top_k: int) -> list[tuple[str, float]]:
        """3-stage funnel search using MRL dimension truncation.

        Stage 1: 128-dim prefix → top_k * 4 candidates
        Stage 2: 256-dim prefix → top_k * 2 candidates
        Stage 3: 768-dim full  → top_k results
        """
        if not self.chunks:
            return []

        # Stage 1: 128-dim
        q128 = query_emb[:128]
        norm_q128 = sum(x * x for x in q128) ** 0.5
        if norm_q128 > 0:
            q128 = [x / norm_q128 for x in q128]
        stage1_scores = []
        for chunk in self.chunks:
            cached = self._funnel_128.get(chunk.chunk_id)
            if cached is None:
                continue
            cosine = sum(a * b for a, b in zip(q128, cached))
            stage1_scores.append((chunk.chunk_id, cosine))
        stage1_scores.sort(key=lambda x: x[1], reverse=True)
        stage1_ids = {cid for cid, _ in stage1_scores[:top_k * 4]}

        if not stage1_ids:
            return []

        # Stage 2: 256-dim
        q256 = query_emb[:256]
        norm_q256 = sum(x * x for x in q256) ** 0.5
        if norm_q256 > 0:
            q256 = [x / norm_q256 for x in q256]
        stage2_scores = []
        for cid in stage1_ids:
            cached = self._funnel_256.get(cid)
            if cached is None:
                continue
            cosine = sum(a * b for a, b in zip(q256, cached))
            stage2_scores.append((cid, cosine))
        stage2_scores.sort(key=lambda x: x[1], reverse=True)
        stage2_ids = {cid for cid, _ in stage2_scores[:top_k * 2]}

        if not stage2_ids:
            return []

        # Stage 3: full 768-dim
        chunk_map = {c.chunk_id: c for c in self.chunks}
        stage3_scores = []
        for cid in stage2_ids:
            chunk = chunk_map.get(cid)
            if chunk is None or chunk.embedding is None:
                continue
            cosine = sum(a * b for a, b in zip(query_emb, chunk.embedding))
            stage3_scores.append((cid, cosine))
        stage3_scores.sort(key=lambda x: x[1], reverse=True)
        return stage3_scores[:top_k]

    def hybrid_retrieve(
        self, query: str, top_k: int = 5, rerank: bool = True,
        store: "ResearchStore | None" = None,
        hyde_emb: "list[float] | None" = None,
        weight_overrides: "dict[str, float] | None" = None,
    ) -> list[RetrievedEntry]:
        """Hybrid retrieval using Weighted Reciprocal Rank Fusion.

        In ingest mode, uses BM25 + text cosine only.
        In query mode, also uses image cosine and graph authority signals,
        followed by cross-encoder reranking (unless rerank=False).
        """
        if not self._built or not self.chunks:
            return []

        retrieve_k = top_k
        if rerank and self.mode == "query":
            retrieve_k = RERANK_CANDIDATES

        # Signal 1: BM25
        bm25_results = self.bm25_retrieve(query, top_k=retrieve_k * 3)
        bm25_ranked = [(r.chunk_id, r.bm25_score) for r in bm25_results]

        # Signal 2: Text cosine (funnel search with MRL truncation)
        cosine_ranked: list[tuple[str, float]] = []
        query_emb: list[float] | None = None
        if _ONNX_AVAILABLE:
            query_emb = _make_embedding(query, mode="query")
            if query_emb is not None:
                if self._funnel_128:
                    cosine_ranked = self._funnel_cosine_rank(query_emb, top_k=retrieve_k * 3)
                else:
                    scores = []
                    for chunk in self.chunks:
                        if chunk.embedding is not None:
                            cosine = sum(a * b for a, b in zip(query_emb, chunk.embedding))
                            scores.append((chunk.chunk_id, cosine))
                    cosine_ranked = sorted(scores, key=lambda x: x[1], reverse=True)[:retrieve_k * 3]

                cosine_map: dict[str, float] = {cid: sc for cid, sc in cosine_ranked}
                for chunk in self.chunks:
                    if chunk.chunk_id not in cosine_map and chunk.embedding is not None:
                        cosine_map[chunk.chunk_id] = sum(
                            a * b for a, b in zip(query_emb, chunk.embedding)
                        )
                need_rerank = False

                for chunk in self.chunks:
                    if chunk.summary_embedding is not None:
                        sc = sum(a * b for a, b in zip(query_emb, chunk.summary_embedding))
                        if sc > cosine_map.get(chunk.chunk_id, -1.0):
                            cosine_map[chunk.chunk_id] = sc
                            need_rerank = True

                key_terms = _extract_query_key_terms(query)
                if key_terms:
                    focused_emb = _make_embedding(key_terms, mode="query")
                    if focused_emb is not None:
                        for chunk in self.chunks:
                            best = cosine_map.get(chunk.chunk_id, -1.0)
                            if chunk.embedding is not None:
                                fc = sum(a * b for a, b in zip(focused_emb, chunk.embedding))
                                if fc > best:
                                    best = fc
                            if chunk.summary_embedding is not None:
                                fc = sum(a * b for a, b in zip(focused_emb, chunk.summary_embedding))
                                if fc > best:
                                    best = fc
                            if best > cosine_map.get(chunk.chunk_id, -1.0):
                                cosine_map[chunk.chunk_id] = best
                                need_rerank = True

                # Jina code embedding: NL-to-code trained model with z-score
                # normalization to make scores comparable to nomic's scale.
                code_chunks = [c for c in self.chunks if c.code_embedding is not None]
                if code_chunks:
                    query_code_emb = _make_code_embedding(query)
                    if query_code_emb is not None:
                        jina_scores: dict[str, float] = {}
                        for chunk in code_chunks:
                            jina_scores[chunk.chunk_id] = sum(
                                a * b for a, b in zip(query_code_emb, chunk.code_embedding)
                            )
                        if len(jina_scores) >= 2:
                            nomic_vals = [cosine_map[cid] for cid in jina_scores if cid in cosine_map]
                            jina_vals = list(jina_scores.values())
                            if nomic_vals:
                                n_mean = sum(nomic_vals) / len(nomic_vals)
                                j_mean = sum(jina_vals) / len(jina_vals)
                                n_std = (sum((v - n_mean) ** 2 for v in nomic_vals) / len(nomic_vals)) ** 0.5
                                j_std = (sum((v - j_mean) ** 2 for v in jina_vals) / len(jina_vals)) ** 0.5
                                if j_std > 1e-9 and n_std > 1e-9:
                                    for cid, jsc in jina_scores.items():
                                        normalized = (jsc - j_mean) / j_std * n_std + n_mean
                                        if normalized > cosine_map.get(cid, -1.0):
                                            cosine_map[cid] = normalized
                                            need_rerank = True

                # Term-based boosts: identifier matching and IDF-weighted
                # text overlap compensate for embedding models conflating
                # chunks that share boilerplate prefixes.
                query_terms = [
                    w.strip("?.,!;:\"'()").lower() for w in query.split()
                    if len(w.strip("?.,!;:\"'()")) >= 4
                    and w.strip("?.,!;:\"'()").lower() not in _QUERY_STOP_WORDS
                ]
                if query_terms:
                    chunk_texts_lower = {c.chunk_id: c.text.lower() for c in self.chunks}
                    n_chunks = len(self.chunks)

                    # IDF-weighted text overlap: terms in many chunks are noise
                    term_doc_freq = {}
                    for t in query_terms:
                        term_doc_freq[t] = sum(
                            1 for txt in chunk_texts_lower.values() if t in txt
                        )
                    term_idf = {
                        t: math.log(n_chunks / (1 + df))
                        for t, df in term_doc_freq.items()
                    }
                    max_idf_sum = sum(term_idf.values()) or 1.0

                    for chunk in self.chunks:
                        text_lower = chunk_texts_lower[chunk.chunk_id]
                        idf_sum = sum(
                            term_idf[t] for t in query_terms if t in text_lower
                        )
                        if idf_sum > 0:
                            boost = 0.09 * (idf_sum / max_idf_sum)
                            new_score = cosine_map.get(chunk.chunk_id, -1.0) + boost
                            if new_score > cosine_map.get(chunk.chunk_id, -1.0):
                                cosine_map[chunk.chunk_id] = new_score
                                need_rerank = True

                    for chunk in self.chunks:
                        ids_lower = [x.lower() for x in _extract_code_identifiers(chunk.text)]
                        if not ids_lower:
                            continue
                        ids_lower_set = set(ids_lower)
                        exact = 0
                        substr = 0
                        for t in query_terms:
                            stems = _simple_stems(t)
                            if any(s in ids_lower_set for s in stems):
                                exact += 1
                            elif any(s in idl for s in stems for idl in ids_lower):
                                substr += 1
                        if exact + substr > 0:
                            boost = (0.08 * exact + 0.03 * substr) / len(query_terms)
                            if exact + substr >= 2:
                                boost += 0.025
                            new_score = cosine_map.get(chunk.chunk_id, -1.0) + boost
                            if new_score > cosine_map.get(chunk.chunk_id, -1.0):
                                cosine_map[chunk.chunk_id] = new_score
                                need_rerank = True

                    query_idents = [m.group() for m in _QUERY_IDENT_RE.finditer(query)]
                    if query_idents:
                        for chunk in self.chunks:
                            hits = sum(1 for ident in query_idents if ident in chunk.text)
                            if hits > 0:
                                boost = 0.06 * (hits / len(query_idents))
                                new_score = cosine_map.get(chunk.chunk_id, -1.0) + boost
                                if new_score > cosine_map.get(chunk.chunk_id, -1.0):
                                    cosine_map[chunk.chunk_id] = new_score
                                    need_rerank = True

                if need_rerank:
                    cosine_ranked = sorted(
                        cosine_map.items(), key=lambda x: x[1], reverse=True,
                    )[:retrieve_k * 3]

        ranked_lists = {
            "bm25": bm25_ranked,
            "text_cosine": cosine_ranked,
        }
        _wo = weight_overrides or {}
        weights = {
            "bm25": _wo.get("bm25", RRF_WEIGHT_BM25),
            "text_cosine": _wo.get("text_cosine", RRF_WEIGHT_TEXT_COSINE),
        }

        # Query-mode signals
        if self.mode == "query":
            # Signal 3: Image cosine
            image_ranked = self._image_cosine_rank(query)
            if image_ranked:
                ranked_lists["image_cosine"] = image_ranked
                weights["image_cosine"] = _wo.get("image_cosine", RRF_WEIGHT_IMAGE_COSINE)

            # Signal 4: Graph authority
            graph_ranked = self._graph_rank()
            if graph_ranked:
                ranked_lists["graph"] = graph_ranked
                weights["graph"] = _wo.get("graph", RRF_WEIGHT_GRAPH)

            # Signal 5: MaxSim (ColBERT late interaction)
            if store is not None:
                maxsim_ranked = self._maxsim_rank(query, store, top_k=retrieve_k * 3)
                if maxsim_ranked:
                    ranked_lists["maxsim"] = maxsim_ranked
                    weights["maxsim"] = _wo.get("maxsim", RRF_WEIGHT_MAXSIM)

            # Signal 6: HyDE cosine (hypothetical document embedding)
            if hyde_emb is not None:
                hyde_scores: list[tuple[str, float]] = []
                for chunk in self.chunks:
                    if chunk.embedding is not None:
                        cosine = sum(a * b for a, b in zip(hyde_emb, chunk.embedding))
                        hyde_scores.append((chunk.chunk_id, cosine))
                hyde_ranked = sorted(hyde_scores, key=lambda x: x[1], reverse=True)[:retrieve_k * 3]
                if hyde_ranked:
                    ranked_lists["hyde_cosine"] = hyde_ranked
                    weights["hyde_cosine"] = _wo.get("hyde_cosine", RRF_WEIGHT_HYDE)

            # Signal 7: Recency (corroboration-aware time decay)
            recency_ranked = self._recency_rank()
            if recency_ranked:
                ranked_lists["recency"] = recency_ranked
                weights["recency"] = _wo.get("recency", RRF_WEIGHT_RECENCY)

            # Signal 8: Entity match
            if ENTITY_RESOLUTION_ENABLED and query_emb is not None:
                entity_ranked = self._entity_rank(query_emb)
                if entity_ranked:
                    ranked_lists["entity"] = entity_ranked
                    weights["entity"] = _wo.get("entity", RRF_WEIGHT_ENTITY)

        # Fuse
        fused = multi_signal_rrf(ranked_lists, weights, k=RRF_K, top_k=retrieve_k)

        chunk_map = {c.chunk_id: c for c in self.chunks}

        # Dynamic rerank depth: expand candidate pool if content-type
        # distribution is homogeneous (minority type < threshold)
        if rerank and self.mode == "query" and RERANK_DIVERSITY_THRESHOLD > 0:
            content_type_map = {c.chunk_id: c.content_type for c in self.chunks}
            code_count = sum(
                1 for cid, _, _ in fused
                if content_type_map.get(cid, "text") == "code"
            )
            total = len(fused)
            if total > 0:
                minority_ratio = min(code_count, total - code_count) / total
                if minority_ratio < RERANK_DIVERSITY_THRESHOLD:
                    expanded_k = min(
                        int(retrieve_k * RERANK_EXPANSION_MULTIPLIER),
                        RERANK_MAX_CANDIDATES,
                    )
                    if expanded_k > retrieve_k:
                        fused = multi_signal_rrf(
                            ranked_lists, weights, k=RRF_K, top_k=expanded_k,
                        )

        results = []
        for cid, rrf_score, signals in fused:
            chunk = chunk_map.get(cid)
            if chunk is None:
                continue
            results.append(RetrievedEntry(
                page_url=chunk.page_url,
                chunk_id=cid,
                section_title=chunk.section_title,
                score=rrf_score,
                bm25_score=signals.get("bm25", 0.0),
                cosine_score=signals.get("text_cosine", 0.0),
                text=chunk.text,
                image_cosine_score=signals.get("image_cosine", 0.0),
                graph_score=signals.get("graph", 0.0),
                maxsim_score=signals.get("maxsim", 0.0),
                hyde_cosine_score=signals.get("hyde_cosine", 0.0),
                recency_score=signals.get("recency", 0.0),
                entity_score=signals.get("entity", 0.0),
            ))

        # Cross-encoder reranking (query mode only)
        if rerank and self.mode == "query" and results:
            rerank_scores = _rerank_pairs(query, [r.text for r in results])
            if rerank_scores is not None:
                for entry, logit in zip(results, rerank_scores):
                    entry.rerank_score = 1.0 / (1.0 + math.exp(-logit))
                results.sort(key=lambda r: r.rerank_score, reverse=True)
                results = results[:top_k]
            else:
                logger.warning("Reranker unavailable, falling back to RRF order")
                results = results[:top_k]
        elif len(results) > top_k:
            results = results[:top_k]

        return results

    @property
    def is_built(self) -> bool:
        return self._built


# ── Code-to-Code Search ──────────────────────────────────────────────────────


def query_similar_code(
    code_text: str,
    index: HybridIndex,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Find structurally similar code using jina embeddings from the code_embedding column.

    Returns a list of (chunk_id, score) tuples sorted by descending cosine similarity.
    Returns an empty list if no code embeddings are cached or the jina model is unavailable.
    """
    if not index._code_embeddings:
        return []
    query_emb = _make_code_embedding(code_text)
    if query_emb is None:
        return []
    query_vec = np.array(query_emb, dtype=np.float32)
    scores: list[tuple[str, float]] = []
    for chunk_id, code_vec in index._code_embeddings.items():
        cosine = float(np.dot(query_vec, code_vec))
        scores.append((chunk_id, cosine))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
