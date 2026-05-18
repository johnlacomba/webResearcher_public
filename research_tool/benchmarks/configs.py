"""Benchmark configuration matrix: all chunking × embedding × retrieval combos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BenchmarkConfig:
    name: str
    tier: Literal["deterministic", "llm_required"]
    module_overrides: dict[str, object] = field(default_factory=dict)
    description: str = ""


# Attributes that must be patched in both store and brain modules.
# brain.py imports these as name bindings from store.py, so both must be set.
DUAL_PATCH_ATTRS = frozenset({
    "PARENT_CHILD_ENABLED",
    "CHILD_CHUNK_MAX_TOKENS",
    "CONTEXTUAL_RETRIEVAL",
    "HYDE_ENABLED",
    "ENTITY_RESOLUTION_ENABLED",
    "ENTITY_RESOLUTION_THRESHOLD",
    "CORROBORATION_DEDUP_INCREMENT",
    "CORROBORATION_GRAPH_INCREMENT",
})

# Attributes only in store (not imported by brain).
STORE_ONLY_ATTRS = frozenset({
    "RRF_K",
    "RRF_WEIGHT_BM25",
    "RRF_WEIGHT_TEXT_COSINE",
    "RRF_WEIGHT_IMAGE_COSINE",
    "RRF_WEIGHT_GRAPH",
    "RRF_WEIGHT_MAXSIM",
    "RRF_WEIGHT_HYDE",
    "RRF_WEIGHT_RECENCY",
    "RRF_WEIGHT_ENTITY",
    "RECENCY_HALF_LIFE_DAYS",
    "CORROBORATION_STRENGTH_CAP",
    "CHUNK_MODE",
    "JINA_CODE_EMBEDDING_ENABLED",
    "TOKEN_EMBEDDING_ENABLED",
    "PARAGRAPH_MAX_TOKENS",
})


def _chunking_configs() -> list[dict]:
    return [
        {"label": "paragraph", "overrides": {"PARENT_CHILD_ENABLED": False, "CHUNK_MODE": "paragraph"}},
        {"label": "parent-child", "overrides": {"PARENT_CHILD_ENABLED": True, "CHUNK_MODE": "auto"}},
        {"label": "wiki-aware", "overrides": {"PARENT_CHILD_ENABLED": False, "CHUNK_MODE": "wiki"}},
        {"label": "ast-code", "overrides": {"PARENT_CHILD_ENABLED": False, "CHUNK_MODE": "code"}},
    ]


def _embedding_configs() -> list[dict]:
    return [
        {"label": "nomic-only", "overrides": {
            "JINA_CODE_EMBEDDING_ENABLED": False,
            "TOKEN_EMBEDDING_ENABLED": False,
        }},
        {"label": "nomic+jina", "overrides": {
            "JINA_CODE_EMBEDDING_ENABLED": True,
            "TOKEN_EMBEDDING_ENABLED": False,
        }},
        {"label": "nomic+colbert+fde", "overrides": {
            "JINA_CODE_EMBEDDING_ENABLED": False,
            "TOKEN_EMBEDDING_ENABLED": True,
        }},
        {"label": "all-embeddings", "overrides": {
            "JINA_CODE_EMBEDDING_ENABLED": True,
            "TOKEN_EMBEDDING_ENABLED": True,
        }},
    ]


def _retrieval_configs() -> list[dict]:
    return [
        {
            "label": "bm25-only",
            "overrides": {
                "RRF_WEIGHT_TEXT_COSINE": 0.0,
                "RRF_WEIGHT_IMAGE_COSINE": 0.0,
                "RRF_WEIGHT_GRAPH": 0.0,
                "RRF_WEIGHT_MAXSIM": 0.0,
                "RRF_WEIGHT_HYDE": 0.0,
                "RRF_WEIGHT_RECENCY": 0.0,
                "RRF_WEIGHT_ENTITY": 0.0,
                "HYDE_ENABLED": False,
            },
        },
        {
            "label": "text-cosine-only",
            "overrides": {
                "RRF_WEIGHT_BM25": 0.0,
                "RRF_WEIGHT_IMAGE_COSINE": 0.0,
                "RRF_WEIGHT_GRAPH": 0.0,
                "RRF_WEIGHT_MAXSIM": 0.0,
                "RRF_WEIGHT_HYDE": 0.0,
                "RRF_WEIGHT_RECENCY": 0.0,
                "RRF_WEIGHT_ENTITY": 0.0,
                "HYDE_ENABLED": False,
            },
        },
        {
            "label": "maxsim-only",
            "overrides": {
                "RRF_WEIGHT_BM25": 0.0,
                "RRF_WEIGHT_TEXT_COSINE": 0.0,
                "RRF_WEIGHT_IMAGE_COSINE": 0.0,
                "RRF_WEIGHT_GRAPH": 0.0,
                "RRF_WEIGHT_HYDE": 0.0,
                "RRF_WEIGHT_RECENCY": 0.0,
                "RRF_WEIGHT_ENTITY": 0.0,
                "HYDE_ENABLED": False,
            },
        },
        {
            "label": "hybrid-no-hyde",
            "overrides": {
                "HYDE_ENABLED": False,
                "RRF_WEIGHT_HYDE": 0.0,
            },
        },
        {
            "label": "hybrid+hyde-cached",
            "overrides": {
                "HYDE_ENABLED": True,
            },
        },
        {
            "label": "hybrid+hyde+entity",
            "overrides": {
                "HYDE_ENABLED": True,
                "ENTITY_RESOLUTION_ENABLED": True,
            },
        },
    ]


def generate_matrix(include_llm: bool = False) -> list[BenchmarkConfig]:
    """Generate the full configuration matrix.

    Returns deterministic-tier configs by default (96 combos).
    Pass include_llm=True to also generate LLM-required tier configs.
    """
    configs: list[BenchmarkConfig] = []

    for chunk_cfg in _chunking_configs():
        for emb_cfg in _embedding_configs():
            for ret_cfg in _retrieval_configs():
                merged = {}
                merged.update(chunk_cfg["overrides"])
                merged.update(emb_cfg["overrides"])
                merged.update(ret_cfg["overrides"])

                name = f"{chunk_cfg['label']}_{emb_cfg['label']}_{ret_cfg['label']}"
                configs.append(BenchmarkConfig(
                    name=name,
                    tier="deterministic",
                    module_overrides=merged,
                    description=f"{chunk_cfg['label']} chunking, {emb_cfg['label']} embedding, {ret_cfg['label']} retrieval",
                ))

    if include_llm:
        for emb_cfg in _embedding_configs():
            for ret_cfg in _retrieval_configs():
                merged = {}
                merged["CONTEXTUAL_RETRIEVAL"] = True
                merged.update(emb_cfg["overrides"])
                merged.update(ret_cfg["overrides"])

                name = f"contextual_{emb_cfg['label']}_{ret_cfg['label']}"
                configs.append(BenchmarkConfig(
                    name=name,
                    tier="llm_required",
                    module_overrides=merged,
                    description=f"contextual chunking, {emb_cfg['label']} embedding, {ret_cfg['label']} retrieval",
                ))

    return configs


def get_config_by_name(name: str, include_llm: bool = False) -> BenchmarkConfig | None:
    for cfg in generate_matrix(include_llm=include_llm):
        if cfg.name == name:
            return cfg
    return None
