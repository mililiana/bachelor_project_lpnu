"""
fast_query_analyzer.py
======================
Drop-in replacement for LLMQueryAnalyzer.
Extracts keywords and optional metadata filters from a user query
without any LLM / API call — uses BM25 scoring over the document corpus.

Output shape (identical to LLMQueryAnalyzer.analyze):
    {
        "keywords": List[str],   # meaningful terms for hybrid-search boosting
        "filters":  dict | None  # ChromaDB 'where' filter, or None
    }
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from loguru import logger
from rank_bm25 import BM25Okapi

# ── Ukrainian / Russian stop-words (inline — no extra file needed) ─────────────
_STOP_WORDS = {
    # Ukrainian
    "і",
    "й",
    "та",
    "або",
    "але",
    "що",
    "як",
    "з",
    "із",
    "зі",
    "до",
    "від",
    "за",
    "на",
    "у",
    "в",
    "про",
    "для",
    "при",
    "після",
    "між",
    "через",
    "де",
    "коли",
    "хто",
    "яка",
    "яке",
    "які",
    "цей",
    "ця",
    "це",
    "ці",
    "той",
    "та",
    "те",
    "ті",
    "він",
    "вона",
    "воно",
    "вони",
    "ми",
    "ви",
    "ж",
    "же",
    "бо",
    "чи",
    "не",
    "так",
    "це",
    "а",
    "ще",
    "вже",
    "більш",
    "також",
    "навіть",
    "лише",
    "тільки",
    "дуже",
    "можна",
    "треба",
    "потрібно",
    "є",
    "має",
    "був",
    "була",
    "були",
    "буде",
    "будуть",
    "зі",
    "своєї",
    "свого",
    "свої",
    "своїм",
    "скільки",
    "яких",
    "якої",
    "яким",
    "яко",
    # Russian (common leakage in Ukrainian texts)
    "и",
    "в",
    "не",
    "на",
    "с",
    "что",
    "по",
    "к",
    "из",
    "или",
    "его",
    "от",
    "для",
    "как",
    "это",
    "все",
    "за",
    "то",
    "же",
    "при",
    "о",
}

# Minimum character length for a token to be kept as a keyword
_MIN_TOKEN_LEN = 3

# How many top BM25-scored tokens to keep as keywords (max)
_MAX_KEYWORDS = 8

# BM25 score threshold — tokens below this fraction of the top score are dropped
_BM25_RELATIVE_THRESHOLD = 0.10

# Default metadata cache path (same file used by build_prompt.py)
_DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "prompt", "vector_db_metadata_cache.json"
)


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = [
        t for t in text.split() if len(t) >= _MIN_TOKEN_LEN and t not in _STOP_WORDS
    ]
    return tokens


class FastQueryAnalyzer:
    """
    Lightweight, token-free query analyzer.

    On construction it builds a small BM25 index from the document titles and
    categories stored in vector_db_metadata_cache.json.  At query time it
    scores the query tokens against that index to pick the most relevant
    keywords, and does a simple substring scan to detect category filters.
    """

    def __init__(self, metadata_cache_path: str = _DEFAULT_CACHE_PATH):
        if not os.path.exists(metadata_cache_path):
            raise FileNotFoundError(
                f"[FastQueryAnalyzer] Metadata cache not found: {metadata_cache_path}"
            )

        with open(metadata_cache_path, encoding="utf-8") as f:
            meta = json.load(f)

        self.categories: List[str] = meta.get("categories", [])
        self.titles: List[str] = meta.get("titles", [])

        # Build BM25 corpus: one "document" per title + one per category
        corpus_texts = self.titles + self.categories
        tokenized_corpus = [_tokenize(doc) for doc in corpus_texts]

        # Guard against empty corpus
        if not any(tokenized_corpus):
            logger.warning(
                "[FastQueryAnalyzer] Corpus is empty — keywords will fall back to raw query tokens"
            )
            self._bm25 = None
            self._corpus_vocab: set = set()
        else:
            self._bm25 = BM25Okapi(tokenized_corpus)
            # Flat set of all unique tokens that appear in the corpus
            self._corpus_vocab: set = {t for doc in tokenized_corpus for t in doc}

        logger.info(
            f"[FastQueryAnalyzer] Initialized — "
            f"{len(self.titles)} titles, {len(self.categories)} categories, "
            f"vocab size: {len(self._corpus_vocab)}"
        )

    # ── Public API (same as LLMQueryAnalyzer.analyze) ────────────────────────

    def analyze(self, query: str) -> Dict:
        """
        Extract keywords and optional filters from *query* without any LLM call.

        Returns:
            {
                "keywords": List[str],
                "filters":  {"where": {"category": str}} | None
            }
        """
        keywords = self._extract_keywords(query)
        filters = self._detect_filter(query)

        result = {"keywords": keywords, "filters": filters}
        logger.info(
            f"[FastQueryAnalyzer] query='{query}' → keywords={keywords}, filters={filters}"
        )
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_keywords(self, query: str) -> List[str]:
        """Use BM25 to rank query tokens by relevance to the corpus vocabulary."""
        query_tokens = _tokenize(query)

        if not query_tokens:
            return [query]  # fallback: return raw query

        if self._bm25 is None:
            # No corpus — just return the cleaned tokens
            return query_tokens[:_MAX_KEYWORDS]

        # Score each query token individually against the full corpus
        token_scores: Dict[str, float] = {}
        for token in query_tokens:
            scores = self._bm25.get_scores([token])
            token_scores[token] = float(scores.max())

        if not token_scores:
            return query_tokens[:_MAX_KEYWORDS]

        max_score = max(token_scores.values())
        threshold = max_score * _BM25_RELATIVE_THRESHOLD

        # Keep tokens above threshold, sorted by score descending
        selected = sorted(
            [(t, s) for t, s in token_scores.items() if s >= threshold],
            key=lambda x: x[1],
            reverse=True,
        )
        keywords = [t for t, _ in selected[:_MAX_KEYWORDS]]

        # If BM25 filtered everything out (all scores very low), fall back to all tokens
        if not keywords:
            keywords = query_tokens[:_MAX_KEYWORDS]

        return keywords

    def _detect_filter(self, query: str) -> Optional[Dict]:
        """
        Check if a known category name appears in the query (case-insensitive).
        Returns a ChromaDB 'where' filter dict, or None.
        """
        q_lower = query.lower()
        for category in self.categories:
            if category.lower() in q_lower:
                logger.info(
                    f"[FastQueryAnalyzer] Category filter detected: '{category}'"
                )
                return {"where": {"category": category}}
        return None
