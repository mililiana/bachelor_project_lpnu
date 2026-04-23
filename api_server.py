"""
Flask API Server for LPNU Assistant Chrome Extension
Exposes a /ask endpoint at http://localhost:5000/ask

Usage:
    python api_server.py

The Chrome extension POSTs to /ask with:
    { "user_query": "<question>" }

And receives back:
    { "answer": "<response>" }
"""

import sys
import json
import time
import os
import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# ── Path setup ─────────────────────────────────────────────────────────────────
# Add the technique_1_basic directory to sys.path so its local imports work
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TECHNIQUE_DIR = os.path.join(BASE_DIR, "src", "technique_1_basic")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")

sys.path.insert(0, TECHNIQUE_DIR)

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ── RAG system import (done after path setup) ──────────────────────────────────
from complete_rag_system import CompleteRAGSystem

# ── Flask app setup ────────────────────────────────────────────────────────────
app = Flask(__name__)

# Allow all origins so the Chrome extension injected into any page can reach us.
# To restrict, replace "*" with the university domain, e.g.:
#   origins=["https://lpnu.ua", "https://www.lpnu.ua"]
CORS(app, resources={r"/*": {"origins": "*"}})

# ── RAG system (lazy-loaded once on first request) ─────────────────────────────
_rag: CompleteRAGSystem | None = None

# ── Persistent cache (survives restarts) ─────────────────────────────────
CACHE_FILE = os.path.join(BASE_DIR, "cache.json")


def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            print(f"Cache loaded: {len(data)} entries from {CACHE_FILE}")
            return data
        except Exception as e:
            print(f"Could not load cache: {e}")
    return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Could not save cache: {e}")


from typing import Any
_cache: dict[str, Any] = _load_cache()


def get_rag() -> CompleteRAGSystem:
    global _rag
    if _rag is None:
        use_lapa = getattr(get_rag, '_use_lapa', False)
        backend = "LapaLLM (local GPU)" if use_lapa else "Gemini API"
        print(f"Initializing RAG system with {backend} backend…")
        _rag = CompleteRAGSystem(
            db_path=VECTOR_DB_PATH,
            collection_name="hybrid_collection",
            use_lapa=use_lapa,
        )
        print("RAG system ready.")
    return _rag


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.route("/health", methods=["GET"])
def health():
    """Simple liveness check — open http://localhost:5001/health in the browser."""
    return jsonify(
        {"status": "ok", "service": "LPNU Assistant API", "cache_size": len(_cache)}
    )


@app.route("/cache", methods=["GET"])
def cache_info():
    """Show all cached queries."""
    return jsonify({"cached_queries": list(_cache.keys()), "total": len(_cache)})


@app.route("/cache", methods=["DELETE"])
def cache_clear():
    """Clear the in-memory and on-disk cache."""
    _cache.clear()
    _save_cache(_cache)
    return jsonify({"status": "cache cleared"})


@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    """
    Expects JSON body: { "user_query": "..." }
    Returns JSON:      { "answer": "...", "sources": [...] }
    """
    if request.method == "OPTIONS":
        resp = jsonify({})
        resp.headers.add("Access-Control-Allow-Origin", "*")
        resp.headers.add("Access-Control-Allow-Headers", "Content-Type")
        resp.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return resp, 204

    data = request.get_json(silent=True)
    if not data or not data.get("user_query", "").strip():
        resp = jsonify({"error": "Missing or empty 'user_query' field."})
        resp.headers.add("Access-Control-Allow-Origin", "*")
        return resp, 400

    user_query = data["user_query"].strip()
    cache_key = user_query.lower()
    print(f"\n[/ask] Query received: {user_query!r}")

    # Return cached answer if available (small delay so the UI typing indicator shows)
    if cache_key in _cache:
        print(f"[/ask] Cache hit! Waiting 800 ms for typing effect…")
        time.sleep(0.8)
        cached_result = _cache[cache_key]
        if isinstance(cached_result, str):
            resp = jsonify({"answer": cached_result})
        else:
            resp = jsonify(cached_result)
        resp.headers.add("Access-Control-Allow-Origin", "*")
        return resp

    try:
        rag = get_rag()
        result = rag.query(user_query, return_sources=True)
        answer = result.get("answer", "Відповідь недоступна.")
        
        sources = []
        if "sources" in result:
            for s in result["sources"]:
                # Default to '#' if no URL is present
                sources.append({
                    "title": s.get("title", "Без назви"),
                    "url": s.get("url", "#")
                })
        
        response_data = {"answer": answer, "sources": sources}
        _cache[cache_key] = response_data  # store in memory and on disk
        _save_cache(_cache)
        print(
            f"[/ask] Answer generated ({len(answer)} chars) — cached ({len(_cache)} total)"
        )
        resp = jsonify(response_data)
        resp.headers.add("Access-Control-Allow-Origin", "*")
        return resp

    except Exception as exc:
        print(f"[/ask] ERROR: {exc}")
        return jsonify({"error": "Internal server error.", "detail": str(exc)}), 500


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPNU Assistant API Server")
    parser.add_argument(
        "--use-lapa",
        action="store_true",
        default=False,
        help="Use local LapaLLM model instead of Gemini API (requires GPU with CUDA)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to run the server on (default: 5001)",
    )
    args = parser.parse_args()

    # Store the flag so get_rag() can read it on first call
    get_rag._use_lapa = args.use_lapa

    backend = "LapaLLM (local GPU)" if args.use_lapa else "Gemini API"
    print("=" * 60)
    print("  LPNU Assistant API Server")
    print(f"  LLM Backend: {backend}")
    print(f"  Listening on http://localhost:{args.port}")
    print(f"  Health check: http://localhost:{args.port}/health")
    print("=" * 60)
    app.run(host="0.0.0.0", port=args.port, debug=False)

