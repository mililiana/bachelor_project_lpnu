"""
lapa_generator.py
=================
Drop-in replacement for Gemini answer generation (Stage 3) using
the local LapaLLM model (lapa-llm/lapa-v0.1.2-instruct).

Runs on Apple Silicon via MPS, falls back to CPU automatically.
No API key or internet connection needed after the model is downloaded (~8 GB).

Usage:
    from lapa_generator import LapaGenerator
    gen = LapaGenerator()                  # downloads model on first run
    answer = gen.generate_answer(query, retrieved_docs)
"""

from __future__ import annotations

from typing import Dict, List

import torch
from loguru import logger
from transformers import pipeline

_MODEL_ID = "lapa-llm/lapa-v0.1.2-instruct"


# Pick best available device: MPS (Apple Silicon) > CUDA > CPU
def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LapaGenerator:
    """
    Wraps the LapaLLM pipeline for text-only answer generation.
    Same interface as the Gemini-based generate_answer() in CompleteRAGSystem.
    """

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        max_new_tokens: int = 512,
    ):
        device = _best_device()
        logger.info(f"[LapaGenerator] Loading '{model_id}' on device='{device}' ...")

        # Use bfloat16 on MPS/CUDA; float32 on CPU
        dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32

        self._pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            device=device,
            torch_dtype=dtype,
        )
        self._max_new_tokens = max_new_tokens
        logger.info("[LapaGenerator] Model loaded and ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_answer(self, query: str, retrieved_documents: List[Dict]) -> str:
        """
        Generate a Ukrainian answer from the retrieved documents.
        Drop-in replacement for CompleteRAGSystem.generate_answer().
        """
        if not retrieved_documents:
            return (
                "Вибачте, не вдалося знайти відповідну інформацію в базі даних університету "
                "для вашого запитання. Спробуйте переформулювати запитання або зверніться "
                "до інформаційної служби університету."
            )

        # Build context string (same format as Gemini version)
        contexts = []
        for i, doc in enumerate(retrieved_documents, 1):
            title = doc.get("title", "Без назви")
            content = doc.get("content", "")
            score = doc.get("combined_score", 0)
            contexts.append(
                f"Документ {i} ({title}, релевантність: {score:.3f}):\n{content}"
            )
        context_text = "\n\n---\n\n".join(contexts)
        num_docs = len(retrieved_documents)

        system_prompt = (
            "Ти - асистент для університетської інформаційної системи "
            "Національного університету «Львівська політехніка».\n"
            "Відповідай ТІЛЬКИ на основі наданих контекстів. "
            "Використовуй конкретні факти, числа, адреси, назви точно як вони зазначені в контекстах. "
            "Відповідай українською мовою, чітко і лаконічно, без зайвих символів (* або **)."
        )

        user_content = (
            f"Знайдено релевантних документів: {num_docs}\n\n"
            f"Контексти:\n{context_text}\n\n"
            f"Запитання: {query}\n\n"
            "Відповідь:"
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_content}],
            },
        ]

        logger.info(f"[LapaGenerator] Generating answer for query: '{query}'")
        try:
            output = self._pipe(
                text=messages,
                max_new_tokens=self._max_new_tokens,
            )
            answer = output[0]["generated_text"][-1]["content"].strip()
            logger.info(f"[LapaGenerator] Answer generated ({len(answer)} chars)")
            return answer
        except Exception as e:
            logger.error(f"[LapaGenerator] Error during generation: {e}")
            return "Вибачте, сталася помилка при генерації відповіді."
