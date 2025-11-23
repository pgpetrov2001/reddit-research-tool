from __future__ import annotations
import os
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import voyageai as voi


load_dotenv()

# Config
VOYAGE_AI_API_SECRET = os.getenv("VOYAGE_AI_API_SECRET", "")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-2")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_CHAT_MODEL = os.getenv("XAI_CHAT_MODEL", "grok-4")


def _xai_client() -> OpenAI:
    if not XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY is not set. Create a .env with XAI_API_KEY=...")
    return OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL)


def _voyage_client() -> voi.Client:
    if not VOYAGE_AI_API_SECRET:
        raise RuntimeError("VOYAGE_AI_API_SECRET is not set. Create a .env with VOYAGE_AI_API_SECRET=...")
    return voi.Client(api_key=VOYAGE_AI_API_SECRET)


def embed_texts(texts: List[str], batch_size: int) -> np.ndarray:
    client = _voyage_client()
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embed(texts=batch, model=VOYAGE_MODEL, input_type="document")
        for vec in resp.embeddings:
            all_vecs.append(vec)
    X = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    return X


def embed_query(text: str) -> np.ndarray:
    client = _voyage_client()
    resp = client.embed(texts=[text], model=VOYAGE_MODEL, input_type="query")
    vec = np.array([resp.embeddings[0]], dtype=np.float32)
    norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
    vec = vec / norms
    return vec


def maybe_xai_answer(system_prompt: str, query: str, context: str) -> Optional[str]:
    if not XAI_API_KEY:
        return None
    try:
        client = _xai_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {query}\nRespond with citations."}
        ]
        resp = client.chat.completions.create(model=XAI_CHAT_MODEL, messages=messages, temperature=0.0)
        return resp.choices[0].message.content
    except Exception:
        return None


def maybe_xai_keywords(user_query: str) -> Optional[str]:
    if not XAI_API_KEY:
        return None
    try:
        client = _xai_client()
        messages = [
            {"role": "system", "content": (
                "You generate concise keyword lists for lexical search (BM25). "
                "Given a user query, output only a comma-separated list of search keywords and short phrases. "
                "Prefer canonical terms, synonyms, acronyms, and key entities. No explanations."
            )},
            {"role": "user", "content": user_query}
        ]
        resp = client.chat.completions.create(model=XAI_CHAT_MODEL, messages=messages, temperature=0.0)
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def maybe_xai_topic(question: str) -> Optional[str]:
    """Extract a single-word topic from a question using AI.

    Args:
        question: The question to extract a topic from.

    Returns:
        A single-word topic string, or None if the API key is missing or an error occurs.
    """
    if not XAI_API_KEY:
        return None
    if not question or not question.strip():
        return None
    try:
        client = _xai_client()
        messages = [
            {"role": "system", "content": (
                "You are a topic classifier. Given a question, respond with exactly ONE word "
                "that best represents the main topic. Output only the single word, nothing else. "
                "Examples: technology, health, finance, sports, politics, science, entertainment."
            )},
            {"role": "user", "content": question}
        ]
        resp = client.chat.completions.create(model=XAI_CHAT_MODEL, messages=messages, temperature=0.0)
        topic = resp.choices[0].message.content.strip().lower()
        # Ensure we only return a single word
        if " " in topic:
            topic = topic.split()[0]
        return topic
    except Exception:
        return None


