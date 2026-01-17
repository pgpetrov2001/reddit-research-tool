from __future__ import annotations
import os
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import voyageai as voi


load_dotenv()

# Config
VOYAGE_AI_API_SECRET = os.getenv("VOYAGE_AI_API_SECRET", "")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-2")

CHAT_AI_VENDOR = os.getenv("CHAT_AI_VENDOR", "deepseek")

if CHAT_AI_VENDOR not in ["xai", "deepseek"]:
    raise RuntimeError(f"Unsupported CHAT_AI_VENDOR: {CHAT_AI_VENDOR}. Supported: xai, deepseek.")

AI_CONFIG = {
    'xai': {
        'api_key': os.getenv("XAI_API_KEY", ""),
        'base_url': os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
        'chat_model': os.getenv("XAI_CHAT_MODEL", "grok-4"),
    },
    'deepseek': {
        'api_key': os.getenv("DEEPSEEK_API_KEY", ""),
        'base_url': os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        'chat_model': os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-reasoner"),
    }
}

AI_API_KEY = AI_CONFIG.get(CHAT_AI_VENDOR, {}).get('api_key')
AI_BASE_URL = AI_CONFIG.get(CHAT_AI_VENDOR, {}).get('base_url')
AI_CHAT_MODEL = AI_CONFIG.get(CHAT_AI_VENDOR, {}).get('chat_model')


def _check_ai_client_configured():
    if not CHAT_AI_VENDOR:
        raise RuntimeError("No CHAT_AI_VENDOR set. Supported: xai, deepseek. Create a .env with CHAT_AI_VENDOR=...")
    if not AI_API_KEY:
        raise RuntimeError(f"No {CHAT_AI_VENDOR.upper()}_API_KEY set. Create a .env with {CHAT_AI_VENDOR.upper()}_API_KEY=...")


def _ai_client() -> OpenAI:
    _check_ai_client_configured()
    return OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)


def _ai_async_client() -> AsyncOpenAI:
    _check_ai_client_configured()
    return AsyncOpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)


def _check_voyage_client_configured():
    if not VOYAGE_AI_API_SECRET:
        raise RuntimeError("VOYAGE_AI_API_SECRET is not set. Create a .env with VOYAGE_AI_API_SECRET=...")


def _voyage_client() -> voi.Client:
    _check_voyage_client_configured()
    return voi.Client(api_key=VOYAGE_AI_API_SECRET)


def _voyage_async_client() -> voi.AsyncClient:
    _check_voyage_client_configured()
    return voi.AsyncClient(api_key=VOYAGE_AI_API_SECRET)


def embed_texts(texts: List[str], batch_size: int) -> np.ndarray:
    client = _voyage_client()
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
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


async def async_embed_query(text: str) -> np.ndarray:
    client = _voyage_async_client()
    resp = await client.embed(texts=[text], model=VOYAGE_MODEL, input_type="query")
    vec = np.array([resp.embeddings[0]], dtype=np.float32)
    norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
    vec = vec / norms
    return vec


def ai_model_query(messages, temperature: float = 0.0) -> Optional[str]:
    if not AI_API_KEY:
        return None
    try:
        client = _ai_client()
        resp = client.chat.completions.create(model=AI_CHAT_MODEL, messages=messages, temperature=temperature)
        return resp.choices[0].message.content
    except Exception:
        return None


async def async_ai_model_query(messages, temperature: float = 0.0) -> Optional[str]:
    if not AI_API_KEY:
        return None
    try:
        client = _ai_async_client()
        resp = await client.chat.completions.create(model=AI_CHAT_MODEL, messages=messages, temperature=temperature)
        return resp.choices[0].message.content
    except Exception:
        return None


def _ai_answer_messages(query: str, context: str) -> List[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a careful assistant. Answer using ONLY the provided context. "
                "If the answer is not in the context, say you don't know. Cite sources as [title](path)."
            )
        },
        {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {query}\nRespond with citations."}
    ]


def maybe_ai_answer(query: str, context: str) -> Optional[str]:
    return ai_model_query(_ai_answer_messages(query, context), temperature=0.0)


async def maybe_async_ai_answer(query: str, context: str) -> Optional[str]:
    return await async_ai_model_query(_ai_answer_messages(query, context), temperature=0.0)


def _ai_keywords_messages(user_query: str) -> List[dict]:
    return [
        {"role": "system", "content": (
            "You generate concise keyword lists for lexical search (BM25). "
            "Given a user query, output only a comma-separated list of search keywords and short phrases. "
            "Prefer canonical terms, synonyms, acronyms, and key entities. No explanations."
        )},
        {"role": "user", "content": user_query}
    ]


def maybe_ai_keywords(user_query: str) -> Optional[str]:
    return ai_model_query(_ai_keywords_messages(user_query), temperature=0.0)


async def maybe_async_ai_keywords(question: str) -> Optional[str]:
    return await async_ai_model_query(_ai_keywords_messages(question), temperature=0.0)


def _ai_topic_messages(question: str) -> List[dict]:
    return [
        {"role": "system", "content": (
            "You are a topic classifier. Given a question, respond with exactly ONE word "
            "that best represents the main topic. Output only the single word, nothing else. "
            "Examples: technology, health, finance, sports, politics, science, entertainment."
        )},
        {"role": "user", "content": question}
    ]


def _parse_topic(topic: Optional[str]) -> Optional[str]:
    if topic is None:
        return None
    topic = topic.strip().lower()
    if " " in topic:
        topic = topic.split()[0]
    return topic


def maybe_ai_topic(question: str) -> Optional[str]:
    """Extract a single-word topic from a question using AI.

    Args:
        question: The question to extract a topic from.

    Returns:
        A single-word topic string, or None if the API key is missing or an error occurs.
    """
    if not question or not question.strip():
        return None
    topic = ai_model_query(_ai_topic_messages(question), temperature=0.0)
    return _parse_topic(topic)


async def maybe_async_ai_topic(question: str) -> Optional[str]:
    """Extract a single-word topic from a question using AI.

    Args:
        question: The question to extract a topic from.

    Returns:
        A single-word topic string, or None if the API key is missing or an error occurs.
    """
    if not question or not question.strip():
        return None
    topic = await async_ai_model_query(_ai_topic_messages(question), temperature=0.0)
    return _parse_topic(topic)
