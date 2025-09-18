from __future__ import annotations
import os
from typing import Dict, List, Tuple

from .models import Candidate
from .retrievers import VectorRetriever, KeywordRetriever
from .ai import maybe_xai_answer


MAX_CONTEXT_WORDS = int(os.getenv("RAG_MAX_CTX_WORDS", 100000))


def trim_by_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


SYSTEM_PROMPT = (
    "You are a careful assistant. Answer using ONLY the provided context. "
    "If the answer is not in the context, say you don't know. Cite sources as [title](path)."
)


def build_context(cands: List[Candidate], max_words: int = MAX_CONTEXT_WORDS) -> str:
    by_doc = {}
    for c in cands:
        key = (c.chunk.title, c.chunk.source)
        by_doc.setdefault(key, []).append(c)
    blocks = []
    for (title, src), items in by_doc.items():
        blocks.append(f"### {title} ({src})\n" + "\n\n".join([i.chunk.text for i in items]))
    ctx = "\n\n".join(blocks)
    return trim_by_words(ctx, max_words)


class Pipeline:
    def __init__(self, store_dir: str):
        self.vec = VectorRetriever(store_dir)
        self.kw = KeywordRetriever(store_dir)

    def ask(self, query: str, k: int = 10, mode: str = "embed", action: str = "ask") -> Dict:
        mode = (mode or "embed").lower()
        action = (action or "ask").lower()  # 'retrieve' or 'ask'
        result: Dict[str, object] = { 'meta': {'k': k, 'mode': mode, 'action': action} }

        def build_result(cands: List[Candidate]) -> Dict[str, object]:
            if action == 'retrieve':
                ans = "(retrieve-only mode)"
            else:
                ctx = build_context(cands)
                llm = maybe_xai_answer(SYSTEM_PROMPT, query, ctx)
                ans = llm if llm else "I don't have a generative model configured."
            cites = [{
                'title': c.chunk.title,
                'source': c.chunk.source,
                'section': c.chunk.section,
                'chunk_id': c.chunk.id
            } for c in cands]
            return {'answer': ans, 'citations': cites}

        if mode == 'embed':
            cands = self.vec.retrieve(query, topk=k)
            result.update(build_result(cands))
            return result
        elif mode == 'keyword':
            cands, expanded = self.kw.retrieve_with_expanded_keywords(query, topk=k)
            result['expanded_keywords'] = expanded
            result.update(build_result(cands))
            return result
        elif mode == 'both':
            c_embed = self.vec.retrieve(query, topk=k)
            c_kw, expanded = self.kw.retrieve_with_expanded_keywords(query, topk=k)
            result['embed'] = build_result(c_embed)
            kw_res = build_result(c_kw)
            kw_res['expanded_keywords'] = expanded
            result['keyword'] = kw_res
            return result
        else:
            cands = self.vec.retrieve(query, topk=k)
            result.update(build_result(cands))
            return result


