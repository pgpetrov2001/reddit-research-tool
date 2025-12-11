"""
Reddit Research Tool Server - FastAPI Async Version

ARCHITECTURAL CHANGE: Converted from synchronous http.server to asynchronous FastAPI + Uvicorn

This module provides an ASYNCHRONOUS HTTP server for processing Reddit research queries
using RAG (Retrieval-Augmented Generation). It retrieves relevant posts from specified
subreddits and generates AI-powered answers.

KEY ARCHITECTURAL CHANGES FROM SYNC VERSION:
==============================================

1. **HTTP Framework**: http.server → FastAPI + Uvicorn
   - BaseHTTPRequestHandler → FastAPI route handlers
   - HTTPServer → Uvicorn ASGI server
   - Benefits: Better performance, built-in OpenAPI docs, middleware support

2. **Request Handling**: Synchronous → Async/await
   - All blocking operations now use async/await pattern
   - Multiple Reddit queries can be processed concurrently
   - Non-blocking I/O for file operations and API calls

3. **CORS Configuration**: Manual headers → FastAPI CORSMiddleware
   - Replaced manual header setting with middleware
   - More robust and standards-compliant
   - Easier to configure and maintain

4. **Input Validation**: Manual JSON parsing → Pydantic models
   - Automatic validation of request/response data
   - Type safety and better error messages
   - Auto-generated API documentation

5. **AI API Calls**: Sync OpenAI client → AsyncOpenAI client
   - Non-blocking API calls to xAI/Grok
   - Multiple API calls can run concurrently
   - Better resource utilization

6. **File I/O**: Synchronous file reading → aiofiles (async file I/O)
   - Non-blocking comment file loading
   - Concurrent file reads across multiple subreddits
   - Better performance for large JSONL files

7. **Heartbeat Mechanism**: Threading → FastAPI startup event
   - Cleaner lifecycle management
   - Integrates with FastAPI's event system
   - Automatic cleanup on shutdown

8. **Concurrent Processing**:
   - Old: Sequential processing (one request blocks all others)
   - New: Concurrent request handling (multiple requests processed simultaneously)
   - Dramatically improved throughput under load

Classes:
    ServerConfig: Configuration settings for the server (unchanged)
    QueryRequest: Pydantic model for request validation (NEW)
    QueryResponse: Pydantic model for response validation (NEW)
    Post: Pydantic model for post data (NEW)
    Comment: Pydantic model for comment data (NEW)
    AsyncQueryProcessor: Async version of QueryProcessor (CONVERTED)
    AsyncCommentLoader: Async version of CommentLoader (CONVERTED)
    ResponseFormatter: Formats query results (mostly unchanged)
"""

# ============================================================================
# Imports - Async versions of libraries
# ============================================================================

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware  # NEW: FastAPI CORS middleware
from pydantic import BaseModel, Field, validator  # NEW: For request/response validation
import uvicorn  # NEW: ASGI server for async handling
import asyncio  # NEW: For async operations
from contextlib import asynccontextmanager  # NEW: For lifespan management

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import sys
import os

# Add parent directory to path to import RAG modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from RAG.retrievers import VectorRetriever
from RAG.pipeline import build_context, SYSTEM_PROMPT
from RAG.models import Candidate

# ARCHITECTURAL CHANGE: Import AsyncOpenAI instead of OpenAI for non-blocking API calls
from openai import AsyncOpenAI
import voyageai as voi
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Configuration (Unchanged from sync version)
# ============================================================================

class ServerConfig:
    """Server configuration settings."""

    # Server settings
    DEFAULT_PORT: int = 8080
    HEARTBEAT_INTERVAL: int = 10  # seconds

    # CORS settings - now used by CORSMiddleware
    CORS_ORIGIN: str = 'http://localhost:5173'
    CORS_METHODS: List[str] = ['POST', 'GET', 'OPTIONS']
    CORS_HEADERS: List[str] = ['Content-Type']

    # Query settings
    DEFAULT_TOP_K: int = 10  # Default number of results to return

    # Path settings
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    SUBREDDITS_DIR: Path = PROJECT_ROOT / "SubReddits"

    # API Keys (loaded from .env)
    VOYAGE_AI_API_SECRET: str = os.getenv("VOYAGE_AI_API_SECRET", "")
    VOYAGE_MODEL: str = os.getenv("VOYAGE_MODEL", "voyage-2")
    XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
    XAI_BASE_URL: str = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    XAI_CHAT_MODEL: str = os.getenv("XAI_CHAT_MODEL", "grok-beta")

    @classmethod
    def get_subreddit_directory(cls, subreddit: str) -> str:
        """
        Get the directory path for a subreddit's vector store.
        Uses case-sensitive matching of subreddit names.

        Args:
            subreddit: Name of the subreddit (case-sensitive)

        Returns:
            Absolute path to the subreddit's RAG store directory
        """
        exact_path = cls.SUBREDDITS_DIR / subreddit / f"{subreddit}_RAG_store"
        return str(exact_path)


# ============================================================================
# Pydantic Models - NEW: For request/response validation
# ============================================================================
# ARCHITECTURAL CHANGE: Replace manual JSON parsing with Pydantic models
# Benefits: Type safety, automatic validation, better error messages, OpenAPI docs

class SubredditItem(BaseModel):
    """
    Model for subreddit item in request (supports dict format).
    Allows both string format and object format for backward compatibility.
    """
    name: str


class QuestionItem(BaseModel):
    """
    Model for question item in request (supports dict format).
    Allows both string format and object format for backward compatibility.
    """
    title: Optional[str] = None
    description: Optional[str] = None

    @validator('title', 'description', pre=True)
    def ensure_string(cls, v):
        """Ensure title/description are strings."""
        if v is None:
            return v
        return str(v)


class QueryRequest(BaseModel):
    """
    Request model for POST /query endpoint.

    ARCHITECTURAL CHANGE: Pydantic model replaces manual JSON parsing in do_POST().
    FastAPI automatically validates incoming JSON against this schema.

    Supports flexible input formats:
    - subreddits: List[str] OR List[SubredditItem]
    - question: str OR QuestionItem

    Examples:
        Simple format:
        {
            "subreddits": ["Hairloss", "tressless"],
            "question": "What helps with hair loss?"
        }

        Object format:
        {
            "subreddits": [{"name": "Hairloss"}, {"name": "tressless"}],
            "question": {"title": "What helps with hair loss?", "description": "Additional context"}
        }
    """
    subreddits: List[Union[str, SubredditItem]] = Field(
        ...,
        min_items=1,
        description="List of subreddit names to search (as strings or objects with 'name' field)"
    )
    question: Union[str, QuestionItem] = Field(
        ...,
        description="Question to answer (as string or object with 'title'/'description' fields)"
    )
    is_paid: bool = Field(
        default=True,
        description="Whether the user has paid access (affects AI summary generation)"
    )

    def get_subreddit_names(self) -> List[str]:
        """Extract subreddit names from flexible input format."""
        names = []
        for item in self.subreddits:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, SubredditItem):
                names.append(item.name)
        return names

    def get_question_text(self) -> str:
        """Extract question text from flexible input format."""
        if isinstance(self.question, str):
            return self.question
        elif isinstance(self.question, QuestionItem):
            return self.question.title or self.question.description or ""
        return ""


class Comment(BaseModel):
    """Model for Reddit comment data."""
    id: str
    author: str
    body: str
    score: int
    created_utc: float


class Post(BaseModel):
    """Model for Reddit post data with metadata."""
    title: str
    source: str
    text: str
    score: float
    chunk_id: str
    section: str
    post_id: str
    link: str
    author: str
    comments: List[Comment] = []


class QueryResponse(BaseModel):
    """
    Response model for successful queries.

    ARCHITECTURAL CHANGE: Pydantic model ensures type-safe responses.
    FastAPI automatically serializes this to JSON.
    """
    status: str = "success"
    answer: str
    posts: List[Post]
    total_candidates: int
    top_k: int
    topic: Optional[str] = None


class ErrorResponse(BaseModel):
    """Response model for errors."""
    status: str = "error"
    message: str


# ============================================================================
# Async AI Helper Functions - NEW
# ============================================================================
# ARCHITECTURAL CHANGE: Convert synchronous AI API calls to async
# Old: Blocking OpenAI().chat.completions.create() calls
# New: Non-blocking await AsyncOpenAI().chat.completions.create() calls
# Benefits: Multiple API calls can run concurrently, server remains responsive

async def async_embed_query(text: str) -> Any:
    """
    Asynchronously embed a query using Voyage AI.

    ARCHITECTURAL CHANGE: This should ideally use async Voyage client,
    but since voyageai doesn't provide async client yet, we run it in
    a thread executor to avoid blocking the event loop.

    Args:
        text: Query text to embed

    Returns:
        Numpy array of embeddings
    """
    # Run blocking Voyage API call in thread executor to avoid blocking event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_embed_query, text)


def _sync_embed_query(text: str):
    """Synchronous helper for embed_query (runs in executor)."""
    import numpy as np
    if not ServerConfig.VOYAGE_AI_API_SECRET:
        raise RuntimeError("VOYAGE_AI_API_SECRET is not set")
    client = voi.Client(api_key=ServerConfig.VOYAGE_AI_API_SECRET)
    resp = client.embed(texts=[text], model=ServerConfig.VOYAGE_MODEL, input_type="query")
    vec = np.array([resp.embeddings[0]], dtype=np.float32)
    norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
    vec = vec / norms
    return vec


async def async_xai_answer(system_prompt: str, query: str, context: str) -> Optional[str]:
    """
    Asynchronously generate an AI answer using xAI/Grok.

    ARCHITECTURAL CHANGE: AsyncOpenAI client instead of OpenAI client.
    This is the BIGGEST performance improvement - AI answer generation can take
    2-10 seconds, and now it doesn't block other requests.

    Args:
        system_prompt: System instructions for the AI
        query: User's question
        context: Retrieved context from Reddit posts

    Returns:
        AI-generated answer or None if API key missing
    """
    if not ServerConfig.XAI_API_KEY:
        return None
    try:
        # ARCHITECTURAL CHANGE: AsyncOpenAI with await
        client = AsyncOpenAI(
            api_key=ServerConfig.XAI_API_KEY,
            base_url=ServerConfig.XAI_BASE_URL
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {query}\nRespond with citations."}
        ]
        # await makes this non-blocking
        resp = await client.chat.completions.create(
            model=ServerConfig.XAI_CHAT_MODEL,
            messages=messages,
            temperature=0.0
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"ERROR: xAI answer generation failed: {e}")
        return None


async def async_xai_topic(question: str) -> Optional[str]:
    """
    Asynchronously extract a single-word topic from a question using AI.

    ARCHITECTURAL CHANGE: AsyncOpenAI client for non-blocking topic classification.

    Args:
        question: The question to extract a topic from

    Returns:
        A single-word topic string, or None if API key missing or error occurs
    """
    if not ServerConfig.XAI_API_KEY:
        return None
    if not question or not question.strip():
        return None
    try:
        client = AsyncOpenAI(
            api_key=ServerConfig.XAI_API_KEY,
            base_url=ServerConfig.XAI_BASE_URL
        )
        messages = [
            {"role": "system", "content": (
                "You are a topic classifier. Given a question, respond with exactly ONE word "
                "that best represents the main topic. Output only the single word, nothing else. "
                "Examples: technology, health, finance, sports, politics, science, entertainment."
            )},
            {"role": "user", "content": question}
        ]
        resp = await client.chat.completions.create(
            model=ServerConfig.XAI_CHAT_MODEL,
            messages=messages,
            temperature=0.0
        )
        topic = resp.choices[0].message.content.strip().lower()
        # Ensure we only return a single word
        if " " in topic:
            topic = topic.split()[0]
        return topic
    except Exception as e:
        print(f"ERROR: xAI topic classification failed: {e}")
        return None


# ============================================================================
# Async Query Processing - CONVERTED
# ============================================================================
# ARCHITECTURAL CHANGE: Convert QueryProcessor.process_query() to async
# Benefits: Can process multiple subreddits concurrently, AI calls don't block

class AsyncQueryProcessor:
    """
    Asynchronous query processor.

    ARCHITECTURAL CHANGE: All methods are now async (use 'async def' and 'await').
    Old version processed subreddits sequentially and blocked on AI calls.
    New version can process everything concurrently.
    """

    @staticmethod
    async def process_query(subreddits: List[str], question: str, k: int = ServerConfig.DEFAULT_TOP_K, is_paid: bool = True) -> Dict[str, Any]:
        """
        ASYNC version of process_query - processes query across multiple subreddits.

        ARCHITECTURAL CHANGES:
        1. Method signature: def → async def
        2. VectorRetriever operations: Could be parallelized (currently sequential for safety)
        3. AI calls: Blocking calls → await async calls
        4. Comment loading: Blocking file I/O → await async file I/O

        PERFORMANCE IMPROVEMENT: While this request is waiting for AI answer generation,
        the server can process other incoming requests concurrently.

        Args:
            subreddits: List of subreddit names to search
            question: The user's question
            k: Number of top results to return
            is_paid: Whether the user has paid access (affects AI summary)

        Returns:
            Dictionary containing status, answer, posts, etc.
        """
        candidates: List[Candidate] = []

        # Retrieve candidates from each subreddit
        # NOTE: VectorRetriever operations could be parallelized with asyncio.gather()
        # but kept sequential for now to match original behavior
        for subreddit in subreddits:
            try:
                rag_store_dir = ServerConfig.get_subreddit_directory(subreddit)

                # Check if RAG store directory exists
                if not os.path.exists(rag_store_dir):
                    print(f"WARNING: RAG store for subreddit '{subreddit}' does not exist at {rag_store_dir}. Skipping...")
                    continue

                # Retrieve candidates from vector database
                # ARCHITECTURAL NOTE: VectorRetriever.retrieve() calls embed_query() which
                # is still synchronous. For true async, we'd need to modify the RAG module.
                # For now, we run it in thread executor to avoid blocking.
                loop = asyncio.get_event_loop()
                vec_db = VectorRetriever(rag_store_dir)
                new_candidates = await loop.run_in_executor(
                    None,
                    vec_db.retrieve,
                    question,
                    k
                )
                candidates.extend(new_candidates)
                print(f"Retrieved {len(new_candidates)} candidates from r/{subreddit}")

            except Exception as e:
                print(f"ERROR: Failed to retrieve from r/{subreddit}: {e}")
                continue

        # Validate that we have at least some results
        if not candidates:
            return {
                "status": "error",
                "message": "No valid subreddits found or no results retrieved"
            }

        # Select the top k candidates by score
        best_candidates = sorted(candidates, key=lambda x: -x.score)[:k]

        # DEBUG: Print selected candidates
        print(f"\n{'='*80}")
        print(f"DEBUG: Selected {len(best_candidates)} best candidates")
        print(f"{'='*80}")
        for i, candidate in enumerate(best_candidates, 1):
            print(f"\nCandidate #{i}:")
            print(f"  Score: {candidate.score:.4f}")
            print(f"  Post ID: {candidate.chunk.doc_id}")
            print(f"  Title: {candidate.chunk.title[:100]}...")
            print(f"  Section: {candidate.chunk.section}")
            print(f"  Author: {candidate.chunk.author}")
            print(f"  Text preview: {candidate.chunk.text[:200]}...")
        print(f"{'='*80}\n")

        # ARCHITECTURAL CHANGE: Classify topic asynchronously (non-blocking)
        # Old: topic = maybe_xai_topic(question) - blocks for ~1 second
        # New: topic = await async_xai_topic(question) - doesn't block event loop
        try:
            topic = await async_xai_topic(question)
            if topic:
                print(f"DEBUG: Question topic classified as: {topic}")
            else:
                print(f"DEBUG: Topic classification returned None")
        except Exception as e:
            print(f"WARNING: Topic classification failed: {e}")
            topic = None

        # ARCHITECTURAL CHANGE: Generate AI answer asynchronously (non-blocking)
        # Old: answer = maybe_xai_answer(...) - blocks for 2-10 seconds
        # New: answer = await async_xai_answer(...) - doesn't block event loop
        # This is the MOST CRITICAL change - AI generation is the slowest operation

        # Check if user has paid access
        if not is_paid:
            print("DEBUG: is_paid is False - returning empty AI summary")
            answer = ""
        else:
            context = build_context(best_candidates)

            # DEBUG: Print context being used
            print(f"\n{'='*80}")
            print(f"DEBUG: Context built for AI (length: {len(context)} chars)")
            print(f"{'='*80}")
            print(context[:500] + "..." if len(context) > 500 else context)
            print(f"{'='*80}\n")

            try:
                print("DEBUG: Calling async_xai_answer...")
                answer = await async_xai_answer(SYSTEM_PROMPT, question, context)
                if not answer:
                    print("DEBUG: async_xai_answer returned None, using fallback answer")
                    answer = "AI summary could not be generated at this time. Please check the relevant posts below for information."
                else:
                    print(f"DEBUG: Got answer from AI (length: {len(answer)} chars)")
            except Exception as e:
                print(f"WARNING: AI answer generation raised exception: {e}")
                answer = "AI summary could not be generated at this time. Please check the relevant posts below for information."

            print(f"\nDEBUG: Final answer: {answer[:200]}..." if len(answer) > 200 else f"\nDEBUG: Final answer: {answer}")

        # ARCHITECTURAL CHANGE: Load comments asynchronously (non-blocking file I/O)
        # Old: posts = ResponseFormatter.format_posts(...) - blocks on file reading
        # New: posts = await AsyncResponseFormatter.format_posts(...) - async file I/O
        print(f"\nDEBUG: Loading comments for {len(best_candidates)} posts...")
        posts = await AsyncResponseFormatter.format_posts(best_candidates, subreddits)
        print(f"DEBUG: Formatted {len(posts)} posts")
        for i, post in enumerate(posts, 1):
            print(f"  Post #{i}: {post['title'][:50]}... ({len(post.get('comments', []))} comments)")

        response = {
            "status": "success",
            "answer": answer,
            "posts": posts,
            "total_candidates": len(candidates),
            "top_k": k,
            "topic": topic
        }

        print(f"\nDEBUG: Returning response with status='{response['status']}', {len(response['posts'])} posts, answer length={len(response['answer'])} chars")
        return response


# ============================================================================
# Async Comment Loading - CONVERTED
# ============================================================================
# ARCHITECTURAL CHANGE: Convert file I/O from synchronous to asynchronous
# Old: open() and f.read() block the entire server
# New: aiofiles allows concurrent file reading without blocking

class AsyncCommentLoader:
    """
    Asynchronous comment loader.

    ARCHITECTURAL CHANGE: Uses thread executor for non-blocking file I/O.
    Large comment files (100MB+) can be read without blocking the event loop.
    """

    @staticmethod
    def _sync_load_comments(post_ids: List[str], subreddits: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Synchronous helper to load comments (runs in executor).
        This is faster than aiofiles for large files.
        """
        print(f"DEBUG [CommentLoader]: Starting SYNC load for {len(post_ids)} posts")
        print(f"DEBUG [CommentLoader]: Post IDs: {post_ids}")
        print(f"DEBUG [CommentLoader]: Subreddits: {subreddits}")

        comments_by_post: Dict[str, List[Dict[str, Any]]] = {pid: [] for pid in post_ids}

        for subreddit in subreddits:
            print(f"DEBUG [CommentLoader]: Processing subreddit r/{subreddit}...")
            try:
                # Find the comments file for this subreddit (case-sensitive)
                subreddit_dir = ServerConfig.SUBREDDITS_DIR / subreddit
                print(f"DEBUG [CommentLoader]:   Subreddit dir: {subreddit_dir}")

                if not subreddit_dir.exists():
                    print(f"DEBUG [CommentLoader]:   Directory does not exist, skipping")
                    continue

                # Look for comments file
                comments_file = subreddit_dir / f"{subreddit}.comments.jsonl"
                print(f"DEBUG [CommentLoader]:   Looking for comments file: {comments_file}")

                if not comments_file.exists():
                    print(f"WARNING: Comments file not found for r/{subreddit}")
                    continue

                print(f"DEBUG [CommentLoader]:   Opening file synchronously...")

                # Synchronous file reading (faster for large files than aiofiles)
                with open(comments_file, 'r', encoding='utf-8', errors='ignore') as f:
                    print(f"DEBUG [CommentLoader]:   File opened, reading lines...")
                    line_count = 0
                    # Read file line by line
                    for line in f:
                        line_count += 1
                        if line_count % 10000 == 0:
                            print(f"DEBUG [CommentLoader]:   Processed {line_count} lines...")
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            comment = json.loads(line)
                            # Extract post ID from link_id (format: "t3_<post_id>")
                            link_id = comment.get("link_id", "")
                            if link_id.startswith("t3_"):
                                post_id = link_id[3:]  # Remove "t3_" prefix
                                if post_id in comments_by_post:
                                    # Add relevant comment data
                                    comments_by_post[post_id].append({
                                        "id": comment.get("id", ""),
                                        "author": comment.get("author", ""),
                                        "body": comment.get("body", ""),
                                        "score": comment.get("score", 0),
                                        "created_utc": comment.get("created_utc", 0),
                                        "parent_id": comment.get("parent_id", "")
                                    })
                        except json.JSONDecodeError:
                            continue

                    print(f"DEBUG [CommentLoader]:   Finished reading file. Total lines: {line_count}")

            except Exception as e:
                print(f"ERROR: Failed to load comments from r/{subreddit}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Count total comments loaded
        total_comments = sum(len(comments) for comments in comments_by_post.values())
        print(f"DEBUG [CommentLoader]: Finished loading comments. Total: {total_comments} comments for {len(post_ids)} posts")

        return comments_by_post

    @staticmethod
    async def load_comments_for_posts(post_ids: List[str], subreddits: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        ASYNC version of load_comments_for_posts - loads comments from JSONL files.

        ARCHITECTURAL CHANGE: Run synchronous file reading in thread executor.
        This is faster than aiofiles for large files while still being non-blocking.

        Args:
            post_ids: List of Reddit post IDs to load comments for
            subreddits: List of subreddit names to search in

        Returns:
            Dictionary mapping post_id → list of comment dictionaries
        """
        # Run synchronous comment loading in thread executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            AsyncCommentLoader._sync_load_comments,
            post_ids,
            subreddits
        )

    def build_comment_tree(comments: List[Dict[str, Any]], post_id: str) -> List[Dict[str, Any]]:
        """
        Build a hierarchical comment tree from a flat list of comments.

        Args:
            comments: Flat list of comment dictionaries with parent_id fields
            post_id: The post ID (without t3_ prefix) to identify top-level comments

        Returns:
            List of top-level comments, each with a 'replies' array containing nested comments
        """
        if not comments:
            return []

        # Create a dictionary to quickly look up comments by their ID
        comments_by_id: Dict[str, Dict[str, Any]] = {}
        for comment in comments:
            comment_id = comment.get("id", "")
            if comment_id:
                # Add a 'replies' array to each comment
                comment["replies"] = []
                comments_by_id[comment_id] = comment

        # Build the tree by linking children to parents
        top_level_comments: List[Dict[str, Any]] = []

        for comment in comments:
            parent_id = comment.get("parent_id", "")

            if not parent_id:
                # No parent_id, treat as top-level
                top_level_comments.append(comment)
            elif parent_id.startswith("t3_"):
                # Parent is the post itself (t3_ prefix), so this is a top-level comment
                top_level_comments.append(comment)
            elif parent_id.startswith("t1_"):
                # Parent is another comment (t1_ prefix)
                parent_comment_id = parent_id[3:]  # Remove "t1_" prefix
                parent_comment = comments_by_id.get(parent_comment_id)
                if parent_comment:
                    # Add this comment to the parent's replies
                    parent_comment["replies"].append(comment)
                else:
                    # Parent comment not found (might be deleted or not in our dataset)
                    # Treat as top-level comment
                    top_level_comments.append(comment)
            else:
                # Unknown parent_id format, treat as top-level
                top_level_comments.append(comment)

        # Sort top-level comments by score (descending) then by created_utc (oldest first)
        top_level_comments.sort(key=lambda c: (-c.get("score", 0), c.get("created_utc", 0)))

        # Recursively sort replies within each comment
        def sort_replies(comment: Dict[str, Any]) -> None:
            """Recursively sort replies by score and timestamp."""
            replies = comment.get("replies", [])
            if replies:
                replies.sort(key=lambda c: (-c.get("score", 0), c.get("created_utc", 0)))
                for reply in replies:
                    sort_replies(reply)

        for comment in top_level_comments:
            sort_replies(comment)

        return top_level_comments


# ============================================================================
# Response Formatting - CONVERTED
# ============================================================================
# ARCHITECTURAL CHANGE: Made format_posts async to support async comment loading

class AsyncResponseFormatter:
    """
    Async response formatter (mostly unchanged except format_posts).
    """

    @staticmethod
    def _generate_reddit_link(candidate: Candidate) -> Optional[str]:
        """
        Generate a Reddit link for a post candidate.
        (Unchanged from sync version)
        """
        source = candidate.chunk.source
        post_id = candidate.chunk.doc_id
        section = candidate.chunk.section or ""

        # Extract subreddit from section (format: "r/subreddit" or "r/subreddit [flair]")
        subreddit = None
        if section.startswith("r/"):
            subreddit_part = section[2:]
            for delimiter in [" ", "[", "("]:
                if delimiter in subreddit_part:
                    subreddit_part = subreddit_part.split(delimiter)[0]
            subreddit = subreddit_part.strip()

        # If we have both subreddit and post_id, construct the Reddit URL
        if subreddit and post_id:
            return f"https://www.reddit.com/r/{subreddit}/comments/{post_id}"

        # Try to extract from source (format: "reddit://subreddit/post_id")
        if source and source.startswith("reddit://"):
            parts = source.replace("reddit://", "").split("/")
            if len(parts) >= 2:
                subreddit = parts[0]
                post_id = parts[1]
                return f"https://www.reddit.com/r/{subreddit}/comments/{post_id}"

        return None

    @staticmethod
    async def format_posts(candidates: List[Candidate], subreddits: List[str]) -> List[Dict[str, Any]]:
        """
        ASYNC version of format_posts - formats candidate posts with comments.

        ARCHITECTURAL CHANGE: Made async to support async comment loading.

        Args:
            candidates: List of Candidate objects from retrieval
            subreddits: List of subreddit names to load comments from

        Returns:
            List of dictionaries with post metadata and comments
        """
        # Extract post IDs from candidates
        post_ids = [candidate.chunk.doc_id for candidate in candidates]

        # ARCHITECTURAL CHANGE: Async comment loading
        # Old: comments_by_post = CommentLoader.load_comments_for_posts(...)
        # New: comments_by_post = await AsyncCommentLoader.load_comments_for_posts(...)
        comments_by_post = await AsyncCommentLoader.load_comments_for_posts(post_ids, subreddits)

        # Format posts with hierarchical comments
        posts = []
        for candidate in candidates:
            post_id = candidate.chunk.doc_id

            # Get flat comments and build hierarchical tree
            flat_comments = comments_by_post.get(post_id, [])
            hierarchical_comments = AsyncCommentLoader.build_comment_tree(flat_comments, post_id)

            # Generate Reddit link from post ID and subreddit info
            reddit_link = AsyncResponseFormatter._generate_reddit_link(candidate)

            posts.append({
                "title": candidate.chunk.title,
                "source": candidate.chunk.source,
                "text": candidate.chunk.text,
                "score": candidate.score,
                "chunk_id": candidate.chunk.id,
                "section": candidate.chunk.section,
                "post_id": post_id,
                "link": reddit_link,
                "author": candidate.chunk.author,
                "comments": hierarchical_comments
            })
        return posts

    @staticmethod
    def print_query_results(response: Dict[str, Any]) -> None:
        """
        Print query results to console in a readable format.
        (Unchanged from sync version)
        """
        if response["status"] != "success":
            return

        # Print AI Summary
        print("\n" + "="*80)
        print("AI SUMMARY".center(80))
        print("="*80)
        print(response["answer"])
        print("="*80)

        # Print statistics
        print(f"\nTotal candidates retrieved: {response.get('total_candidates', 'N/A')}")
        print(f"Top posts displayed: {response.get('top_k', len(response['posts']))}")
        if response.get('topic'):
            print(f"Question topic: {response['topic']}")

        # Print posts
        print("\n" + "="*80)
        print("TOP RELEVANT POSTS".center(80))
        print("="*80)

        for i, post in enumerate(response["posts"], 1):
            print(f"\n{'─'*80}")
            print(f"POST #{i}")
            print(f"{'─'*80}")
            print(f"Title:      {post['title']}")
            print(f"Source:     {post['source']}")
            print(f"Score:      {post['score']:.4f}")
            print(f"Section:    {post.get('section', 'N/A')}")
            print(f"Chunk ID:   {post.get('chunk_id', 'N/A')}")
            print(f"Post ID:    {post.get('post_id', 'N/A')}")
            print(f"Link:       {post.get('link', 'N/A')}")
            print(f"Comments:   {len(post.get('comments', []))} comments")
            print(f"\nFull Text:\n{'-'*80}")
            print(post['text'])
            print(f"{'-'*80}")

            # Print comments if available
            comments = post.get('comments', [])
            if comments:
                print(f"\nComments ({len(comments)}):")
                print(f"{'─'*80}")
                for j, comment in enumerate(comments[:5], 1):
                    print(f"\n  Comment #{j} by {comment.get('author', 'Unknown')} (score: {comment.get('score', 0)})")
                    print(f"  {'-'*76}")
                    comment_body = comment.get('body', '')
                    if len(comment_body) > 200:
                        comment_body = comment_body[:200] + "..."
                    print(f"  {comment_body}")
                if len(comments) > 5:
                    print(f"\n  ... and {len(comments) - 5} more comments")
                print(f"{'─'*80}")

        print("\n" + "="*80 + "\n")


# ============================================================================
# FastAPI Application Setup - NEW
# ============================================================================
# ARCHITECTURAL CHANGE: Replaced HTTPServer + BaseHTTPRequestHandler with FastAPI
# Benefits: Better routing, automatic docs, middleware, dependency injection

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    ARCHITECTURAL CHANGE: FastAPI lifespan for startup/shutdown events.
    Replaces threading.Thread(target=heartbeat) with cleaner async approach.

    This runs:
    - Before startup: Initialize resources
    - After startup: Start background tasks (heartbeat)
    - On shutdown: Cleanup resources
    """
    # Startup
    print(f"\n{'='*80}")
    print("FASTAPI ASYNC SERVER STARTING".center(80))
    print(f"{'='*80}\n")
    print(f"Reddit Data Server running on port {ServerConfig.DEFAULT_PORT}...")
    print("Waiting for subreddit requests from the client...\n")

    # Start heartbeat background task
    heartbeat_task = asyncio.create_task(heartbeat())

    yield  # Server is running

    # Shutdown
    heartbeat_task.cancel()
    print("\nShutting down server...")

async def heartbeat():
    """
    ARCHITECTURAL CHANGE: Async heartbeat instead of threaded heartbeat.
    Old: threading.Thread with time.sleep() (blocking)
    New: asyncio task with asyncio.sleep() (non-blocking)

    Prints periodic heartbeat messages to console without blocking the event loop.
    """
    while True:
        print("listening...")
        await asyncio.sleep(ServerConfig.HEARTBEAT_INTERVAL)


# Create FastAPI app instance
# ARCHITECTURAL CHANGE: FastAPI app replaces HTTPServer
app = FastAPI(
    title="Reddit Research Tool API",
    description="Async API for processing Reddit research queries using RAG",
    version="2.0.0-async",
    lifespan=lifespan
)

# ARCHITECTURAL CHANGE: CORSMiddleware replaces manual _set_cors_headers()
# Benefits: More robust, standards-compliant, handles preflight automatically
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ServerConfig.CORS_ORIGIN],  # Frontend origin
    allow_credentials=True,
    allow_methods=ServerConfig.CORS_METHODS,
    allow_headers=ServerConfig.CORS_HEADERS,
)


# ============================================================================
# API Endpoints - NEW
# ============================================================================
# ARCHITECTURAL CHANGE: FastAPI route decorators replace do_POST(), do_OPTIONS()
# Benefits: Cleaner code, automatic OpenAPI docs, better error handling


@app.get("/")
async def root():
    """
    Health check endpoint.

    ARCHITECTURAL CHANGE: FastAPI automatically handles routing and responses.
    No need for manual send_response(), send_header(), wfile.write().
    """
    return {
        "message": "Reddit Research Tool API",
        "status": "running",
        "version": "2.0.0-async"
    }


@app.get("/favicon.ico")
async def favicon():
    """Return empty response for favicon to prevent 404 logs."""
    from fastapi.responses import Response
    return Response(status_code=204)


@app.post("/")
async def root_query(request: QueryRequest):
    """
    Process query at root endpoint (for backward compatibility).
    This is the same as /query endpoint.
    """
    return await query(request)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a Reddit research query across multiple subreddits.

    ARCHITECTURAL CHANGES:
    1. Decorator-based routing: @app.post() replaces do_POST()
    2. Pydantic validation: FastAPI automatically validates request against QueryRequest model
    3. Async handler: 'async def' allows non-blocking processing
    4. Auto serialization: FastAPI automatically serializes QueryResponse to JSON
    5. Error handling: HTTPException replaces manual error responses

    OLD FLOW (sync):
    - do_POST() called
    - Manual content_length reading
    - Manual JSON parsing
    - Manual validation with if statements
    - Manual error responses
    - Blocking query processing
    - Manual JSON serialization

    NEW FLOW (async):
    - FastAPI routes to query()
    - Automatic request body parsing
    - Automatic Pydantic validation
    - Automatic 422 errors for invalid input
    - Non-blocking query processing
    - Automatic response serialization

    Args:
        request: QueryRequest object (automatically parsed and validated by FastAPI)

    Returns:
        QueryResponse object (automatically serialized to JSON by FastAPI)

    Raises:
        HTTPException: If query processing fails
    """
    print("\n" + "="*80)
    print("DEBUG: /query endpoint called")
    print("="*80)

    try:
        # Extract and log request data
        subreddits = request.get_subreddit_names()
        question = request.get_question_text()

        print("\n--- Request Received ---")
        print(f"Subreddits: {subreddits}")
        print(f"Question: {question}")
        print(f"is_paid: {request.is_paid}")
        print("------------------------\n")

        # Validate inputs (FastAPI already ensures non-empty, but let's double-check)
        if not subreddits:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No subreddits provided"
            )

        if not question:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No question provided"
            )

        # Log query details
        print(f"\nProcessing query: '{question}'")
        print(f"Subreddits: {', '.join(subreddits)}\n")

        # ARCHITECTURAL CHANGE: Async query processing
        # Old: response = QueryProcessor.process_query(...) - blocks server
        # New: response = await AsyncQueryProcessor.process_query(...) - doesn't block
        # While this request waits for AI/embeddings, other requests can be processed
        response = await AsyncQueryProcessor.process_query(
            subreddits=subreddits,
            question=question,
            k=ServerConfig.DEFAULT_TOP_K,
            is_paid=request.is_paid
        )

        # Check if query processing returned an error
        if response.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response.get("message", "Query processing failed")
            )

        # Print results to console for monitoring (same as sync version)
        AsyncResponseFormatter.print_query_results(response)

        # FastAPI automatically serializes the response to JSON
        print("\n" + "="*80)
        print("DEBUG: Sending response back to client")
        print(f"  Status: {response.get('status')}")
        print(f"  Answer length: {len(response.get('answer', ''))} chars")
        print(f"  Number of posts: {len(response.get('posts', []))}")
        print(f"  Total candidates: {response.get('total_candidates')}")
        print(f"  Topic: {response.get('topic')}")
        print("="*80 + "\n")

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# Main Entry Point - NEW
# ============================================================================
# ARCHITECTURAL CHANGE: uvicorn.run() replaces HTTPServer.serve_forever()
# Uvicorn is a high-performance ASGI server that supports async/await

def run(port: int = ServerConfig.DEFAULT_PORT, host: str = "0.0.0.0") -> None:
    """
    Run the FastAPI Reddit data server with Uvicorn.

    ARCHITECTURAL CHANGE: Uvicorn ASGI server replaces http.server HTTPServer

    OLD (sync):
        server = HTTPServer(('', port), RedditDataRequestHandler)
        server.serve_forever()

    NEW (async):
        uvicorn.run(app, host=host, port=port)

    Benefits of Uvicorn:
    - ASGI support (async/await)
    - Better performance (concurrent request handling)
    - Production-ready features (graceful shutdown, signal handling)
    - Hot reload in development
    - Worker processes for multi-core scaling

    Args:
        port: Port number to run the server on (default: 8080)
        host: Host address to bind to (default: "0.0.0.0" for all interfaces)

    Example:
        >>> run(8080)  # Start async server on port 8080
    """
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        # For production, you can add:
        # workers=4,  # Multiple worker processes for better CPU utilization
        # reload=False,  # Disable auto-reload in production
    )


if __name__ == "__main__":
    """
    ARCHITECTURAL CHANGE: Same entry point, different server architecture.

    To run:
        python server.py

    The server will:
    1. Load environment variables from .env
    2. Initialize FastAPI app with CORS middleware
    3. Start Uvicorn ASGI server on port 8080
    4. Begin heartbeat background task
    5. Listen for async POST /query requests
    6. Process multiple requests concurrently
    """
    # Start the async server
    run()
