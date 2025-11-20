"""
Reddit Research Tool Server

This module provides an HTTP server for processing Reddit research queries using RAG
(Retrieval-Augmented Generation). It retrieves relevant posts from specified subreddits
and generates AI-powered answers.

Classes:
    ServerConfig: Configuration settings for the server
    QueryProcessor: Handles query processing and retrieval logic
    ResponseFormatter: Formats query results for HTTP responses
    RedditDataRequestHandler: HTTP request handler
    RedditDataServer: Main HTTP server class
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import os

# Add parent directory to path to import RAG modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from RAG.retrievers import VectorRetriever
from RAG.pipeline import build_context, SYSTEM_PROMPT
from RAG.ai import maybe_xai_answer
from RAG.models import Candidate


# ============================================================================
# Configuration
# ============================================================================

class ServerConfig:
    """Server configuration settings."""

    # Server settings
    DEFAULT_PORT: int = 8080
    HEARTBEAT_INTERVAL: int = 10  # seconds

    # CORS settings
    CORS_ORIGIN: str = 'http://localhost:5173'
    CORS_METHODS: str = 'POST, GET, OPTIONS'
    CORS_HEADERS: str = 'Content-Type'

    # Query settings
    DEFAULT_TOP_K: int = 10  # Default number of results to return

    # Path settings
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    SUBREDDITS_DIR: Path = PROJECT_ROOT / "SubReddits"

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
        # Use exact case-sensitive match only
        exact_path = cls.SUBREDDITS_DIR / subreddit / f"{subreddit}_RAG_store"
        return str(exact_path)


# ============================================================================
# Query Processing
# ============================================================================

class QueryProcessor:
    """Handles query processing across multiple subreddits."""

    @staticmethod
    def process_query(subreddits: List[str], question: str, k: int = ServerConfig.DEFAULT_TOP_K) -> Dict[str, Any]:
        """
        Process a query across multiple subreddits using vector retrieval and AI generation.

        This method:
        1. Retrieves relevant candidates from each subreddit's vector store
        2. Sorts candidates by relevance score
        3. Generates an AI-powered answer based on the top candidates

        Args:
            subreddits: List of subreddit names to search
            question: The user's question
            k: Number of top results to return (default: ServerConfig.DEFAULT_TOP_K)

        Returns:
            Dictionary containing:
                - status: "success" or "error"
                - answer: AI-generated answer (if successful)
                - posts: List of relevant posts with metadata (if successful)
                - total_candidates: Total number of candidates retrieved (if successful)
                - top_k: Number of top results returned (if successful)
                - message: Error message (if error)
        """
        candidates: List[Candidate] = []

        # Retrieve candidates from each subreddit
        for subreddit in subreddits:
            try:
                rag_store_dir = ServerConfig.get_subreddit_directory(subreddit)

                # Check if RAG store directory exists
                if not os.path.exists(rag_store_dir):
                    print(f"WARNING: RAG store for subreddit '{subreddit}' does not exist at {rag_store_dir}. Skipping...")
                    continue

                # Retrieve candidates from vector database
                vec_db = VectorRetriever(rag_store_dir)
                new_candidates = vec_db.retrieve(question, topk=k)
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

        # Generate AI answer using the retrieved context
        context = build_context(best_candidates)
        answer = maybe_xai_answer(SYSTEM_PROMPT, question, context)

        if not answer:
            answer = "AI answer generation is not configured (missing XAI_API_KEY)"

        # Format the response with comments
        posts = ResponseFormatter.format_posts(best_candidates, subreddits)

        return {
            "status": "success",
            "answer": answer,
            "posts": posts,
            "total_candidates": len(candidates),
            "top_k": k
        }


# ============================================================================
# Comment Loading
# ============================================================================

class CommentLoader:
    """Handles loading and filtering comments from JSONL files."""

    @staticmethod
    def load_comments_for_posts(post_ids: List[str], subreddits: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load comments for specific post IDs from subreddit comment files.

        Args:
            post_ids: List of Reddit post IDs to load comments for
            subreddits: List of subreddit names to search in

        Returns:
            Dictionary mapping post_id -> list of comment dictionaries
        """
        comments_by_post: Dict[str, List[Dict[str, Any]]] = {pid: [] for pid in post_ids}

        for subreddit in subreddits:
            try:
                # Find the comments file for this subreddit (case-sensitive)
                subreddit_dir = ServerConfig.SUBREDDITS_DIR / subreddit
                if not subreddit_dir.exists():
                    continue

                # Look for comments file
                comments_file = subreddit_dir / f"{subreddit}.comments.jsonl"
                if not comments_file.exists():
                    print(f"WARNING: Comments file not found for r/{subreddit}")
                    continue

                # Read and filter comments
                with open(comments_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
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
                                        "created_utc": comment.get("created_utc", 0)
                                    })
                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                print(f"ERROR: Failed to load comments from r/{subreddit}: {e}")
                continue

        return comments_by_post


# ============================================================================
# Response Formatting
# ============================================================================

class ResponseFormatter:
    """Formats query results for HTTP responses."""

    @staticmethod
    def format_posts(candidates: List[Candidate], subreddits: List[str]) -> List[Dict[str, Any]]:
        """
        Format candidate posts for JSON response, including comments.

        Args:
            candidates: List of Candidate objects from retrieval
            subreddits: List of subreddit names to load comments from

        Returns:
            List of dictionaries with post metadata and comments
        """
        # Extract post IDs from candidates
        post_ids = [candidate.chunk.doc_id for candidate in candidates]

        # Load comments for all posts
        comments_by_post = CommentLoader.load_comments_for_posts(post_ids, subreddits)

        # Format posts with comments
        posts = []
        for candidate in candidates:
            post_id = candidate.chunk.doc_id
            posts.append({
                "title": candidate.chunk.title,
                "source": candidate.chunk.source,
                "text": candidate.chunk.text,
                "score": candidate.score,
                "chunk_id": candidate.chunk.id,
                "section": candidate.chunk.section,
                "post_id": post_id,
                "comments": comments_by_post.get(post_id, [])
            })
        return posts

    @staticmethod
    def create_error_response(message: str, status_code: int = 400) -> tuple[Dict[str, str], int]:
        """
        Create a standardized error response.

        Args:
            message: Error message to return
            status_code: HTTP status code (default: 400)

        Returns:
            Tuple of (response_dict, status_code)
        """
        return {"status": "error", "message": message}, status_code

    @staticmethod
    def print_query_results(response: Dict[str, Any]) -> None:
        """
        Print query results to console in a readable format.

        Args:
            response: The response dictionary from process_query
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
            print(f"Comments:   {len(post.get('comments', []))} comments")
            print(f"\nFull Text:\n{'-'*80}")
            print(post['text'])
            print(f"{'-'*80}")

            # Print comments if available
            comments = post.get('comments', [])
            if comments:
                print(f"\nComments ({len(comments)}):")
                print(f"{'─'*80}")
                for j, comment in enumerate(comments[:5], 1):  # Show first 5 comments
                    print(f"\n  Comment #{j} by {comment.get('author', 'Unknown')} (score: {comment.get('score', 0)})")
                    print(f"  {'-'*76}")
                    comment_body = comment.get('body', '')
                    # Truncate long comments for readability
                    if len(comment_body) > 200:
                        comment_body = comment_body[:200] + "..."
                    print(f"  {comment_body}")
                if len(comments) > 5:
                    print(f"\n  ... and {len(comments) - 5} more comments")
                print(f"{'─'*80}")

        print("\n" + "="*80 + "\n")


# ============================================================================
# HTTP Request Handler
# ============================================================================

class RedditDataRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for processing Reddit research queries.

    Handles:
        - OPTIONS: CORS preflight requests
        - POST: Query processing requests with subreddit and question data
    """

    def _set_cors_headers(self) -> None:
        """
        Add CORS (Cross-Origin Resource Sharing) headers to the response.

        This allows the frontend application to make requests to this server
        from a different origin (localhost:5173 by default).
        """
        self.send_header('Access-Control-Allow-Origin', ServerConfig.CORS_ORIGIN)
        self.send_header('Access-Control-Allow-Methods', ServerConfig.CORS_METHODS)
        self.send_header('Access-Control-Allow-Headers', ServerConfig.CORS_HEADERS)

    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200) -> None:
        """
        Send a JSON response to the client.

        Args:
            data: Dictionary to serialize as JSON
            status_code: HTTP status code (default: 200)
        """
        self.send_response(status_code)
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self) -> None:
        """
        Handle OPTIONS requests for CORS preflight.

        This is called by browsers before making actual POST requests
        to verify that the server accepts cross-origin requests.
        """
        self._send_json_response({}, 200)

    def do_POST(self) -> None:
        """
        Handle POST requests containing query data.

        Expected JSON format:
            {
                "subreddits": ["subreddit1", "subreddit2", ...],
                "question": "User's question here"
            }

        Returns JSON response with query results or error message.
        """
        # Read and parse request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            # Parse JSON request
            data = json.loads(body)
            print("\n--- JSON Received ---")
            print(json.dumps(data, indent=4))
            print("---------------------\n")

            # Extract and validate request parameters
            subreddits_raw = data.get("subreddits", [])
            question_data = data.get("question", "")

            # Handle subreddits - can be list of strings or list of dicts with 'name' field
            subreddits = []
            if isinstance(subreddits_raw, list):
                for item in subreddits_raw:
                    if isinstance(item, str):
                        subreddits.append(item)
                    elif isinstance(item, dict) and "name" in item:
                        subreddits.append(item["name"])
                    else:
                        print(f"WARNING: Skipping invalid subreddit item: {item}")

            # Handle question - can be string or dict with 'title'/'description' fields
            if isinstance(question_data, dict):
                question = question_data.get("title", "")
                if not question:
                    question = question_data.get("description", "")
            else:
                question = question_data

            # Validate subreddits parameter
            if not subreddits:
                error_response, status_code = ResponseFormatter.create_error_response(
                    "No subreddits provided", 400
                )
                self._send_json_response(error_response, status_code)
                return

            # Validate question parameter
            if not question:
                error_response, status_code = ResponseFormatter.create_error_response(
                    "No question provided", 400
                )
                self._send_json_response(error_response, status_code)
                return

            # Log query details
            print(f"\nProcessing query: '{question}'")
            print(f"Subreddits: {', '.join(subreddits)}\n")

            # Process the query using QueryProcessor
            response = QueryProcessor.process_query(
                subreddits=subreddits,
                question=question,
                k=ServerConfig.DEFAULT_TOP_K
            )

            # Print results to console for monitoring
            ResponseFormatter.print_query_results(response)

            # Send successful response
            self._send_json_response(response, 200)

        except json.JSONDecodeError:
            # Handle invalid JSON
            print("ERROR: Received invalid JSON")
            error_response, status_code = ResponseFormatter.create_error_response(
                "Invalid JSON", 400
            )
            self._send_json_response(error_response, status_code)

        except Exception as e:
            # Handle unexpected errors
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            error_response, status_code = ResponseFormatter.create_error_response(
                str(e), 500
            )
            self._send_json_response(error_response, status_code)


# ============================================================================
# HTTP Server
# ============================================================================

class RedditDataServer:
    """
    HTTP server for receiving and processing Reddit research queries.

    This server:
    - Listens for HTTP POST requests containing subreddit queries
    - Processes queries using RAG (Retrieval-Augmented Generation)
    - Returns AI-generated answers with relevant Reddit posts
    - Provides periodic heartbeat messages to indicate server is running

    Attributes:
        port: Port number the server listens on
        heartbeat_interval: Seconds between heartbeat console messages
        server: The underlying HTTPServer instance
    """

    def __init__(self, port: int = ServerConfig.DEFAULT_PORT,
                 heartbeat_interval: int = ServerConfig.HEARTBEAT_INTERVAL):
        """
        Initialize the Reddit data server.

        Args:
            port: Port number to run the server on (default: ServerConfig.DEFAULT_PORT)
            heartbeat_interval: Seconds between heartbeat messages (default: ServerConfig.HEARTBEAT_INTERVAL)
        """
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.server: Optional[HTTPServer] = None

    def _heartbeat(self) -> None:
        """
        Print periodic heartbeat messages to console.

        Runs in a background thread to indicate the server is alive and listening.
        This helps with monitoring and debugging during development.
        """
        while True:
            print("listening...")
            time.sleep(self.heartbeat_interval)

    def start(self) -> None:
        """
        Start the HTTP server and begin listening for requests.

        This method:
        1. Creates an HTTPServer instance
        2. Starts a background heartbeat thread
        3. Begins serving requests indefinitely
        4. Handles graceful shutdown on KeyboardInterrupt (Ctrl+C)
        """
        # Create the HTTP server
        self.server = HTTPServer(('', self.port), RedditDataRequestHandler)

        print(f"Reddit Data Server running on port {self.port}...")
        print("Waiting for subreddit requests from the client...\n")

        # Start heartbeat in background daemon thread
        # Daemon thread will automatically terminate when main program exits
        heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
        heartbeat_thread.start()

        # Start serving requests
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            if self.server:
                self.server.shutdown()


# ============================================================================
# Main Entry Point
# ============================================================================

def run(port: int = ServerConfig.DEFAULT_PORT) -> None:
    """
    Run the Reddit data server.

    This is the main entry point for starting the server. It creates
    a RedditDataServer instance and starts it.

    Args:
        port: Port number to run the server on (default: ServerConfig.DEFAULT_PORT)

    Example:
        >>> run(8080)  # Start server on port 8080
    """
    server = RedditDataServer(port=port)
    server.start()


if __name__ == "__main__":
    # Start the server when script is run directly
    run()

