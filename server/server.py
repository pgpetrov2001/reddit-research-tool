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

        Args:
            subreddit: Name of the subreddit

        Returns:
            Absolute path to the subreddit's directory
        """
        return str(cls.SUBREDDITS_DIR / subreddit)


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
                subreddit_dir = ServerConfig.get_subreddit_directory(subreddit)

                # Check if subreddit data exists
                if not os.path.exists(subreddit_dir):
                    print(f"WARNING: Subreddit '{subreddit}' does not exist in SubReddits folder. Skipping...")
                    continue

                # Retrieve candidates from vector database
                vec_db = VectorRetriever(subreddit_dir)
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

        # Format the response
        posts = ResponseFormatter.format_posts(best_candidates)

        return {
            "status": "success",
            "answer": answer,
            "posts": posts,
            "total_candidates": len(candidates),
            "top_k": k
        }


# ============================================================================
# Response Formatting
# ============================================================================

class ResponseFormatter:
    """Formats query results for HTTP responses."""

    @staticmethod
    def format_posts(candidates: List[Candidate]) -> List[Dict[str, Any]]:
        """
        Format candidate posts for JSON response.

        Args:
            candidates: List of Candidate objects from retrieval

        Returns:
            List of dictionaries with post metadata
        """
        posts = []
        for candidate in candidates:
            posts.append({
                "title": candidate.chunk.title,
                "source": candidate.chunk.source,
                "text": candidate.chunk.text,
                "score": candidate.score,
                "chunk_id": candidate.chunk.id,
                "section": candidate.chunk.section
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

        print("\n=== AI SUMMARY ===")
        print(response["answer"])
        print("\n=== TOP POSTS ===")

        for i, post in enumerate(response["posts"], 1):
            print(f"\n{i}. {post['title']} (score: {post['score']:.4f})")
            print(f"   Source: {post['source']}")
            print(f"   Text preview: {post['text'][:200]}...")

        print("\n" + "="*50 + "\n")


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
            subreddits = data.get("subreddits", [])
            question = data.get("question", "")

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

