from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
import time
import subprocess
import datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Optional


class SubredditManager:
    """Manages subreddit data checking and scraping operations."""

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the SubredditManager.

        Args:
            project_root: Root directory of the project. Defaults to parent of server/ directory.
        """
        self.project_root = project_root or Path(__file__).parent.parent.resolve()
        self.subreddits_dir = self.project_root / "SubReddits"
        self.scraper_script = self.project_root / "Get_reddit_data_class.py"

        # Ensure SubReddits directory exists
        self.subreddits_dir.mkdir(parents=True, exist_ok=True)

    def extract_subreddit_names(self, data: dict) -> List[str]:
        """
        Extract subreddit names from various JSON structures.

        Args:
            data: JSON data containing subreddit information

        Returns:
            List of subreddit names
        """
        if "subreddits" in data:
            # Handle {"subreddits": ["sub1", "sub2"]}
            return data["subreddits"]
        elif "subreddit" in data:
            # Handle {"subreddit": "sub1"}
            return [data["subreddit"]]
        elif isinstance(data, list):
            # Handle ["sub1", "sub2"]
            return data
        else:
            print("WARNING: Could not find subreddit data in JSON")
            print("Expected format: {'subreddits': ['sub1', 'sub2']} or {'subreddit': 'sub1'}")
            return []

    def check_subreddit_exists(self, subreddit: str) -> bool:
        """
        Check if a subreddit folder exists.

        Args:
            subreddit: Name of the subreddit

        Returns:
            True if folder exists, False otherwise
        """
        subreddit_folder = self.subreddits_dir / subreddit
        return subreddit_folder.exists() and subreddit_folder.is_dir()

    def scrape_subreddit(self, subreddit: str, after: Optional[str] = None,
                        before: Optional[str] = None, what: str = "both") -> bool:
        """
        Scrape a subreddit using the Get_reddit_data_class.py script.

        Args:
            subreddit: Name of the subreddit to scrape
            after: Start date (YYYY-MM-DD or ISO 8601)
            before: End date (YYYY-MM-DD or ISO 8601)
            what: What to fetch ("submissions", "comments", or "both")

        Returns:
            True if scraping was successful, False otherwise
        """
        if not self.scraper_script.exists():
            print(f"ERROR: Scraper script not found at {self.scraper_script}")
            return False

        # Create output directory for this subreddit
        outdir = self.subreddits_dir / subreddit

        # Build command
        cmd = [
            "python3",
            str(self.scraper_script),
            "-s", subreddit,
            "--what", what,
            "--outdir", str(outdir)
        ]

        # Add date filters if provided
        if after:
            cmd.extend(["--after", after])
        if before:
            cmd.extend(["--before", before])

        print(f"\n[SCRAPING] Starting scrape for r/{subreddit}...")
        print(f"[SCRAPING] Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )

            print(f"[SCRAPING] Success for r/{subreddit}")
            print(result.stdout)
            return True

        except subprocess.CalledProcessError as e:
            print(f"[SCRAPING] Error scraping r/{subreddit}:")
            print(f"[SCRAPING] Exit code: {e.returncode}")
            print(f"[SCRAPING] stderr: {e.stderr}")
            return False

        except subprocess.TimeoutExpired:
            print(f"[SCRAPING] Timeout scraping r/{subreddit} (exceeded 5 minutes)")
            return False

    def process_subreddits(self, data: dict, auto_scrape: bool = True) -> Dict[str, Any]:
        """
        Process subreddits from received JSON data.

        Checks if subreddit folders exist and optionally scrapes missing ones.

        Args:
            data: JSON data containing subreddit information
            auto_scrape: If True, automatically scrape subreddits that don't exist

        Returns:
            Dict with processing results
        """
        subreddits = self.extract_subreddit_names(data)

        if not subreddits:
            print("WARNING: No subreddits found in the received data")
            return {"processed": 0, "existing": 0, "scraped": 0, "failed": 0}

        print(f"\nProcessing {len(subreddits)} subreddit(s)...")

        results = {
            "processed": len(subreddits),
            "existing": 0,
            "scraped": 0,
            "failed": 0
        }

        for subreddit in subreddits:
            if self.check_subreddit_exists(subreddit):
                print(f"✓ Data for r/{subreddit} already exists")
                results["existing"] += 1
            else:
                print(f"✗ Data for r/{subreddit} does not exist")

                if auto_scrape:
                    # Scrape the subreddit - you can customize date ranges here
                    success = self.scrape_subreddit(subreddit, what="both")

                    if success:
                        results["scraped"] += 1
                    else:
                        results["failed"] += 1

        print(f"\n[SUMMARY] Processed: {results['processed']} | Existing: {results['existing']} | "
              f"Scraped: {results['scraped']} | Failed: {results['failed']}\n")

        return results


class RedditDataRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for receiving subreddit requests."""

    def _set_cors(self):
        """Add CORS headers to responses."""
        self.send_header('Access-Control-Allow-Origin', 'http://localhost:5173')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        """Handle browser preflight requests."""
        self.send_response(200)
        self._set_cors()
        self.end_headers()

    def do_POST(self):
        """Handle POST requests with subreddit data."""
        self.send_response(200)
        self._set_cors()
        self.end_headers()

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            print("\n--- JSON Received ---")
            print(json.dumps(data, indent=4))
            print("---------------------\n")

            # Process subreddits using the manager
            manager = SubredditManager()
            results = manager.process_subreddits(data, auto_scrape=True)

            # Send back results
            response = {
                "status": "success",
                "results": results
            }
            self.wfile.write(json.dumps(response).encode())

        except json.JSONDecodeError:
            print("ERROR: Received invalid JSON")
            response = {"status": "error", "message": "Invalid JSON"}
            self.wfile.write(json.dumps(response).encode())


class RedditDataServer:
    """HTTP server for receiving and processing subreddit requests."""

    def __init__(self, port: int = 8080, heartbeat_interval: int = 10):
        """
        Initialize the Reddit data server.

        Args:
            port: Port number to run the server on
            heartbeat_interval: Seconds between heartbeat messages
        """
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.server = None

    def _heartbeat(self):
        """Print periodic heartbeat messages."""
        while True:
            print("listening...")
            time.sleep(self.heartbeat_interval)

    def start(self):
        """Start the HTTP server."""
        self.server = HTTPServer(('', self.port), RedditDataRequestHandler)
        print(f"Reddit Data Server running on port {self.port}...")
        print("Waiting for subreddit requests from the client...\n")

        # Start heartbeat in background thread
        thread = threading.Thread(target=self._heartbeat, daemon=True)
        thread.start()

        # Start serving
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            self.server.shutdown()


def run(port: int = 8080):
    """
    Run the Reddit data server.

    Args:
        port: Port number to run the server on
    """
    server = RedditDataServer(port=port)
    server.start()


if __name__ == "__main__":
    run()

