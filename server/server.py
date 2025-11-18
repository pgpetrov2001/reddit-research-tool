from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
import time
import subprocess
import datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Optional


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

            # Send back results
            response = {
                "status": "success",
                "message": "Data received successfully"
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

