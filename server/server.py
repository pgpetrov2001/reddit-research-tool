from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
import time

class SimpleJSONServer(BaseHTTPRequestHandler):

    # Add CORS to all responses
    def _set_cors(self):
        self.send_header('Access-Control-Allow-Origin', 'http://localhost:5173')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    # Handle browser preflight requests
    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors()
        self.end_headers()

    def do_POST(self):
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
        except json.JSONDecodeError:
            print("Received invalid JSON")

        self.wfile.write(b"JSON received")

def heartbeat():
    while True:
        print("listening...")
        time.sleep(10)

def run(port=8080):
    server = HTTPServer(('', port), SimpleJSONServer)
    print(f"Python server running on port {port}...")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()

    server.serve_forever()

if __name__ == "__main__":
    run()

