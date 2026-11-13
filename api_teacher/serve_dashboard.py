"""Serve the dashboard over localhost so browser media policies are consistent."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os


ROOT = Path(__file__).resolve().parents[1] / "api_runs"
os.chdir(ROOT)
print("Open http://127.0.0.1:8765/ablation_dashboard.html")
ThreadingHTTPServer(("127.0.0.1", 8765), SimpleHTTPRequestHandler).serve_forever()
