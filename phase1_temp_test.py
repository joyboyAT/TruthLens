import threading
import time
from typing import Tuple

from http.server import BaseHTTPRequestHandler, HTTPServer

from src.data_collection import fetch_url
from src.preprocessing import html_to_text
from src.utils.text_cleaning import clean_text, normalize_text


HTML_SAMPLE = (
    """
<!doctype html>
<html>
  <head><title>Test Page</title><script>var x=1;</script></head>
  <body>
    <h1>Headline</h1>
    <p>Paragraph A with   extra   spaces.</p>
    <p>Paragraph B &amp; entities.</p>
  </body>
  <script>console.log('ignore me');</script>
</html>
"""
).strip()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # type: ignore[override]
        body = HTML_SAMPLE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # silence server logs
        return


def _start_server() -> Tuple[HTTPServer, int]:
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    thr = threading.Thread(target=srv.serve_forever, daemon=True)
    thr.start()
    time.sleep(0.1)
    return srv, port


def main() -> None:
    print("[Phase1] Starting local HTTP server…", flush=True)
    srv, port = _start_server()
    try:
        url = f"http://127.0.0.1:{port}/"
        print(f"[Phase1] Fetching URL: {url}", flush=True)
        rec = fetch_url(url)
        assert rec.get("status") == 200, f"unexpected status: {rec}"
        html = rec.get("html") or ""
        print("[Phase1] Converting HTML to text…", flush=True)
        text = html_to_text(html)
        print("[Phase1] Cleaning and normalizing text…", flush=True)
        cleaned = clean_text(text)
        normalized = normalize_text(cleaned)
        assert "Headline" in normalized
        assert "Paragraph A with extra spaces." in normalized
        assert "Paragraph B & entities." in normalized
        print("[Phase1] Text/URL ingestion + normalization: OK", flush=True)

        # OCR/Translation presence checks (best-effort informational)
        try:
            import pytesseract  # noqa: F401
            print("[Phase1] OCR lib detected (pytesseract)", flush=True)
        except Exception:
            print("[Phase1] OCR not found → skipping OCR test", flush=True)

        try:
            from google.cloud import translate  # type: ignore  # noqa: F401
            print("[Phase1] Google Cloud Translate detected", flush=True)
        except Exception:
            print("[Phase1] Vertex/Google Translate not found → skipping translation test", flush=True)

        print("[Phase1] All checks passed.", flush=True)
    finally:
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    main()


