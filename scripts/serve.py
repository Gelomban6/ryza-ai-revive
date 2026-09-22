#encoding: utf-8
"""Local static server + LLM/TTS proxy.

Browser pages on 127.0.0.1 cannot call Aliyun/Xiaomi APIs (CORS).
POST /_proxy?u=<https url> forwards the JSON body and Authorization header.

  python scripts/serve.py
  # http://127.0.0.1:8765/
"""
from __future__ import annotations

import json
import os
import sys
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen, build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
PROVIDERS = ROOT / "config" / "providers.json"
PORT = 8765
# Cloudflare (opencode.ai etc.) returns 1010 for the default Python-urllib UA.
UA = "RyzaChat/1.2.13"


class Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/_proxy":
            self._proxy_get(parsed)
            return
        if path == "/_edge_tts":
            self._handle_edge_tts(parsed)
            return
        if path == "/config/providers.json" and PROVIDERS.is_file():
            raw = PROVIDERS.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
            return
        return SimpleHTTPRequestHandler.do_GET(self)

    def _open(self, req, timeout=180, target=""):
        host = ""
        try:
            host = (urlparse(target).hostname or "").lower()
        except Exception:
            pass
        # Bypass system proxies for local loopback / LAN endpoints (Ollama, LM Studio, etc.)
        if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.startswith("192.168.") or host.startswith("10.") or host.startswith("172."):
            opener = build_opener(ProxyHandler({}))
            return opener.open(req, timeout=timeout)
        return urlopen(req, timeout=timeout)

    def _proxy_get(self, parsed):
        """GET /_proxy?u=<https url> — forwards a GET (Qwen TTS returns
        time-limited OSS audio URLs; the page pulls them through here so
        the blob is same-origin for the lip-sync analyser)."""
        target = (parse_qs(parsed.query).get("u") or [""])[0]
        if not (target.startswith("https://") or target.startswith("http://")):
            self.send_error(400, "proxy target must be http or https")
            return
        try:
            headers = {"User-Agent": UA}
            auth = self.headers.get("Authorization")
            if auth:
                headers["Authorization"] = auth
            apikey = self.headers.get("api-key") or self.headers.get("Api-Key")
            if apikey:
                headers["api-key"] = apikey
            req = Request(target, headers=headers, method="GET")
            with self._open(req, timeout=120, target=target) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type") or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except HTTPError as e:
            data = e.read() if e.fp else str(e).encode("utf-8")
            self.send_response(e.code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (URLError, TimeoutError, OSError) as e:
            msg = json.dumps({"error": {"message": str(e)}}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def _handle_edge_tts(self, parsed):
        qs = parse_qs(parsed.query)
        text = (qs.get("text") or [""])[0]
        voice = (qs.get("voice") or ["ja-JP-NanamiNeural"])[0]
        if not text:
            self.send_error(400, "text parameter required")
            return
        try:
            import edge_tts
            import asyncio
        except ImportError:
            msg = json.dumps({"error": {"message": "edge-tts not installed on server. Run: pip install edge-tts"}}).encode("utf-8")
            self.send_response(503)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(msg)
            return

        async def _synthesize():
            communicate = edge_tts.Communicate(text, voice)
            audio_bytes = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes.extend(chunk["data"])
            return bytes(audio_bytes)

        try:
            audio = asyncio.run(_synthesize())
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(audio)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(audio)
        except Exception as e:
            msg = json.dumps({"error": {"message": str(e)}}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(msg)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, api-key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/_proxy":
            self.send_error(404, "use POST /_proxy")
            return
        target = (parse_qs(parsed.query).get("u") or [""])[0]
        if not (target.startswith("https://") or target.startswith("http://")):
            self.send_error(400, "proxy target must be http or https")
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        headers = {
            "Content-Type": self.headers.get("Content-Type") or "application/json",
            "User-Agent": UA,
        }
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        apikey = self.headers.get("api-key") or self.headers.get("Api-Key")
        if apikey:
            headers["api-key"] = apikey
        req = Request(target, data=body, headers=headers, method="POST")
        try:
            with self._open(req, timeout=180, target=target) as resp:
                data = resp.read()
                self.send_response(resp.status)
                ctype = resp.headers.get("Content-Type") or "application/json"
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except HTTPError as e:
            data = e.read() if e.fp else (str(e).encode("utf-8"))
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type") or "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (URLError, TimeoutError, OSError) as e:
            msg = json.dumps({"error": {"message": str(e)}}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)


def get_lan_ip() -> str:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def main():
    os.chdir(WEB)
    httpd = Server(("0.0.0.0", PORT), partial(Handler, directory=str(WEB)))
    lan = get_lan_ip()
    print("Ryza chat running:", flush=True)
    print("  Laptop:   http://127.0.0.1:%d/" % PORT, flush=True)
    if lan != "127.0.0.1":
        print("  HP / LAN: http://%s:%d/" % (lan, PORT), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
        httpd.shutdown()


if __name__ == "__main__":
    main()
