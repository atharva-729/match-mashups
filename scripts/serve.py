"""Local static server for the v2 player: serves the repo folder (player/, outputs/, captions/) with HTTP Range
support, so the browser can seek in video.mp4 and the audio stems. Local only (binds 127.0.0.1); never deploy it.
usage: python scripts/serve.py [port]      (or: python run.py --serve)  ->  http://127.0.0.1:8765/player/
"""
import http.server, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = ("player/", "outputs/", "captions/")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def log_message(self, fmt, *args):
        pass

    def send_head(self):
        path = self.path.split("?")[0].lstrip("/")
        if path in ("", "player"):
            self.send_response(302); self.send_header("Location", "/player/"); self.end_headers(); return None
        if not path.startswith(ALLOWED):            # only the player and its data, never inputs/ or work/
            self.send_error(404); return None
        f = Path(self.translate_path(self.path))
        if f.is_dir():
            f = f / "index.html"
        if not f.exists():
            self.send_error(404); return None
        size = f.stat().st_size
        rng = self.headers.get("Range")
        ctype = self.guess_type(str(f))
        if f.suffix == ".m4a":
            ctype = "audio/mp4"
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            a = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
            b = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
            b = min(b, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            length = b - a + 1
        else:
            a, length = 0, size
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        fh = open(f, "rb")
        fh.seek(a)
        self._left = length
        return fh

    def copyfile(self, src, dst):
        left = getattr(self, "_left", None)
        while left is None or left > 0:
            chunk = src.read(min(1 << 16, left) if left is not None else 1 << 16)
            if not chunk:
                break
            try:
                dst.write(chunk)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                break
            if left is not None:
                left -= len(chunk)


def main(port=8765):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Match Mashups player: http://127.0.0.1:{port}/player/   (Ctrl+C to stop)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8765)
