"""
Minimal local test client for the vidgen-pipeline server.

    python scripts/test_client.py --url http://localhost:8000 \
        --image ref.png --c1 control1.mp4 --c2 control2.mp4 [--save-dir out]

Posts one reference image + two control videos, prints timing/latency, and
optionally downloads the two result videos.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="server base URL")
    ap.add_argument("--image", required=True)
    ap.add_argument("--c1", required=True)
    ap.add_argument("--c2", required=True)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    try:
        import requests
    except ImportError:
        raise SystemExit("pip install requests")

    base = args.url.rstrip("/")
    t0 = time.time()
    with open(args.image, "rb") as fi, open(args.c1, "rb") as f1, open(args.c2, "rb") as f2:
        resp = requests.post(
            f"{base}/generate",
            files={
                "user_image": (os.path.basename(args.image), fi, "image/png"),
                "control_video_1": (os.path.basename(args.c1), f1, "video/mp4"),
                "control_video_2": (os.path.basename(args.c2), f2, "video/mp4"),
            },
            timeout=1800,
        )
    client_rtt = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    print(json.dumps(data, indent=2))
    print(f"\nclient round-trip: {client_rtt:.2f}s")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for r in data["results"]:
            out = os.path.join(args.save_dir, os.path.basename(r["video_url"]))
            urllib.request.urlretrieve(base + r["video_url"], out)
            print(f"saved {out}")


if __name__ == "__main__":
    main()
