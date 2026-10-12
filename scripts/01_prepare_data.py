"""Stage 1 (Member A): build data/manifest.csv and the 256x256 image cache.

    python scripts/01_prepare_data.py            # first time
    python scripts/01_prepare_data.py --force    # rebuild the cache

Commit data/manifest.csv once; everyone then uses the identical file ordering and split.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.data import build_image_cache, build_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild the image cache")
    ap.add_argument("--threads", type=int, default=8, help="reader threads (I/O bound; more helps on Google Drive)")
    args = ap.parse_args()

    cfg = load_config()
    df = build_manifest(cfg)
    build_image_cache(cfg, df, force=args.force, n_threads=args.threads)


if __name__ == "__main__":
    main()
