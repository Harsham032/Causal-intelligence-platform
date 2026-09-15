#!/usr/bin/env python
"""Download the Criteo Uplift Prediction Dataset.

**Not run for the published results.** The environment used to produce
``docs/results.md`` has no network route to Criteo's hosts; every endpoint
returns HTTP 403 at the proxy. This script exists so the path is available on a
machine that can reach it, and so the specification's primary source is not
silently dropped.

The dataset is roughly 25 million rows of a real randomised advertising
experiment: a binary treatment, two binary outcomes (visit and conversion) and
eleven anonymised features.

Usage::

    python scripts/download_criteo.py --output data/external
    python scripts/download_criteo.py --output data/external --check-only
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cip.config import load_settings
from cip.logging_utils import configure_logging, get_logger

logger = get_logger("download_criteo")

EXPECTED_COLUMNS = (
    "f0",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "treatment",
    "conversion",
    "visit",
    "exposure",
)


def verify(path: Path) -> bool:
    """Read the header and a few rows, so a truncated download is caught here."""
    import gzip
    import io

    import pandas as pd

    try:
        with gzip.open(path, "rb") as handle:
            head = handle.read(1_000_000)
        frame = pd.read_csv(io.BytesIO(head), nrows=100)
    except Exception as exc:
        logger.error("verification_failed", path=str(path), error=str(exc))
        return False

    missing = [c for c in ("treatment", "visit", "conversion") if c not in frame.columns]
    if missing:
        logger.error("unexpected_schema", missing=missing, found=list(frame.columns)[:20])
        return False
    logger.info("verified", path=str(path), columns=len(frame.columns))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", default="data/external")
    parser.add_argument("--url", default=None, help="override the configured source URL")
    parser.add_argument(
        "--check-only", action="store_true", help="test reachability without downloading"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    settings = load_settings()
    url = args.url or settings.criteo_url
    destination = Path(args.output) / "criteo-uplift-v2.1.csv.gz"

    if destination.is_file():
        logger.info("already_downloaded", path=str(destination))
        return 0 if verify(destination) else 1

    if args.check_only:
        try:
            request = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(request, timeout=30) as response:
                logger.info("reachable", url=url, status=response.status)
                return 0
        except urllib.error.HTTPError as exc:
            print(f"unreachable: HTTP {exc.code} for {url}")
            return 1
        except Exception as exc:
            print(f"unreachable: {exc}")
            return 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading", url=url, destination=str(destination))
    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as exc:
        print(
            f"\nCould not download the Criteo dataset: {exc}\n\n"
            "This is expected in a sandboxed environment. The published results in\n"
            "docs/results.md do not use this dataset and do not claim to; see\n"
            "docs/data-sources.md for what is used instead.\n"
        )
        return 1

    if not verify(destination):
        destination.unlink(missing_ok=True)
        print("the downloaded file did not verify and has been removed")
        return 1

    size_mb = destination.stat().st_size / 1_048_576
    print(f"downloaded {destination} ({size_mb:,.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
