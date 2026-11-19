"""Download a small public PDF corpus and report real page totals."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pypdf import PdfReader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("download_sample_pdfs")
MAX_DOWNLOAD_BYTES = 40 * 1024 * 1024


def read_url_list(path: Path) -> list[tuple[str, str]]:
    """Read filename and URL records separated by a tab."""
    records: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t", maxsplit=1)
        if len(parts) != 2:
            logger.warning("Skipping malformed URL record: %s", stripped)
            continue
        records.append((Path(parts[0]).name, parts[1].strip()))
    return records


def download_pdf(url: str, destination: Path) -> int:
    """Download one PDF safely and return its page count."""
    request = Request(url, headers={"User-Agent": "DocumentQAAgent/1.0 (research sample downloader)"})
    with urlopen(request, timeout=40) as response:
        content_type = response.headers.get_content_type()
        content = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"PDF exceeds {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB download limit")
    if not content.startswith(b"%PDF-") and content_type != "application/pdf":
        raise ValueError(f"Response is not a PDF (content-type={content_type})")
    destination.write_bytes(content)
    try:
        return len(PdfReader(str(destination), strict=False).pages)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls", type=Path, default=PROJECT_ROOT / "data/pdfs/urls.txt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/pdfs")
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    records = read_url_list(args.urls)[: args.limit]
    successes = 0
    pages = 0
    failures = 0
    for filename, url in tqdm(records, desc="Sample PDFs"):
        destination = args.output / filename
        try:
            if destination.is_file():
                page_count = len(PdfReader(str(destination), strict=False).pages)
            else:
                page_count = download_pdf(url, destination)
            successes += 1
            pages += page_count
            logger.info("%s: %d pages", filename, page_count)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            failures += 1
            logger.warning("Could not download %s: %s", filename, exc)
    logger.info(
        "Download summary: %d/%d PDFs succeeded, %d pages, %d failures",
        successes,
        len(records),
        pages,
        failures,
    )
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())

