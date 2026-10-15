"""Fetch one allowed US-government page and save its main text as Markdown in data/corpus/.

Usage: uv run python scripts/fetch_doc.py <doc-id> <https-url>

The text is extracted deterministically (no language model touches it), so excerpts copied
from the saved file stay verbatim. Review the file by hand before keeping it.
"""

import re
import sys
from pathlib import Path
from urllib.parse import urlparse

ALLOWED_DOMAINS = ("nhlbi.nih.gov", "cdc.gov")
CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"
DOC_ID = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
COPYRIGHT_MARKERS = (
    "©",
    "copyright",
    "used with permission",
    "reprinted",
    "courtesy of",
    "adapted from",
)


def is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host:
        return False
    return any(host == domain or host.endswith("." + domain) for domain in ALLOWED_DOMAINS)


def flag_markers(text: str) -> list[str]:
    """Lines that mention third-party copyright, so a human can check them."""
    findings = []
    for number, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        if any(marker in lowered for marker in COPYRIGHT_MARKERS):
            findings.append(f"line {number}: {line.strip()[:120]}")
    return findings


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    doc_id, url = args
    if not DOC_ID.match(doc_id):
        print(
            f"error: doc id '{doc_id}' must be lowercase words joined by hyphens", file=sys.stderr
        )
        return 2
    if not is_allowed(url):
        print(f"error: {url} is not an https page on {', '.join(ALLOWED_DOMAINS)}", file=sys.stderr)
        return 2

    import trafilatura  # imported here so the helpers above work without it

    html = trafilatura.fetch_url(url)
    text = (
        trafilatura.extract(
            html,
            output_format="markdown",
            include_formatting=True,
            include_links=False,
            include_images=False,
            include_tables=True,
            favor_precision=True,
        )
        if html
        else None
    )
    if not text:
        print(f"error: could not download or extract text from {url}", file=sys.stderr)
        return 1

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    path = CORPUS_DIR / f"{doc_id}.md"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"Wrote {path} ({len(text)} characters)")
    for finding in flag_markers(text):
        print(f"REVIEW {finding}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
