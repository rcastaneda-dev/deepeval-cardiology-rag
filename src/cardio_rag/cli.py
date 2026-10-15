"""Command line: `cardio ingest` builds the index, `cardio ask "question"` answers one."""

import argparse
import re
import sys

from cardio_rag import wiring
from cardio_rag.config import DISCLAIMER, INDEX_PATH, MissingSettingError, Settings

_CONTROL_SEQUENCES = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|[\x00-\x08\x0b-\x1f\x7f-\x9f]"
)


def _printable(text: str) -> str:
    """Drop ANSI escape sequences and control characters (keeping newline and tab) so model
    output cannot drive the terminal."""
    return _CONTROL_SEQUENCES.sub("", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cardio", description=DISCLAIMER)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="embed the corpus and write the index file")
    ask = commands.add_parser("ask", help="answer one question from the corpus")
    ask.add_argument("question")
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_env()
        if args.command == "ingest":
            index = wiring.ingest(settings)
            documents = len({chunk.doc_id for chunk in index.chunks})
            print(f"Indexed {len(index.chunks)} chunks from {documents} documents -> {INDEX_PATH}")
            return 0
        result = wiring.build_pipeline(settings).ask(args.question)
    except MissingSettingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(DISCLAIMER, end="\n\n")
    print(_printable(result.answer.text))
    if result.answer.cited_ids:
        print("\nSources: " + ", ".join(result.answer.cited_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
