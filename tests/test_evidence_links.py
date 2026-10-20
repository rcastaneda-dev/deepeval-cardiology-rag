"""The comments and documents that quote a measurement link its file; the links must resolve."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LINK = re.compile(r"evals/snapshots/[\w.\-]+\.(?:md|json)")
QUOTING = [
    "README.md",
    "docs/findings.md",
    "src/cardio_rag/config.py",
    *(f"evals/{path.name}" for path in (REPO / "evals").glob("*.py")),
    *(f"scripts/{path.name}" for path in (REPO / "scripts").glob("*.py")),
]


def missing_links(files: list[str], root: Path) -> list[str]:
    """`file -> link` for every evals/snapshots/... file named in `files` (relative to `root`)
    that does not exist under `root`."""
    missing = []
    for name in files:
        for link in sorted(set(LINK.findall((root / name).read_text(encoding="utf-8")))):
            if not (root / link).is_file():
                missing.append(f"{name} -> {link}")
    return missing


def test_every_snapshot_file_the_code_and_docs_name_exists():
    assert missing_links(QUOTING, REPO) == []


def test_a_link_to_a_file_that_is_not_there_is_reported(tmp_path):
    (tmp_path / "evals" / "snapshots").mkdir(parents=True)
    (tmp_path / "evals" / "snapshots" / "here.md").write_text("x")
    (tmp_path / "doc.md").write_text(
        "See evals/snapshots/here.md and evals/snapshots/gone.json, or evals/snapshots/gone.json."
    )
    assert missing_links(["doc.md"], tmp_path) == ["doc.md -> evals/snapshots/gone.json"]


def test_the_scan_reads_the_files_that_quote_the_trials():
    assert {"README.md", "src/cardio_rag/config.py", "evals/thresholds.py"} <= set(QUOTING)
