"""The README states what the original design's success criterion 5 asks (P18 in
evals/promises.py): the corpus sources, the disclaimer and the limits of the evaluation. It also
links the QA report before its first section, where a reader who asks "how do you know?" looks."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text(encoding="utf-8")


def _section(heading: str) -> str:
    """The text of the README section under `heading` (an exact "## " line), up to the next."""
    _, found, rest = README.partition(f"\n## {heading}\n")
    assert found, f"README has no section '## {heading}'"
    return rest.split("\n## ", 1)[0]


def test_the_readme_names_the_corpus_sources_and_links_their_license_checks():
    assert "nhlbi.nih.gov" in README and "cdc.gov" in README
    assert "](data/SOURCES.md)" in README
    assert (REPO / "data" / "SOURCES.md").is_file()


def test_the_readme_carries_the_disclaimer_before_the_first_section():
    intro = README.split("\n## ", 1)[0]
    assert "**Not medical advice.**" in intro
    assert "call 911" in intro


def test_the_readme_states_the_limits_of_the_evaluation():
    limits = _section("Limits")
    assert "not that its content is clinically right" in limits
    assert "judges are noisy" in limits


def test_the_readme_links_the_qa_report_before_its_first_section():
    intro = README.split("\n## ", 1)[0]
    assert "](docs/qa-report.md)" in intro
    assert (REPO / "docs" / "qa-report.md").is_file()
