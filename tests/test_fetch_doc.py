import pytest

from scripts.fetch_doc import flag_markers, is_allowed, main


@pytest.mark.parametrize(
    "url",
    [
        "https://www.nhlbi.nih.gov/health/heart-attack",
        "https://nhlbi.nih.gov/health/heart-attack",
        "https://www.cdc.gov/heart-disease/about/index.html",
        "https://cdc.gov",
    ],
)
def test_pages_on_allowed_domains_are_accepted(url):
    assert is_allowed(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://www.nhlbi.nih.gov/health/heart-attack",
        "https://medlineplus.gov/heartattack.html",
        "https://nhlbi.nih.gov.attacker.example/page",
        "https://evilcdc.gov/page",
        "https://example.com/?next=https://www.cdc.gov",
        "not a url",
        "",
    ],
)
def test_everything_else_is_refused(url):
    assert not is_allowed(url)


def test_copyright_mentions_are_flagged_with_their_line_numbers():
    text = "Clean line.\nImage © Some Publisher\nMore text.\nPhoto courtesy of a hospital"
    assert flag_markers(text) == [
        "line 2: Image © Some Publisher",
        "line 4: Photo courtesy of a hospital",
    ]


def test_clean_text_raises_no_flags():
    assert flag_markers("The heart pumps blood.\nIt has four chambers.") == []


def test_bad_arguments_are_rejected_before_anything_is_downloaded(capsys):
    # trafilatura is not importable in some environments; reaching it would raise ImportError.
    assert main(["heart-attack"]) == 2
    assert main(["Heart Attack", "https://www.cdc.gov/x"]) == 2
    assert main(["heart-attack", "https://example.com/x"]) == 2
    assert "not an https page" in capsys.readouterr().err
