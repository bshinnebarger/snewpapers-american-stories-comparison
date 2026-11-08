from pathlib import Path

import pytest

from snewpapers_tools.american_stories import ArchiveInfo, ScanName
from snewpapers_tools.fetching import IssuePair, fetch_both
from snewpapers_tools.issues import IssueKey
from snewpapers_tools.selection import Estimate

KEY = IssueKey("iss-1", "sn85025007", "1856-04-22", 1)
ESTIMATE = Estimate(KEY, "alexandria gazette", 3, {"browse": 1, "text": 9, "total": 10}, 500,
                    ArchiveInfo(1856, "faro_1856.tar.gz", 10, Path("/nowhere/faro_1856.tar.gz")), as_pages=[1, 2])
NO_ARCHIVE_ESTIMATE = Estimate(KEY, "alexandria gazette", 3, {"browse": 1, "text": 9, "total": 10}, 500, None)


class FakeClient:
    def __init__(self, drop: list[str] | None = None):
        self.text_ids = None
        self.balance = 500
        self.browsed = []
        self.drop = drop or []

    def browse_issue(self, issue_uuid, *, refresh=False):
        self.browsed.append(issue_uuid)
        hits = [
            {"uuid": "a", "pages": [1], "boxes": [{"page": 1, "rects": [[0.1, 0.1, 0.2, 0.2]]}]},
            {"uuid": "b", "pages": [1, 2]},
            {"uuid": "c", "pages": [2]},
        ]
        return {"issue_uuid": issue_uuid, "total_components": 3, "hits": hits}

    def components_text(self, ids):
        self.text_ids = ids
        ids = [u for u in ids if u not in self.drop]
        self.balance -= 3 * len(ids)
        texts = {"a": "x" * 10, "b": "y" * 30, "c": "z" * 5}
        return [{"uuid": u, "content": texts[u], "char_count": len(texts[u]), "pages": None, "label": "news"}
                for u in ids]

    def tokens(self):
        return {"balance": self.balance}


def scan(page):
    return ScanName("1856-04-22", page, "sn85025007", "00280761023", 1, f"04{page}")


AS_PAGES = [
    (scan(1), {"bboxes": [{"legibility": "Legible"}, {"legibility": "Illegible"}],
               "full articles": [{"article": "aaaa"}, {"article": "bb"}]}),
    (scan(2), {"bboxes": [{"legibility": "Illegible"}],
               "full articles": [{"article": "cccccc"}]}),
]


def test_fetch_both_browses_fetches_text_and_reads_american_stories():
    client = FakeClient()
    pair = fetch_both(client, ESTIMATE, american_stories=lambda est: AS_PAGES)
    assert client.text_ids == ["a", "b", "c"]
    assert [c["uuid"] for c in pair.components] == ["a", "b", "c"]
    assert pair.components[1]["pages"] == [1, 2]  # page list from the browse hit when the text lacks it
    assert pair.pages == AS_PAGES
    assert pair.tokens_spent == 9
    assert pair.total_components == 3
    assert pair.missing == []


def test_fetch_both_carries_each_browse_hits_boxes_onto_the_merged_component():
    client = FakeClient()
    pair = fetch_both(client, ESTIMATE, american_stories=lambda est: AS_PAGES)
    assert pair.components[0]["boxes"] == [{"page": 1, "rects": [[0.1, 0.1, 0.2, 0.2]]}]
    assert pair.components[1]["boxes"] is None  # hit "b" carried no boxes


def test_fetch_both_needs_a_confirmed_estimate():
    with pytest.raises(ValueError, match="Compare this issue"):
        fetch_both(FakeClient(), None, american_stories=lambda est: [])


def test_fetch_both_refuses_to_spend_when_american_stories_has_no_archive():
    client = FakeClient()
    calls = []
    with pytest.raises(ValueError, match="no archive for 1856"):
        fetch_both(client, NO_ARCHIVE_ESTIMATE, american_stories=lambda est: calls.append(est) or [])
    assert calls == []  # american_stories was never even called
    assert client.browsed == []
    assert client.text_ids is None


def test_fetch_both_refuses_to_spend_when_american_stories_has_no_pages_for_the_issue():
    client = FakeClient()
    with pytest.raises(ValueError, match="no pages for sn85025007 1856-04-22 ed-1"):
        fetch_both(client, ESTIMATE, american_stories=lambda est: [])
    assert client.browsed == []  # nothing spent
    assert client.text_ids is None


def test_fetch_both_reports_browse_hits_that_text_did_not_return():
    client = FakeClient(drop=["b"])
    pair = fetch_both(client, ESTIMATE, american_stories=lambda est: AS_PAGES)
    assert pair.missing == ["b"]
    assert [c["uuid"] for c in pair.components] == ["a", "c"]


def test_per_page_counts_both_sides():
    pair = fetch_both(FakeClient(), ESTIMATE, american_stories=lambda est: AS_PAGES)
    table = pair.per_page()
    assert table.to_dict("records") == [
        {"page": 1, "api_components": 2, "api_chars": 40, "as_articles": 2, "as_chars": 6, "as_illegible_boxes": 1},
        {"page": 2, "api_components": 2, "api_chars": 5, "as_articles": 1, "as_chars": 6, "as_illegible_boxes": 1},
    ]


def test_per_page_keys_american_stories_scans_by_sequence_when_names_all_say_p1():
    linked = [{"uuid": "a", "pages": [1], "content": "x", "char_count": 1,
               "links": {"loc_pdf_pages": [{"page_number": 1, "url": "https://tile.loc.gov/d/041.pdf"}]}},
              {"uuid": "b", "pages": [2], "content": "y", "char_count": 1,
               "links": {"loc_pdf_pages": [{"page_number": 2, "url": "https://tile.loc.gov/d/042.pdf"}]}}]
    p1_twice = [(ScanName("1856-04-22", 1, "sn85025007", "00280761023", 1, "041"), AS_PAGES[0][1]),
                (ScanName("1856-04-22", 1, "sn85025007", "00280761023", 1, "042"), AS_PAGES[1][1])]
    pair = IssuePair(KEY, linked, p1_twice, 0, 2, [])
    table = pair.per_page()
    assert table[["page", "as_articles", "as_chars"]].to_dict("records") == [
        {"page": 1, "as_articles": 2, "as_chars": 6},
        {"page": 2, "as_articles": 1, "as_chars": 6},
    ]
