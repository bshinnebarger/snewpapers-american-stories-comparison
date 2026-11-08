import random
from pathlib import Path

import pytest

from snewpapers_tools.american_stories import ArchiveInfo
from snewpapers_tools.issues import IssueKey
from snewpapers_tools.selection import Estimate, estimate, hits_table, pick


def hit(n):
    return {"uuid": f"u{n}", "newspaper": "the weekly comet", "issue_date": "1853-09-11", "pages": [2],
            "label": "notice", "quality_tier": "green", "summary": f"summary {n}", "issue_uuid": f"iss-{n}",
            "lccn": "sn86053662"}


class FakeClient:
    def __init__(self, total=40, balance=500, issue_pages=None):
        self.total = total
        self.balance = balance
        self.issue_pages = issue_pages
        self.browsed = []

    def browse_issue_start(self, issue_uuid):
        self.browsed.append(issue_uuid)
        return {"total_components": self.total, "hits": [], "cursor": "c", "issue_pages": self.issue_pages}

    def tokens(self):
        return {"balance": self.balance}


COMPONENT = {
    "uuid": "u1",
    "issue_uuid": "iss-1",
    "newspaper": "the weekly comet",
    "links": {"loc_issue_url": "https://www.loc.gov/item/sn86053662/1853-09-11/ed-1/"},
}


def archive(year):
    return ArchiveInfo(year, f"faro_{year}.tar.gz", 1_358_708_574, Path("/nowhere") / f"faro_{year}.tar.gz")


HIT_COLUMNS = ["newspaper", "issue_date", "pages", "label", "quality_tier", "summary", "in_american_stories"]


def test_hits_table_has_one_row_per_hit_in_reading_columns():
    table = hits_table([hit(1), hit(2)])
    assert list(table.columns) == HIT_COLUMNS
    assert table["summary"].tolist() == ["summary 1", "summary 2"]


def test_hits_table_of_no_hits_still_has_the_columns():
    assert list(hits_table([]).columns) == HIT_COLUMNS


def test_hits_table_in_american_stories_is_none_without_coverage():
    table = hits_table([hit(1)])
    assert table["in_american_stories"].tolist() == [None]


def test_hits_table_in_american_stories_comes_from_the_covered_mapping():
    table = hits_table([hit(1), hit(2)], covered={"u1": True, "u2": False})
    assert table["in_american_stories"].tolist() == [True, False]
    # a hit missing from the mapping is unknown, not assumed uncovered
    assert hits_table([hit(3)], covered={"u1": True})["in_american_stories"].tolist() == [None]


def test_pick_is_random_but_reproducible_with_a_seed():
    hits = [hit(n) for n in range(10)]
    assert pick(hits, random.Random(7)) == pick(hits, random.Random(7))


def test_pick_avoids_the_current_hit_when_it_can():
    hits = [hit(1), hit(2)]
    assert all(pick(hits, random.Random(s), avoid=hits[0]) is hits[1] for s in range(20))
    assert pick([hits[0]], random.Random(0), avoid=hits[0]) is hits[0]


def test_pick_prefers_a_different_issue_than_avoid():
    a, b, c = hit(1), hit(2), hit(3)
    b["issue_uuid"] = a["issue_uuid"]  # b is a second hit from a's issue
    hits = [a, b, c]
    assert all(pick(hits, random.Random(s), avoid=a) is c for s in range(20))


def test_pick_of_no_hits_raises():
    with pytest.raises(ValueError):
        pick([], random.Random(0))


def test_estimate_sizes_the_issue_and_prices_it():
    client = FakeClient(total=40, balance=500)
    est = estimate(client, COMPONENT, archive_lookup=archive, coverage=lambda a, k: [1, 2])
    assert est.issue == IssueKey("iss-1", "sn86053662", "1853-09-11", 1)
    assert est.newspaper == "the weekly comet"
    assert est.total_components == 40
    assert est.cost == {"browse": 2, "text": 120, "total": 122}
    assert est.balance == 500
    assert est.archive.filename == "faro_1853.tar.gz"
    assert est.as_pages == [1, 2]
    assert est.issue_pages is None  # the browse response did not report it
    assert est.ready
    assert client.browsed == ["iss-1"]


def test_estimate_captures_issue_pages_from_browse_page_one():
    client = FakeClient(total=88, balance=500, issue_pages=4)
    est = estimate(client, COMPONENT, archive_lookup=archive, coverage=lambda a, k: [1, 2, 3, 4])
    assert est.issue_pages == 4


def test_estimate_passes_the_archive_and_key_to_coverage():
    client = FakeClient(total=40, balance=500)
    calls = []

    def coverage(a, k):
        calls.append((a, k))
        return [1]

    est = estimate(client, COMPONENT, archive_lookup=archive, coverage=coverage)
    assert calls == [(est.archive, est.issue)]


def test_estimate_is_not_ready_without_tokens_or_without_the_year():
    assert not estimate(
        FakeClient(total=40, balance=100), COMPONENT, archive_lookup=archive, coverage=lambda a, k: [1]
    ).ready
    assert not estimate(FakeClient(), COMPONENT, archive_lookup=lambda year: None).ready


def test_estimate_is_not_ready_when_american_stories_lacks_the_issue():
    est = estimate(FakeClient(), COMPONENT, archive_lookup=archive, coverage=lambda a, k: [])
    assert est.as_pages == []
    assert not est.ready
    assert not est.needs_download  # known, just absent


def test_estimate_needs_download_when_coverage_is_unknown():
    est = estimate(FakeClient(), COMPONENT, archive_lookup=archive, coverage=lambda a, k: None)
    assert est.as_pages is None
    assert est.needs_download
    assert not est.ready


def test_estimate_does_not_need_download_without_a_year_archive():
    est = estimate(FakeClient(), COMPONENT, archive_lookup=lambda year: None)
    assert est.archive is None
    assert not est.needs_download
