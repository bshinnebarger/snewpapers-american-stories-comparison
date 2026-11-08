import random
from pathlib import Path

from snewpapers_tools.american_stories import ArchiveInfo
from snewpapers_tools.api import InsufficientTokens
from snewpapers_tools.form import DENSE_PER_PAGE, ComparisonForm, render_pick
from snewpapers_tools.issues import IssueKey
from snewpapers_tools.selection import Estimate


def hit(n):
    return {"uuid": f"u{n}", "newspaper": "the weekly comet", "issue_date": "1853-09-11", "pages": [2],
            "label": "notice", "quality_tier": "green", "summary": f"Yellow fever <b>{n}</b>",
            "issue_uuid": f"iss-{n}", "site_url": f"https://snewpapers.com/c/u{n}", "lccn": "sn86053662"}


def component(uuid):
    return {
        "uuid": uuid,
        "issue_uuid": f"iss-{uuid}",
        "newspaper": "the weekly comet",
        "links": {"loc_issue_url": "https://www.loc.gov/item/sn86053662/1853-09-11/ed-1/"},
        "thumbs": {"clip_lg": [{"page_number": 2, "url": f"https://img.example/{uuid}_clip_2_lg.webp"}]},
        "citation": {"formatted": "The Weekly Comet, 11 September 1853, p. 2"},
    }


def click(button) -> None:
    """Fire a button's on_click handlers synchronously, the way ipywidgets does for a real click."""
    button.click()


class FakeClient:
    def __init__(self, hits, total=40, balance=500, fail=None):
        self.hits = hits
        self.total = total
        self.balance = balance
        self.fail = fail
        self.searches = []

    def find_articles(self, start_year, end_year, keywords="", *, topic="", refresh=False):
        self.searches.append((start_year, end_year, keywords, topic))
        if self.fail:
            raise self.fail
        return {"mode": "keywords", "total_components": len(self.hits), "hits": self.hits}

    def component(self, uuid, *, refresh=False):
        return component(uuid)

    def browse_issue_start(self, issue_uuid, *, refresh=False):
        return {"total_components": self.total, "hits": [], "cursor": "c"}

    def tokens(self):
        return {"balance": self.balance}


def archive(year):
    return ArchiveInfo(year, f"faro_{year}.tar.gz", 1_358_708_574, Path("/nowhere") / f"faro_{year}.tar.gz")


def make_form(
    client,
    lookup=archive,
    coverage=lambda a, k: [1, 2],
    download=lambda a: None,
    day_coverage=lambda lccn, date: None,
):
    return ComparisonForm(
        client, rng=random.Random(0), archive_lookup=lookup, coverage=coverage, download=download,
        day_coverage=day_coverage,
    )


def test_search_sends_the_form_values_and_picks_a_hit():
    client = FakeClient([hit(1), hit(2), hit(3)])
    form = make_form(client)
    form.start_year.value, form.end_year.value = 1850, 1855
    form.keywords.value = " yellow fever quarantine "
    click(form.search_button)
    assert client.searches == [(1850, 1855, "yellow fever quarantine", "")]
    assert form.picked in client.hits
    assert form.estimate.total_components == 40
    assert not form.confirm_button.disabled
    assert not form.repick_button.disabled
    assert form.download_button.disabled  # coverage is known (the default fake), so no download is needed


def test_the_form_searches_by_keywords_with_an_optional_topic():
    form = make_form(FakeClient([hit(1)]))
    assert not hasattr(form, "phrase")
    assert form.keywords.value and "Keywords" in form.keywords.description
    assert form.topic.value == "" and "Topic" in form.topic.description


def test_a_topic_alone_or_with_keywords_is_searched():
    client = FakeClient([hit(1)])
    form = make_form(client)
    form.keywords.value, form.topic.value = "", " a yellow fever epidemic in a port city "
    click(form.search_button)
    form.keywords.value = "quarantine"
    click(form.search_button)
    assert client.searches == [(1850, 1859, "", "a yellow fever epidemic in a port city"),
                               (1850, 1859, "quarantine", "a yellow fever epidemic in a port city")]


def test_search_needs_keywords_or_a_topic_of_3_characters_or_more():
    client = FakeClient([hit(1)])
    form = make_form(client)
    form.keywords.value, form.topic.value = "", ""
    click(form.search_button)
    assert "keywords or a topic" in form.message.value
    form.keywords.value, form.topic.value = "yellow fever", "ox"
    click(form.search_button)
    assert "at least 3 characters" in form.message.value
    assert client.searches == []


def test_search_refuses_short_keywords_or_backwards_years():
    client = FakeClient([hit(1)])
    form = make_form(client)
    form.keywords.value = "ox"
    click(form.search_button)
    assert "at least 3 characters" in form.message.value
    form.keywords.value = "yellow fever"
    form.start_year.value, form.end_year.value = 1860, 1850
    click(form.search_button)
    assert "before the start year" in form.message.value
    assert client.searches == []
    assert form.picked is None


def test_search_with_no_hits_leaves_nothing_to_confirm():
    form = make_form(FakeClient([]))
    click(form.search_button)
    assert form.picked is None
    assert form.confirm_button.disabled


def test_search_reports_a_short_balance_instead_of_raising():
    form = make_form(FakeClient([hit(1)], fail=InsufficientTokens("not enough tokens", 1, 0)))
    click(form.search_button)
    assert form.picked is None
    assert form.estimate is None
    assert form.confirm_button.disabled
    assert "not enough tokens" in form.message.value


def test_pick_another_changes_the_pick():
    form = make_form(FakeClient([hit(1), hit(2)]))
    click(form.search_button)
    first = form.picked
    click(form.repick_button)
    assert form.picked is not first


def test_pick_another_skips_hits_known_to_be_uncovered():
    calls = {"n": 0}

    def day_coverage(lccn, date):
        calls["n"] += 1
        return [None, False, True][calls["n"] - 1]  # u1 unknown, u2 known absent, u3 known present

    form = make_form(FakeClient([hit(1), hit(2), hit(3)]), day_coverage=day_coverage)
    click(form.search_button)
    assert form.covered == {"u1": None, "u2": False, "u3": True}
    picks = set()
    for _ in range(20):
        click(form.repick_button)
        picks.add(form.picked["uuid"])
    assert "u2" not in picks
    assert picks  # sanity: picks did happen


def test_pick_another_falls_back_to_all_hits_when_every_hit_is_known_uncovered():
    form = make_form(FakeClient([hit(1), hit(2)]), day_coverage=lambda lccn, date: False)
    click(form.search_button)
    assert form.covered == {"u1": False, "u2": False}
    assert form.picked is not None  # a pick still happens despite no "covered" candidates


def test_confirm_is_disabled_when_the_issue_cannot_be_compared():
    unaffordable = make_form(FakeClient([hit(1)], total=40, balance=10))
    click(unaffordable.search_button)
    assert unaffordable.confirm_button.disabled
    no_year = make_form(FakeClient([hit(1)]), lookup=lambda year: None)
    click(no_year.search_button)
    assert no_year.confirm_button.disabled
    not_covered = make_form(FakeClient([hit(1)]), coverage=lambda a, k: [])
    click(not_covered.search_button)
    assert not_covered.confirm_button.disabled
    unknown = make_form(FakeClient([hit(1)]), coverage=lambda a, k: None)
    click(unknown.search_button)
    assert unknown.confirm_button.disabled


def test_confirm_keeps_the_estimate_for_the_next_cells():
    form = make_form(FakeClient([hit(1)]))
    click(form.search_button)
    click(form.confirm_button)
    assert form.confirmed is form.estimate


def test_a_failing_pick_clears_the_estimate_and_disables_confirm():
    def offline(year):
        raise OSError("offline")
    form = make_form(FakeClient([hit(1), hit(2)]), lookup=offline)
    click(form.search_button)
    assert form.estimate is None
    assert form.confirm_button.disabled
    assert form.download_button.disabled
    assert "OSError: offline" in form.message.value


def test_a_failing_repick_clears_a_previously_good_estimate():
    calls = {"n": 0}

    def flaky(year):
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("offline")
        return archive(year)

    form = make_form(FakeClient([hit(1), hit(2)]), lookup=flaky)
    click(form.search_button)
    assert form.estimate is not None
    click(form.repick_button)
    assert form.estimate is None
    assert form.confirm_button.disabled


def test_a_render_error_is_caught_by_the_guard_not_swallowed_by_the_output():
    import snewpapers_tools.form as form_module

    def boom(*args, **kwargs):
        raise ValueError("bad render")

    form = make_form(FakeClient([hit(1), hit(2)]))
    original = form_module.render_pick
    form_module.render_pick = boom
    try:
        click(form.search_button)
    finally:
        form_module.render_pick = original
    assert form.estimate is None
    assert form.confirm_button.disabled
    assert "ValueError: bad render" in form.message.value


# -- download button --


def test_download_button_enabled_only_when_the_estimate_needs_a_download():
    needs_it = make_form(FakeClient([hit(1)]), coverage=lambda a, k: None)
    click(needs_it.search_button)
    assert needs_it.estimate.needs_download
    assert not needs_it.download_button.disabled
    assert needs_it.confirm_button.disabled

    covered = make_form(FakeClient([hit(1)]), coverage=lambda a, k: [1, 2])
    click(covered.search_button)
    assert covered.download_button.disabled
    assert not covered.confirm_button.disabled

    no_year = make_form(FakeClient([hit(1)]), lookup=lambda year: None)
    click(no_year.search_button)
    assert no_year.download_button.disabled


def test_download_button_click_downloads_then_reestimates_with_no_new_pick_or_tokens():
    calls = {"n": 0}

    def coverage(a, k):
        calls["n"] += 1
        return None if calls["n"] == 1 else [1, 2]

    downloaded = []

    def fake_download(info):
        downloaded.append(info)
        return info.path

    form = make_form(FakeClient([hit(1)]), coverage=coverage, download=fake_download)
    click(form.search_button)
    picked_before = form.picked
    component_before = form.component
    assert form.estimate.needs_download
    assert not form.download_button.disabled

    click(form.download_button)

    assert len(downloaded) == 1
    assert downloaded[0] is form.estimate.archive or downloaded[0].filename == form.estimate.archive.filename
    assert form.picked is picked_before  # no new pick
    assert form.component is component_before  # no new component fetch
    assert form.estimate.as_pages == [1, 2]
    assert not form.estimate.needs_download
    assert not form.confirm_button.disabled
    assert form.download_button.disabled
    assert "Downloading" in form.message.value
    assert "GB" in form.message.value


def test_download_button_click_error_is_guarded():
    def boom(info):
        raise OSError("network down")

    form = make_form(FakeClient([hit(1)]), coverage=lambda a, k: None, download=boom)
    click(form.search_button)
    click(form.download_button)
    assert form.estimate is None
    assert form.confirm_button.disabled
    assert form.download_button.disabled
    assert "OSError: network down" in form.message.value


# -- render_pick --


def estimate_for(total=40, balance=500, archive_info=None, as_pages=None, issue_pages=None):
    return Estimate(IssueKey("iss-1", "sn86053662", "1853-09-11", 1), "the weekly comet", total,
                    {"browse": 2, "text": 3 * total, "total": 2 + 3 * total}, balance, archive_info, as_pages,
                    issue_pages)


def test_render_pick_shows_the_article_issue_cost_and_a_pending_download():
    page = render_pick(hit(1), component("u1"), estimate_for(archive_info=archive(1853)))
    assert "The Weekly Comet" in page
    assert "Yellow fever &lt;b&gt;1&lt;/b&gt;" in page  # summaries are escaped
    assert 'src="https://img.example/u1_clip_2_lg.webp"' in page
    assert "sn86053662" in page and "ed-1" in page
    assert "122" in page and "500" in page
    assert "1.4 GB" in page
    assert "before we can check it has this issue" in page.lower()


def test_render_pick_shows_american_stories_coverage_when_known():
    page = render_pick(hit(1), component("u1"), estimate_for(archive_info=archive(1853), as_pages=[1, 2, 3]))
    assert "has this issue: 3 pages" in page.lower()
    assert "faro_1853.tar.gz" in page


def test_render_pick_warns_when_american_stories_lacks_the_issue():
    page = render_pick(hit(1), component("u1"), estimate_for(archive_info=archive(1853), as_pages=[]))
    assert "does not have this issue: pick another" in page.lower()


def test_render_pick_warns_about_a_dense_issue_and_a_short_balance():
    total = DENSE_PER_PAGE * 2 + 1
    page = render_pick(hit(1), component("u1"), estimate_for(total=total, balance=10, archive_info=archive(1853),
                                                              as_pages=[1, 2]))
    assert "dense issue" in page.lower()
    assert f"{total} components on 2 pages" in page.lower()
    assert "not enough tokens" in page.lower()


def test_render_pick_does_not_warn_about_density_when_the_page_count_is_unknown():
    page = render_pick(hit(1), component("u1"), estimate_for(total=DENSE_PER_PAGE * 5, archive_info=archive(1853)))
    assert "dense issue" not in page.lower()


def test_render_pick_dense_warning_prefers_issue_pages_over_as_pages_count():
    # issue_pages (4, from the API) says this is dense; as_pages (10 American Stories scans) would not.
    total = DENSE_PER_PAGE * 4 + 1
    page = render_pick(hit(1), component("u1"), estimate_for(
        total=total, archive_info=archive(1853), as_pages=list(range(1, 11)), issue_pages=4
    ))
    assert "dense issue" in page.lower()
    assert f"{total} components on 4 pages" in page.lower()
    assert f"{total} components on 10 pages" not in page.lower()


def test_render_pick_names_the_page_count_when_known():
    page = render_pick(hit(1), component("u1"), estimate_for(
        total=88, archive_info=archive(1853), as_pages=[1, 2, 3, 4], issue_pages=4
    ))
    assert "88 components on 4 pages" in page.lower()


def test_render_pick_omits_the_page_count_when_not_reported():
    page = render_pick(hit(1), component("u1"), estimate_for(total=40, archive_info=archive(1853)))
    assert "on 4 pages" not in page.lower()
    assert "40 components" in page.lower()


def test_render_pick_warns_about_a_missing_year():
    page = render_pick(hit(1), component("u1"), estimate_for(archive_info=None))
    assert "no archive for 1853" in page.lower()


def test_render_pick_without_thumbs_has_no_image():
    comp = component("u1") | {"thumbs": None}
    assert "<img" not in render_pick(hit(1), comp, estimate_for(archive_info=archive(1853)))


def test_render_pick_without_site_url_has_no_snewpapers_link():
    page = render_pick(hit(1) | {"site_url": None}, component("u1"), estimate_for(archive_info=archive(1853)))
    assert "snewpapers.com" not in page


def test_render_pick_title_cases_the_newspaper_with_apostrophes():
    page = render_pick(hit(1) | {"newspaper": "the people's press"}, component("u1"),
                        estimate_for(archive_info=archive(1853)))
    assert "The People&#x27;s Press" in page  # capwords, not .title(), which would yield "People'S"


def test_render_pick_escapes_a_quote_in_the_clip_url():
    comp = component("u1")
    comp["thumbs"]["clip_lg"][0]["url"] = 'https://img.example/u1"onload=alert(1).webp'
    page = render_pick(hit(1), comp, estimate_for(archive_info=archive(1853)))
    assert "&quot;" in page
    assert 'onload=alert(1)"' not in page
