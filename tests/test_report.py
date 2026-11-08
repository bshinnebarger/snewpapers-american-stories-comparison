import re

from PIL import Image

from snewpapers_tools.american_stories import ScanName
from snewpapers_tools.fetching import IssuePair
from snewpapers_tools.issues import IssueKey
from snewpapers_tools.report import crop_box, crop_group, segmentation_report, word_diff_html
from snewpapers_tools.segmentation import (
    AS_MERGE, AS_ONLY, AS_ONLY_TINY, AS_SPLIT, IN_AS_MERGE, MATCH, OURS_ONLY, segment_issue,
)

# --- word diff --------------------------------------------------------------------------------


def test_word_diff_marks_words_only_on_one_side():
    ours, theirs = word_diff_html("The river is very low today", "The river is low today.")
    assert ours == 'The river is <span class="only-ours">very</span> low today'
    assert theirs == "The river is low today."  # case and punctuation alone are no difference


def test_word_diff_marks_both_sides_of_a_replaced_run_and_inserted_words():
    ours, theirs = word_diff_html("fever rages here", "fcver rages here and there")
    assert ours == '<span class="only-ours">fever</span> rages here'
    assert theirs == '<span class="only-as">fcver</span> rages here <span class="only-as">and there</span>'


def test_word_diff_escapes_html_and_keeps_paragraph_breaks():
    ours, theirs = word_diff_html("A <b> & B\n\nNext para", "A <b> & B\n\nNext para")
    assert ours == theirs == "A &lt;b&gt; &amp; B<br><br>Next para"


def test_word_diff_with_one_side_empty_leaves_the_other_unmarked():
    assert word_diff_html("only ours", "") == ("only ours", "")


def test_crop_box_adds_a_margin_and_stays_on_the_page():
    assert crop_box((0.02, 0.5, 0.3, 0.98), 0.04) == (0.0, 0.46, 0.34, 1.0)


# --- the report -------------------------------------------------------------------------------

DIR = "https://tile.loc.gov/storage-services/service/ndnp/x/batch_x/data/sn1/00000000001/1856010101"
SCAN_SIZE = (2000, 2600)


def test_crop_group_of_a_sliver_at_the_page_edge_is_never_empty():
    crop = crop_group(Image.new("RGB", (400, 520)), (0.999981, 0.015421, 1.0, 0.824393), (), 520)
    assert crop.width >= 1 and crop.height >= 1


def test_report_has_no_crops_for_an_as_box_wholly_outside_the_scan(tmp_path):
    """The Alexandria Advertiser 1802 p1: AS box 40 lies at x 1.04-1.39 of the scan's width."""
    pair = small_pair()
    pair.pages[0][1]["bboxes"].append({"id": 9, "class": "article", "raw_text": "x", "legibility": "Legible",
                                       "bbox": {"x0": 2080, "y0": 100, "x1": 2700, "y1": 2000}})
    seg = segment_issue(pair, {1: SCAN_SIZE, 2: SCAN_SIZE})
    html, _path = segmentation_report(pair, seg, load_pages=fake_load_pages, page_width=400, reports_dir=tmp_path)
    assert "the boxes lie outside the page scan" in html


def component(uuid, page, rects, content, label="notice"):
    return {"uuid": uuid, "label": label, "content": content, "pages": [page],
            "boxes": [{"page": page, "rects": rects}],
            "links": {"loc_issue_url": "https://www.loc.gov/item/sn1/1856-01-01/ed-1/",
                      "loc_pdf_pages": [{"page_number": page, "url": f"{DIR}/000{page}.pdf"}],
                      "site_component_url": f"https://snewpapers.com/components/{uuid}_evt_0"}}


def as_page(boxes):
    """AS page data from (article id or None for an orphan, (x0, y0, x1, y1) in fractions, text), in scan pixels;
    boxes with the same article id make one article."""
    w, h = SCAN_SIZE
    bboxes = [{"id": k, "class": "article", "reading_order_id": 0, "raw_text": text, "legibility": "Legible",
               "bbox": {"x0": r[0] * w, "y0": r[1] * h, "x1": r[2] * w, "y1": r[3] * h}}
              for k, (_aid, r, text) in enumerate(boxes)]
    articles: dict[int, list[int]] = {}
    for k, (aid, _r, _text) in enumerate(boxes):
        if aid is not None:
            articles.setdefault(aid, []).append(k)
    full = [{"full_article_id": aid, "object_ids": ids, "headline": "", "article": "", "byline": ""}
            for aid, ids in articles.items()]
    return {"scan": {"width": 100, "height": 100}, "bboxes": bboxes, "full articles": full}


def small_pair():
    """Page 1: a match and an as_split. Page 2: ours only, two components inside one AS article (an
    as_merge, and each in it), an AS-only advertisement and a tiny AS-only heading."""
    comps = [
        component("a", 1, [[0.1, 0.1, 0.3, 0.4]], "The river is low and falling."),
        component("b", 1, [[0.5, 0.1, 0.7, 0.4]], "We learn that the boats run again. NOTICE. All persons are warned."),
        component("c", 2, [[0.1, 0.1, 0.3, 0.3]], "DISSOLUTION OF PARTNERSHIP. The firm is dissolved."),
        component("d", 2, [[0.1, 0.5, 0.3, 0.6]], "MARRIED. In this town, Mr. Smith to Miss Doe."),
        component("e", 2, [[0.1, 0.6, 0.3, 0.7]], "DIED. In this town, Mr. Jones, aged 80."),
    ]
    pages = [
        as_page([(1, (0.1, 0.1, 0.3, 0.4), "The river is low and falling."),
                 (2, (0.5, 0.1, 0.7, 0.25), "We learn that the boats ran again."),
                 (3, (0.5, 0.25, 0.7, 0.4), "NOTICE. All persons are warned.")]),
        as_page([(4, (0.6, 0.6, 0.9, 0.9), "Just received and for sale cheap, $1.50."),
                 (5, (0.1, 0.5, 0.3, 0.6), "MARRIED. In this town, Mr. Smith to Miss Doe."),
                 (5, (0.1, 0.6, 0.3, 0.7), "DIED. In this town, Mr. Jones, aged 80."),
                 (None, (0.6, 0.1, 0.9, 0.12), "NOTICES.")]),
    ]
    scans = [ScanName("1856-01-01", 1, "sn1", "00000000001", 1, f"000{n}") for n in (1, 2)]  # both named p1
    return IssuePair(IssueKey("i", "sn1", "1856-01-01", 1), comps, list(zip(scans, pages)), 0, 5, [])


def fake_load_pages(pair, width):
    """Plain grey scans at `width`, in place of the downloaded JP2s."""
    height = round(width * SCAN_SIZE[1] / SCAN_SIZE[0])
    return {n: (Image.new("RGB", (width, height), (230, 230, 230)), data, SCAN_SIZE)
            for n, (_scan, data) in enumerate(pair.pages, start=1)}


def build(tmp_path, **kwargs):
    pair = small_pair()
    seg = segment_issue(pair, {1: SCAN_SIZE, 2: SCAN_SIZE})
    html, path = segmentation_report(pair, seg, load_pages=fake_load_pages, page_width=400,
                                     reports_dir=tmp_path, **kwargs)
    return seg, html, path


GROUPS = 8


def test_the_fixture_has_every_case():
    seg = segment_issue(small_pair(), {1: SCAN_SIZE, 2: SCAN_SIZE})
    assert sorted(g.case for g in seg.groups) == sorted(
        [MATCH, AS_SPLIT, OURS_ONLY, IN_AS_MERGE, IN_AS_MERGE, AS_MERGE, AS_ONLY, AS_ONLY_TINY])


def test_report_is_written_where_asked_and_named_for_the_issue(tmp_path):
    _seg, html, path = build(tmp_path)
    assert path == tmp_path / "sn1_1856-01-01_ed-1" / "index.html"
    assert path.read_text() == html
    assert html.startswith("<!doctype html>")


def test_report_embedded_is_written_as_one_flat_file_named_for_the_issue(tmp_path):
    _seg, html, path = build(tmp_path, embed=True)
    assert path == tmp_path / "sn1_1856-01-01_ed-1_segmentation.html"
    assert path.read_text() == html
    assert html.startswith("<!doctype html>")


def test_report_has_a_section_per_page_and_a_badge_per_group(tmp_path):
    seg, html, _path = build(tmp_path)
    assert len(re.findall(r'<section class="page"', html)) == 2
    assert len(re.findall(r'<summary><span class="badge ', html)) == len(seg.groups) == GROUPS


def test_report_shows_neutral_case_labels_not_the_bare_machine_keys(tmp_path):
    _seg, html, _path = build(tmp_path)
    for label in ("1:1 match", "1 ours : N AS", "N ours : 1 AS", "ours only", "AS only", "AS only, tiny"):
        assert label in html
    assert '<span class="badge as-split">' in html and '<span class="badge as-only-tiny">' in html
    for key in ("as_only", "as_split", "as_merge", "ours_only"):
        assert key not in html  # the machine keys never leak into the page text


def test_report_writes_crops_as_lazy_loaded_sidecar_webp_files(tmp_path):
    _seg, html, path = build(tmp_path)
    assert len(re.findall(r'<img src="crops/[^"]+\.webp" alt="[^"]*" loading="lazy">', html)) == 2 * GROUPS
    assert "base64" not in html
    crops_dir = path.parent / "crops"
    crop_files = sorted(p.name for p in crops_dir.glob("*.webp"))
    assert len(crop_files) == 2 * GROUPS
    for name in crop_files:
        assert f'<img src="crops/{name}"' in html  # every file the html references actually exists


def test_report_can_embed_crops_as_base64_webp_instead(tmp_path):
    _seg, html, path = build(tmp_path, embed=True)
    assert len(re.findall(r'<img src="data:image/webp;base64,', html)) == 2 * GROUPS
    assert not (path.parent / "crops").exists()  # nothing written alongside: the file stands alone


def test_report_starts_with_every_group_collapsed_and_opens_the_one_a_link_points_to(tmp_path):
    seg, html, _path = build(tmp_path)
    details = re.findall(r'<details class="group ([a-z-]+)" id="g-p\d+-\d+"( open)?>', html)
    assert len(details) == len(seg.groups)
    assert all(opened == "" for _case, opened in details)
    assert "<script>" in html and "location.hash" in html and ".open = true" in html


def test_report_shows_the_word_diff_the_clipped_as_pieces_and_summary(tmp_path):
    _seg, html, _path = build(tmp_path)
    assert '<span class="only-ours">run</span>' in html
    assert '<span class="only-as">ran</span>' in html
    assert ("<li>[1] AS #2: begins: &quot;We learn that the boats ran again.&quot; … "
            "ends: &quot;We learn that the boats ran again.&quot;</li>") in html
    assert "<li>[2] AS #3: begins: &quot;NOTICE. All persons are warned.&quot;" in html
    assert "<li>[1] notice: begins: &quot;MARRIED." in html  # the as_merge's pieces are ours
    assert "as_covered_by_ours" not in html  # summary keys are given readable names
    assert "AS covered by ours" in html


def test_report_still_lists_groups_on_a_page_without_a_scan(tmp_path):
    pair = small_pair()
    seg = segment_issue(pair, {1: SCAN_SIZE, 2: SCAN_SIZE})
    only_page_1 = lambda pair, width: {1: fake_load_pages(pair, width)[1]}  # noqa: E731
    html, _path = segmentation_report(pair, seg, load_pages=only_page_1, page_width=400, reports_dir=tmp_path)
    assert len(re.findall(r'<summary><span class="badge ', html)) == GROUPS
    assert len(re.findall(r'<img src="crops/[^"]+\.webp"', html)) == 2 * 2
    assert "no scan" in html


# --- verdicts ---------------------------------------------------------------------------------


def verdicts_for(seg):
    """The as_split decided "one" both ways, the as_merge unstable, the AS-only advertisement no
    verdict, the rest not asked."""
    from snewpapers_tools.adjudicate import (
        AMERICAN_STORIES, MATCHED, NOT_CAPTURED, SNEWPAPERS, TINY, Answer, Asked, GroupVerdict, group_ids,
    )

    verdicts = []
    for g, gid in zip(seg.groups, group_ids(seg)):
        if g.case == AS_SPLIT:
            asked = [Asked(SNEWPAPERS, Answer({"verdict": "one", "reason": "one <notice> throughout"}, 0.001, "{}"), SNEWPAPERS),
                     Asked(AMERICAN_STORIES, Answer({"verdict": "one", "reason": "a single item"}, 0.001, "{}"), SNEWPAPERS)]
            verdicts.append(GroupVerdict(gid, g.case, SNEWPAPERS, asked))
        elif g.case == AS_MERGE:
            asked = [Asked(SNEWPAPERS, Answer({"verdict": "several", "reason": "two notices"}, 0.001, "{}"), AMERICAN_STORIES),
                     Asked(AMERICAN_STORIES, Answer({"verdict": "one", "reason": "one column"}, 0.001, "{}"), AMERICAN_STORIES)]
            verdicts.append(GroupVerdict(gid, g.case, "unstable", asked))
        elif g.case == AS_ONLY:
            verdicts.append(GroupVerdict(gid, g.case, "no verdict", [Asked(AMERICAN_STORIES, Answer(None, 0.002, "I think it is an <ad>"))]))
        elif g.case == IN_AS_MERGE:  # the merge's outcome
            verdicts.append(GroupVerdict(gid, g.case, "unstable"))
        else:
            verdicts.append(GroupVerdict(gid, g.case, {MATCH: MATCHED, OURS_ONLY: NOT_CAPTURED, AS_ONLY_TINY: TINY}[g.case]))
    return verdicts


def test_report_without_verdicts_has_no_scorecard(tmp_path):
    _seg, html, _path = build(tmp_path)
    assert "Scorecard" not in html and 'class="verdict' not in html


def test_report_puts_the_scorecard_first_and_each_verdict_on_its_group(tmp_path):
    pair = small_pair()
    seg = segment_issue(pair, {1: SCAN_SIZE, 2: SCAN_SIZE})
    html, _path = segmentation_report(pair, seg, load_pages=fake_load_pages, page_width=400,
                                      reports_dir=tmp_path, verdicts=verdicts_for(seg))
    assert html.index("<h2>Scorecard</h2>") < html.index("groups</th>") < html.index('<section class="page"')
    scorecard = html[html.index("<h2>Scorecard</h2>"):html.index("<h2>How to read this")]
    differ = scorecard[scorecard.index("Where the two segmentations agree or differ"):]
    assert ("<tr><th></th><th>times</th><th>snewpapers right</th><th>AS right</th><th>neither</th><th>unsure</th>"
            "<th>page furniture, not counted</th></tr>" in differ)
    assert ("<tr><th>Same article on both sides (=)</th><td>1</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr>"
            in differ)
    assert ("<tr><th>One snewpapers component, several AS articles (1 : N)</th>"
            "<td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>–</td></tr>") in differ
    assert ("<tr><th>Several snewpapers components, one AS article (N : 1)</th>"
            "<td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>–</td></tr>") in differ
    assert ("<tr><th>One snewpapers component, one AS article running on into other text</th>"
            "<td>0</td><td>0</td><td>0</td><td>–</td><td>0</td><td>0</td></tr>") in differ
    one_side = scorecard[scorecard.index("Where only one side has text"):]
    assert "<tr><th>snewpapers component, AS has no text, no AS box there (an AS error)</th><td>1</td></tr>" in one_side
    assert "<tr><th>snewpapers component, AS has no text, AS marked it illegible (an AS error)</th><td>0</td></tr>" in one_side
    assert "<tr><th>AS text in no snewpapers component: too short to ask</th><td>1</td></tr>" in one_side
    assert "<tr><th>AS boxes in no snewpapers component with no text at all (AS read nothing)</th><td>0</td></tr>" in one_side
    assert "<tr><th>AS text in no snewpapers component: no verdict</th><td>1</td></tr>" in one_side
    judged = scorecard[scorecard.index("What the decisions judged"):]
    assert "<tr><th>American Stories articles</th><td>2</td><td>0</td></tr>" in judged
    assert "<tr><th>snewpapers components</th><td>0</td><td>1</td></tr>" in judged
    errors = scorecard[scorecard.index("American Stories errors found"):]
    assert "<tr><th>fragment articles AS split our components into</th><td>2</td><td>1</td></tr>" in errors
    quality = scorecard[scorecard.index("Text quality"):]
    assert "<tr><th>snewpapers</th><td>" in quality and "<tr><th>American Stories</th><td>" in quality
    assert "Components where more of the text is real words:" in quality
    assert "<details" in scorecard and "How this was scored" in scorecard
    assert "5 questions, 0 from the cache; 0 input + 0 output tokens, $0.0060" in scorecard
    assert '<span class="badge as-split">1 ours : N AS</span><span class="verdict snewpapers">snewpapers right</span>' in html
    assert ("stated first: one component (snewpapers): <b>snewpapers right (said one)</b>: one &lt;notice&gt; throughout"
            in html)
    assert "stated first: several components (snewpapers): <b>snewpapers right (said several)</b>: two notices" in html
    assert '<span class="verdict unstable">unstable</span>' in html
    assert "<b>no verdict</b><pre>I think it is an &lt;ad&gt;</pre>" in html  # the raw response, escaped
    assert '<span class="verdict not-captured-by-as">not captured by AS</span>' in html
    assert html.count('class="verdict ') == GROUPS


def test_one_vs_many_reasons_say_which_belief_was_stated_first():
    from snewpapers_tools.adjudicate import AMERICAN_STORIES, SNEWPAPERS, Answer, Asked, GroupVerdict
    from snewpapers_tools.report import _reasons_html

    asked = [Asked(SNEWPAPERS, Answer({"verdict": "one", "reason": "one notice"}, 0.001, "{}"), SNEWPAPERS),
             Asked(AMERICAN_STORIES, Answer({"verdict": "mixed", "reason": "[2] is separate"}, 0.001, "{}"), SNEWPAPERS)]
    html = _reasons_html(GroupVerdict("p1-1", AS_SPLIT, "unstable", asked))
    assert "stated first: one component (snewpapers): <b>snewpapers right (said one)</b>: one notice" in html
    assert "stated first: several components (American Stories): <b>neither (said mixed)</b>: [2] is separate" in html
    assert "A = " not in html


# --- shared phrases and duplicated text -------------------------------------------------------


def test_report_names_the_containment_metric_shared_4_word_phrases():
    from snewpapers_tools.report import _metrics_line
    from snewpapers_tools.segmentation import Group

    g = Group(1, MATCH, [], [], metrics={"ratio": 90.0, "token_set_ratio": 95.0, "shared_phrases": 80.0}, other_pages=[2])
    assert _metrics_line(g) == "ratio 90.0 · token set 95.0 · shared 4-word phrases 80.0 · continues on p. 2"


def duplicated_pair():
    """Page 1: our item, which AS has twice, in articles 5 and 83."""
    item = "LEARNED FLEAS. The Italian correspondent of the Newark Daily Advertiser writes of a show of fleas."
    twice = "LEARNED FLEAs.-The Italian correspondent { the Newark Daily Advertiser writes of a show of fleas"
    page = as_page([(5, (0.1, 0.1, 0.3, 0.25), item), (83, (0.1, 0.25, 0.3, 0.4), twice)])
    scan = ScanName("1856-01-01", 1, "sn1", "00000000001", 1, "0001")
    return IssuePair(IssueKey("i", "sn1", "1856-01-01", 1), [component("a", 1, [[0.1, 0.1, 0.3, 0.4]], item)],
                     [(scan, page)], 0, 1, [])


def test_report_says_american_stories_duplicated_the_text_and_counts_it(tmp_path):
    from snewpapers_tools.adjudicate import MATCHED, GroupVerdict

    pair = duplicated_pair()
    seg = segment_issue(pair, {1: SCAN_SIZE})
    [g] = seg.groups
    assert g.case == MATCH and g.as_duplicated
    verdicts = [GroupVerdict("p1-1", MATCH, MATCHED, as_duplicated=True)]
    html, _path = segmentation_report(pair, seg, load_pages=fake_load_pages, page_width=400, reports_dir=tmp_path,
                                      verdicts=verdicts)
    assert "American Stories duplicated this text (articles #5 and #83)" in html
    assert "<tr><th>text AS has twice</th><td>1</td><td>–</td></tr>" in html[html.index("American Stories errors found"):]
    assert "<tr><th>AS duplicated text</th><td>1</td></tr>" in html  # the summary


# --- AS text attached outside our component ---------------------------------------------------


def attached_pair():
    """Page 1: AS article 31 (a college notice, no component of ours) ends with its date line; AS
    article 7 is that date line again, just above our notice, then the notice our component covers."""
    college = "GOLDSBORO' FEMALE COLLEGE. The third Session will commence in January. W. K. LANE.\nDec Is, 1854. l5-tf"
    body = "NOTICE. THE CO-PARTNERSHIP heretofore existing under the name of MERRIMAN, WOODRUFF & CO. is dissolved."
    page = as_page([(31, (0.1, 0.1, 0.3, 0.39), college), (7, (0.1, 0.39, 0.16, 0.395), "Dec. Is, 1854."),
                    (7, (0.1, 0.4, 0.3, 0.6), body)])
    scan = ScanName("1856-01-01", 1, "sn1", "00000000001", 1, "0001")
    return IssuePair(IssueKey("i", "sn1", "1856-01-01", 1), [component("n", 1, [[0.1, 0.4, 0.3, 0.6]], body)],
                     [(scan, page)], 0, 1, [])


def test_report_shows_the_text_american_stories_attaches_outside_with_its_verdict(tmp_path):
    from snewpapers_tools.adjudicate import (
        AMERICAN_STORIES, EXCLUDED, MATCHED, SNEWPAPERS, Answer, Asked, GroupVerdict, group_ids,
    )

    pair = attached_pair()
    seg = segment_issue(pair, {1: SCAN_SIZE})
    notice = next(g for g in seg.groups if g.case == MATCH)
    assert notice.attached_text == "Dec. Is, 1854." and notice.attached_duplicates == [("#31", "#7")]
    asked = [Asked(SNEWPAPERS, Answer({"verdict": "separate", "reason": "it closes the <college> notice"}, 0.001, "{}"),
                   AMERICAN_STORIES),
             Asked(AMERICAN_STORIES, Answer({"verdict": "separate", "reason": "a date line above"}, 0.001, "{}"),
                   AMERICAN_STORIES)]
    verdicts = []
    for g, gid in zip(seg.groups, group_ids(seg)):
        if g is notice:
            verdicts.append(GroupVerdict(gid, MATCH, MATCHED, as_duplicated=True, attached_outcome=SNEWPAPERS,
                                         attached_asked=asked))
        else:
            verdicts.append(GroupVerdict(gid, g.case, EXCLUDED))
    html, _path = segmentation_report(pair, seg, load_pages=fake_load_pages, page_width=400, reports_dir=tmp_path,
                                      verdicts=verdicts)
    assert ("American Stories also attaches: «Dec. Is, 1854.» (outside this component; AS article #7) "
            '<span class="verdict snewpapers">snewpapers right</span>') in html
    assert "American Stories duplicated it: #31 also holds the text of #7" in html
    assert ("stated first: the component does not include it (snewpapers): <b>snewpapers right (said separate)</b>: "
            "it closes the &lt;college&gt; notice") in html
    assert "stated first: the component includes it (American Stories): <b>snewpapers right (said separate)</b>" in html
    assert "American Stories duplicated this text" not in html  # nothing duplicated inside the component
    assert ("<tr><th>AS text attached outside a snewpapers component</th>"
            "<td>1</td><td>1</td><td>0</td><td>–</td><td>0</td><td>0</td></tr>") in html
    assert "<tr><th>AS text attached outside ours</th><td>1</td></tr>" in html
    assert "2 questions, 0 from the cache" in html


# --- same LoC scan ----------------------------------------------------------------------------


def test_report_header_says_how_many_pages_are_the_same_loc_scan(tmp_path):
    _seg, html, _path = build(tmp_path)
    assert "<p>Same LoC scan for 2 of 2 pages.</p>" in html
    assert "not meaningful" not in html


def test_report_warns_when_american_stories_segmented_another_digitisation(tmp_path):
    pair = small_pair()
    scan, data = pair.pages[1]
    pair.pages[1] = (ScanName(scan.date, 2, scan.lccn, "01000718192", 1, "0007"), data)  # page 2: an older scan
    seg = segment_issue(pair, {1: SCAN_SIZE, 2: SCAN_SIZE})
    assert {g.page for g in seg.groups} == {1}
    html, _path = segmentation_report(pair, seg, load_pages=fake_load_pages, page_width=400, reports_dir=tmp_path)
    assert "<b>Same LoC scan for 1 of 2 pages.</b>" in html
    assert "another LoC digitisation of p. 2 than" in html and "IoU on those pages is not meaningful" in html
    assert "<tr><th>pages left out: AS scan is another digitisation</th><td>[2]</td></tr>" in html
    assert "pages American Stories lacks" not in html


# --- the benchmark index ----------------------------------------------------------------------


def test_benchmark_row_regroups_the_scorecard_and_totals_add_up():
    import json

    from snewpapers_tools.report import benchmark_index_html, benchmark_row, benchmark_text, benchmark_totals

    pair = small_pair()
    seg = segment_issue(pair, {1: SCAN_SIZE, 2: SCAN_SIZE})
    row = benchmark_row("1850s", "The <Herald>", "1856-01-01", seg, verdicts_for(seg), "../sn1_1856-01-01_ed-1/index.html")
    assert row["same_scan"] == [2, 2]
    assert row["components"] == {"correct as is": 2, "AS split it correctly": 0, "AS merged it correctly": 0,
                                 "neither": 0, "unstable": 2, "no verdict": 0, "partly captured by AS": 0,
                                 "AS's wider article right": 0,
                                 "not captured by AS": 1}
    assert row["nibbles"] == {"snewpapers": 0, "american_stories": 0, "other": 0}
    assert row["merges"] == {"snewpapers": 0, "american_stories": 0, "other": 1}  # the unstable as_merge
    assert row["attached"] == {"snewpapers": 0, "american_stories": 0, "other": 0}
    assert row["as_only"] == {"correctly not served": 0, "missed by snewpapers": 0}  # the one asked had no verdict
    assert row["as_defects"] == {"AS duplicated text": 0, "nested boxes": 0, "ads kept": 0, "illegible read": 0}
    assert row["as_illegible_area"] == [0, row["as_illegible_area"][1]] and row["as_illegible_area"][1] > 0
    assert row["judged_wrong"] == {"as_articles": 2, "components": 0}  # one decision, its two AS articles
    assert row["scorecard"]["judged"]["components"] == {"wrong": 0, "right": 1}
    assert row["as_errors"] == {"merge_decisions": 0, "items AS merged": 0, "split_decisions": 1, "fragment articles": 2,
                                "nibble_decisions": 0, "lines in another AS article": 0,
                                "AS articles missing part of ours": 0, "AS articles running on into other items": 0,
                                "our components AS has no text for": 1, "pages of our components AS lacks the text for": 0}
    assert row["text_median"] == seg.summary()["median_match_ratio"] and len(row["match_ratios"]) == 1
    assert row["cost"] == 0.006
    json.dumps(row)  # saved as scorecards.json

    total = benchmark_totals([row, row])
    start = benchmark_index_html([{**row, "overlay": "overlays/x.webp"}] * 2, intro="<p>Start here.</p>")
    summary = start[start.index("Start here."):start.index("All the scorecard numbers")]
    assert summary.count('<a href="../sn1_1856-01-01_ed-1/index.html">report</a>') == 2
    assert summary.count('<a href="overlays/x.webp">overlay</a>') == 2
    for heading in ("our components", "AS articles judged wrong", "snewpapers components judged wrong",
                    "real words, snewpapers", "real words, AS"):
        assert heading in summary
    assert "<td>2 issues</td>" in summary  # the totals row
    assert "All the scorecard numbers" not in benchmark_index_html([row])  # without an intro, the table alone
    assert total["same_scan"] == [4, 4] and total["components"]["correct as is"] == 4 and total["merges"]["other"] == 2
    assert total["as_errors"]["fragment articles"] == 4 and total["as_errors"]["split_decisions"] == 2
    assert total["cost"] == 0.012 and total["text_median"] == row["text_median"]
    assert total["judged_wrong"] == {"as_articles": 4, "components": 0}

    html = benchmark_index_html([row, row])
    assert html.startswith("<!doctype html>") and ".seg-report" in html
    assert html.count('<a href="../sn1_1856-01-01_ed-1/index.html">report</a>') == 2
    assert "overlay</a>" not in html
    with_overlay = benchmark_index_html([{**row, "overlay": "overlays/sn1_1856-01-01_ed-1_verdicts.webp"}])
    assert ('<a href="../sn1_1856-01-01_ed-1/index.html">report</a> · '
            '<a href="overlays/sn1_1856-01-01_ed-1_verdicts.webp">overlay</a>') in with_overlay
    assert "The &lt;Herald&gt;" in html and html.count("<tr") == 2 + 2 + 1  # two header rows, two issues, the totals
    assert '<tr class="total"><td class="">total</td><td class="">2 issues</td>' in html
    assert '<td class="n first">4 of 4</td>' in html and "$0.0120" in html
    for heading in ("snewpapers components", "N ours : 1 AS", "attached outside", "AS only", "AS defects", "judged wrong",
                    "AS articles", "snewpapers comps", "AS errors (decided for snewpapers)", "items AS merged", "lines in another AS article",
                    "nibbles",
                    "fragment articles", "LLM cost"):
        assert heading in html

    text = benchmark_text([row])
    assert len(text.splitlines()) == 3 and "1850s" in text and "total" in text and "2/0/0/0/2/0/0/0/1" in text and "wrong as/ours" in text
    assert " 2/0 " in text.splitlines()[1]  # AS articles / snewpapers components judged wrong
    assert " 0/0/1/2/0/0/0/1 " in text.splitlines()[1]  # AS errors: N:1 decisions / items merged / 1:N decisions / fragments / lines elsewhere


# --- pieces inside an AS merge ------------------------------------------------------------------

ITEMS = ["MARRIED. In this town, on Tuesday last, Mr. John Smith to Miss Mary Doe, both of this county.",
         "DIED. At his residence, Mr. Peter Brown, aged 80.",
         "FIRE. On Monday night the stable of Mr. Jones was burnt to the ground, with two horses and all his hay."]


def items_pair():
    """One AS article, a headline and ONE body box over three items of ours."""
    comps = [component("m", 1, [[0.1, 0.1, 0.3, 0.3]], ITEMS[0], "local_news"),
             component("d", 1, [[0.1, 0.3, 0.3, 0.4]], ITEMS[1], "local_news"),
             component("f", 1, [[0.1, 0.4, 0.3, 0.9]], ITEMS[2], "local_news")]
    page = as_page([(9, (0.1, 0.08, 0.3, 0.1), "LOCAL ITEMS."), (9, (0.1, 0.1, 0.3, 0.9), "\n".join(ITEMS))])
    page["bboxes"][1]["reading_order_id"] = 1
    scans = [ScanName("1856-01-01", 1, "sn1", "00000000001", 1, "0001")]
    return IssuePair(IssueKey("i", "sn1", "1856-01-01", 1), comps, list(zip(scans, [page])), 0, 3, [])


def test_report_outlines_only_real_as_boxes_and_lists_the_slices_under_the_merge(tmp_path, monkeypatch):
    from snewpapers_tools import report

    pair = items_pair()
    seg = segment_issue(pair, {1: SCAN_SIZE})
    assert sorted(g.case for g in seg.groups) == sorted([AS_ONLY_TINY, AS_MERGE] + [IN_AS_MERGE] * 3)
    outlined = []
    real = report.crop_group
    monkeypatch.setattr(report, "crop_group", lambda drawn, box, own, w: outlined.append(own) or real(drawn, box, own, w))
    html, _path = segmentation_report(pair, seg, load_pages=fake_load_pages, page_width=400, reports_dir=tmp_path)
    as_outlines = {r for own in outlined[1::2] for r in own}  # each group's AS crop comes second
    slices = {r for g in seg.groups if g.case == IN_AS_MERGE for r in g.as_rects}
    assert not as_outlines & slices  # never a slice drawn as an AS box
    assert {r[1:4:2] for r in as_outlines} <= {(0.08, 0.1), (0.1, 0.9)}  # the headline's and body box's own extent
    merge_id = next(f"p1-{k}" for k, g in enumerate(seg.groups, start=1) if g.case == AS_MERGE)
    assert html.count(f"inside AS's merged article, decided by {merge_id}") == 3
    inside = html[html.index("Text agreement inside AS's merged article"):]
    assert "(3 pieces)" in inside and "these are no matches" in inside
    for text in ITEMS:
        assert text.split(".")[0] in inside
    assert "American Stories (slice of #9)" in inside and "ratio 100.0" in inside


def cross_page_pair():
    """A story on pages 1-2: AS #1 and #2 hold it on page 1, AS #7 on page 2."""
    story = {"uuid": "s", "label": "domestic_news", "content": "MESSAGE. The President speaks. And ends here.",
             "pages": [1, 2],
             "boxes": [{"page": 1, "rects": [[0.1, 0.1, 0.3, 0.9]]}, {"page": 2, "rects": [[0.1, 0.1, 0.3, 0.4]]}],
             "links": {"loc_pdf_pages": [{"page_number": n, "url": f"{DIR}/000{n}.pdf"} for n in (1, 2)]}}
    pages = [as_page([(1, (0.1, 0.1, 0.3, 0.5), "MESSAGE. The President speaks."),
                      (2, (0.1, 0.5, 0.3, 0.9), "He speaks on.")]),
             as_page([(7, (0.1, 0.1, 0.3, 0.4), "And ends here.")])]
    scans = [ScanName("1856-01-01", 1, "sn1", "00000000001", 1, f"000{n}") for n in (1, 2)]
    return IssuePair(IssueKey("i", "sn1", "1856-01-01", 1), [story], list(zip(scans, pages)), 0, 1, [])


def test_report_shows_a_component_across_pages_once_with_a_crop_pair_per_page_and_a_pointer_on_the_later_page(tmp_path):
    pair = cross_page_pair()
    seg = segment_issue(pair, {1: SCAN_SIZE, 2: SCAN_SIZE})
    assert [(g.page, g.case) for g in seg.groups] == [(1, AS_SPLIT), (2, "continued")]
    html, path = segmentation_report(pair, seg, load_pages=fake_load_pages, page_width=400, reports_dir=tmp_path)
    split = html[html.index('id="g-p1-1"'):html.index('id="g-p2-1"')]
    assert "one component across p. 1–2" in split and "1 snewpapers : 3 AS" in split
    assert "AS #1 (p. 1), AS #2 (p. 1), AS #7 (p. 2)" in split
    assert "[3] AS #7 (p. 2): begins:" in split
    assert 'continued in <a href="#g-p2-1">p2-1</a>' in split
    crops = sorted(f.name for f in (path.parent / "crops").iterdir())
    assert crops == ["p1-1_p1_as.webp", "p1-1_p1_ours.webp", "p1-1_p2_as.webp", "p1-1_p2_ours.webp",
                     "p2-1_as.webp", "p2-1_ours.webp"]
    pointer = html[html.index('id="g-p2-1"'):]
    assert 'continues <a href="#g-p1-1">p1-1</a>: compared, asked and decided there' in pointer
    assert 'class="text"' not in pointer  # its texts are compared in p1-1


# --- coverage, IoU and nibbles --------------------------------------------------------------------


def test_report_metrics_line_says_how_much_of_ours_as_covers_and_the_iou(tmp_path):
    _seg, html, _path = build(tmp_path)
    assert "AS covers 100% · IoU 1.00" in html


def nibble_pair():
    """Page 1: our address (title and body); AS article 17 is the dateline above it and the title,
    AS article 4 the body. The title is a nibble: one line of ours in another AS article."""
    body = "IF ever there was a time when national exertion, firmness and unanimity were necessary. " * 6
    page = as_page([(17, (0.1, 0.05, 0.3, 0.07), "Monday, June 11, 1798."),
                    (17, (0.1, 0.1, 0.3, 0.11), "To the CITIZENS of the United States."),
                    (4, (0.1, 0.11, 0.3, 0.6), body)])
    scan = ScanName("1856-01-01", 1, "sn1", "00000000001", 1, "0001")
    comp = component("n", 1, [[0.1, 0.1, 0.3, 0.6]], "To the Citizens of the United States. " + body, label="letter_to_editor")
    return IssuePair(IssueKey("i", "sn1", "1856-01-01", 1), [comp], [(scan, page)], 0, 1, [])


def test_report_shows_a_line_american_stories_puts_in_another_article_with_its_verdict(tmp_path):
    from snewpapers_tools.adjudicate import AMERICAN_STORIES, MATCHED, SNEWPAPERS, Answer, Asked, GroupVerdict

    pair = nibble_pair()
    seg = segment_issue(pair, {1: SCAN_SIZE})
    [g] = [g for g in seg.groups if g.pieces]
    assert g.case == MATCH and [p.short_name for p in g.as_nibbles] == ["#17"]
    asked = [Asked(s, Answer({"verdict": "belongs", "reason": "the address's title"}, 0.001, "{}"), SNEWPAPERS)
             for s in (SNEWPAPERS, AMERICAN_STORIES)]
    verdicts = [GroupVerdict(gid, x.case, MATCHED if x is g else "correctly not served",
                             nibble_outcome=SNEWPAPERS if x is g else None, nibble_asked=asked if x is g else [])
                for gid, x in zip([f"p1-{k}" for k in range(1, len(seg.groups) + 1)], seg.groups)]
    html, _path = segmentation_report(pair, seg, load_pages=fake_load_pages, page_width=400, reports_dir=tmp_path,
                                      verdicts=verdicts)
    assert ("American Stories puts part of this component in another article: «To the CITIZENS of the United States.» "
            '(inside this component; AS article #17) <span class="verdict snewpapers">snewpapers right</span>') in html
    assert "stated first: the component includes it (snewpapers): <b>snewpapers right (said belongs)</b>" in html
    assert ("<tr><th>A line of a snewpapers component in another AS article</th>"
            "<td>1</td><td>1</td><td>0</td><td>–</td><td>0</td><td>0</td></tr>") in html
    assert "<tr><th>lines of ours AS put in another article</th><td>1</td><td>1</td></tr>" in html  # AS errors


def test_report_metrics_line_gives_each_sides_real_word_rate(tmp_path):
    _seg, html, _path = build(tmp_path)
    assert "real words: snewpapers 100%, AS 100%" in html


def test_benchmark_row_sums_real_words_over_the_components_both_sides_have_text_for():
    from snewpapers_tools.report import benchmark_index_html, benchmark_row, benchmark_totals

    seg = segment_issue(nibble_pair(), {1: SCAN_SIZE})
    row = benchmark_row("1790s", "Gazette", "1798-06-11", seg, verdicts_for(seg), "r.html")
    real, total = row["real_words"]["snewpapers"]
    assert total > 0 and real == total  # ours: every word real
    assert row["real_words"]["american_stories"][1] > 0
    assert benchmark_totals([row, row])["real_words"]["snewpapers"] == [2 * real, 2 * total]
    assert "real words" in benchmark_index_html([row])


def test_real_word_summary_counts_both_sides_and_which_reads_better_per_component():
    from snewpapers_tools.report import real_word_summary

    seg = segment_issue(small_pair(), {1: SCAN_SIZE, 2: SCAN_SIZE})
    summary = real_word_summary(seg)
    assert summary["snewpapers"][1] > 0 and summary["american_stories"][1] > 0
    assert sum(summary["more_real"].values()) == summary["components"] == 4  # the match, the split and the two in the merge


def test_report_notes_as_boxes_nested_inside_another_and_how_they_are_treated():
    from snewpapers_tools.report import nested_note_html

    seg = segment_issue(small_pair(), {1: SCAN_SIZE, 2: SCAN_SIZE})
    assert nested_note_html(seg) == ""
    seg.as_nested = [(2, "#67", "#24"), (2, "#40", "#24"), (2, "#36", "#27"), (2, "#36", "#27")]
    note = nested_note_html(seg)
    assert "4 American Stories boxes lie inside another of its boxes" in note
    assert "p. 2: #67 and #40 inside #24; p. 2: #36 (2 boxes) inside #27" in note
    assert "text AS has twice" in note



def test_report_counts_as_boxes_marked_illegible_that_snewpapers_read_and_our_real_words_there(tmp_path):
    from snewpapers_tools.report import illegible_summary

    seg = segment_issue(small_pair(), {1: SCAN_SIZE, 2: SCAN_SIZE})
    assert illegible_summary(seg) == {"boxes": 0, "components": 0, "real_words": [0, 0]}
    seg.illegible_read = [(1, 9, "#1", ("a",)), (1, 10, "#1", ("a",))]
    summary = illegible_summary(seg)
    assert summary["boxes"] == 2 and summary["components"] == 1 and summary["real_words"] == [5, 5]  # "The river is low and falling.": 5 words of 3+ letters
    html, _path = segmentation_report(small_pair(), seg, load_pages=fake_load_pages, page_width=400, reports_dir=tmp_path,
                                      verdicts=verdicts_for(seg))
    assert "<tr><th>AS boxes it marked illegible that snewpapers read</th><td>2</td><td>–</td></tr>" in html
    assert "AS marked 2 boxes illegible over 1 of our components" in html



def test_metrics_line_names_the_share_as_marked_illegible():
    from snewpapers_tools.report import _metric

    assert _metric("illegible_cover", 0.84) == "AS marked 84% illegible"


def test_the_verdict_badge_says_what_as_has_of_a_component_it_lacks_text_for():
    from snewpapers_tools.adjudicate import NOT_CAPTURED, PARTLY_CAPTURED, GroupVerdict
    from snewpapers_tools.report import _verdict_badge
    from snewpapers_tools.segmentation import AS_PARTIAL, OURS_ONLY, AsBox, AsPart, Group, OurPiece

    piece = OurPiece("u", "notice", 4, ((0.2, 0.23, 0.34, 0.29),), "LETTER BOX. Some text.", (4,))
    head = AsPart(23, (AsBox(4, 40, 23, (0.235, 0.235, 0.304, 0.244), "LETTER BOX.", "headline", "Legible"),))
    body = AsPart(23, (AsBox(4, 41, 23, (0.2, 0.245, 0.34, 0.28), "Some text", "article", "Legible"),))
    headline_only = Group(4, AS_PARTIAL, [piece], [head], metrics={"as_cover": 0.2, "illegible_cover": 0.8})
    assert ">AS has the headline only (body marked illegible)<" in _verdict_badge(GroupVerdict("p4-12", AS_PARTIAL, PARTLY_CAPTURED), headline_only)
    headline_only.metrics = {"as_cover": 0.2}
    assert ">AS has the headline only (body in no AS box)<" in _verdict_badge(GroupVerdict("p4-12", AS_PARTIAL, PARTLY_CAPTURED), headline_only)
    some = Group(4, AS_PARTIAL, [piece], [head, body], metrics={"as_cover": 0.7, "illegible_cover": 0.3})
    assert ">partly captured by AS (the rest marked illegible)<" in _verdict_badge(GroupVerdict("p4-1", AS_PARTIAL, PARTLY_CAPTURED), some)
    none = Group(4, OURS_ONLY, [piece], [], metrics={"illegible_cover": 1.0})
    assert ">not captured by AS (marked illegible)<" in _verdict_badge(GroupVerdict("p4-2", OURS_ONLY, NOT_CAPTURED), none)



def test_text_quality_table_gives_the_share_of_as_boxed_area_marked_illegible_and_unread(tmp_path):
    seg = segment_issue(small_pair(), {1: SCAN_SIZE, 2: SCAN_SIZE})
    seg.as_illegible_area = (250, 1000)
    html, _path = segmentation_report(small_pair(), seg, load_pages=fake_load_pages, page_width=400, reports_dir=tmp_path,
                                      verdicts=verdicts_for(seg))
    quality = html[html.index("Text quality"):]
    assert "<tr><th>American Stories boxed area it marked illegible and did not read</th><td>–</td><td>–</td><td>25.0%</td></tr>" in quality


def test_every_case_has_a_badge_colour():
    from snewpapers_tools.report import CSS, _css_case
    from snewpapers_tools.segmentation import CASES

    assert [c for c in CASES if f".seg-report .badge.{_css_case(c)} {{" not in CSS] == []


def test_each_group_shows_its_id_after_its_names_so_it_can_be_found_with_ctrl_f(tmp_path):
    _seg, html, _path = build(tmp_path)
    for summary in re.findall(r'<details class="group [a-z-]+" id="g-(p\d+-\d+)"[^>]*><summary>(.*?)</summary>', html, re.S):
        gid, text = summary
        assert f'<span class="gid">{gid}</span>' in text


def test_the_match_rule_is_explained_as_coverage_both_ways_line_by_line():
    from snewpapers_tools.adjudicate import scorecard
    from snewpapers_tools.report import CASE_HELP, _scorecard_html
    from snewpapers_tools.segmentation import AS_PARTIAL, AS_WIDER

    method = _scorecard_html(scorecard([]))
    method = method[method.index("How this was scored"):]
    assert "both ways" in method and "line by line" in method
    assert "90% of our component's area" not in method
    assert "line by line" in CASE_HELP[AS_PARTIAL] and "area" not in CASE_HELP[AS_PARTIAL]
    assert "marked it illegible" in CASE_HELP[AS_PARTIAL]
    assert "line by line" in CASE_HELP[AS_WIDER]


def test_a_published_report_and_index_can_leave_the_llm_usage_out(tmp_path):
    from snewpapers_tools.report import benchmark_index_html, benchmark_row

    pair = small_pair()
    seg = segment_issue(pair, {1: SCAN_SIZE, 2: SCAN_SIZE})
    html, _path = segmentation_report(pair, seg, load_pages=fake_load_pages, page_width=400,
                                      reports_dir=tmp_path, verdicts=verdicts_for(seg), usage=False)
    assert "<h2>Scorecard</h2>" in html and "from the cache" not in html and "output tokens" not in html
    row = benchmark_row("1850s", "The Herald", "1856-01-01", seg, verdicts_for(seg), "x/index.html")
    assert "LLM cost" in benchmark_index_html([row])
    index = benchmark_index_html([row], usage=False)
    assert "LLM cost" not in index and "$" not in index
