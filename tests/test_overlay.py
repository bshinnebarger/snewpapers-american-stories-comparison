from PIL import Image

from snewpapers_tools.american_stories import ScanName
from snewpapers_tools.overlay import (
    AS_COLORS,
    FAMILY_COLORS,
    LABEL_INK,
    components_by_page,
    draw_american_stories_page,
    draw_snewpapers_page,
    family_of,
    tile_pages,
    two_row_sheet,
)


def _closer_to_white_than_to(pixel: tuple, color: tuple) -> bool:
    to_white = sum(abs(c - 255) for c in pixel)
    to_color = sum(abs(p - c) for p, c in zip(pixel, color))
    return to_white < to_color


def page_scan():
    """A 1000 x 2000 scan with one two-box article (headline + body) and an orphan author box."""
    return {
        "scan": {"width": 1000, "height": 2000},
        "bboxes": [
            {"id": 0, "class": "headline", "bbox": {"x0": 100, "y0": 100, "x1": 900, "y1": 300}, "reading_order_id": -1},
            {"id": 1, "class": "article", "bbox": {"x0": 100, "y0": 400, "x1": 900, "y1": 1800}, "reading_order_id": 0},
            {"id": 2, "class": "author", "bbox": {"x0": 50, "y0": 1900, "x1": 300, "y1": 1950}, "reading_order_id": 0},
        ],
        "full articles": [{"full_article_id": 7, "object_ids": [0, 1]}],
    }


def test_draw_page_outlines_each_box_in_its_class_colour_at_display_scale():
    image = Image.new("RGB", (500, 1000), "white")  # half the scan size
    drawn = draw_american_stories_page(image, page_scan())
    assert drawn.size == (500, 1000)
    assert drawn.getpixel((50, 50)) == AS_COLORS["headline"]  # headline box top-left corner, scaled by 0.5
    assert drawn.getpixel((50, 200)) == AS_COLORS["article"]  # article box top-left corner
    assert drawn.getpixel((25, 950)) == AS_COLORS["author"]
    assert image.getpixel((50, 50)) == (255, 255, 255)  # the input is not modified


def test_draw_page_scales_by_the_real_scan_size_when_the_stated_one_is_approximate():
    # The JSON says 1000 x 2000 (dimensions_approx), but the JP2 is 2000 x 4000: boxes are in
    # the real scan's pixels, so on a 500 x 1000 display they land at a quarter, not a half.
    page = page_scan() | {"scan": {"width": 1000, "height": 2000, "dimensions_approx": True}}
    drawn = draw_american_stories_page(Image.new("RGB", (500, 1000), "white"), page, scan_size=(2000, 4000))
    assert drawn.getpixel((25, 25)) == AS_COLORS["headline"]
    # (300, 50) sits inside the headline box only if it were wrongly scaled by the stated (not the
    # real) size; at the correct quarter scale no box reaches it, so it stays plain white.
    assert drawn.getpixel((300, 50)) == (255, 255, 255)


def fill_label_scan():
    """A 1000 x 1000 scan (1:1 display scale) with one large box and one 6 x 6 tiny box."""
    return {
        "scan": {"width": 1000, "height": 1000},
        "bboxes": [
            {"id": 0, "class": "article", "bbox": {"x0": 100, "y0": 100, "x1": 400, "y1": 300}, "reading_order_id": 0},
            {"id": 1, "class": "author", "bbox": {"x0": 600, "y0": 100, "x1": 606, "y1": 106}, "reading_order_id": 0},
        ],
        "full articles": [],
    }


def test_draw_page_fills_a_box_with_a_translucent_blend_of_its_colour():
    drawn = draw_american_stories_page(Image.new("RGB", (1000, 1000), "white"), fill_label_scan())
    pixel = drawn.getpixel((110, 290))  # inside the box, away from the outline and the centred label
    color = AS_COLORS["article"]
    assert pixel != (255, 255, 255)
    assert pixel != color
    assert _closer_to_white_than_to(pixel, color)


def test_draw_page_outline_stays_the_pure_colour_over_the_fill():
    drawn = draw_american_stories_page(Image.new("RGB", (1000, 1000), "white"), fill_label_scan())
    assert drawn.getpixel((100, 150)) == AS_COLORS["article"]  # left edge of the box's outline


def test_draw_page_skips_the_label_on_a_box_too_small_to_fit_it():
    drawn = draw_american_stories_page(Image.new("RGB", (1000, 1000), "white"), fill_label_scan())
    pixel = drawn.getpixel((603, 103))  # centre of the 6 x 6 "author" box
    color = AS_COLORS["author"]
    assert pixel not in ((255, 255, 255), LABEL_INK)  # no label backing or ink was drawn here
    assert pixel != color  # not on the outline either
    assert _closer_to_white_than_to(pixel, color)  # it's the translucent fill blend


def test_draw_page_draws_the_label_near_the_centre_of_a_large_box():
    drawn = draw_american_stories_page(Image.new("RGB", (1000, 1000), "white"), fill_label_scan())
    centre_region = [drawn.getpixel((x, y)) for x in range(200, 301) for y in range(170, 231)]
    assert LABEL_INK in centre_region


def test_tile_pages_lays_pages_out_in_rows_with_a_header():
    pages = {p: Image.new("RGB", (100, 150), "white") for p in (1, 2, 3, 4, 5)}
    sheet = tile_pages(pages, columns=4, title="Wilmington Journal 1856-08-08")
    assert sheet.width == 4 * 100 + 5 * 12  # four pages and five gutters
    assert sheet.height > 2 * 150  # two rows plus the header and page labels


def snewpapers_component(label="editorial", boxes=None):
    """One snewpapers component with `boxes` as the API's POST /search shape gives them."""
    return {
        "uuid": "c1", "label": label,
        "boxes": boxes if boxes is not None else [
            {"page": 1, "rects": [[0.1, 0.1, 0.3, 0.3], [0.5, 0.1, 0.7, 0.3]]},
            {"page": 2, "rects": [[0.1, 0.1, 0.3, 0.3]]},
        ],
    }


def test_family_of_maps_labels_to_their_family_and_falls_back_to_other():
    assert family_of("editorial") == "opinion"
    assert family_of("domestic_news") == "news"
    assert family_of("crossword_puzzle") == "literary"
    assert family_of("radio_listings") == "listings"
    assert family_of("something_unknown") == "other"
    assert family_of(None) == "other"


def test_draw_snewpapers_page_outlines_each_rect_in_its_family_colour_on_each_page():
    comp = snewpapers_component()  # "editorial" -> family "opinion"
    image = Image.new("RGB", (1000, 1000), "white")

    page1 = draw_snewpapers_page(image, [comp], 1)
    assert page1.getpixel((100, 100)) == FAMILY_COLORS["opinion"]  # first rect, top-left corner
    assert page1.getpixel((500, 100)) == FAMILY_COLORS["opinion"]  # second rect on the same page

    page2 = draw_snewpapers_page(image, [comp], 2)
    assert page2.getpixel((100, 100)) == FAMILY_COLORS["opinion"]  # the single rect on page 2
    assert image.getpixel((100, 100)) == (255, 255, 255)  # the input is not modified


def test_draw_snewpapers_page_unknown_label_is_grey():
    comp = snewpapers_component(label="mystery_label", boxes=[{"page": 1, "rects": [[0.1, 0.1, 0.3, 0.3]]}])
    drawn = draw_snewpapers_page(Image.new("RGB", (1000, 1000), "white"), [comp], 1)
    assert drawn.getpixel((100, 100)) == FAMILY_COLORS["other"] == (138, 138, 138)


def test_draw_snewpapers_page_skips_a_component_without_boxes_for_this_page():
    comp = snewpapers_component(boxes=[{"page": 2, "rects": [[0.1, 0.1, 0.3, 0.3]]}])
    drawn = draw_snewpapers_page(Image.new("RGB", (1000, 1000), "white"), [comp], 1)
    assert drawn.getpixel((100, 100)) == (255, 255, 255)


def test_draw_snewpapers_page_skips_a_component_with_no_boxes_at_all():
    comp = {"uuid": "c2", "label": "editorial"}
    drawn = draw_snewpapers_page(Image.new("RGB", (1000, 1000), "white"), [comp], 1)
    assert drawn.getpixel((500, 500)) == (255, 255, 255)


def test_two_row_sheet_stacks_pages_aligned_in_two_rows_with_a_header():
    top = {1: Image.new("RGB", (100, 150), "white"), 2: Image.new("RGB", (100, 150), "white")}
    bottom = {1: Image.new("RGB", (100, 150), "white"), 2: Image.new("RGB", (100, 150), "white")}
    sheet = two_row_sheet(top, bottom, title="Wilmington Journal 1856-08-08")
    assert sheet.width == 2 * 100 + 3 * 12  # two page columns and three gutters
    assert sheet.height > 2 * 150  # two rows plus header, page labels and gutters


def test_components_by_page_lists_our_components_under_each_page():
    comps = [
        {"uuid": "a", "pages": [1], "label": "news", "char_count": 900, "content": "Yellow fever at New York quarantine..."},
        {"uuid": "b", "pages": [1, 2], "label": "editorial", "char_count": 3000, "content": "THE FEVER.\nWe learn"},
    ]
    table = components_by_page(comps)
    assert table.to_dict("records")[:2] == [
        {"page": 1, "label": "news", "chars": 900, "spans_pages": "1", "opening": "Yellow fever at New York quarantine..."},
        {"page": 1, "label": "editorial", "chars": 3000, "spans_pages": "1-2", "opening": "THE FEVER. We learn"},
    ]
    assert len(table) == 3  # b listed on page 2 as well


TILE_DIR = "https://tile.loc.gov/storage-services/service/ndnp/x/batch_x/data/sn1/00000000001/1814010101"


def p1_twice_pair():
    """An issue whose two American Stories scans are both named p1; the API links them as pages 1 and 2."""
    from snewpapers_tools.fetching import IssuePair
    from snewpapers_tools.issues import IssueKey

    comps = [{"uuid": u, "label": "notice", "links": {"loc_pdf_pages": [{"page_number": n, "url": f"{TILE_DIR}/{seq}.pdf"}]}}
             for u, n, seq in (("a", 1, "0001"), ("b", 2, "0002"))]
    scans = [ScanName("1814-01-01", 1, "sn1", "00000000001", 1, seq) for seq in ("0001", "0002")]
    pages = [(s, {"tag": s.seq, "scan": {"width": 100, "height": 100}, "bboxes": [], "full articles": []}) for s in scans]
    return IssuePair(IssueKey("i", "sn1", "1814-01-01", 1), comps, pages, 0, 2, [])


def fake_download(tmp_path, sizes):
    """A download_jp2 stand-in writing a small JP2 of the given (w, h) per seq into tmp_path."""
    def download(url, folder):
        seq = url.rsplit("/", 1)[-1][:-4]
        path = tmp_path / f"{seq}.jp2"
        Image.new("RGB", sizes[seq], "white").save(path)
        return path
    return download


def test_load_issue_pages_keys_pages_by_sequence_when_every_scan_is_named_p1(tmp_path):
    from snewpapers_tools.overlay import load_issue_pages

    download = fake_download(tmp_path, {"0001": (80, 100), "0002": (60, 100)})
    loaded = load_issue_pages(p1_twice_pair(), 40, download=download)
    assert sorted(loaded) == [1, 2]
    assert loaded[1][1]["tag"] == "0001" and loaded[2][1]["tag"] == "0002"
    assert loaded[1][2] == (80, 100) and loaded[2][2] == (60, 100)  # real JP2 sizes
    assert loaded[1][0].width == 40


def test_issue_scan_sizes_reads_each_pages_jp2_size(tmp_path):
    from snewpapers_tools.overlay import issue_scan_sizes

    download = fake_download(tmp_path, {"0001": (80, 100), "0002": (60, 100)})
    assert issue_scan_sizes(p1_twice_pair(), download=download) == {1: (80, 100), 2: (60, 100)}


def test_draw_page_can_leave_the_labels_off():
    drawn = draw_american_stories_page(Image.new("RGB", (1000, 1000), "white"), fill_label_scan(), labels=False)
    centre_region = [drawn.getpixel((x, y)) for x in range(200, 301) for y in range(170, 231)]
    assert LABEL_INK not in centre_region


def test_draw_snewpapers_page_can_leave_the_labels_off():
    comp = snewpapers_component(boxes=[{"page": 1, "rects": [[0.1, 0.1, 0.5, 0.5]]}])
    with_labels = draw_snewpapers_page(Image.new("RGB", (1000, 1000), "white"), [comp], 1)
    without = draw_snewpapers_page(Image.new("RGB", (1000, 1000), "white"), [comp], 1, labels=False)
    centre = [(x, y) for x in range(250, 351) for y in range(280, 321)]
    assert LABEL_INK in [with_labels.getpixel(p) for p in centre]
    assert LABEL_INK not in [without.getpixel(p) for p in centre]


def test_draw_snewpapers_page_draws_heavier_reading_order_dashes_on_request():
    comp = snewpapers_component(boxes=[{"page": 1, "rects": [[0.1, 0.1, 0.3, 0.2], [0.1, 0.6, 0.3, 0.7]]}])
    color = FAMILY_COLORS["opinion"]
    # a patch over the line between the two centres (x 200), a few dashes long
    column = lambda img: [img.getpixel((x, y)) for x in range(190, 211) for y in range(380, 430)]  # noqa: E731
    thin = draw_snewpapers_page(Image.new("RGB", (1000, 1000), "white"), [comp], 1, labels=False)
    heavy = draw_snewpapers_page(Image.new("RGB", (1000, 1000), "white"), [comp], 1, labels=False, line_width=3)
    assert column(heavy).count(color) > column(thin).count(color) >= 1


def test_labels_once_go_in_each_components_largest_rect_and_each_as_articles_largest_box_per_class():
    from snewpapers_tools.overlay import american_stories_label_sets, queue_labels, snewpapers_label_sets

    comps = [{"uuid": "a", "label": "domestic_news",
              "boxes": [{"page": 1, "rects": [[0.0, 0.0, 0.3, 0.2], [0.4, 0.0, 0.7, 0.9], [0.8, 0.0, 1.0, 0.5]]}]}]
    sets = snewpapers_label_sets(comps, 1, 700, 1000)
    assert len(queue_labels(sets, 700)) == 3
    [(cx, cy, _font, text)] = queue_labels(sets, 700, once=True)
    assert text == "domestic news" and (cx, cy) == (385.0, 450.0)
    page = {"bboxes": [{"id": 1, "class": "article", "bbox": {"x0": 0, "y0": 0, "x1": 300, "y1": 200}},
                       {"id": 2, "class": "article", "bbox": {"x0": 0, "y0": 300, "x1": 300, "y1": 900}},
                       {"id": 3, "class": "headline", "bbox": {"x0": 0, "y0": 200, "x1": 300, "y1": 290}},
                       {"id": 4, "class": "article", "bbox": {"x0": 400, "y0": 0, "x1": 700, "y1": 300}}],
            "full articles": [{"full_article_id": 7, "object_ids": [1, 2, 3]}]}
    rect = lambda b: (b["bbox"]["x0"], b["bbox"]["y0"], b["bbox"]["x1"], b["bbox"]["y1"])  # noqa: E731
    labels = queue_labels(american_stories_label_sets(page, rect), 700, once=True)
    assert sorted((t, cy) for _cx, cy, _f, t in labels) == [("article", 150.0), ("article", 600.0), ("headline", 245.0)]
