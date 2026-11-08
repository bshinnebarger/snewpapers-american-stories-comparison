from dataclasses import replace

import pytest

from snewpapers_tools.american_stories import ScanName
from snewpapers_tools.fetching import IssuePair
from snewpapers_tools.issues import IssueKey
from snewpapers_tools.segmentation import (
    AS_MERGE,
    AS_ONLY,
    AS_ONLY_TINY,
    AS_ONLY_EMPTY,
    AS_PARTIAL,
    AS_SPLIT,
    AS_WIDER,
    CASE_LABELS,
    CASES,
    CONTINUED,
    IN_AS_MERGE,
    MATCH,
    OURS_ONLY,
    AsBox,
    AsPart,
    IssueSegmentation,
    OurPiece,
    as_boxes,
    off_page,
    _repair_gaps,
    adjacent,
    as_parts,
    box_claims,
    attached_duplicates,
    collapse_duplicates,
    link_pages,
    nested_boxes,
    illegible_read,
    illegible_area,
    fragment_owner,
    grid_shape,
    holds_text,
    intersection_area,
    our_pieces,
    phrase_containment,
    reading_order,
    region_area,
    segment_issue,
    segment_page,
    slice_box_text,
    slice_span,
    text_metrics,
)


def ours(uuid, *rects, text="", page=1, pages=(1,), label="notice"):
    return OurPiece(uuid, label, page, tuple(rects), text, tuple(pages))


def box(box_id, article_id, rect, text="", cls="article", order=0, page=1):
    return AsBox(page, box_id, article_id, rect, text, cls, "Legible", order)


def by_case(groups):
    found = {}
    for g in groups:
        found.setdefault(g.case, []).append(g)
    return found


# --- geometry ---------------------------------------------------------------------------------


def test_grid_is_1000_wide_and_as_tall_as_the_page_aspect():
    assert grid_shape(1.0) == (1000, 1000)
    assert grid_shape(1.3) == (1300, 1000)


def test_region_area_counts_grid_cells_of_the_union_of_its_rects():
    assert region_area(((0.1, 0.1, 0.3, 0.3),), 1.0) == 200 * 200
    # two rects overlapping by a 100 x 200 strip count it once
    assert region_area(((0.1, 0.1, 0.3, 0.3), (0.2, 0.1, 0.4, 0.3)), 1.0) == 300 * 200
    assert region_area(((0.1, 0.1, 0.3, 0.3),), 2.0) == 200 * 400  # taller page, taller cells


def test_intersection_area_of_two_unions():
    assert intersection_area(((0.1, 0.1, 0.3, 0.5),), ((0.1, 0.2, 0.3, 0.6),), 1.0) == 200 * 300
    assert intersection_area(((0.1, 0.1, 0.2, 0.2),), ((0.5, 0.5, 0.6, 0.6),), 1.0) == 0
    assert intersection_area((), ((0.5, 0.5, 0.6, 0.6),), 1.0) == 0


def test_reading_order_is_column_major_with_a_centred_headline_kept_above_its_body():
    regions = [
        ((0.40, 0.12, 0.60, 0.5),),  # body, column 2
        ((0.45, 0.10, 0.55, 0.12),),  # centred headline above it
        ((0.10, 0.60, 0.30, 0.9),),  # column 1, lower
        ((0.10, 0.10, 0.30, 0.5),),  # column 1, top
    ]
    assert reading_order(regions) == [3, 2, 1, 0]


# --- slicing AS boxes among our pieces --------------------------------------------------------


def test_a_piece_claims_a_box_overlapping_15_percent_of_the_box_or_half_of_the_piece():
    pieces = [
        ours("top", (0.1, 0.1, 0.3, 0.3)),  # 25% of the box
        ours("sliver", (0.1, 0.3, 0.3, 0.32)),  # 2.5% of the box, but all of itself
        ours("beside", (0.25, 0.4, 0.5, 0.5)),  # 2.5% of the box and 20% of itself: no claim
        ours("bottom", (0.1, 0.5, 0.3, 0.9), (0.6, 0.6, 0.7, 0.7)),  # 50% of the box; its far rect is not inside
    ]
    big = box(1, 1, (0.1, 0.1, 0.3, 0.9))
    claims = box_claims(big, pieces, 1.0)
    assert [(i, round(t, 3), round(b, 3)) for i, t, b in claims] == [(0, 0.0, 0.25), (1, 0.25, 0.275), (3, 0.5, 1.0)]
    assert box_claims(box(2, 1, (0.6, 0.1, 0.8, 0.2)), pieces, 1.0) == []


def test_a_claim_spans_the_rows_the_piece_really_covers_not_a_neighbouring_columns_sliver():
    # the top of the box, plus the next column's edge overlapping the box's left edge all the way down
    piece = ours("editorial", (0.3, 0.1, 0.5, 0.3), (0.1, 0.1, 0.31, 0.9))
    [(_, top, bottom)] = box_claims(box(1, 1, (0.3, 0.1, 0.5, 0.9)), [piece], 1.0)
    assert (top, bottom) == (0.0, 0.25)


def test_slice_box_text_cuts_proportionally_and_pushes_each_end_out_to_a_word_boundary():
    text = "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj"
    assert slice_box_text(text, 0.33, 0.55, "") == "dddd eeee ffff"  # 16 and 27 fall inside words
    assert slice_box_text(text, 0.0, 1.0, "") == text
    assert slice_span(text, 0.33, 0.55, "") == (15, 29)


EDITORIAL_END = "for he has been tried and found true, and to him we give our vote; and so we close."
STORY = ("A CANINE NOBLEMAN. We take the following from an English paper. A gentleman of Kent had a dog "
         "of rare sagacity, who would fetch the post from the village every morning and lay it at his master's "
         "door, and would suffer no stranger to touch it on the way.")
COLUMN = EDITORIAL_END + "\n" + STORY  # the editorial's end is a quarter of the text
EDITORIAL = "THE MODEL PRESIDENT. The supporters of Mr. Fillmore claim much for him. " + EDITORIAL_END


def test_slice_box_text_moves_the_cut_to_where_the_piece_opens_and_closes():
    # the overlap puts the boundary at 37.5% of the box's height; the text says a quarter
    assert slice_box_text(COLUMN, 0.0, 0.375, EDITORIAL) == EDITORIAL_END  # its closing words found
    assert slice_box_text(COLUMN, 0.375, 1.0, STORY) == STORY  # its opening words found


def test_slice_box_text_finds_the_edges_through_ocr_errors():
    noisy = COLUMN.replace("We take the following", "We tako the followiug").replace("and so we close", "aud so we c1ose")
    assert slice_box_text(noisy, 0.375, 1.0, STORY).startswith("A CANINE NOBLEMAN. We tako")
    assert slice_box_text(noisy, 0.0, 0.375, EDITORIAL).endswith("aud so we c1ose.")


def test_slice_box_text_keeps_the_proportional_cut_when_the_piece_is_not_found():
    unrelated = "Wanted, a smart lad to learn the printing business; apply at this office forthwith."
    rough = slice_box_text(COLUMN, 0.0, 0.375, "")
    assert rough.endswith("We take the following from")  # 37.5% of the characters, to a word end
    assert slice_box_text(COLUMN, 0.0, 0.375, unrelated) == rough


def test_slice_box_text_ignores_an_edge_matched_only_in_part_at_the_end_of_the_text_searched():
    closing = "the boats can no longer pass the shoals above town."
    text = " ".join(["lorem"] * 100) + " " + closing + " " + " ".join(["ipsum"] * 100)
    # the cut ends at 503; the search runs to 629, halfway through the closing words
    assert slice_span(text, 0.0, 0.4, "We learn that the river is low and " + closing) == (0, 503)


def test_as_parts_group_by_article_keep_orphans_single_and_follow_as_reading_order():
    boxes = [
        box(5, 2, (0.5, 0.2, 0.7, 0.5), "body two", order=1),
        box(6, 2, (0.5, 0.1, 0.7, 0.2), "HEAD TWO", cls="headline", order=0),
        box(7, None, (0.1, 0.6, 0.3, 0.7), "J. SMITH.", cls="author"),
        box(8, None, (0.1, 0.8, 0.3, 0.9), "orphan two"),
        box(9, 1, (0.1, 0.1, 0.3, 0.5), "body one"),
    ]
    parts = as_parts(boxes)
    assert [(p.article_id, [b.box_id for b in p.boxes]) for p in parts] == [
        (1, [9]), (None, [7]), (None, [8]), (2, [6, 5])]
    assert parts[3].text == "HEAD TWO\n\nbody two"
    assert parts[1].name == "AS orphan author" and parts[3].name == "AS #2"


# --- cases on one page ------------------------------------------------------------------------


def test_match_compares_our_text_with_its_slices_in_as_reading_order():
    piece = ours("a", (0.1, 0.1, 0.3, 0.5), text="NOTICE.\nThe river is low.")
    boxes = [box(1, 7, (0.1, 0.12, 0.3, 0.5), "The river is low.", order=1),
             box(2, 7, (0.12, 0.1, 0.28, 0.12), "NOTICE.", cls="headline", order=0)]
    [g] = segment_page([piece], boxes, 1.0)[0]
    assert g.case == MATCH
    assert g.as_text == "NOTICE.\n\nThe river is low."
    assert g.metrics["ratio"] == 100.0 and "shared_phrases" not in g.metrics


def test_an_orphan_box_alone_is_a_match():
    [g] = segment_page([ours("a", (0.1, 0.1, 0.3, 0.3), text="J. SMITH.")],
                       [box(1, None, (0.1, 0.1, 0.3, 0.3), "J. SMITH.", cls="author")], 1.0)[0]
    assert g.case == MATCH and g.parts[0].article_id is None


def test_as_split_pieces_hold_only_the_text_of_the_boxes_inside_our_piece():
    piece = ours("a", (0.1, 0.1, 0.3, 0.7), text="THE FEVER. It rages. NOTICE. All persons.")
    boxes = [
        box(1, 1, (0.1, 0.1, 0.3, 0.4), "THE FEVER. It rages."),
        box(2, 2, (0.1, 0.4, 0.3, 0.7), "NOTICE. All persons."),
        box(3, 2, (0.1, 0.7, 0.3, 0.9), "are hereby warned not to trespass.", order=1),  # article 2 runs on below our piece
    ]
    groups, _ = segment_page([piece], boxes, 1.0)
    split = by_case(groups)[AS_SPLIT][0]
    assert [p.text for p in split.parts] == ["THE FEVER. It rages.", "NOTICE. All persons."]
    assert "warned" not in split.as_text
    # the box outside is article 2's, which only our piece claims: attached outside it, not AS only
    assert AS_ONLY not in by_case(groups)
    assert [(p.article_id, p.text) for p in split.as_attached_outside] == [(2, "are hereby warned not to trespass.")]


def model_president_page():
    """Our editorial: column 1 and the top of column 2. Our story: the rest of column 2.
    AS article 1 is column 1 (headline and body); AS article 2 is ONE box, all of column 2: the
    editorial's end and then the story."""
    editorial = ours("editorial", (0.1, 0.1, 0.3, 0.9), (0.3, 0.1, 0.5, 0.4), text=EDITORIAL, label="editorial")
    story = ours("story", (0.3, 0.4, 0.5, 0.9), text=STORY, label="story")
    boxes = [
        box(10, 1, (0.12, 0.1, 0.28, 0.12), "THE MODEL PRESIDENT.", cls="headline", order=0),
        box(11, 1, (0.1, 0.12, 0.3, 0.9), "The supporters of Mr. Fillmore claim much for him.", order=1),
        box(20, 2, (0.3, 0.1, 0.5, 0.9), COLUMN, order=0),
    ]
    return [editorial, story], boxes


def test_model_president_a_box_straddling_two_pieces_is_sliced_between_them():
    pieces, boxes = model_president_page()
    groups, _ = segment_page(pieces, boxes, 1.0)
    cases = {g.pieces[0].uuid: g for g in groups if g.pieces and g.case != AS_MERGE}
    assert set(cases) == {"editorial", "story"} and len(groups) == 3  # and the merge; nothing left over
    editorial, story = cases["editorial"], cases["story"]
    assert editorial.case == AS_SPLIT
    assert [(p.article_id, [b.box_id for b in p.boxes]) for p in editorial.parts] == [(1, [10, 11]), (2, [20])]
    assert [p.text for p in editorial.parts] == [
        "THE MODEL PRESIDENT.\n\nThe supporters of Mr. Fillmore claim much for him.", EDITORIAL_END]
    # AS's column box holds the editorial's end (37% of it) and the story: two of ours in one AS article
    [merge] = by_case(groups)[AS_MERGE]
    assert [p.uuid for p in merge.pieces] == ["editorial", "story"]
    assert story.case == IN_AS_MERGE and story.merge is merge
    assert [(p.article_id, [b.box_id for b in p.boxes]) for p in story.parts] == [(2, [20])]
    assert story.as_text == STORY
    assert story.metrics["ratio"] == 100.0
    # each slice is drawn as the band of the box its piece overlaps
    assert editorial.parts[1].rects == ((0.3, 0.1, 0.5, pytest.approx(0.4)),)
    assert story.parts[0].rects == ((0.3, pytest.approx(0.4), 0.5, 0.9),)


ITEMS = ["MARRIED. In this town, on Tuesday last, Mr. John Smith to Miss Mary Doe, both of this county.",
         "DIED. At his residence, Mr. Peter Brown, aged 80.",
         "FIRE. On Monday night the stable of Mr. Jones was burnt to the ground, with two horses and all his hay."]


def vermont_items_page():
    """The Vermont Phoenix 1862-02-20 p2 in small: AS article 9 is a headline box and ONE body box
    over three separate items of ours."""
    pieces = [ours("married", (0.1, 0.1, 0.3, 0.3), text=ITEMS[0]),
              ours("died", (0.1, 0.3, 0.3, 0.4), text=ITEMS[1]),  # 12.5% of the box: claims by being inside it
              ours("fire", (0.1, 0.4, 0.3, 0.9), text=ITEMS[2])]
    boxes = [box(0, 9, (0.1, 0.08, 0.3, 0.1), "LOCAL ITEMS.", cls="headline", order=0),
             box(1, 9, (0.1, 0.1, 0.3, 0.9), "\n".join(ITEMS), order=1)]
    return pieces, boxes


def test_one_as_body_box_over_three_of_our_items_is_one_merge_with_three_pieces_and_no_match():
    groups, _ = segment_page(*vermont_items_page(), 1.0)
    found = by_case(groups)
    assert MATCH not in found  # the slices are cut along our boundaries, not AS's: no agreement
    [merge] = found[AS_MERGE]
    assert [p.uuid for p in merge.pieces] == ["married", "died", "fire"]
    assert [b.box_id for b in merge.parts[0].boxes] == [0, 1]  # the AS article whole: its real boxes
    inside = found[IN_AS_MERGE]
    assert [g.pieces[0].uuid for g in inside] == ["married", "died", "fire"]
    assert all(g.merge is merge for g in inside)
    # each keeps its slice and text metrics: the text agreement inside AS's merged article
    assert [g.as_text for g in inside] == ITEMS
    assert all(g.metrics["ratio"] == 100.0 for g in inside)
    # a slice remembers the real box it was cut from
    assert {b.box_rect for g in inside for b in g.parts[0].boxes} == {(0.1, 0.1, 0.3, 0.9)}
    assert inside[1].as_box_rects == ((0.1, 0.1, 0.3, 0.9),) and inside[1].as_rects != inside[1].as_box_rects
    assert set(found) == {AS_MERGE, IN_AS_MERGE, AS_ONLY_TINY}  # and the headline nobody claimed


def test_a_shared_article_short_of_80_percent_is_still_a_merge_when_its_slices_cover_most_of_two_pieces():
    # each piece is 60% inside article 9's one box and claims a slice of it; the rest lies on no AS box
    pieces = [ours("top", (0.1, 0.1, 0.3, 0.3), text=ITEMS[0]), ours("bottom", (0.1, 0.3, 0.3, 0.5), text=ITEMS[1])]
    groups, _ = segment_page(pieces, [box(1, 9, (0.1, 0.18, 0.3, 0.42), "\n".join(ITEMS[:2]))], 1.0)
    found = by_case(groups)
    [merge] = found[AS_MERGE]
    assert [p.uuid for p in merge.pieces] == ["top", "bottom"]
    assert [g.pieces[0].uuid for g in found[IN_AS_MERGE]] == ["top", "bottom"] and MATCH not in found


def test_a_shared_article_one_piece_mostly_inside_and_another_marginal_keeps_its_match():
    # "main" lies in the box; "edge" claims the box's bottom tenth (its overlap is 15% of the box) but
    # only 20% of it lies inside: no merge, and main's slice is still a match
    pieces = [ours("main", (0.1, 0.1, 0.3, 0.45), text=ITEMS[0]), ours("edge", (0.1, 0.45, 0.3, 0.9), text=ITEMS[1])]
    groups, _ = segment_page(pieces, [box(1, 9, (0.1, 0.1, 0.3, 0.5), "\n".join(ITEMS[:2]))], 1.0)
    found = by_case(groups)
    assert AS_MERGE not in found and IN_AS_MERGE not in found
    assert [g.pieces[0].uuid for g in found[MATCH]] == ["main"]
    assert [g.pieces[0].uuid for g in found[AS_PARTIAL]] == ["edge"]  # AS has text for a tenth of it


def test_box_text_between_two_slices_is_as_only():  # its article is claimed by two pieces: not attached outside
    top_text = "We learn that the river is low and the boats can no longer pass the shoals above town."
    middle = "JUST RECEIVED, and for sale cheap for cash, a fine lot of bacon, lard and coffee at Brown's store."
    bottom_text = "MARRIED. In this town, on Tuesday last, Mr. John Smith to Miss Mary Doe, of this county."
    pieces = [ours("top", (0.1, 0.1, 0.3, 0.25), text=top_text), ours("bottom", (0.1, 0.65, 0.3, 0.9), text=bottom_text)]
    groups, _ = segment_page(pieces, [box(1, 4, (0.1, 0.1, 0.3, 0.9), " ".join([top_text, middle, bottom_text]))], 1.0)
    found = by_case(groups)
    assert [g.as_text for g in found[IN_AS_MERGE]] == [top_text, bottom_text]  # both inside article 4: a merge
    assert len(found[AS_MERGE]) == 1 and MATCH not in found
    [left] = found[AS_ONLY]
    assert left.as_text == middle and left.parts[0].article_id == 4
    assert left.rects[0][1] > 0.25 and left.rects[0][3] < 0.65  # drawn as the middle band of the box


def test_two_of_our_pieces_inside_one_as_article_are_an_as_merge():
    top = ours("top", (0.1, 0.1, 0.3, 0.3), text="We learn that the river is low.")
    bottom = ours("bottom", (0.1, 0.3, 0.3, 0.5), text="MARRIED. In this town, Mr. Smith to Miss Doe.")
    boxes = [box(1, 9, (0.1, 0.1, 0.3, 0.3), "We learn that the river is low.", order=0),
             box(2, 9, (0.1, 0.3, 0.3, 0.5), "MARRIED. In this town, Mr. Smith to Miss Doe.", order=1)]
    groups, _ = segment_page([bottom, top], boxes, 1.0)
    found = by_case(groups)
    [merge] = found[AS_MERGE]  # AS has the two as one article...
    # ...so neither piece is a match, though each has a box of its own: AS's boundary is not there
    assert MATCH not in found and [g.merge for g in found[IN_AS_MERGE]] == [merge, merge]
    assert [p.uuid for p in merge.pieces] == ["top", "bottom"]
    assert merge.parts[0].article_id == 9 and merge.as_text.startswith("We learn")
    assert merge.rects == ((0.1, 0.1, 0.3, 0.3), (0.1, 0.3, 0.3, 0.5))  # the AS article's area
    assert "shared_phrases" in merge.metrics


def test_a_word_or_two_the_cuts_leave_over_goes_to_the_next_slice_not_to_as_only():
    text = "THE NEWS. We learn that the river is low and falling fast, and the boats are laid up."
    piece = ours("news", (0.1, 0.3, 0.3, 0.9))  # covers the box from 25% of its height down: "THE NEWS." is above
    groups, _ = segment_page([piece], [box(1, 4, (0.1, 0.1, 0.3, 0.9), text)], 1.0)
    assert [(g.case, g.as_text) for g in groups] == [(MATCH, text)]


def test_one_big_as_box_over_two_of_our_pieces_is_sliced_between_them_and_is_a_merge():
    pieces = [ours("top", (0.1, 0.1, 0.3, 0.3)), ours("bottom", (0.1, 0.3, 0.3, 0.5))]  # no text to find the edges by
    groups, _ = segment_page(pieces, [box(1, 9, (0.1, 0.1, 0.3, 0.5), "Marriages of the week. Deaths of the week.")], 1.0)
    assert sorted(g.case for g in groups) == sorted([IN_AS_MERGE, IN_AS_MERGE, AS_MERGE])
    # cut at half the characters, each slice to the end of the word the cut falls in
    assert [g.as_text for g in groups if g.case == IN_AS_MERGE] == ["Marriages of the week.", "week. Deaths of the week."]


def test_a_piece_less_than_80_percent_inside_counts_towards_a_merge_only_with_a_slice_and_half_of_it_or_a_quarter_of_the_article():
    article = [box(1, 9, (0.1, 0.1, 0.3, 0.3)), box(2, 9, (0.1, 0.3, 0.3, 0.5), order=1)]
    less = [ours("in", (0.1, 0.1, 0.3, 0.3)), ours("less", (0.1, 0.3, 0.3, 0.8))]  # 40% inside, claiming box 2...
    [merge] = by_case(segment_page(less, article, 1.0)[0])[AS_MERGE]  # ...which is half the article
    assert [p.uuid for p in merge.pieces] == ["in", "less"]
    small = [box(1, 9, (0.1, 0.1, 0.3, 0.42)), box(2, 9, (0.1, 0.42, 0.3, 0.5), order=1)]
    less = [ours("in", (0.1, 0.1, 0.3, 0.42)), ours("less", (0.1, 0.42, 0.3, 0.8))]  # 21% inside, a fifth of the article
    assert AS_MERGE not in by_case(segment_page(less, small, 1.0)[0])
    half = [ours("in", (0.1, 0.1, 0.3, 0.3)), ours("half", (0.1, 0.3, 0.3, 0.7))]  # 50% inside, claiming box 2
    [merge] = by_case(segment_page(half, article, 1.0)[0])[AS_MERGE]
    assert [p.uuid for p in merge.pieces] == ["in", "half"]
    # half inside without a slice of the article does not count: "beside" straddles two big boxes of
    # article 9, a quarter of it over each (6% of either box), the rest outside
    tall = [box(1, 9, (0.1, 0.1, 0.3, 0.5)), box(2, 9, (0.1, 0.5, 0.3, 0.9), order=1)]
    beside = [ours("in", (0.1, 0.1, 0.3, 0.3)), ours("beside", (0.2, 0.45, 0.4, 0.55))]
    assert AS_MERGE not in by_case(segment_page(beside, tall, 1.0)[0])


def test_left_over_boxes_are_as_only_per_article_and_orphans_singly():
    boxes = [box(1, 4, (0.6, 0.1, 0.9, 0.2), "JUST RECEIVED, for sale cheap", order=0),
             box(2, 4, (0.6, 0.2, 0.9, 0.4), "a fine lot of bacon and lard, $1.50.", order=1),
             box(3, None, (0.6, 0.5, 0.9, 0.7), "Subscription terms: two dollars per annum.")]
    groups, _ = segment_page([ours("a", (0.1, 0.1, 0.3, 0.3))], boxes, 1.0)
    as_only = by_case(groups)[AS_ONLY]
    assert [(g.parts[0].article_id, g.as_text) for g in as_only] == [
        (4, "JUST RECEIVED, for sale cheap\n\na fine lot of bacon and lard, $1.50."),
        (None, "Subscription terms: two dollars per annum.")]
    assert all(g.pieces == [] and g.metrics == {} for g in as_only)


def test_tiny_left_overs_are_as_only_tiny():
    groups, _ = segment_page([], [box(1, None, (0.6, 0.1, 0.9, 0.12), "  GENERAL NOTICES.  ", cls="headline"),
                                  box(2, None, (0.6, 0.5, 0.9, 0.7), "x" * 20)], 1.0)
    assert [g.case for g in groups] == [AS_ONLY_TINY, AS_ONLY]


def test_ours_only():
    groups, _ = segment_page([ours("a", (0.1, 0.1, 0.3, 0.3))], [], 1.0)
    assert [g.case for g in groups] == [OURS_ONLY]
    assert groups[0].parts == [] and groups[0].metrics == {}


def test_groups_on_a_page_come_in_position_order():
    pieces = [ours("right", (0.6, 0.1, 0.8, 0.3)), ours("left-low", (0.1, 0.5, 0.3, 0.7)), ours("left", (0.1, 0.1, 0.3, 0.3))]
    groups, _ = segment_page(pieces, [], 1.0)
    assert [g.pieces[0].uuid for g in groups] == ["left", "left-low", "right"]


def test_page_coverage_by_area_each_way():
    piece = ours("a", (0.1, 0.1, 0.3, 0.5))
    boxes = [box(1, 1, (0.1, 0.1, 0.3, 0.3)), box(2, 2, (0.6, 0.6, 0.8, 0.8), "Cheap for cash, $2.")]
    _, cov = segment_page([piece], boxes, 1.0)
    assert cov.ours_area == 200 * 400 and cov.ours_covered == 200 * 200
    assert cov.as_area == 2 * 200 * 200 and cov.as_covered == 200 * 200


def test_text_metrics_add_shared_phrases_on_request():
    assert text_metrics("The  River\nis low.", "the river is low.", partial=False) == {"ratio": 100.0, "token_set_ratio": 100.0}
    m = text_metrics("the river is low", "the river is low and falling", partial=True)
    assert m["shared_phrases"] == 100.0 and m["ratio"] < 100
    # the side with fewer words is looked for in the other, whichever side it is
    assert text_metrics("we learn the river is low and falling", "the river is low", partial=True)["shared_phrases"] == 100.0


# --- shared phrases ---------------------------------------------------------------------------


def test_phrase_containment_is_the_share_of_a_texts_four_word_phrases_found_in_the_other():
    assert phrase_containment("The river is low, and falling.", "We learn THE RIVER is low and falling fast") == 100.0
    # 3 phrases: "the river is low" and "river is low and" are in b, "is low and rising" is not
    assert phrase_containment("the river is low and rising", "the river is low and falling") == pytest.approx(200 / 3)
    assert phrase_containment("the river is low and falling", "") == 0.0
    assert phrase_containment("", "the river is low") == 0.0


def test_phrase_containment_needs_whole_phrases_not_scattered_letters_or_words():
    a = "the boats can no longer pass the shoals"
    scattered = "pass the boats, no, the shoals can longer"  # every word of a, in no phrase of it
    assert phrase_containment(a, scattered) == 0.0
    assert phrase_containment(a, "x " * 50 + a + " y" * 50) == 100.0


def test_phrase_containment_of_a_text_shorter_than_a_phrase_counts_its_words():
    assert phrase_containment("River low.", "the river is low") == 100.0
    assert phrase_containment("river high", "the river is low") == 50.0
    assert phrase_containment("12 $3.50", "the river is low") == 0.0  # no words at all


# --- repairing the gaps the cuts leave --------------------------------------------------------

LOW = "We learn that the river is low and the boats are laid up at the wharf for want of water."
GAP = "The captains say that no boat will run until the autumn rains have come to raise it."
WED = "MARRIED. In this town, on Tuesday last, Mr. John Smith to Miss Mary Doe, of this county."


def between():
    text = " ".join([LOW, GAP, WED])
    return text, [[0, len(LOW)], [text.index(WED), len(text)]]


def test_a_gap_between_two_slices_goes_to_the_piece_sharing_its_phrases():
    text, spans = between()
    assert _repair_gaps(text, spans, [LOW + " " + GAP, WED]) == ([], [])
    assert spans == [[0, text.index(WED)], [text.index(WED), len(text)]]  # the slice before takes it
    text, spans = between()
    assert _repair_gaps(text, spans, [LOW, GAP + " " + WED]) == ([], [])
    assert spans == [[0, len(LOW)], [len(LOW), len(text)]]  # the slice after takes it


def test_a_gap_both_pieces_share_equally_goes_to_the_slice_before():
    text, spans = between()
    _repair_gaps(text, spans, [LOW + " " + GAP, GAP + " " + WED])
    assert spans[0][1] == text.index(WED)


def test_a_gap_neither_piece_shares_a_phrase_with_is_left_over():
    text, spans = between()
    assert _repair_gaps(text, spans, [LOW, WED]) == ([(len(LOW), text.index(WED))], [])
    assert spans == between()[1]


EDGE = "Fresh Goes the river along by the mill and the boats wait there for spring."  # 15 words: 12 phrases


@pytest.mark.parametrize("words, joins", [(7, True), (6, False)])  # 4 of 12 phrases (33%); 3 of 12 (25%)
def test_a_run_at_a_boxs_start_or_end_joins_the_slice_next_to_it_from_30_percent_shared(words, joins):
    piece_text = " ".join(EDGE.split()[:words]) + " " + LOW
    text = EDGE + " " + LOW  # the run at the start
    spans = [[len(EDGE) + 1, len(text)]]
    left, _ = _repair_gaps(text, spans, [piece_text])
    assert (spans[0][0] == 0, left) == ((True, []) if joins else (False, [(0, len(EDGE) + 1)]))
    text = LOW + " " + EDGE  # the run at the end
    spans = [[0, len(LOW)]]
    left, _ = _repair_gaps(text, spans, [LOW + " " + " ".join(EDGE.split()[:words])])
    assert (spans[0][1] == len(text), left) == ((True, []) if joins else (False, [(len(LOW), len(text))]))


def test_a_run_at_a_boxs_edge_goes_to_an_overlapping_piece_without_a_slice_when_it_shares_more():
    text = LOW + " " + EDGE
    spans = [[0, len(LOW)]]
    left, taken = _repair_gaps(text, spans, [LOW + " Fresh goes the river along"], [EDGE + " And more."])
    assert (left, taken, spans) == ([], [(0, len(LOW), len(text))], [[0, len(LOW)]])


EDITORIAL_OPENING = "THE MODEL PRESIDENT. The supporters of Mr. Fillmore claim much for him."


def test_a_box_ending_with_the_opening_of_a_piece_too_little_inside_it_to_claim_a_slice_gives_it_that_text():
    """AS box 1 is a story and, below it, the first lines of our editorial; our editorial overlaps
    the box too little to claim a slice (7.5% of the box, 7% of the editorial)."""
    story = ours("story", (0.1, 0.1, 0.3, 0.45), text=STORY, label="story")
    editorial = ours("editorial", (0.1, 0.47, 0.3, 0.9), text=EDITORIAL, label="editorial")
    boxes = [box(1, 1, (0.1, 0.1, 0.3, 0.5), STORY + "\n" + EDITORIAL_OPENING),
             box(2, 2, (0.1, 0.5, 0.3, 0.9), EDITORIAL_END)]
    groups, _ = segment_page([story, editorial], boxes, 1.0)
    cases = {g.pieces[0].uuid: g for g in groups}
    assert set(cases) == {"story", "editorial"}  # nothing left over
    assert (cases["story"].case, cases["story"].as_text) == (MATCH, STORY)
    assert cases["editorial"].case == AS_SPLIT
    assert [(p.article_id, p.text) for p in cases["editorial"].parts] == [(1, EDITORIAL_OPENING), (2, EDITORIAL_END)]
    # an unrelated run in the same place is not the editorial's: box 1's article is only the story's,
    # so the run is attached outside the story
    ad = "JUST RECEIVED, and for sale cheap for cash, a fine lot of bacon and lard."
    groups, _ = segment_page([story, editorial], [replace(boxes[0], text=STORY + "\n" + ad), boxes[1]], 1.0)
    cases = {g.pieces[0].uuid: g for g in groups}
    assert set(cases) == {"story", "editorial"}
    assert (cases["story"].as_text, cases["story"].attached_text) == (STORY, ad)
    assert cases["editorial"].case == MATCH and cases["editorial"].as_attached_outside == []


# --- American Stories duplicating text --------------------------------------------------------

ITEM = "LEARNED FLEAS. The Italian correspondent of the Newark Daily Advertiser writes of a show of fleas."
ITEM_TWICE = "LEARNED FLEAs.-The Italian correspondent { the Newark Daily Advertiser writes of a show of fleas"


def test_two_as_articles_holding_the_same_text_in_our_piece_are_a_match_flagged_as_duplicated():
    piece = ours("fleas", (0.1, 0.1, 0.3, 0.5), text=ITEM)
    boxes = [box(1, 5, (0.1, 0.1, 0.3, 0.3), ITEM), box(2, 83, (0.1, 0.3, 0.3, 0.5), ITEM_TWICE)]
    [g] = segment_page([piece], boxes, 1.0)[0]
    assert g.case == MATCH and g.as_duplicated
    assert [p.article_id for p in g.parts] == [5]  # the longer is kept
    assert g.duplicates == [("#5", "#83")]
    assert g.metrics["ratio"] == 100.0


def test_three_as_articles_two_duplicating_each_other_are_an_as_split_of_two():
    piece = ours("fleas", (0.1, 0.1, 0.3, 0.7), text=ITEM + " " + WED)
    boxes = [box(1, 5, (0.1, 0.1, 0.3, 0.3), ITEM), box(2, 83, (0.1, 0.3, 0.3, 0.5), ITEM_TWICE),
             box(3, 9, (0.1, 0.5, 0.3, 0.7), WED)]
    [g] = segment_page([piece], boxes, 1.0)[0]
    assert g.case == AS_SPLIT and g.as_duplicated
    assert [p.article_id for p in g.parts] == [5, 9]
    assert g.duplicates == [("#5", "#83")]


def test_a_short_headline_whose_words_recur_in_the_body_is_no_duplicate():
    BOX_A, BOX_B = (0.1, 0.1, 0.3, 0.2), (0.1, 0.2, 0.3, 0.5)
    head = AsPart(None, (box(1, None, BOX_A, "THE FEVER.", cls="headline"),))
    body = AsPart(2, (box(2, 2, BOX_B, "The fever rages in the town and the fever is feared."),))
    assert collapse_duplicates([head, body]) == ([head, body], [])


def test_summary_counts_the_groups_american_stories_duplicated():
    piece = ours("fleas", (0.1, 0.1, 0.3, 0.5), text=ITEM)
    boxes = [box(1, 5, (0.1, 0.1, 0.3, 0.3), ITEM), box(2, 83, (0.1, 0.3, 0.3, 0.5), ITEM_TWICE)]
    groups, cov = segment_page([piece, ours("other", (0.6, 0.1, 0.8, 0.3))], boxes, 1.0)
    assert IssueSegmentation(groups, {1: cov}, []).summary()["as_duplicated"] == 1


LEWIS_22D = ("0a-Mr. D. c. Lewis, writes to, US under date Of Bryants Swamp. Bladen county, N. c.. July 22d, "
             "to state that he no longer belongs to the American Party")
LEWIS_3D = ("d 0a-Mr. D. c. Lewis, writes to, us under date @ It Bryants Swamp, Bladen county, N. c., July 3d. "
            "state that ne $\" no longer belongs to the American I'll Ds. -.")
BROWN = ("Mr. J. Brown, writes to us under date of Elizabethtown, Bladen county, to say that he will be "
         "a candidate for the Commons.")


def test_the_same_notice_printed_twice_with_different_ocr_collapses_from_half_its_phrases_shared():
    """AS has the "Mr. D. C. Lewis" notice in two articles, OCR'd differently (July 22d / July 3d):
    a little over half the phrases shared, which the old 70% missed."""
    assert 50 <= phrase_containment(LEWIS_22D, LEWIS_3D) < 70 and 50 <= phrase_containment(LEWIS_3D, LEWIS_22D) < 70
    piece = ours("lewis", (0.1, 0.1, 0.3, 0.5), text=LEWIS_22D)
    boxes = [box(1, 17, (0.1, 0.1, 0.3, 0.3), LEWIS_3D), box(2, 28, (0.1, 0.3, 0.3, 0.5), LEWIS_22D)]
    [g] = segment_page([piece], boxes, 1.0)[0]
    assert g.case == MATCH and g.duplicates == [("#17", "#28")]  # the longer (in characters) is kept


def test_two_notices_of_the_same_form_but_different_text_are_no_duplicates():
    assert phrase_containment(BROWN, LEWIS_22D) < 50
    a = AsPart(17, (box(1, 17, (0.1, 0.1, 0.3, 0.3), LEWIS_22D),))
    b = AsPart(28, (box(2, 28, (0.1, 0.3, 0.3, 0.5), BROWN),))
    assert collapse_duplicates([a, b]) == ([a, b], [])


# --- AS text attached outside our component ---------------------------------------------------

COLLEGE = ("GOLDSBORO' FEMALE COLLEGE. The third Session of this Institution will commence on the first "
           "Monday in January. Thanks for the support we have had. VIM. K LANE, President Stockholders.\n"
           "Dec Is, 1854. l5-tf")
NOTICE_BODY = ("THE CO-PARTNERSHIP heretofore existing under the name and firm MERRIMAN, WOODRUFF & CO., of "
               "Stricklandsville, N. C., is this day, by mutual consent, dissolved.")


def dec_18_page(date="Dec. Is, 1854."):
    """Wilmington p1: AS article 31 is a college notice (no component of ours: an advertisement)
    ending with its date line; AS article 7 is a small `author` box holding that date line again,
    printed just above our notice, then the notice's headline and body, which our notice covers."""
    notice = ours("notice", (0.1, 0.4, 0.3, 0.6), text="NOTICE.\n\n" + NOTICE_BODY)
    boxes = [box(1, 31, (0.1, 0.1, 0.3, 0.39), COLLEGE),
             box(2, 7, (0.1, 0.39, 0.16, 0.395), date, cls="author", order=0),
             box(3, 7, (0.15, 0.4, 0.25, 0.41), "NOTICE.", cls="headline", order=1),
             box(4, 7, (0.1, 0.41, 0.3, 0.6), NOTICE_BODY, order=2)]
    return [notice], boxes


def test_as_text_of_our_components_own_article_printed_outside_it_is_attached_outside_not_as_only():
    groups, _ = segment_page(*dec_18_page(), 1.0)
    found = by_case(groups)
    assert set(found) == {MATCH, AS_ONLY}  # no as_only_tiny for the date line
    [notice] = found[MATCH]
    assert notice.as_text == "NOTICE.\n\n" + NOTICE_BODY  # the comparison leaves the date out...
    assert [(p.article_id, p.classes, p.text) for p in notice.as_attached_outside] == [(7, ("author",), "Dec. Is, 1854.")]
    assert notice.attached_text == "Dec. Is, 1854."  # ...and keeps it beside it
    assert notice.attached_duplicates == [("#31", "#7")] and notice.as_duplicated  # article 31 already ends with it
    assert notice.duplicates == []
    [college] = found[AS_ONLY]  # an article none of our pieces claimed stays AS only
    assert college.parts[0].article_id == 31
    summary = IssueSegmentation(groups, {}, []).summary()
    assert (summary["attached_outside"], summary["as_duplicated"]) == (1, 1)


def test_attached_text_no_other_article_holds_is_not_flagged_duplicated():
    groups, _ = segment_page(*dec_18_page(date="Stricklandsville, June 3, 1856."), 1.0)
    [notice] = by_case(groups)[MATCH]
    assert notice.attached_text == "Stricklandsville, June 3, 1856."
    assert (notice.attached_duplicates, notice.as_duplicated) == ([], False)


def test_the_date_line_directly_under_the_article_ending_with_it_is_still_duplicated_when_apart_a_little():
    notice, boxes = dec_18_page()
    boxes[1] = replace(boxes[1], rect=(0.1, 0.3905, 0.16, 0.3955))  # a hair below article 31's box, not touching
    groups, _ = segment_page(notice, boxes, 1.0)
    [g] = by_case(groups)[MATCH]
    assert g.attached_duplicates == [("#31", "#7")]


def general_notices_page():
    """Wilmington p1: two notices in different columns, each under its own "GENERAL NOTICES." heading,
    which AS gives each notice's article; our notices leave the headings out."""
    left = ours("left", (0.1, 0.2, 0.3, 0.6), text=NOTICE_BODY)
    right = ours("right", (0.6, 0.2, 0.8, 0.6), text=LEWIS_22D)
    boxes = [box(1, 7, (0.1, 0.15, 0.3, 0.18), "GENERAL NOTICES.", cls="headline", order=0),
             box(2, 7, (0.1, 0.2, 0.3, 0.6), NOTICE_BODY, order=1),
             box(3, 9, (0.6, 0.15, 0.8, 0.18), "GENERAL NOTICES.", cls="headline", order=0),
             box(4, 9, (0.6, 0.2, 0.8, 0.6), LEWIS_22D, order=1)]
    return [left, right], boxes


def test_the_same_heading_attached_to_notices_in_different_columns_is_no_duplicate():
    groups, _ = segment_page(*general_notices_page(), 1.0)
    notices = by_case(groups)[MATCH]
    assert [g.attached_text for g in notices] == ["GENERAL NOTICES.", "GENERAL NOTICES."]
    assert [(g.attached_duplicates, g.as_duplicated) for g in notices] == [([], False), ([], False)]


def test_adjacent_needs_half_the_narrower_width_overlapping_and_a_gap_of_at_most_one_and_a_half_heights():
    date = (0.1, 0.39, 0.16, 0.395)  # 0.005 high
    assert adjacent(date, (0.1, 0.1, 0.3, 0.39))  # touching, from above
    assert adjacent(date, (0.1, 0.4, 0.3, 0.6))  # 0.005 below: one height
    assert adjacent(date, (0.1, 0.4025, 0.3, 0.6))  # 1.5 heights
    assert not adjacent(date, (0.1, 0.41, 0.3, 0.6))  # 3 heights
    assert adjacent(date, (0.13, 0.1, 0.3, 0.39))  # half the date's width overlaps
    assert not adjacent(date, (0.14, 0.1, 0.3, 0.39))  # a third
    assert not adjacent(date, (0.6, 0.1, 0.8, 0.39))  # another column


def test_holds_text_short_texts_by_their_exact_words_and_digits_long_ones_by_phrases():
    assert holds_text(COLLEGE, "Dec. Is, 1854.")  # punctuation aside
    assert not holds_text(COLLEGE, "Dec. 18, 1854.")
    assert not holds_text(COLLEGE, "Is, 18")  # whole runs only: "18" is not "1854"
    assert holds_text("x " + LEWIS_22D, "Mr. D. c. Lewis, writes to us under date of Bryants Swamp")
    assert not holds_text(LEWIS_22D, BROWN)
    a = AsPart(7, (box(2, 7, (0.1, 0.39, 0.16, 0.395), "Dec. Is, 1854."),))
    whole_7 = AsPart(7, (box(2, 7, (0.1, 0.39, 0.16, 0.395), "Dec. Is, 1854."), box(4, 7, (0.1, 0.41, 0.3, 0.6), "x")))
    assert attached_duplicates([a], [whole_7]) == []  # its own article does not count


# --- short left-over fragments found in our text ----------------------------------------------

PRATT = ("The letter of Senator Pratt, of Maryland, will commend itself to the attention of every "
         "reasonable man in the State.")
PRATT_GARBLED = ("90 The lettcr OF Senatur Pratt, Of Narv,ana ill\ncommend itsclf TO the attentiun of everv "
                 "reAsot' man iu the Statc.")
LETTER = "To the Editor. We have read with care the papers of the week and the speeches made at the meeting."


def test_fragment_owner_finds_a_garbled_short_run_in_a_pieces_text_by_partial_ratio():
    assert phrase_containment(PRATT_GARBLED, PRATT) == 0  # no whole phrase survives the OCR
    pieces = [ours("wed", (0.1, 0.1, 0.2, 0.2), text=WED), ours("letter", (0.1, 0.3, 0.2, 0.4), text=LETTER + " " + PRATT)]
    assert fragment_owner(PRATT_GARBLED, pieces) == 1
    assert fragment_owner("JUST RECEIVED, for sale cheap, a fine lot of bacon and lard.", pieces) is None
    assert fragment_owner("the papers", pieces) is None  # tiny: not looked for
    assert fragment_owner((PRATT_GARBLED + " ") * 6, pieces) is None  # 600 characters or more: not looked for
    assert fragment_owner(PRATT_GARBLED, [ours("short", (0.1, 0.1, 0.2, 0.2), text="Senator Pratt")]) is None


def test_a_short_left_over_found_in_our_text_joins_that_pieces_as_text_before_anything_else():
    """AS has a garbled line of our letter in a box of the letter's own article printed apart from
    it: found in the letter's text, it joins the comparison (not attached outside, not AS only)."""
    letter = ours("letter", (0.1, 0.1, 0.3, 0.5), text=LETTER + " " + PRATT, label="letter_to_editor")
    boxes = [box(1, 10, (0.1, 0.1, 0.3, 0.5), LETTER),
             box(2, 10, (0.6, 0.7, 0.9, 0.75), PRATT_GARBLED, order=1),
             box(3, None, (0.6, 0.1, 0.9, 0.15), "JUST RECEIVED, for sale cheap, a fine lot of bacon and lard.")]
    groups, _ = segment_page([letter], boxes, 1.0)
    found = by_case(groups)
    [g] = found[MATCH]
    assert g.as_text == LETTER + "\n\n" + PRATT_GARBLED and g.as_attached_outside == []
    assert [x.parts[0].boxes[0].box_id for x in found[AS_ONLY]] == [3]  # an unrelated short orphan stays AS only

def test_case_labels_are_neutral_about_the_counts_each_side_has():
    assert CASE_LABELS == {
        MATCH: "1:1 match",
        AS_SPLIT: "1 ours : N AS",
        AS_MERGE: "N ours : 1 AS",
        IN_AS_MERGE: "in N ours : 1 AS",
        CONTINUED: "continued",
        OURS_ONLY: "ours only",
        AS_ONLY: "AS only",
        AS_ONLY_TINY: "AS only, tiny",
        AS_ONLY_EMPTY: "AS only, no text",
        AS_PARTIAL: "1:1, AS partial",
        AS_WIDER: "1:1, AS article wider",
    }
    assert set(CASE_LABELS) == set(CASES)


# --- inputs -----------------------------------------------------------------------------------


def test_our_pieces_one_per_component_page():
    comp = {"uuid": "c", "label": "editorial", "content": "text", "pages": [1, 2],
            "boxes": [{"page": 1, "rects": [[0.1, 0.1, 0.2, 0.2]]}, {"page": 2, "rects": [[0.3, 0.3, 0.4, 0.4]]}]}
    pieces = our_pieces([comp, {"uuid": "d", "label": "notice", "content": "", "pages": [1]}])
    assert pieces == [
        OurPiece("c", "editorial", 1, ((0.1, 0.1, 0.2, 0.2),), "text", (1, 2)),
        OurPiece("c", "editorial", 2, ((0.3, 0.3, 0.4, 0.4),), "text", (1, 2)),
    ]


def as_page(stated=(1000, 1000)):
    """An AS page in full-scan pixels: a headline + body article (listed headline first) and an orphan author box."""
    return {
        "scan": {"width": stated[0], "height": stated[1]},
        "bboxes": [
            {"id": 1, "class": "article", "bbox": {"x0": 200, "y0": 300, "x1": 600, "y1": 1000}, "raw_text": "All persons.",
             "legibility": "Legible"},
            {"id": 0, "class": "headline", "bbox": {"x0": 200, "y0": 200, "x1": 600, "y1": 300}, "raw_text": "NOTICE.",
             "legibility": "Legible"},
            {"id": 2, "class": "author", "bbox": {"x0": 1200, "y0": 1800, "x1": 1600, "y1": 1900}, "raw_text": "J. SMITH.",
             "legibility": "Illegible"},
        ],
        "full articles": [{"full_article_id": 7, "object_ids": [0, 1], "headline": "NOTICE.", "article": "All persons.", "byline": ""}],
    }


def test_as_boxes_scale_by_the_real_scan_size_and_know_their_article_and_order():
    body, head, orphan = as_boxes(as_page(), 3, (2000, 2000))
    assert body == AsBox(3, 1, 7, (0.1, 0.15, 0.3, 0.5), "All persons.", "article", "Legible", 1)
    assert head == AsBox(3, 0, 7, (0.1, 0.1, 0.3, 0.15), "NOTICE.", "headline", "Legible", 0)
    assert orphan == AsBox(3, 2, None, (0.6, 0.9, 0.8, 0.95), "J. SMITH.", "author", "Illegible", 0)
    assert AsPart(7, (head, body)).text == "NOTICE.\n\nAll persons."


def test_as_boxes_leave_out_empty_boxes_beyond_the_scan():
    page = as_page()
    page["bboxes"] += [
        {"id": 40, "class": "article", "legibility": "Illegible", "raw_text": "",  # the margin past a tall page
         "bbox": {"x0": 2080, "y0": 120, "x1": 2780, "y1": 1560}, "reading_order_id": 0},
        {"id": 41, "class": "article", "legibility": "Legible", "raw_text": "Kept: it has text.",
         "bbox": {"x0": 2080, "y0": 120, "x1": 2780, "y1": 1560}, "reading_order_id": 0},
    ]
    ids = [b.box_id for b in as_boxes(page, 3, (2000, 2000))]
    assert 40 not in ids and 41 in ids
    assert off_page((1.04, 0.06, 1.39, 0.78)) and not off_page((0.9, 0.1, 1.05, 0.2))


# --- a whole issue ----------------------------------------------------------------------------

DIR = "https://tile.loc.gov/storage-services/service/ndnp/x/batch_x/data/sn1/00000000001/1856010101"


def test_segment_issue_compares_a_cross_page_component_once_across_its_pages():
    """One AS article on each page is two articles for one story: one question across both pages,
    listed on the first page; the second page points to it."""
    comp = {"uuid": "c", "label": "editorial", "content": "NOTICE. All persons. More text.", "pages": [1, 2],
            "boxes": [{"page": 1, "rects": [[0.1, 0.1, 0.3, 0.5]]}, {"page": 2, "rects": [[0.1, 0.1, 0.3, 0.5]]}],
            "links": {"loc_pdf_pages": [{"page_number": 1, "url": f"{DIR}/0001.pdf"},
                                        {"page_number": 2, "url": f"{DIR}/0002.pdf"}]}}
    scans = [ScanName("1856-01-01", 1, "sn1", "00000000001", 1, seq) for seq in ("0001", "0002")]  # both named p1
    pair = IssuePair(IssueKey("i", "sn1", "1856-01-01", 1), [comp], [(s, as_page()) for s in scans], 0, 1, [])
    seg = segment_issue(pair, {1: (2000, 2000), 2: (2000, 2000)})
    assert [(g.page, g.case, g.other_pages) for g in seg.groups] == [
        (1, AS_SPLIT, []), (1, AS_ONLY_TINY, []), (2, CONTINUED, [1]), (2, AS_ONLY_TINY, [])]
    split, pointer = seg.groups[0], seg.groups[2]
    assert split.pages == (1, 2) and split.cross_page and split.continued == [pointer] and pointer.continues is split
    assert [(part.page, part.short_name) for part in split.parts] == [(1, "#7"), (2, "#7")]
    assert split.as_text == "NOTICE.\n\nAll persons.\n\nNOTICE.\n\nAll persons."
    assert split.ours_text == comp["content"]  # our component's text once, not once per page
    assert "shared_phrases" not in split.metrics  # every page of it compared
    assert [part.page for part in pointer.parts] == [2] and pointer.metrics == {}


def cross_page_groups(page_two_parts=("#3",)):
    """A component on pages 1-2 over AS #1 and #2 on page 1 and `page_two_parts` on page 2, per page."""
    text = "the whole story " * 20
    p1 = ours("c", (0.1, 0.1, 0.3, 0.9), text=text, pages=(1, 2))
    p2 = ours("c", (0.6, 0.1, 0.9, 0.5), text=text, page=2, pages=(1, 2))
    one = segment_page([p1], [box(1, 1, (0.1, 0.1, 0.3, 0.5), "the whole story begins"),
                              box(2, 2, (0.1, 0.5, 0.3, 0.9), "and goes on here")], 1.0)[0]
    two_boxes = [box(10 + k, int(name[1:]), (0.6, 0.1 + 0.2 * k, 0.9, 0.3 + 0.2 * k), f"and ends {k}", page=2)
                 for k, name in enumerate(page_two_parts)]
    two = segment_page([p2], two_boxes, 1.0)[0]
    return one + two


def test_a_cross_page_component_over_two_as_articles_on_one_page_and_one_on_the_next_is_one_as_split_of_three():
    groups = cross_page_groups()
    assert [(g.page, g.case) for g in groups] == [(1, AS_SPLIT), (2, AS_PARTIAL)]  # per page (AS covers half of p. 2's)
    link_pages(groups)
    split, pointer = groups
    assert (split.case, pointer.case, pointer.continues) == (AS_SPLIT, CONTINUED, split)
    assert [(part.page, part.short_name) for part in split.parts] == [(1, "#1"), (1, "#2"), (2, "#3")]
    assert [p.page for p in split.pieces] == [1, 2] and split.pages == (1, 2)
    assert split.ours_text == "the whole story " * 20  # once
    assert split.our_rects == ((0.1, 0.1, 0.3, 0.9),)  # geometry: the group's own page
    assert split.as_box_rects_on(2) == ((0.6, 0.1, 0.9, 0.3),)


def test_a_cross_page_component_with_one_as_article_on_each_page_is_still_a_question_and_never_a_match():
    text = "the whole story " * 20
    p1 = ours("c", (0.1, 0.1, 0.3, 0.9), text=text, pages=(1, 2))
    p2 = ours("c", (0.6, 0.1, 0.9, 0.5), text=text, page=2, pages=(1, 2))
    groups = (segment_page([p1], [box(1, 1, (0.1, 0.1, 0.3, 0.9), "the whole story")], 1.0)[0]
              + segment_page([p2], [box(9, 1, (0.6, 0.1, 0.9, 0.5), "the whole story", page=2)], 1.0)[0])
    assert [g.case for g in groups] == [MATCH, MATCH]
    link_pages(groups)
    assert [g.case for g in groups] == [AS_SPLIT, CONTINUED]
    assert [(part.page, part.short_name) for part in groups[0].parts] == [(1, "#1"), (2, "#1")]  # 2 articles, 1 story


def test_a_cross_page_component_with_as_text_from_one_article_in_all_keeps_its_per_page_cases():
    text = "the whole story " * 20
    p1 = ours("c", (0.1, 0.1, 0.3, 0.9), text=text, pages=(1, 2))
    p2 = ours("c", (0.6, 0.1, 0.9, 0.5), text=text, page=2, pages=(1, 2))
    groups = (segment_page([p1], [box(1, 1, (0.1, 0.1, 0.3, 0.9), "the whole story")], 1.0)[0]
              + segment_page([p2], [], 1.0)[0])
    link_pages(groups)
    assert [g.case for g in groups] == [MATCH, OURS_ONLY]


def test_segment_issue_falls_back_to_the_stated_scan_size_and_lists_pages_without_a_scan():
    comp = {"uuid": "c", "label": "notice", "content": "x", "pages": [2],
            "boxes": [{"page": 2, "rects": [[0.1, 0.1, 0.3, 0.5]]}],
            "links": {"loc_pdf_pages": [{"page_number": 2, "url": f"{DIR}/0002.pdf"}]}}
    on_page_1 = {"uuid": "d", "label": "notice", "content": "y", "pages": [1],
                 "links": {"loc_pdf_pages": [{"page_number": 1, "url": f"{DIR}/0001.pdf"}]}}
    scan = ScanName("1856-01-01", 1, "sn1", "00000000001", 1, "0001")
    pair = IssuePair(IssueKey("i", "sn1", "1856-01-01", 1), [comp, on_page_1], [(scan, as_page((2000, 2000)))], 0, 1, [])
    seg = segment_issue(pair, {})
    assert seg.pages_without_as == [2]
    head = next(b for g in seg.groups for part in g.parts for b in part.boxes if b.box_id == 0)
    assert head.rect == (0.1, 0.1, 0.3, 0.15)


def test_segment_issue_leaves_out_pages_whose_as_scan_is_another_digitisation():
    """The Patowmac Guardian 1791: AS segmented reel 01000718192 seq 0007-0008, the API's boxes are on
    .../print/1791071101/0133-0134; here page 2's AS scan is the API's, page 1's is not."""
    other = "https://tile.loc.gov/storage-services/service/ndnp/x/batch_x/data/sn1/print/1856010101"
    comp = {"uuid": "c", "label": "editorial", "content": "NOTICE. All persons.", "pages": [1, 2],
            "boxes": [{"page": 1, "rects": [[0.1, 0.1, 0.3, 0.5]]}, {"page": 2, "rects": [[0.1, 0.1, 0.3, 0.5]]}],
            "links": {"loc_pdf_pages": [{"page_number": 1, "url": f"{other}/0133.pdf"},
                                        {"page_number": 2, "url": f"{DIR}/0002.pdf"}]}}
    scans = [ScanName("1856-01-01", 1, "sn1", "01000718192", 1, "0007"),
             ScanName("1856-01-01", 2, "sn1", "00000000001", 1, "0002")]
    pair = IssuePair(IssueKey("i", "sn1", "1856-01-01", 1), [comp], [(s, as_page()) for s in scans], 0, 1, [])
    seg = segment_issue(pair, {1: (2000, 2000), 2: (2000, 2000)})
    assert {g.page for g in seg.groups} == {2} and sorted(seg.coverage) == [2]
    assert (seg.same_scan, seg.pages_other_scan, seg.pages_without_as) == ((1, 2), [1], [])
    summary = seg.summary()
    assert (summary["same_scan"], summary["pages_other_scan"]) == ((1, 2), [1])


def test_summary_counts_cases_coverage_and_median_match_ratio():
    pieces = [ours("a", (0.1, 0.1, 0.3, 0.5), text="the river is low"),
              ours("b", (0.4, 0.1, 0.6, 0.5), text="the boats run"),
              ours("z", (0.7, 0.1, 0.9, 0.2))]
    boxes = [box(1, 1, (0.1, 0.1, 0.3, 0.5), "the river is low"),
             box(2, 2, (0.4, 0.1, 0.6, 0.5), "the boats ran"),
             box(3, 3, (0.1, 0.6, 0.3, 0.8), "GENERAL NOTICES.")]
    groups, cov = segment_page(pieces, boxes, 1.0)
    summary = IssueSegmentation(groups, {1: cov}, []).summary()
    assert summary["cases"] == {MATCH: 2, AS_SPLIT: 0, AS_MERGE: 0, IN_AS_MERGE: 0, CONTINUED: 0, OURS_ONLY: 1,
                                AS_ONLY: 0, AS_ONLY_TINY: 1, AS_ONLY_EMPTY: 0, AS_PARTIAL: 0, AS_WIDER: 0}
    assert summary["ours_covered_by_as"] == pytest.approx(16 / 18, abs=1e-3)
    assert summary["as_covered_by_ours"] == pytest.approx(16 / 20, abs=1e-3)
    assert summary["median_match_ratio"] == pytest.approx((100 + text_metrics("the boats run", "the boats ran", partial=False)["ratio"]) / 2)


# --- empty AS boxes, nibbles and how much of our piece AS covers --------------------------------


def test_an_empty_as_box_inside_our_piece_is_no_part_of_a_split():
    piece = ours("a", (0.1, 0.1, 0.3, 0.5), text="NOTICE. The river is low.")
    boxes = [box(1, 1, (0.1, 0.1, 0.3, 0.5), "NOTICE. The river is low."),
             box(2, None, (0.12, 0.3, 0.28, 0.32), "   ")]  # an AS box with no text: an artifact
    [g] = segment_page([piece], boxes, 1.0)[0]
    assert g.case == MATCH and [p.article_id for p in g.parts] == [1]


def test_a_line_of_our_piece_in_another_as_article_is_a_nibble_not_a_split():
    """p1-1 of The Gazette (ME) 1798-06-11: AS put the address's title in the dateline's article."""
    body = "IF ever there was a time when national exertion, firmness and unanimity were necessary " * 8
    piece = ours("a", (0.0, 0.16, 0.25, 0.77), text="To the Citizens of the United States. " + body)
    boxes = [box(9, 17, (0.03, 0.12, 0.23, 0.137), "Monday, June 11, 1798.", cls="headline"),
             box(16, 17, (0.04, 0.165, 0.24, 0.178), "To the CITIZENS of the United States.", order=2),
             box(3, 4, (0.0, 0.171, 0.24, 0.77), body)]
    [g] = [g for g in segment_page([piece], boxes, 0.6)[0] if g.pieces]
    assert g.case == MATCH
    assert [p.article_id for p in g.parts] == [4]
    assert [(p.article_id, p.text) for p in g.as_nibbles] == [(17, "To the CITIZENS of the United States.")]
    assert [p.article_id for p in g.as_attached_outside] == [17]  # the dateline: still a boundary question


def test_a_nibble_is_at_most_12_words_and_5_percent_of_our_piece():
    body = "word " * 400
    piece = ours("a", (0.1, 0.1, 0.3, 0.9), text="A short heading here. " + body)
    big = "a whole paragraph of thirteen words that some other AS article holds here"  # 13 words
    boxes = [box(1, 1, (0.1, 0.12, 0.3, 0.9), body), box(2, 2, (0.1, 0.1, 0.3, 0.115), big)]
    [g] = segment_page([piece], boxes, 1.0)[0]
    assert g.case == AS_SPLIT and g.as_nibbles == []
    boxes = [box(1, 1, (0.1, 0.2, 0.3, 0.9), body), box(2, 2, (0.1, 0.1, 0.3, 0.2), "A short heading here.")]  # 12.5% of ours
    [g] = segment_page([piece], boxes, 1.0)[0]
    assert g.case == AS_SPLIT and g.as_nibbles == []


def test_two_scraps_alone_over_a_component_as_has_little_text_for_are_partial_not_nibbles():
    piece = ours("a", (0.1, 0.1, 0.3, 0.9), text="FOUND. A purse. Apply here.")
    boxes = [box(1, 1, (0.1, 0.1, 0.3, 0.11), "FOUND."), box(2, 2, (0.1, 0.5, 0.3, 0.51), "A purse.")]
    [g] = segment_page([piece], boxes, 1.0)[0]
    assert g.case == AS_PARTIAL and g.as_nibbles == [] and len(g.parts) == 2


def test_one_as_article_covering_under_90_percent_of_our_piece_is_partial_not_a_match():
    """AS boxed only the notice's heading ("MEETINGS."), or left a paragraph of it in no box at all."""
    piece = ours("a", (0.1, 0.1, 0.3, 0.5), text="MEETINGS. The annual meeting will be held.")
    [g] = segment_page([piece], [box(1, 1, (0.1, 0.1, 0.3, 0.12), "MEETINGS.", cls="headline")], 1.0)[0]
    assert g.case == AS_PARTIAL
    assert g.metrics["as_cover"] == pytest.approx(0.05, abs=0.01)
    [g] = segment_page([piece], [box(1, 1, (0.1, 0.1, 0.3, 0.47), "MEETINGS. The annual meeting")], 1.0)[0]
    assert g.case == MATCH and g.metrics["as_cover"] == pytest.approx(0.935, abs=0.01)


def test_a_match_carries_the_iou_of_our_piece_and_the_as_articles_boxes():
    piece = ours("a", (0.1, 0.1, 0.3, 0.5), text="The river is low.")
    [g] = segment_page([piece], [box(1, 1, (0.1, 0.1, 0.3, 0.6), "The river is low.")], 1.0)[0]
    assert g.case == MATCH and g.metrics["iou"] == pytest.approx(0.8, abs=0.01)


def test_partial_is_its_own_case_with_a_neutral_label():
    assert AS_PARTIAL in CASES and CASE_LABELS[AS_PARTIAL] == "1:1, AS partial"


def test_coverage_counts_rows_so_a_centred_headline_narrower_than_the_column_covers_its_row():
    """Small Pox (The Gazette (ME) 1798-06-11, p. 3): AS's headline box is only as wide as its words."""
    piece = ours("a", (0.1, 0.1, 0.3, 0.3), text="Small Pox. All persons who intend being inoculated.")
    boxes = [box(1, 5, (0.17, 0.1, 0.23, 0.13), "Small Pox.", cls="headline"),
             box(2, 5, (0.1, 0.132, 0.3, 0.3), "All persons who intend being inoculated.", order=1)]
    [g] = segment_page([piece], boxes, 1.0)[0]
    assert g.case == MATCH and g.metrics["as_cover"] == pytest.approx(1.0, abs=0.01)


def test_an_empty_box_of_the_matched_article_covers_its_rows_but_an_empty_orphan_does_not():
    """From The Task (p. 4): AS boxed the headline in the poem's article but read no text in it."""
    piece = ours("a", (0.1, 0.1, 0.3, 0.3), text="FROM THE TASK. Oh for a lodge in some vast wilderness.")
    body = box(2, 7, (0.1, 0.13, 0.3, 0.3), "Oh for a lodge in some vast wilderness.", order=1)
    [g] = segment_page([piece], [box(1, 7, (0.15, 0.1, 0.25, 0.13), "", cls="headline"), body], 1.0)[0]
    assert g.case == MATCH and g.metrics["as_cover"] == pytest.approx(1.0, abs=0.01)
    [g] = segment_page([piece], [box(1, None, (0.15, 0.1, 0.25, 0.13), "", cls="headline"), body], 1.0)[0]
    assert g.case == AS_PARTIAL and g.metrics["as_cover"] == pytest.approx(0.87, abs=0.01)


def test_coverage_weights_each_of_our_rects_by_its_area():
    """p2-2: the opening paragraph at the foot of column 1 is in no AS box; the rest, column 2, is."""
    piece = ours("a", (0.1, 0.8, 0.3, 0.9), (0.35, 0.1, 0.55, 0.5), text="Committees were appointed. A committee was.")
    [g] = segment_page([piece], [box(1, 1, (0.35, 0.1, 0.55, 0.5), "A committee was.")], 1.0)[0]
    assert g.case == AS_PARTIAL and g.metrics["as_cover"] == pytest.approx(0.8, abs=0.01)


# --- an AS article running beyond our component ----------------------------------------------------

REWARD = ("Ten Dollars Reward. STRAYED from Mount Washington, a hound dog; whoever brings him home shall "
          "receive the above Reward. WILLIAM SPENCE.")
LEMONS = ("Just received and for Sale by the Subscriber, Fresh Lemons by the box, Filberts, Almonds and English "
          "Walnuts, Large sweet Oranges and Tamarinds, Cocoa Nuts, Raisins. A. WILLIS. Nov. 4.")


def test_an_as_article_running_well_beyond_our_component_is_as_wider_not_a_match():
    """The Alexandria Advertiser 1802-11-04 p3: AS's one box is our reward notice and the lemons ad below it."""
    piece = ours("reward", (0.7, 0.7, 0.92, 0.8), text=REWARD)
    [g] = [g for g in segment_page([piece], [box(1, 13, (0.7, 0.7, 0.92, 0.9), REWARD + " " + LEMONS)], 1.0)[0]
           if g.pieces]
    assert g.case == AS_WIDER and g.metrics["ours_cover"] == pytest.approx(0.55, abs=0.05)  # text cut by proportion
    assert "Lemons" in g.attached_text  # the boundary question decides it


def test_a_dateline_attached_outside_leaves_a_match():
    groups, _ = segment_page(*dec_18_page(), 1.0)
    [notice] = by_case(groups)[MATCH]
    assert notice.metrics["ours_cover"] >= 0.9


def test_text_after_a_piece_in_an_article_two_pieces_hold_is_attached_outside_the_piece_it_follows():
    """p4: AS's #24 is the last line of "To be Rented" (ours, above), our auction, then the text below it."""
    rented = "To be Rented, the House in which I now live. Apply to the subscriber. JOHN GILL."
    auction = "ADVERTISEMENT. On Thursday the 18th day of November will be sold the lands of the late J. Fitzgerald. " * 3
    below = "teen months. The Lands will be conveyed on payment of the fourth instalment, and the Ladies are invited."
    pieces = [ours("rented", (0.77, 0.35, 0.98, 0.42), text=rented), ours("auction", (0.77, 0.42, 0.98, 0.83), text=auction)]
    boxes = [box(1, 26, (0.77, 0.35, 0.98, 0.41), rented[:-11]),
             box(2, 24, (0.77, 0.41, 0.98, 0.96), "JOHN GILL. " + auction + " " + below)]
    groups, _ = segment_page(pieces, boxes, 0.6)
    [wider] = [g for g in groups if g.pieces and g.pieces[0].uuid == "auction"]
    assert wider.case == AS_WIDER and "Lands will be conveyed" in wider.attached_text
    assert AS_ONLY not in by_case(groups)


def test_as_wider_has_a_neutral_label():
    assert AS_WIDER in CASES and CASE_LABELS[AS_WIDER] == "1:1, AS article wider"


def test_left_over_text_of_a_box_lying_inside_the_piece_that_holds_the_box_is_that_pieces_not_attached():
    """Southport Telegraph 1840-11-17 p3: the cut fell where our closing words recur in a land table,
    leaving the table's garbled tail, printed inside our component, looking as if it were outside it."""
    ours_text = "SALE OF LANDS FOR TAXES. The following lands will be sold. sec four town one range val tax two cts"
    tail = "h n or sec twenty town ronge rs n,ecad rol too hundred forty dol S ffac or sec thirty one"
    piece = ours("lands", (0.1, 0.1, 0.3, 0.9), text=ours_text)
    [g] = segment_page([piece], [box(1, 25, (0.1, 0.1, 0.3, 0.9), ours_text + " " + tail)], 1.0)[0]
    assert g.as_attached_outside == [] and "thirty one" in g.as_text


WHOLE_HOG_1 = ('"IHE WHOLE HoG."-Lorenzo Brown OF Vernon slaughtered hog a few days sncc, that weighed 1000 pounds '
               "the LIST of January, and which dressed $50 pounds. The hog was sold To Mr.. Amos W. Ross Ct Norhfed,")
WHOLE_HOG_2 = ('Us "lHE WHOLE Hog. "-Lorenzo Brown G D. Vernon slaughtered hog few days sncC Is that weighed 1000 pounds '
               "the LIST of January er and which dressed $50 pounds. The hog wa ne sold To Mr.. Amos W. Ross Ct")


def test_the_same_item_ocrd_twice_too_noisily_to_share_half_its_phrases_is_a_duplicate_by_character_ratio():
    """Vermont Phoenix 1862-02-20 p2: AS has "The Whole Hog" in its own box and again in a column box."""
    assert phrase_containment(WHOLE_HOG_2, WHOLE_HOG_1) < 50
    a = AsPart(5, (box(1, 5, (0.1, 0.7, 0.3, 0.8), WHOLE_HOG_1),))
    b = AsPart(9, (box(2, 9, (0.1, 0.0, 0.3, 0.8), WHOLE_HOG_2),))
    kept, found = collapse_duplicates([a, b])
    assert kept == [b] and found == [("#9", "#5")]  # the longer kept


def test_nested_as_boxes_are_text_boxes_mostly_inside_another_articles_box():
    """Vermont Phoenix 1862-02-20 p2: one AS box is all of column 3; item boxes inside it are OCR'd again."""
    boxes = [box(23, 24, (0.3, 0.04, 0.44, 0.79), "boxes and packages ... WHITINGHAM ... Lorenzo Brown"),
             box(66, 67, (0.31, 0.6, 0.44, 0.67), "WHITINGHAM.-A Mrs. Stockwell"),  # inside #24
             box(39, 40, (0.31, 0.73, 0.44, 0.8), '"THE WHOLE HOG."'),  # mostly inside (6/7 of its height)
             box(5, 24, (0.32, 0.05, 0.42, 0.06), "HEADLINE", cls="headline"),  # #24's own headline: no nesting
             box(7, None, (0.32, 0.3, 0.42, 0.31), "")]  # an empty box: no text, not counted
    assert nested_boxes(boxes) == [(1, "#67", "#24"), (1, "#40", "#24")]


def test_a_few_pixels_on_a_one_line_component_are_within_tolerance_both_ways():
    """Daily Argus 1875-01-20 p1: "Failure of B. F. Allen, in Chicago." AS's box is a hair taller."""
    piece = ours("allen", (0.3, 0.3522, 0.42, 0.358), text="Failure of B. F. Allen, in Chicago.")
    [g] = segment_page([piece], [box(45, 46, (0.3, 0.352, 0.421, 0.361), "Failure of b. z. Allen, in Chicago.")], 1.4)[0]
    assert g.case == MATCH
    [g] = segment_page([piece], [box(45, 46, (0.3, 0.3545, 0.421, 0.358), "Failure of b. z. Allen, in Chicago.")], 1.4)[0]
    assert g.case == MATCH and g.metrics["as_cover"] == 1.0


def test_a_box_in_no_listed_article_falls_back_to_its_own_full_article_id():
    """Vermont Phoenix 1862-02-20 p2: AS's article 9 (two headlines and an empty body box) is missing
    from its "full articles" list, but each box still says full_article_id 9."""
    data = {"scan": {"width": 100, "height": 100}, "full articles": [{"full_article_id": 4, "object_ids": [3]}],
            "bboxes": [{"id": 3, "class": "article", "raw_text": "Official Thanks", "full_article_id": 4,
                        "reading_order_id": 0, "bbox": {"x0": 80, "y0": 5, "x1": 95, "y1": 30}},
                       {"id": 25, "class": "headline", "raw_text": "BURNSIDE MARCHING ON.", "full_article_id": 9,
                        "reading_order_id": -2, "bbox": {"x0": 84, "y0": 31, "x1": 92, "y1": 32}},
                       {"id": 33, "class": "headline", "raw_text": "More Towns Occupied.", "full_article_id": 9,
                        "reading_order_id": -1, "bbox": {"x0": 83, "y0": 32, "x1": 94, "y1": 33}},
                       {"id": 8, "class": "article", "raw_text": "", "full_article_id": 9,
                        "reading_order_id": 0, "bbox": {"x0": 82, "y0": 33, "x1": 95, "y1": 51}}]}
    found = {b.box_id: (b.article_id, b.order) for b in as_boxes(data, 2, (100, 100))}
    assert found == {3: (4, 0), 25: (9, -2), 33: (9, -1), 8: (9, 0)}


def test_an_empty_body_box_is_no_text_so_a_story_whose_body_as_could_not_read_is_partial():
    piece = ours("burnside", (0.82, 0.31, 0.95, 0.51), text="BURNSIDE MARCHING ON. More Towns Occupied. A dispatch states")
    boxes = [box(25, 9, (0.84, 0.31, 0.92, 0.32), "BURNSIDE MARCHING ON.", cls="headline", order=-2),
             box(33, 9, (0.83, 0.32, 0.94, 0.33), "More Towns Occupied.", cls="headline", order=-1),
             box(8, 9, (0.82, 0.33, 0.95, 0.51), "", order=0)]
    [g] = segment_page([piece], boxes, 1.3)[0]
    assert g.case == AS_PARTIAL and [p.short_name for p in g.parts] == ["#9"]



def test_a_long_component_whose_end_fills_a_good_part_of_an_as_article_counts_towards_its_merge():
    """Vermont Phoenix 1862-02-20 p2: AS's #4 is one box holding the end of our Roanoke Island story
    (which runs over several columns, so little of it lies inside #4) and then "Official Thanks"."""
    roanoke_text = "GEN. BURNSIDE SUCCESSFUL. " + "the expedition of his gunboats went on " * 20 + "as more full details will show."
    thanks_text = "Official Thanks to Our Heroes. The following document, expressing the thanks of the president."
    roanoke = ours("roanoke", (0.4, 0.05, 0.55, 0.95), (0.56, 0.05, 0.69, 0.95), (0.70, 0.05, 0.81, 0.95),
                   (0.82, 0.05, 0.95, 0.16), text=roanoke_text)
    thanks = ours("thanks", (0.82, 0.165, 0.95, 0.31), text=thanks_text)
    boxes = [box(3, 4, (0.82, 0.05, 0.95, 0.31), "as more full details will show. " + thanks_text)]
    groups, _ = segment_page([roanoke, thanks], boxes, 1.3)
    [merge] = by_case(groups)[AS_MERGE]
    assert [p.uuid for p in merge.pieces] == ["roanoke", "thanks"]
    assert [g.case for g in groups if g.pieces and g.pieces[0].uuid == "thanks" and g.case != AS_MERGE] == [IN_AS_MERGE]


def test_illegible_as_boxes_we_read_are_empty_illegible_boxes_mostly_inside_our_components():
    """Workmen's Advocate 1886-04-25 p1: AS marked the "Lynn K. of L. Fair" body Illegible and read no text."""
    pieces = [ours("fair", (0.49, 0.749, 0.627, 0.833), text="LYNN K. OF L. FAIR. The great Fair will take place.")]
    boxes = [AsBox(1, 0, 1, (0.49, 0.758, 0.626, 0.834), "", "article", "Illegible"),
             AsBox(1, 17, 1, (0.509, 0.747, 0.605, 0.756), "LYNN K. OF /. FAIR.", "headline", "Legible"),
             AsBox(1, 5, 2, (0.7, 0.1, 0.9, 0.3), "", "article", "Illegible"),  # nothing of ours there
             AsBox(1, 6, 3, (0.49, 0.6, 0.627, 0.7), "", "article", "Legible")]  # empty but not marked illegible
    assert illegible_read(pieces, boxes, 1.4) == [(1, 0, "#1", ("fair",))]


def test_a_component_whose_missing_text_lies_under_boxes_as_marked_illegible_is_as_illegible():
    piece = ours("fair", (0.49, 0.749, 0.627, 0.833), text="LYNN K. OF L. FAIR. The great Fair will take place.")
    head = AsBox(1, 17, 1, (0.509, 0.747, 0.605, 0.756), "LYNN K. OF /. FAIR.", "headline", "Legible")
    body = AsBox(1, 0, 1, (0.49, 0.758, 0.626, 0.834), "", "article", "Illegible")
    [g] = segment_page([piece], [head, body], 1.4)[0]
    assert g.case == AS_PARTIAL and g.as_illegible
    [g] = segment_page([piece], [head], 1.4)[0]  # no AS box over the body at all
    assert g.case == AS_PARTIAL and not g.as_illegible
    [g] = segment_page([piece], [body], 1.4)[0]  # only the illegible box: AS has no text for any of it
    assert g.case == OURS_ONLY and g.as_illegible


def test_scraps_of_several_as_articles_over_a_component_as_has_no_body_text_for_are_partial_not_a_split():
    """Workmen's Advocate 1886-04-25 p4: "Woman's Rights": AS's headline, an illegible empty body box,
    and a garbled one-word signature in another AS article. Nothing to ask whether it is one or several."""
    piece = ours("rights", (0.198, 0.073, 0.339, 0.23), text="WOMAN'S RIGHTS. The Knights of Labor have done more. " * 4)
    boxes = [AsBox(4, 37, 2, (0.216, 0.072, 0.314, 0.08), "WOMAN'S RIGHTS.", "headline", "Legible"),
             AsBox(4, 1, 2, (0.197, 0.082, 0.337, 0.23), "", "article", "Illegible"),
             AsBox(4, 68, 23, (0.202, 0.225, 0.227, 0.23), "I0jI1o'.", "author", "Legible")]
    [g] = segment_page([piece], boxes, 1.4)[0]
    assert g.case == AS_PARTIAL and g.as_illegible and sorted(p.short_name for p in g.parts) == ["#2", "#23"]


def test_a_short_item_as_covers_split_into_two_scraps_is_still_a_split():
    piece = ours("found", (0.1, 0.1, 0.3, 0.12), text="FOUND. A purse. Apply here.")
    boxes = [box(1, 1, (0.1, 0.1, 0.3, 0.11), "FOUND."), box(2, 2, (0.1, 0.11, 0.3, 0.12), "A purse. Apply here.")]
    [g] = segment_page([piece], boxes, 1.0)[0]
    assert g.case == AS_SPLIT



def gala_groups():
    """Workmen's Advocate 1886-04-25: our "Gala Labor Day" runs from p. 1 (two AS articles) to p. 3,
    where AS has only the continuation heading and marked the body box illegible, reading nothing."""
    text = "A GALA LABOR DAY. Two monster meetings of workmen. " * 10
    p1 = ours("gala", (0.06, 0.15, 0.34, 0.93), text=text, pages=(1, 3))
    p3 = ours("gala", (0.2, 0.16, 0.34, 0.69), text=text, page=3, pages=(1, 3))
    one = segment_page([p1], [box(1, 12, (0.06, 0.15, 0.2, 0.93), "A GALA LABOR DAY. Two monster meetings"),
                              box(2, 19, (0.2, 0.15, 0.34, 0.93), "of workmen gathered")], 1.4)[0]
    three = segment_page([p3], [AsBox(3, 5, 3, (0.21, 0.155, 0.33, 0.164), "A GALA DAY FOR LABOR.", "headline", "Legible"),
                                AsBox(3, 6, 3, (0.2, 0.165, 0.34, 0.69), "", "article", "Illegible")], 1.4)[0]
    groups = one + three
    link_pages(groups)
    return groups


def test_a_component_across_pages_knows_the_pages_as_marked_illegible_or_lacks_the_text_of():
    split, pointer = gala_groups()
    assert (split.case, pointer.case) == (AS_SPLIT, CONTINUED)
    assert split.illegible_part and split.missing_pages == [3]
    plain = cross_page_groups()
    link_pages(plain)
    assert not plain[0].illegible_part and plain[0].missing_pages == [2]  # p. 2: AS covers half of it


def test_an_as_article_outside_our_components_with_no_text_at_all_is_as_only_empty_not_tiny():
    """Workmen's Advocate 1886-04-25 p3: whole AS articles (headline and body) marked Illegible, no text."""
    boxes = [AsBox(3, 54, 34, (0.389, 0.205, 0.442, 0.211), "", "headline", "Illegible"),
             AsBox(3, 33, 34, (0.344, 0.212, 0.485, 0.257), "", "article", "Illegible"),
             AsBox(3, 40, 23, (0.372, 0.831, 0.452, 0.841), "\\\\ ducrti scnlcnts.", "headline", "Questionable")]
    groups, _ = segment_page([], boxes, 1.3)
    assert [(g.case, g.parts[0].short_name) for g in groups] == [(AS_ONLY_EMPTY, "#34"), (AS_ONLY_TINY, "#23")]
    assert CASE_LABELS[AS_ONLY_EMPTY] == "AS only, no text"


def test_illegible_area_is_as_boxed_area_marked_illegible_and_unread_over_all_its_boxed_area():
    boxes = [AsBox(1, 1, 1, (0.0, 0.0, 0.5, 0.5), "text", "article", "Legible"),
             AsBox(1, 2, 2, (0.5, 0.0, 1.0, 0.5), "", "article", "Illegible"),
             AsBox(1, 3, 3, (0.5, 0.5, 1.0, 1.0), "read anyway", "article", "Illegible")]  # has text: AS read it
    illegible, total = illegible_area(boxes, 1.0)
    assert total == 750_000 and illegible == 250_000
