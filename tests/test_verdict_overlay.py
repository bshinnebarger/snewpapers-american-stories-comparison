from PIL import Image

from snewpapers_tools.adjudicate import (
    AMERICAN_STORIES,
    EXCLUDED,
    MATCHED,
    MISSED,
    NEITHER,
    NO_VERDICT,
    NOT_CAPTURED,
    SNEWPAPERS,
    Judgement,
    TINY,
    UNSTABLE,
    GroupVerdict,
)
from snewpapers_tools.overlay import SheetLayout
from snewpapers_tools.segmentation import (
    AS_MERGE,
    AS_ONLY,
    AS_ONLY_TINY,
    AS_ONLY_EMPTY,
    AS_PARTIAL,
    AS_SPLIT,
    AS_WIDER,
    CONTINUED,
    IN_AS_MERGE,
    MATCH,
    OURS_ONLY,
    AsBox,
    AsPart,
    Group,
    IssueSegmentation,
    OurPiece,
)
from snewpapers_tools.verdict_overlay import (
    AS_ONLY_TAG,
    AS_PARTIAL_TAG,
    DUP,
    EQUAL,
    NOT_SERVED,
    RIGHT,
    SNEW_ONLY,
    UNSURE,
    WRONG,
    ITEMS_IN_ONE,
    Mark,
    _stacks,
    arrow_route,
    attached_links,
    cross_page_links,
    group_marks,
    issue_marks,
    link_labels,
    place_stacks,
    reduce_marks,
    scorecard_line,
    tallies,
    tally_text,
    verdict_overlay,
    wrap,
)

PIECE = (0.1, 0.1, 0.4, 0.5)


def piece(rects=(PIECE,), page=1, uuid="u1"):
    return OurPiece(uuid, "editorial", page, tuple(rects), "some text", (page,))


def box(box_id, article_id, rect, page=1, text="words", whole=None):
    return AsBox(page, box_id, article_id, rect, text, "article", whole=whole)


def part(article_id, *boxes):
    return AsPart(article_id, tuple(boxes))


def kinds(marks, side):
    return [(m.kind, m.rect) for m in marks if m.side == side]


def verdict(case, outcome, attached=None, gid="p1-1"):
    return GroupVerdict(gid, case, outcome, attached_outcome=attached)


# --- marks per case -----------------------------------------------------------------------------


def test_match_puts_equal_on_our_piece_and_on_the_real_as_boxes_never_on_slices():
    column = (0.1, 0.0, 0.4, 0.9)  # the AS box; our piece holds a slice of it (the rest is a left-over nobody claims)
    group = Group(1, MATCH, [piece()], [part(7, box(1, 7, (0.1, 0.1, 0.4, 0.5), whole=column),
                                             box(2, 7, (0.1, 0.0, 0.4, 0.1)))])
    marks = group_marks(group, verdict(MATCH, MATCHED), "p1-1")
    assert kinds(marks, SNEWPAPERS) == [(EQUAL, PIECE)]
    assert kinds(marks, AMERICAN_STORIES) == [(EQUAL, column), (EQUAL, (0.1, 0.0, 0.4, 0.1))]
    assert {m.gid for m in marks} == {"p1-1"}


def split_group():
    return Group(1, AS_SPLIT, [piece()], [part(1, box(1, 1, (0.1, 0.1, 0.4, 0.3))), part(2, box(2, 2, (0.1, 0.3, 0.4, 0.5)))])


def test_as_split_decided_for_snewpapers_checks_ours_and_crosses_every_as_part():
    marks = group_marks(split_group(), verdict(AS_SPLIT, SNEWPAPERS), "p1-1")
    assert [k for k, _ in kinds(marks, SNEWPAPERS)] == [RIGHT]
    assert [k for k, _ in kinds(marks, AMERICAN_STORIES)] == [WRONG, WRONG]


def test_as_split_decided_for_american_stories_is_the_reverse():
    marks = group_marks(split_group(), verdict(AS_SPLIT, AMERICAN_STORIES), "p1-1")
    assert [k for k, _ in kinds(marks, SNEWPAPERS)] == [WRONG]
    assert [k for k, _ in kinds(marks, AMERICAN_STORIES)] == [RIGHT, RIGHT]


def test_unstable_missing_or_no_verdict_is_a_question_mark_on_both_sides_and_neither_a_cross_on_both():
    for v in (verdict(AS_SPLIT, UNSTABLE), verdict(AS_SPLIT, NO_VERDICT), None):
        marks = group_marks(split_group(), v, "p1-1")
        assert {m.kind for m in marks} == {UNSURE}
        assert {m.side for m in marks} == {SNEWPAPERS, AMERICAN_STORIES}
    marks = group_marks(split_group(), verdict(AS_SPLIT, NEITHER), "p1-1")
    assert {(m.side, m.kind) for m in marks} == {(SNEWPAPERS, WRONG), (AMERICAN_STORIES, WRONG)}


def test_as_merge_marks_each_of_our_pieces_and_the_whole_as_article():
    article_box = (0.0, 0.0, 0.5, 0.9)  # the AS article, wider than our pieces: not clipped
    group = Group(1, AS_MERGE, [piece(uuid="a"), piece(((0.1, 0.6, 0.4, 0.8),), uuid="b")], [part(3, box(1, 3, article_box))])
    marks = group_marks(group, verdict(AS_MERGE, SNEWPAPERS), "p1-1")
    assert [k for k, _ in kinds(marks, SNEWPAPERS)] == [RIGHT, RIGHT]
    assert kinds(marks, AMERICAN_STORIES) == [(WRONG, article_box), ("2 items in 1", article_box)]


def test_items_inside_an_as_merge_get_no_mark_and_the_merged_article_says_how_many_items_it_holds():
    """The Vermont Phoenix 1862-02-20 p2: AS #1, a headline and one body box over seven items of ours."""
    head, body = (0.1, 0.30, 0.3, 0.32), (0.1, 0.32, 0.3, 0.9)
    items = [piece(((0.1, 0.32 + 0.08 * k, 0.3, 0.4 + 0.08 * k),), uuid=f"item{k}") for k in range(7)]
    merge = Group(1, AS_MERGE, items, [part(1, box(18, 1, head), box(0, 1, body))])
    inside = [Group(1, IN_AS_MERGE, [p], [part(1, box(0, 1, p.rects[0], whole=body))], merge=merge) for p in items]
    seg = IssueSegmentation([merge, *inside], {}, [])
    marks = issue_marks(seg, [verdict(AS_MERGE, SNEWPAPERS)] + [verdict(IN_AS_MERGE, SNEWPAPERS, gid=f"p1-{k}")
                                                                for k in range(2, 9)])
    assert {m.gid for m in marks} == {"p1-1"}  # nothing from the items' own groups: no "=" on them
    assert [m.kind for m in marks if m.side == SNEWPAPERS] == [RIGHT] * 7
    assert [(m.kind, m.rect) for m in marks if m.side == AMERICAN_STORIES] == [
        (WRONG, head), (WRONG, body), ("7 items in 1", head), ("7 items in 1", body)]  # AS's real boxes only
    shown, outlines = reduce_marks(marks)
    assert [(m.kind, m.rect) for m in shown if m.side == AMERICAN_STORIES] == [(WRONG, body), ("7 items in 1", body)]
    assert {m.rect for m in outlines if m.side == AMERICAN_STORIES} == {head, body}
    counts = tallies(marks, [])
    assert counts[(AMERICAN_STORIES, 1)][ITEMS_IN_ONE] == (7, 7)
    [(_rect, badges)] = [s for s in _stacks([m for m in shown if m.side == AMERICAN_STORIES], 100, 100)]
    assert badges == [((WRONG, "7 items in 1"), "p1-1")]


def test_ours_only_tags_our_piece():
    marks = group_marks(Group(1, OURS_ONLY, [piece()], []), verdict(OURS_ONLY, NOT_CAPTURED), "p1-1")
    assert kinds(marks, SNEWPAPERS) == [(SNEW_ONLY, PIECE)] and not kinds(marks, AMERICAN_STORIES)


def test_as_only_is_tagged_by_the_classifier_and_tiny_is_not_marked():
    rect = (0.5, 0.5, 0.9, 0.6)
    group = Group(1, AS_ONLY, [], [part(4, box(1, 4, rect))])
    assert kinds(group_marks(group, verdict(AS_ONLY, MISSED), "g"), AMERICAN_STORIES) == [(AS_ONLY_TAG, rect)]
    assert kinds(group_marks(group, verdict(AS_ONLY, EXCLUDED), "g"), AMERICAN_STORIES) == [(NOT_SERVED, rect)]
    assert kinds(group_marks(group, verdict(AS_ONLY, NO_VERDICT), "g"), AMERICAN_STORIES) == [(UNSURE, rect)]
    tiny = Group(1, AS_ONLY_TINY, [], [part(4, box(1, 4, rect, text="x"))])
    assert group_marks(tiny, verdict(AS_ONLY_TINY, TINY), "g") == []


def test_attached_outside_marks_our_piece_and_the_attached_box_by_the_boundary_verdict():
    dateline = (0.1, 0.05, 0.4, 0.09)
    group = Group(1, MATCH, [piece()], [part(7, box(1, 7, PIECE))], as_attached_outside=[part(7, box(2, 7, dateline))])
    marks = group_marks(group, verdict(MATCH, MATCHED, attached=SNEWPAPERS), "p1-1")
    assert kinds(marks, SNEWPAPERS) == [(EQUAL, PIECE), (RIGHT, PIECE)]
    assert kinds(marks, AMERICAN_STORIES) == [(EQUAL, PIECE), (WRONG, dateline)]
    marks = group_marks(group, verdict(MATCH, MATCHED, attached=UNSTABLE), "p1-1")
    assert (UNSURE, dateline) in kinds(marks, AMERICAN_STORIES)


def test_duplicated_as_text_gets_dup_tags_on_the_left_out_boxes_and_on_duplicated_attached_text():
    dropped = box(5, 61, (0.0, 0.4, 0.9, 0.45))  # article #61, its part inside our piece left out as a duplicate of #24
    elsewhere = box(6, 61, (0.6, 0.6, 0.9, 0.9))  # #61 elsewhere on the page, not on our piece
    group = Group(1, MATCH, [piece()], [part(24, box(1, 24, PIECE))], duplicates=[("#24", "#61")])
    marks = group_marks(group, verdict(MATCH, MATCHED), "p1-1", [dropped, elsewhere, box(1, 24, PIECE)])
    assert (DUP, (0.0, 0.4, 0.9, 0.45)) in kinds(marks, AMERICAN_STORIES)  # on the real box
    assert [k for k, _ in kinds(marks, AMERICAN_STORIES)].count(DUP) == 1
    attached = part(7, box(2, 7, (0.1, 0.05, 0.4, 0.09)))
    group = Group(1, MATCH, [piece()], [part(7, box(1, 7, PIECE))], as_attached_outside=[attached],
                  attached_duplicates=[("#9", "#7")])
    assert (DUP, (0.1, 0.05, 0.4, 0.09)) in kinds(group_marks(group, verdict(MATCH, MATCHED, SNEWPAPERS), "p1-1"),
                                                  AMERICAN_STORIES)


def test_issue_marks_numbers_groups_per_page_and_matches_verdicts_by_id():
    seg = IssueSegmentation([Group(1, OURS_ONLY, [piece()], []), split_group(),
                             Group(2, OURS_ONLY, [piece(page=2)], [])], {}, [])
    marks = issue_marks(seg, [verdict(AS_SPLIT, AMERICAN_STORIES, gid="p1-2")])
    assert [(m.gid, m.kind) for m in marks if m.side == SNEWPAPERS] == [("p1-1", SNEW_ONLY), ("p1-2", WRONG), ("p2-1", SNEW_ONLY)]


def test_tallies_count_groups_for_the_other_marks_and_judged_units_with_their_decisions_for_checks_and_crosses():
    marks = [Mark(AMERICAN_STORIES, 1, (0, 0, 0.1, 0.1), EQUAL, "p1-3"), Mark(AMERICAN_STORIES, 1, (0, 0.2, 0.1, 0.3), EQUAL, "p1-3"),
             Mark(AMERICAN_STORIES, 1, (0, 0, 0.1, 0.1), WRONG, "p1-1"), Mark(SNEWPAPERS, 1, (0, 0, 0.1, 0.1), RIGHT, "p1-1"),
             Mark(SNEWPAPERS, 2, (0, 0, 0.1, 0.1), UNSURE, "p2-1"), Mark(SNEWPAPERS, 2, (0, 0.5, 0.1, 0.6), UNSURE, "p2-2")]
    split = [Judgement(AMERICAN_STORIES, 1, f"#{k}", "wrong", ("p1-1", "case")) for k in range(15)]
    split.append(Judgement(SNEWPAPERS, 1, "u1", "right", ("p1-1", "case")))
    merge = [Judgement(SNEWPAPERS, 1, f"u{k}", "right", ("p1-2", "case")) for k in (2, 3)]
    merge.append(Judgement(AMERICAN_STORIES, 1, "#30", "wrong", ("p1-2", "case")))
    assert tallies(marks, split + merge) == {
        (AMERICAN_STORIES, 1): {EQUAL: (1, 1), WRONG: (16, 2)},  # "=" once for its group however many boxes
        (SNEWPAPERS, 1): {RIGHT: (3, 2)},
        (SNEWPAPERS, 2): {UNSURE: (2, 2)},
    }


def test_tally_text_names_the_decisions_only_when_fewer():
    assert tally_text(15, 1) == " 15 (1 decision)"
    assert tally_text(6, 5) == " 6 (5 decisions)"
    assert tally_text(2, 2) == " 2" and tally_text(0, 0) == " 0"


def test_reduce_marks_keeps_one_mark_per_group_kind_and_row_on_its_largest_box_and_outlines_every_judged_box():
    small, big, other = (0.1, 0.1, 0.2, 0.2), (0.3, 0.1, 0.6, 0.5), (0.7, 0.1, 0.8, 0.2)
    marks = [Mark(AMERICAN_STORIES, 1, small, WRONG, "p1-2"), Mark(AMERICAN_STORIES, 1, big, WRONG, "p1-2"),
             Mark(AMERICAN_STORIES, 1, other, WRONG, "p1-2"), Mark(AMERICAN_STORIES, 2, small, WRONG, "p1-2"),
             Mark(SNEWPAPERS, 1, small, EQUAL, "p1-3"), Mark(SNEWPAPERS, 1, big, EQUAL, "p1-3"),
             Mark(SNEWPAPERS, 1, small, SNEW_ONLY, "p1-4"), Mark(SNEWPAPERS, 1, big, SNEW_ONLY, "p1-4")]
    kept, outlines = reduce_marks(marks)
    assert kept == [marks[1], marks[3], marks[5], marks[7]]
    assert outlines == marks[:4]  # crosses, checks and question marks only: "=" and tags keep normal outlines


def test_stacks_put_a_groups_marks_on_one_box_in_one_badge_most_important_first():
    rect = (0.1, 0.1, 0.5, 0.5)
    marks = [Mark(SNEWPAPERS, 1, rect, EQUAL, "p1-2"), Mark(SNEWPAPERS, 1, rect, RIGHT, "p1-2"),
             Mark(SNEWPAPERS, 1, rect, EQUAL, "p1-5")]
    [(box, badges)] = _stacks(marks, 100, 200)
    assert box == (10, 20, 50, 100) and badges == [((RIGHT, EQUAL), "p1-2"), (EQUAL, "p1-5")]


# --- lines --------------------------------------------------------------------------------------


def test_cross_page_links_join_a_components_last_rect_on_a_page_to_its_first_on_the_next():
    comps = [{"uuid": "a", "label": "news", "boxes": [{"page": 2, "rects": [[0.1, 0.1, 0.2, 0.2], [0.8, 0.8, 0.9, 0.95]]},
                                                       {"page": 3, "rects": [[0.05, 0.05, 0.2, 0.1], [0.3, 0.3, 0.4, 0.4]]}]},
             {"uuid": "b", "label": "news", "boxes": [{"page": 1, "rects": [[0, 0, 1, 1]]}]},
             {"uuid": "c", "label": "news"}]
    [link] = cross_page_links(comps)
    assert (link.uuid, link.from_page, link.from_rect, link.to_page, link.to_rect) == \
        ("a", 2, (0.8, 0.8, 0.9, 0.95), 3, (0.05, 0.05, 0.2, 0.1))


def test_cross_page_links_take_the_rects_in_the_order_the_api_gives_them():
    # the API lists a component's rects in reading order: here the text begins in the right column
    right, left = [0.5, 0.1, 0.9, 0.6], [0.1, 0.1, 0.48, 0.9]
    comps = [{"uuid": "a", "label": "news", "boxes": [{"page": 1, "rects": [right, left]},
                                                       {"page": 2, "rects": [right, left]}]}]
    [link] = cross_page_links(comps)
    assert (link.from_rect, link.to_rect) == (tuple(left), tuple(right))


def test_attached_links_run_from_the_attached_box_to_the_nearest_box_of_its_article():
    near, far = box(1, 7, (0.1, 0.1, 0.4, 0.2)), box(3, 7, (0.1, 0.6, 0.4, 0.9))
    dateline = box(2, 7, (0.1, 0.22, 0.4, 0.25))
    group = Group(1, MATCH, [piece()], [part(7, near, far)], as_attached_outside=[part(7, dateline)])
    assert attached_links(group, "p1-1") == [(1, dateline.rect, near.rect, "p1-1")]
    assert attached_links(Group(1, MATCH, [piece()], [part(7, near)]), "p1-1") == []
    # text attached from the rest of a box our piece has a slice of: one real box, no line within it
    column = (0.1, 0.1, 0.4, 0.9)
    group = Group(1, MATCH, [piece()], [part(7, box(1, 7, (0.1, 0.1, 0.4, 0.5), whole=column))],
                  as_attached_outside=[part(7, box(1, 7, (0.1, 0.5, 0.4, 0.9), whole=column))])
    assert attached_links(group, "p1-1") == []


def test_arrow_route_runs_down_the_gutter_between_neighbouring_pages():
    # page 2's column is x 100-200, page 3's 212-312 (gutter 12)
    route = arrow_route((180, 900, 195, 950), (220, 50, 260, 60), (100, 200), (212, 312), detour_y=10, gutter=12)
    assert route == [(195, 925.0), (206.0, 925.0), (206.0, 55.0), (220, 55.0)]


def test_arrow_route_detours_above_the_pages_between_pages_further_apart():
    route = arrow_route((180, 900, 195, 950), (330, 50, 360, 60), (100, 200), (324, 424), detour_y=10, gutter=12)
    assert route == [(195, 925.0), (206.0, 925.0), (206.0, 10), (318.0, 10), (318.0, 55.0), (330, 55.0)]


# --- placing --------------------------------------------------------------------------------------


def _size(badges, compact):
    return (20, 10 * len(badges)) if compact else (60, 30 * len(badges))


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_place_stacks_puts_a_stack_below_the_label_in_its_rect():
    label = (40, 90, 160, 110)
    [[(spot, compact, badges)]] = place_stacks([((0, 0, 200, 300), [(EQUAL, "p1-1")])], _size, [label], (500, 500))
    assert not compact and badges == [(EQUAL, "p1-1")]
    assert spot[1] >= label[3] and not _overlap(spot, label)
    assert 0 <= spot[0] and spot[2] <= 200 and spot[3] <= 300


def test_place_stacks_keeps_stacks_in_the_same_area_apart():
    rect = (0, 0, 200, 300)
    spots = place_stacks([(rect, [(RIGHT, "p1-1")]), ((0, 0, 200, 290), [(EQUAL, "p1-2")])], _size, [], (500, 500))
    assert not _overlap(spots[0][0][0], spots[1][0][0])


def test_place_stacks_puts_a_compact_mark_at_a_tiny_rects_corner_and_drops_one_shown_elsewhere():
    big, tiny, tiny2 = (0, 0, 200, 300), (300, 300, 310, 305), (300, 400, 310, 405)
    spots = place_stacks([(big, [(WRONG, "p1-1")]), (tiny, [(WRONG, "p1-1")]), (tiny2, [(WRONG, "p1-2")])],
                         _size, [], (500, 500))
    assert spots[1] == []  # p1-1's cross is already on the big rect
    [(spot, compact, badges)] = spots[2]
    assert compact and badges == [(WRONG, "p1-2")] and spot[:2] == (301, 401)


def test_place_stacks_moves_a_fallback_mark_off_a_label():
    tiny, label = (300, 300, 310, 305), (295, 295, 340, 315)
    [[(spot, compact, _)]] = place_stacks([(tiny, [(WRONG, "p1-1")])], _size, [label], (500, 500))
    assert compact and not _overlap(spot, label)


def test_place_stacks_splits_a_stack_that_does_not_fit_into_badges_inside_its_rect():
    rect, label = (0, 0, 70, 50), (10, 18, 60, 32)  # room for one compact badge above the label and one below
    [spots] = place_stacks([(rect, [(WRONG, "p1-1"), (EQUAL, "p1-2")])], _size, [label], (500, 500))
    assert [badges for _, _, badges in spots] == [[(WRONG, "p1-1")], [(EQUAL, "p1-2")]]
    for spot, compact, _ in spots:
        assert compact and not _overlap(spot, label) and 0 <= spot[1] and spot[3] <= 50
    assert not _overlap(spots[0][0], spots[1][0])


def test_place_stacks_puts_a_badge_wider_than_its_rect_over_the_rects_own_label_before_going_far():
    rect, own = (100, 100, 115, 120), (101, 105, 114, 115)
    neighbours = [(100, 70, 140, 99), (100, 121, 140, 150)]  # labels just above and below
    [[(spot, compact, _)]] = place_stacks([(rect, [(SNEW_ONLY, "p1-1")])], _size, [own] + neighbours, (500, 500))
    assert compact and _overlap(spot, rect) and not any(_overlap(spot, n) for n in neighbours)


# --- the sheet -------------------------------------------------------------------------------------


def test_sheet_layout_places_page_columns_and_rows():
    layout = SheetLayout((1, 2, 3), 100, 150, header_h=50, label_h=20)
    assert layout.size == (3 * 100 + 4 * 12, 50 + 2 * (20 + 150 + 12) + 12)
    assert layout.origin(0, 1) == (12, 70)
    assert layout.origin(1, 3) == (12 + 2 * 112, 50 + 182 + 20)


def test_scorecard_line_lists_the_non_zero_counts_and_the_cost():
    line = scorecard_line([verdict(MATCH, MATCHED, gid="p1-1"), verdict(AS_ONLY, EXCLUDED, gid="p1-2")])
    assert "our components 1: correct as is 1" in line
    assert "AS only 1: correctly not served 1" in line
    assert "unstable" not in line and "$0.0000" in line and "judged wrong" not in line


def test_scorecard_line_counts_what_was_judged_given_the_segmentation():
    seg = IssueSegmentation([split_group()], {}, [])
    line = scorecard_line([verdict(AS_SPLIT, SNEWPAPERS)], seg)
    assert "judged wrong: 2 AS articles, 0 snewpapers components (right: 0 AS, 1 snewpapers)" in line


def test_scorecard_line_counts_the_as_errors_decisions_for_snewpapers_found():
    merge = Group(1, AS_MERGE, [piece(uuid=f"u{k}") for k in range(7)], [part(1, box(0, 1, PIECE))])
    seg = IssueSegmentation([merge, split_group()], {}, [])
    line = scorecard_line([verdict(AS_MERGE, SNEWPAPERS), verdict(AS_SPLIT, SNEWPAPERS, gid="p1-2")], seg)
    assert "AS errors (decided for snewpapers): 7 items AS merged (1 decision), 2 fragment articles (1 decision)" in line


def test_wrap_fills_lines_greedily_and_indents_the_ones_after_the_first():
    assert wrap([40, 40, 40], [0, 10, 10], 100) == [[0, 1], [2]]
    assert wrap([40, 40, 40, 40], [0, 10, 10, 10], 100, indent=10) == [[0, 1], [2, 3]]
    assert wrap([40, 40, 40, 40], [0, 10, 10, 10], 100, indent=30) == [[0, 1], [2], [3]]
    assert wrap([150, 20], [0, 10], 100) == [[0], [1]]  # too wide for any line: a line of its own


def _page_data():
    return {"scan": {"width": 1000, "height": 1400},
            "bboxes": [{"id": 1, "class": "article", "bbox": {"x0": 100, "y0": 140, "x1": 400, "y1": 700},
                        "raw_text": "Some words of the article here", "reading_order_id": 0}],
            "full articles": [{"full_article_id": 7, "object_ids": [1]}]}


def test_verdict_overlay_draws_both_rows_greys_another_digitisation_and_saves_a_webp(tmp_path):
    from snewpapers_tools.fetching import IssuePair
    from snewpapers_tools.issues import IssueKey

    comps = [{"uuid": "a", "label": "editorial", "content": "Some words of the article here",
              "boxes": [{"page": 1, "rects": [[0.1, 0.1, 0.4, 0.3]]}, {"page": 2, "rects": [[0.1, 0.1, 0.4, 0.3]]}]}]
    pair = IssuePair(IssueKey("i", "sn1", "1850-01-01", 1), comps, [], 0, 1, [])
    scan = Image.new("RGB", (200, 280), (250, 245, 230))
    pages = {1: (scan, _page_data(), (1000, 1400)), 2: (scan, _page_data(), (1000, 1400))}
    seg = IssueSegmentation([Group(1, MATCH, [piece(((0.1, 0.1, 0.4, 0.3),))], [part(7, box(1, 7, (0.1, 0.1, 0.4, 0.5)))]),
                             Group(1, AS_MERGE, [piece(((0.1, 0.1, 0.4, 0.3),), uuid="a"),
                                                 piece(((0.1, 0.3, 0.4, 0.5),), uuid="b")],
                                   [part(7, box(1, 7, (0.1, 0.1, 0.4, 0.5)))])],
                            {}, [], pages_other_scan=[2])
    sheet, path = verdict_overlay(pair, seg, [verdict(MATCH, MATCHED), verdict(AS_MERGE, SNEWPAPERS, gid="p1-2")],
                                  width=200, load_pages=lambda p, w: pages, out_dir=tmp_path)
    assert path == tmp_path / "sn1_1850-01-01_ed-1_verdicts.webp" and path.exists()
    layout_left_2 = 12 + 200 + 12
    grey = sheet.getpixel((layout_left_2 + 5, sheet.height - 20))
    assert grey[0] == grey[1] == grey[2]  # page 2 is greyed out
    assert sheet.width == 2 * 200 + 3 * 12
    # the header wraps to the sheet: nothing drawn in the last column of pixels above the pages
    edge = [sheet.getpixel((sheet.width - 1, y)) for y in range(0, 200)]
    assert all(p == (255, 255, 255) for p in edge)


# --- a mark on every AS article judged, and a component across pages -----------------------------


def test_every_as_article_judged_gets_its_own_mark_on_its_largest_box_and_the_group_id_once_per_page():
    small1, big1 = (0.1, 0.1, 0.4, 0.15), (0.1, 0.15, 0.4, 0.3)
    small2, big2 = (0.1, 0.3, 0.4, 0.32), (0.1, 0.32, 0.4, 0.5)
    only3 = (0.5, 0.1, 0.6, 0.12)
    group = Group(1, AS_SPLIT, [piece()], [part(1, box(1, 1, small1), box(2, 1, big1)),
                                           part(2, box(3, 2, small2), box(4, 2, big2)), part(3, box(5, 3, only3))])
    kept, outlines = reduce_marks(group_marks(group, verdict(AS_SPLIT, SNEWPAPERS), "p1-1"))
    crosses = [m for m in kept if m.side == AMERICAN_STORIES]
    assert [(m.kind, m.unit, m.rect) for m in crosses] == [(WRONG, "#1", big1), (WRONG, "#2", big2), (WRONG, "#3", only3)]
    assert [m.labelled for m in crosses] == [False, True, False]  # the id beside the largest box marked (#2's) only
    [check] = [m for m in kept if m.side == SNEWPAPERS]
    assert (check.kind, check.labelled) == (RIGHT, True)
    assert len([m for m in outlines if m.side == AMERICAN_STORIES]) == 5  # every box outlined
    # one badge each: the labelled one with the id, the others without
    badges = [b for _box, stack in _stacks(crosses, 100, 100) for b in stack]
    assert sorted(badges) == sorted([(WRONG, "p1-1"), (WRONG, ""), (WRONG, "")])


def test_place_stacks_keeps_each_articles_mark_without_an_id_even_when_another_like_it_is_shown():
    big, tiny, tiny2 = (0, 0, 200, 300), (300, 300, 310, 305), (300, 400, 310, 405)
    spots = place_stacks([(big, [(WRONG, "p1-1")]), (tiny, [(WRONG, "")]), (tiny2, [(WRONG, "")])], _size, [], (500, 500))
    assert all(spots) and [b for _, _, b in spots[2]] == [[(WRONG, "")]]


def cross_page_seg():
    """Our component on pages 1-2 over AS #1 and #2 on page 1 and AS #3 on page 2, decided across both."""
    p1, p2 = piece(page=1, uuid="story"), piece(((0.5, 0.1, 0.9, 0.4),), page=2, uuid="story")
    parts = [part(1, box(1, 1, (0.1, 0.1, 0.4, 0.3))), part(2, box(2, 2, (0.1, 0.3, 0.4, 0.5))),
             part(3, box(3, 3, (0.5, 0.1, 0.9, 0.4), page=2))]
    split = Group(1, AS_SPLIT, [p1, p2], parts)
    pointer = Group(2, CONTINUED, [p2], parts[2:], other_pages=[1])
    split.continued, pointer.continues = [pointer], split
    return IssueSegmentation([split, pointer], {}, [])


def test_a_component_across_pages_is_marked_on_every_page_by_its_one_verdict_and_id():
    seg = cross_page_seg()
    verdicts = [verdict(AS_SPLIT, SNEWPAPERS, gid="p1-1"), verdict(CONTINUED, SNEWPAPERS, gid="p2-1")]
    kept, _ = reduce_marks(issue_marks(seg, verdicts))
    assert sorted((m.side, m.page, m.kind, m.unit, m.gid) for m in kept) == sorted([
        (SNEWPAPERS, 1, RIGHT, "story", "p1-1"), (AMERICAN_STORIES, 1, WRONG, "#1", "p1-1"),
        (AMERICAN_STORIES, 1, WRONG, "#2", "p1-1"),
        (SNEWPAPERS, 2, RIGHT, "story", "p1-1"), (AMERICAN_STORIES, 2, WRONG, "#3", "p1-1")])  # labelled p1-1 there too
    assert [m.labelled for m in kept if m.side == AMERICAN_STORIES and m.page == 2] == [True]
    from snewpapers_tools.adjudicate import issue_judgements

    counts = tallies(kept, issue_judgements(seg, verdicts))
    assert counts[(AMERICAN_STORIES, 1)][WRONG] == (2, 1) and counts[(AMERICAN_STORIES, 2)][WRONG] == (1, 1)
    assert counts[(SNEWPAPERS, 1)][RIGHT] == (1, 1) and counts[(SNEWPAPERS, 2)][RIGHT] == (1, 1)


def test_the_cross_page_arrow_runs_from_the_last_box_to_the_first_and_says_where_it_continues_and_comes_from():
    comps = [{"uuid": "a", "label": "news", "boxes": [{"page": 2, "rects": [[0.1, 0.1, 0.2, 0.2], [0.8, 0.8, 0.9, 0.95]]},
                                                       {"page": 3, "rects": [[0.05, 0.05, 0.2, 0.1], [0.3, 0.3, 0.4, 0.4]]}]}]
    [link] = cross_page_links(comps)
    # page 2's column is x 100-200, page 3's 212-312; the rects in sheet pixels
    src, dst = (180, 900, 190, 950), (217, 50, 232, 60)
    route = arrow_route(src, dst, (100, 200), (212, 312), detour_y=10, gutter=12)
    assert route[0] == (190, 925.0) and route[-1] == (217, 55.0)  # out of the last box's right side, into the first's left
    assert link_labels(link, route) == [("continues p. 3", (190, 925.0 - 8), "rb"), ("from p. 2", (219, 55.0 - 8), "lb")]  # lifted ARROW_WIDTH + 2
    # a link running back to a page on the left labels the other sides
    back = [(100.0, 50.0), (94.0, 50.0), (94.0, 900.0), (80.0, 900.0)]
    assert [anchor for _, _, anchor in link_labels(link, back)] == ["lb", "rb"]


# --- nibbles and partial ------------------------------------------------------------------------


def test_a_nibble_marks_our_piece_and_only_the_nibbling_as_article_by_its_verdict():
    title = (0.1, 0.1, 0.4, 0.12)
    group = Group(1, MATCH, [piece()], [part(4, box(1, 4, PIECE))], as_nibbles=[part(17, box(2, 17, title))])
    marks = group_marks(group, GroupVerdict("p1-1", MATCH, MATCHED, nibble_outcome=SNEWPAPERS), "p1-1")
    assert kinds(marks, SNEWPAPERS) == [(EQUAL, PIECE), (RIGHT, PIECE)]
    assert kinds(marks, AMERICAN_STORIES) == [(EQUAL, PIECE), (WRONG, title)]  # #4 stays "="
    assert [m.unit for m in marks if m.kind == WRONG] == ["#17"]


def test_a_partial_tags_our_piece_as_partly_captured_and_marks_no_agreement():
    group = Group(1, AS_PARTIAL, [piece()], [part(7, box(1, 7, (0.1, 0.1, 0.4, 0.12)))])
    marks = group_marks(group, GroupVerdict("p1-1", AS_PARTIAL, "partly captured by AS"), "p1-1")
    assert kinds(marks, SNEWPAPERS) == [(AS_PARTIAL_TAG, PIECE)]
    assert kinds(marks, AMERICAN_STORIES) == [(WRONG, (0.1, 0.1, 0.4, 0.12))]  # AS missing part of it: an AS error
    assert [m.unit for m in marks if m.kind == WRONG] == ["#7"]


def test_marks_and_lines_are_sized_to_read_at_a_glance():
    from snewpapers_tools.verdict_overlay import ARROW_WIDTH, ORDER_LINE_WIDTH, Scale

    assert Scale.for_width(700).icon == 35 and ORDER_LINE_WIDTH == 5 and ARROW_WIDTH == 6


def test_a_verdict_mark_gets_the_groups_id_even_where_its_equal_mark_already_has_it():
    """p2-4: "=" on the editorial's AS box, and a cross on the "PORTLAND." line AS attaches outside it."""
    body, portland = (0.3, 0.5, 0.5, 0.85), (0.32, 0.48, 0.5, 0.5)
    group = Group(1, MATCH, [piece()], [part(2, box(1, 2, body))], as_attached_outside=[part(2, box(2, 2, portland))])
    kept, _ = reduce_marks(group_marks(group, verdict(MATCH, MATCHED, attached=SNEWPAPERS), "p1-1"))
    assert sorted((m.kind, m.labelled) for m in kept if m.side == AMERICAN_STORIES) == [(EQUAL, True), (WRONG, True)]



def test_as_wider_has_no_equal_and_is_marked_by_its_attached_text_question():
    group = Group(1, AS_WIDER, [piece()], [part(13, box(1, 13, (0.1, 0.1, 0.4, 0.5), whole=(0.1, 0.1, 0.4, 0.9)))],
                  as_attached_outside=[part(13, box(1, 13, (0.1, 0.5, 0.4, 0.9), whole=(0.1, 0.1, 0.4, 0.9)))])
    marks = group_marks(group, verdict(AS_WIDER, SNEWPAPERS, attached=SNEWPAPERS), "p1-1")
    assert kinds(marks, SNEWPAPERS) == [(RIGHT, PIECE)]
    assert kinds(marks, AMERICAN_STORIES) == [(WRONG, (0.1, 0.1, 0.4, 0.9))]



def test_a_furniture_verdict_puts_no_check_or_cross():
    from snewpapers_tools.adjudicate import FURNITURE

    dateline = (0.1, 0.05, 0.4, 0.09)
    group = Group(1, MATCH, [piece()], [part(7, box(1, 7, PIECE))], as_attached_outside=[part(7, box(2, 7, dateline))])
    marks = group_marks(group, verdict(MATCH, MATCHED, attached=FURNITURE), "p1-1")
    assert kinds(marks, SNEWPAPERS) == [(EQUAL, PIECE)] and kinds(marks, AMERICAN_STORIES) == [(EQUAL, PIECE)]



def test_an_as_box_marked_illegible_that_we_read_gets_an_illegible_tag_without_an_id():
    from snewpapers_tools.segmentation import IssueSegmentation
    from snewpapers_tools.verdict_overlay import ILLEGIBLE_TAG

    body = (0.49, 0.758, 0.626, 0.834)
    seg = IssueSegmentation([], {}, [], illegible_read=[(1, 0, "#1", ("fair",))])
    marks = issue_marks(seg, [], {1: [AsBox(1, 0, 1, body, "", "article", "Illegible")]})
    assert [(m.side, m.kind, m.rect, m.gid) for m in marks] == [(AMERICAN_STORIES, ILLEGIBLE_TAG, body, "")]


def test_every_illegible_box_on_a_page_keeps_its_tag():
    from snewpapers_tools.segmentation import IssueSegmentation
    from snewpapers_tools.verdict_overlay import ILLEGIBLE_TAG

    a, b = (0.49, 0.758, 0.626, 0.834), (0.8, 0.2, 0.95, 0.3)
    seg = IssueSegmentation([], {}, [], illegible_read=[(1, 0, "#1", ("x",)), (1, 4, "#5", ("y",))])
    kept, _ = reduce_marks(issue_marks(seg, [], {1: [AsBox(1, 0, 1, a, "", "article", "Illegible"),
                                                      AsBox(1, 4, 5, b, "", "article", "Illegible")]}))
    assert sorted(m.rect for m in kept if m.kind == ILLEGIBLE_TAG) == sorted([a, b])



def test_a_component_as_lacks_because_it_marked_it_illegible_is_tagged_as_illegible_not_snew_only_or_partial():
    from snewpapers_tools.verdict_overlay import AS_ILLEGIBLE_TAG

    only = Group(1, OURS_ONLY, [piece()], [], metrics={"illegible_cover": 1.0})
    assert kinds(group_marks(only, verdict(OURS_ONLY, "not captured by AS"), "p1-1"), SNEWPAPERS) == [(AS_ILLEGIBLE_TAG, PIECE)]
    part_ = Group(1, AS_PARTIAL, [piece()], [part(7, box(1, 7, (0.1, 0.1, 0.4, 0.12)))],
                  metrics={"as_cover": 0.1, "illegible_cover": 0.9})
    marks = group_marks(part_, GroupVerdict("p1-1", AS_PARTIAL, "partly captured by AS"), "p1-1")
    assert kinds(marks, SNEWPAPERS) == [(AS_ILLEGIBLE_TAG, PIECE)]
    assert kinds(marks, AMERICAN_STORIES) == [(WRONG, (0.1, 0.1, 0.4, 0.12))]


def test_the_tally_counts_every_illegible_box_not_one_per_page():
    from snewpapers_tools.verdict_overlay import ILLEGIBLE_TAG

    marks = [Mark(AMERICAN_STORIES, 2, (0.1, 0.1 * k, 0.3, 0.1 * k + 0.05), ILLEGIBLE_TAG, "", f"box {k}") for k in range(1, 4)]
    assert tallies(marks, [])[(AMERICAN_STORIES, 2)][ILLEGIBLE_TAG] == (3, 3)



def test_an_as_only_article_with_no_text_is_tagged_no_text():
    from snewpapers_tools.verdict_overlay import NO_TEXT_TAG

    body = (0.344, 0.212, 0.485, 0.257)
    group = Group(3, AS_ONLY_EMPTY, [], [part(34, box(33, 34, body, page=3, text=""))])
    assert kinds(group_marks(group, verdict(AS_ONLY_EMPTY, "no text, not asked"), "p3-7"), AMERICAN_STORIES) == [(NO_TEXT_TAG, body)]


def test_scorecard_line_can_leave_the_llm_usage_out():
    verdicts = [verdict(MATCH, MATCHED, gid="p1-1")]
    assert "questions" in scorecard_line(verdicts)
    assert "questions" not in scorecard_line(verdicts, usage=False) and "$" not in scorecard_line(verdicts, usage=False)
