"""One image per issue: where American Stories and snewpapers agree, and who was judged right where they don't.

The sheet is issue_overlay's (overlay.py): American Stories (AS) on top, snewpapers below, page
columns aligned, each side's boxes filled as there and labelled once per component (per AS article
and class), in its largest box. On top of that, every group of the comparison (segmentation.py)
marks the boxes it is about, with the group's id (adjudicate.group_ids, as in the report):

- a match: "=" on our piece and on the AS boxes of the matched article; an as_wider (AS's article runs
  on beyond our piece into other text) no "=": its attached-text question's check and cross mark it;
- a dispute (as_split, as_merge) and the boundary question of AS text attached outside our
  component: a green check on the side judged right, a red cross on the other; amber "?" on both
  when the verdict was unstable or missing, a red cross on both when it was "neither". An as_merge's
  AS article also gets a tag "N items in 1", N the items of ours it holds; those items' own groups
  (in_as_merge: their AS text a slice of the merged article, cut along our boundaries) get no mark,
  no "=" above all: the merge decides them;
- ours_only: a blue "SNEW ONLY" tag on our piece; as_partial (one AS article, but AS text for under 90%
  of our piece): a blue "AS PARTIAL" tag on our piece and a red cross on the AS article, an AS error
  found without asking (AS has no text for the rest); as_only: an orange "AS ONLY" tag on the AS boxes
  when the classifier said it is content snewpapers does not serve, a grey "NOT SERVED" tag when it
  is an advertisement, terms or furniture (left out by policy); as_only_tiny: nothing;
- a nibble (a line or two of our component AS has in another article, segmentation's
  `Group.as_nibbles`): its question's check, cross or "?" on our piece and on the nibbling AS
  article only, the article holding the rest keeping its "=";
- AS text American Stories holds twice: a small "DUP" tag on the duplicate AS boxes;
- a component of ours across pages (one as_split across them, segmentation's `link_pages`): its
  verdict's marks on every page it is on, those on a later page (its `continued` group) labelled
  with the id of the group on its first page, where it was decided.

On the AS row marks sit only on American Stories' real boxes (a slice's whole box, `AsBox.box_rect`),
never on the slices segmentation cuts them into along our boundaries. A check, cross or question
mark goes on every AS article a group judged (`Mark.unit`), on its largest box; every other kind of
mark a group puts on a row of a page once, on the largest of its boxes there (`reduce_marks`). The
group's id is shown once per row and page for its checks, crosses and question marks and once for
its other marks, beside those on the largest box marked. Every box of a
check, cross or question mark gets a thick outline in its colour, so a story American Stories split
into 15 articles is 15 crosses, one with the id, and 15 red outlines.

Lines: each side's reading order within a page (dashed, heavier than issue_overlay's; on the AS row
through AS's real boxes, in AS's order, as overlay.draw_american_stories_page draws them); a thick
blue arrow on the snewpapers row from a component's last rect on one page to its first on the next,
across the gutter, labelled "continues p. M" at its start and "from p. N" at its end
(`link_labels`); a purple line, a dot at each end, on the AS row from text AS attaches outside our
component to the AS article it belongs to. Above each page and row, its tally (`tallies`):
checks and crosses count what was judged, AS articles on the AS row and snewpapers components on
ours (adjudicate.judgements), with the decisions behind them where fewer; "items in 1" counts the
items of ours in AS merges; "=", "?" and the other tags count groups. Pages AS segmented on another digitisation are greyed out. The header carries the legend
and the scorecard, wrapped to the sheet's width.

Marks are drawn as shapes (no font has to carry the check and cross glyphs), on white halos, a
group's marks on one box side by side before its id, and placed below the box's label, clear of
the labels and of each other; where they do not fit, smaller, one by one, and failing that at the
box's corner or straddling its edge (over its own label if need be) before further out.

The helpers that decide what goes where (`group_marks`, `issue_marks`, `reduce_marks`, `tallies`,
`cross_page_links`, `link_labels`, `attached_links`, `arrow_route`, `place_stacks`, `wrap`) are pure; `verdict_overlay`
loads the scans and draws.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageOps

from snewpapers_tools.adjudicate import (
    usage_text,
    AMERICAN_STORIES,
    COMPONENT_OUTCOMES,
    DISPUTE_OUTCOMES,
    AS_ONLY_OUTCOMES,
    ATTACHED_OUTCOMES,
    AS_DUPLICATED,
    EXCLUDED,
    FRAGMENT_ARTICLES,
    FURNITURE,
    ITEMS_MERGED,
    MISSED,
    NEITHER,
    OUTCOME_LABELS,
    SNEWPAPERS,
    JUDGED_RIGHT,
    JUDGED_WRONG,
    GroupVerdict,
    Judgement,
    group_ids,
    issue_judgements,
    scorecard,
)
from snewpapers_tools.overlay import (
    AS_COLORS,
    FAMILY_COLORS,
    GUTTER,
    american_stories_label_sets,
    queue_labels,
    label_backing,
    snewpapers_label_sets,
    HALO,
    LABEL_INK,
    OVERLAY_DIR,
    SheetLayout,
    _font,
    arrow_head,
    draw_american_stories_page,
    draw_snewpapers_page,
    load_issue_pages,
    two_row_canvas,
)
from snewpapers_tools.segmentation import (
    AS_MERGE,
    IN_AS_MERGE,
    AS_ONLY,
    AS_ONLY_EMPTY,
    AS_SPLIT,
    CONTINUED,
    AS_PARTIAL,
    MATCH,
    OURS_ONLY,
    AsBox,
    AsPart,
    Group,
    IssueSegmentation,
    Rect,
    Region,
    as_boxes,
)

# --- marks ------------------------------------------------------------------------------------

EQUAL = "="  # both sides agree
RIGHT = "right"  # judged right (a check)
WRONG = "wrong"  # judged wrong (a cross)
UNSURE = "?"  # unstable or no verdict
SNEW_ONLY = "SNEW ONLY"
AS_PARTIAL_TAG = "AS PARTIAL"  # one AS article, but AS has text for under 90% of our piece
AS_ILLEGIBLE_TAG = "AS ILLEGIBLE"  # AS lacks text for our piece mostly because it marked the boxes there illegible
AS_ONLY_TAG = "AS ONLY"
NOT_SERVED = "NOT SERVED"
NO_TEXT_TAG = "NO TEXT"  # an AS article outside our components with no text at all
ILLEGIBLE_TAG = "ILLEGIBLE"  # an AS box AS marked illegible (no text) that lies inside our components
DUP = "DUP"
ITEMS_IN_ONE = "items in 1"  # an as_merge's AS article: tagged "7 items in 1" (items_tag); its tally counts the items
ICONS = (EQUAL, RIGHT, WRONG, UNSURE)
TAGS = (SNEW_ONLY, AS_PARTIAL_TAG, AS_ILLEGIBLE_TAG, AS_ONLY_TAG, NOT_SERVED, DUP, ITEMS_IN_ONE, ILLEGIBLE_TAG, NO_TEXT_TAG)
MARK_COLORS = {
    EQUAL: LABEL_INK,
    RIGHT: (0x00, 0x83, 0x00),
    WRONG: (0xE3, 0x49, 0x48),
    UNSURE: (0xED, 0xA1, 0x00),
    SNEW_ONLY: (0x2A, 0x78, 0xD6),
    AS_PARTIAL_TAG: (0x2A, 0x78, 0xD6),
    AS_ILLEGIBLE_TAG: (0x2A, 0x78, 0xD6),
    AS_ONLY_TAG: (0xEB, 0x68, 0x34),
    NOT_SERVED: (0x8A, 0x8A, 0x8A),
    ILLEGIBLE_TAG: (0x5A, 0x5A, 0x5A),
    NO_TEXT_TAG: (0xA8, 0xA8, 0xA8),
    DUP: (0x4A, 0x3A, 0xA7),
    ITEMS_IN_ONE: (0xB0, 0x30, 0x6A),
}
# What each row's tally counts, in order (the icons always, zeros too).
ROW_KINDS = {
    AMERICAN_STORIES: (EQUAL, RIGHT, WRONG, UNSURE, ITEMS_IN_ONE, AS_ONLY_TAG, NOT_SERVED, DUP, ILLEGIBLE_TAG, NO_TEXT_TAG),
    SNEWPAPERS: (EQUAL, RIGHT, WRONG, UNSURE, SNEW_ONLY, AS_PARTIAL_TAG, AS_ILLEGIBLE_TAG),
}
ROWS = (AMERICAN_STORIES, SNEWPAPERS)  # row 0 on top
# A decided outcome -> (our mark, AS's mark); anything else (unstable, no verdict, none) is UNSURE on both.
DECIDED = {SNEWPAPERS: (RIGHT, WRONG), AMERICAN_STORIES: (WRONG, RIGHT), NEITHER: (WRONG, WRONG)}
# Stacks are placed in this order of their most important mark (disputes first, agreement last).
PRIORITY = {RIGHT: 0, WRONG: 0, UNSURE: 0, ITEMS_IN_ONE: 0, SNEW_ONLY: 1, AS_PARTIAL_TAG: 1, AS_ILLEGIBLE_TAG: 1, AS_ONLY_TAG: 1, NOT_SERVED: 2, EQUAL: 3,
            DUP: 4, ILLEGIBLE_TAG: 4, NO_TEXT_TAG: 4}


def items_tag(n: int) -> str:
    """The tag on an AS article holding `n` items of ours: "7 items in 1"."""
    return f"{n} {ITEMS_IN_ONE}"


def base_kind(kind: str) -> str:
    """A mark's kind, an items tag ("7 items in 1") as ITEMS_IN_ONE."""
    return ITEMS_IN_ONE if kind.endswith(ITEMS_IN_ONE) else kind


def items_in(kind: str) -> int:
    """The N of an items tag ("7 items in 1" -> 7)."""
    return int(kind.split()[0])


def _rank(kind: str) -> int:
    return PRIORITY[base_kind(kind)]

OTHER_SCAN_NOTE = "different digitisation - not compared"
ORDER_LINE_WIDTH = 5  # px, the reading-order dashes (issue_overlay's are 1)
ARROW_WIDTH = 6  # px, the cross-page arrow
ATTACHED_WIDTH = 4  # px, the attached-outside line
ATTACHED_INK = (0x4A, 0x3A, 0xA7)
LINK_INK = (0x2A, 0x78, 0xD6)  # the cross-page arrow and its labels
OUTLINED = (WRONG, UNSURE, RIGHT)  # the marks whose boxes get a thick outline in their colour (outermost first)
JUDGED_KIND = {JUDGED_RIGHT: RIGHT, JUDGED_WRONG: WRONG}


@dataclass(frozen=True)
class Mark:
    """One mark: on `side`'s row (SNEWPAPERS or AMERICAN_STORIES), page `page`, inside `rect`
    (page fractions), of `kind` (ICONS or TAGS), for the group `gid`. `unit` is what it judges, an
    AS article's short name or our component's uuid ("" for a mark about the group as a whole);
    `labelled`: whether the group's id is drawn beside it (`reduce_marks` shows it once per row and page)."""

    side: str
    page: int
    rect: Rect
    kind: str
    gid: str
    unit: str = ""
    labelled: bool = True


def _overlap(a: Rect, b: Rect) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _short_name(box: AsBox) -> str:
    return AsPart(box.article_id, (box,)).short_name


def group_marks(group: Group, verdict: GroupVerdict | None, gid: str, page_boxes: list[AsBox] = (),
                case_gid: str | None = None) -> list[Mark]:
    """The marks one group puts on its page (see the module docstring), in order: its case's, the
    boundary question's, then DUP tags. AS marks go on AS's real boxes only (`Group.as_box_rects`),
    never on slices; a check, cross or question mark on each AS article's boxes, the article as its
    `unit`. A cross-page as_split marks its own (first) page; each later page's `continued` group
    marks that page as the as_split would, by its (copied) outcome, labelled `case_gid` (the
    as_split's id). `page_boxes` are the page's whole AS boxes, where the AS parts a piece's
    duplicates were left out of are found (on the piece); without them those parts get no DUP tag."""
    marks: list[Mark] = []
    page, piece = group.page, group.our_rects
    uuid = group.pieces[0].uuid if len({p.uuid for p in group.pieces}) == 1 else ""

    def ours(kind: str, rects, label: str = gid) -> None:
        marks.extend(Mark(SNEWPAPERS, page, r, kind, label, uuid) for r in rects)

    def theirs(kind: str, rects, label: str = gid) -> None:
        marks.extend(Mark(AMERICAN_STORIES, page, r, kind, label) for r in rects)

    def each(kind: str, parts, label: str = gid) -> None:
        marks.extend(Mark(AMERICAN_STORIES, page, r, kind, label, part.short_name) for part in parts for r in part.box_rects)

    outcome = verdict.outcome if verdict is not None else None
    parts = group.parts_on(page)
    if group.case == MATCH:
        ours(EQUAL, piece)
        theirs(EQUAL, group.as_box_rects)
    elif group.case in (AS_SPLIT, AS_MERGE, CONTINUED):
        label = case_gid or gid if group.case == CONTINUED else gid
        mine, other = DECIDED.get(outcome, (UNSURE, UNSURE))
        if group.case == AS_MERGE:
            marks.extend(Mark(SNEWPAPERS, page, r, mine, gid, p.uuid) for p in group.pieces for r in p.rects)
        else:
            ours(mine, piece, label)
        each(other, parts, label)
        if group.case == AS_MERGE:
            theirs(items_tag(len(dict.fromkeys(p.uuid for p in group.pieces))), group.as_box_rects)
    elif group.case == OURS_ONLY:
        ours(AS_ILLEGIBLE_TAG if group.as_illegible else SNEW_ONLY, piece)
    elif group.case == AS_PARTIAL:  # AS has no text for part of it: an AS error, without asking
        ours(AS_ILLEGIBLE_TAG if group.as_illegible else AS_PARTIAL_TAG, piece)
        each(WRONG, parts)
    elif group.case == AS_ONLY:
        theirs({EXCLUDED: NOT_SERVED, MISSED: AS_ONLY_TAG}.get(outcome, UNSURE), group.as_box_rects)
    elif group.case == AS_ONLY_EMPTY:
        theirs(NO_TEXT_TAG, group.as_box_rects)
    # in_as_merge: its merge marks it; as_only_tiny: no mark
    attached = verdict.attached_outcome if verdict is not None else None
    if group.as_attached_outside and attached != FURNITURE:  # page furniture decides nothing: no mark
        mine, other = DECIDED.get(attached, (UNSURE, UNSURE))
        ours(mine, piece)
        each(other, group.as_attached_outside)
    nibbled = verdict.nibble_outcome if verdict is not None else None
    if group.as_nibbles and nibbled != FURNITURE:
        mine, other = DECIDED.get(nibbled, (UNSURE, UNSURE))
        ours(mine, piece)
        each(other, group.as_nibbles)
    for _kept, *dropped in group.duplicates:
        theirs(DUP, [b.box_rect for b in page_boxes
                     if _short_name(b) in dropped and any(_overlap(b.box_rect, r) for r in piece)])
    attached_dups = {name for _other, name in group.attached_duplicates}
    theirs(DUP, [r for part in group.as_attached_outside if part.short_name in attached_dups for r in part.box_rects])
    unique = list(dict.fromkeys(marks))
    return unique


def issue_marks(seg: IssueSegmentation, verdicts: list[GroupVerdict] | None,
                as_boxes_by_page: dict[int, list[AsBox]] | None = None) -> list[Mark]:
    """Every group's marks, in segmentation order; verdicts are matched to groups by id, and a
    continued group's verdict marks are labelled with its cross-page as_split's id."""
    by_id = {v.group_id: v for v in verdicts or []}
    boxes = as_boxes_by_page or {}
    ids = group_ids(seg)
    gid_of = {id(g): gid for g, gid in zip(seg.groups, ids)}
    marks = [m for g, gid in zip(seg.groups, ids)
             for m in group_marks(g, by_id.get(gid), gid, boxes.get(g.page, []),
                                  gid_of.get(id(g.continues)) if g.continues is not None else None)]
    for page, box_id, _article, _ours in seg.illegible_read:  # AS read nothing there; we did
        marks += [Mark(AMERICAN_STORIES, page, b.rect, ILLEGIBLE_TAG, "", f"box {box_id}")
                  for b in boxes.get(page, []) if b.box_id == box_id]
    return marks


def _area(rect: Rect) -> float:
    return (rect[2] - rect[0]) * (rect[3] - rect[1])


def reduce_marks(marks: list[Mark]) -> tuple[list[Mark], list[Mark]]:
    """(the marks to draw, the boxes to outline): of each group's marks of one kind and unit on one
    row of a page only the one on the largest box (the first of equal ones), in input order, the
    group's id shown (`labelled`) only beside the marks on the largest of those boxes, once for its
    checks, crosses and question marks and once for its other marks; and every
    mark of a kind in OUTLINED, once per box and kind, each to be outlined in its kind's colour."""
    best: dict[tuple, Mark] = {}
    for m in marks:
        key = (m.side, m.page, m.gid, m.kind, m.unit)
        if key not in best or _area(m.rect) > _area(best[key].rect):
            best[key] = m
    kept = [m for m in dict.fromkeys(marks) if best[(m.side, m.page, m.gid, m.kind, m.unit)] is m]
    family = lambda m: (m.side, m.page, m.gid, m.kind in OUTLINED)  # noqa: E731  verdict marks apart from the rest
    label_at: dict[tuple, Rect] = {}
    for m in kept:
        if family(m) not in label_at or _area(m.rect) > _area(label_at[family(m)]):
            label_at[family(m)] = m.rect
    kept = [m if m.rect == label_at[family(m)] else replace(m, labelled=False) for m in kept]
    outlines = list({(m.side, m.page, m.rect, m.kind): m for m in marks if m.kind in OUTLINED}.values())
    return kept, outlines


def tallies(marks: list[Mark], judged: list[Judgement]) -> dict[tuple[str, int], dict[str, tuple[int, int]]]:
    """(side, page) -> kind -> (count, decisions). Checks and crosses count what was judged there
    (`judged`, adjudicate.judgements): the AS articles on American Stories' row, our components on
    ours, each once, and the decisions (questions) that judged them. Items tags count, under
    ITEMS_IN_ONE, the items of ours their AS merges hold (7 for "7 items in 1"; count and decisions
    alike). Every other kind counts the groups that put it there, once however many boxes it sits
    on (count and decisions alike); a mark of no group (ILLEGIBLE) counts its box."""
    out: dict[tuple[str, int], dict[str, tuple[int, int]]] = {}
    groups: dict[tuple, set] = {}
    items: dict[tuple, dict[str, int]] = {}
    for m in marks:
        if base_kind(m.kind) == ITEMS_IN_ONE:
            items.setdefault((m.side, m.page), {})[m.gid] = items_in(m.kind)
        elif m.kind not in (RIGHT, WRONG):
            groups.setdefault((m.side, m.page, m.kind), set()).add(m.gid or m.unit)  # a mark of no group: its box
    for (side, page, kind), gids in groups.items():
        out.setdefault((side, page), {})[kind] = (len(gids), len(gids))
    for (side, page), per_group in items.items():
        n = sum(per_group.values())
        out.setdefault((side, page), {})[ITEMS_IN_ONE] = (n, n)
    units: dict[tuple, tuple[set, set]] = {}
    for j in judged:
        found = units.setdefault((j.side, j.page, JUDGED_KIND[j.judged]), (set(), set()))
        found[0].add(j.unit)
        found[1].add(j.decision)
    for (side, page, kind), (found, decisions) in units.items():
        out.setdefault((side, page), {})[kind] = (len(found), len(decisions))
    return out


@dataclass(frozen=True)
class CrossLink:
    """A component running on from page `from_page` (its last rect there) to `to_page` (its first rect there)."""

    uuid: str
    label: str | None
    from_page: int
    from_rect: Rect
    to_page: int
    to_rect: Rect


def cross_page_links(components: list[dict]) -> list[CrossLink]:
    """One link per page break of every component whose `boxes` sit on more than one page, from its last
    rect on one page to its first on the next (the API lists a component's rects in reading order)."""
    links = []
    for comp in components:
        entries = [e for e in comp.get("boxes") or [] if e.get("rects")]
        for a, b in zip(entries, entries[1:]):
            if a["page"] != b["page"]:
                last, first = tuple(a["rects"][-1]), tuple(b["rects"][0])
                links.append(CrossLink(comp["uuid"], comp.get("label"), a["page"], last, b["page"], first))
    return links


def link_labels(link: CrossLink, route: list[tuple[float, float]]) -> list[tuple[str, tuple[float, float], str]]:
    """The cross-page arrow's two labels as (text, point, PIL text anchor), in the route's units:
    "continues p. M" by its start and "from p. N" by its end, each just above the line and on the
    side of the point facing into the component's box (away from the arrow)."""
    (sx, sy), (ax, _ay) = route[0], route[1]
    (bx, _by), (ex, ey) = route[-2], route[-1]
    lift = ARROW_WIDTH + 2
    start_anchor = "rb" if ax >= sx else "lb"  # the arrow leaves rightwards: the label sits to the left, in the box
    end_anchor = "lb" if ex >= bx else "rb"  # it enters rightwards: the label sits to the right, in the box
    return [(f"continues p. {link.to_page}", (sx, sy - lift), start_anchor),
            (f"from p. {link.from_page}", (ex + (2 if end_anchor == "lb" else -2), ey - lift), end_anchor)]


def _nearest_point(point: tuple[float, float], rect: Rect) -> tuple[float, float]:
    return min(max(point[0], rect[0]), rect[2]), min(max(point[1], rect[1]), rect[3])


def _centre(rect: Rect) -> tuple[float, float]:
    return (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2


def attached_links(group: Group, gid: str) -> list[tuple[int, Rect, Rect, str]]:
    """(page, attached box, AS box it is attached to, gid) per attached-outside part: of the real AS
    boxes (never slices) the part's text and its article's text inside our piece are in, the two
    that lie closest together; none when the attached text is in a box our piece has a slice of."""
    links = []
    for part in group.as_attached_outside:
        targets = [r for p in group.parts if p.article_id == part.article_id for r in p.box_rects]
        sources = [r for r in part.box_rects if r not in targets]
        if not targets or not sources:
            continue
        dist = lambda pair: math.dist(_centre(pair[0]), _nearest_point(_centre(pair[0]), pair[1]))  # noqa: E731
        source, target = min(((r, t) for r in sources for t in targets), key=dist)
        links.append((group.page, source, target, gid))
    return links


def arrow_route(src: Rect, dst: Rect, src_column: tuple[float, float], dst_column: tuple[float, float],
                detour_y: float, gutter: float = GUTTER) -> list[tuple[float, float]]:
    """The points of a cross-page arrow, in sheet pixels: out of `src`'s side facing `dst`'s page,
    along to the gutter beside its page, up or down that gutter to `dst`'s height and into `dst`'s
    near side. Pages not side by side are joined through `detour_y` (just above the pages), from
    gutter to gutter. `src_column` and `dst_column` are the pages' (left, right) x."""
    forward = dst_column[0] >= src_column[1]
    sy, ty = (src[1] + src[3]) / 2, (dst[1] + dst[3]) / 2
    if forward:
        start, end = (src[2], sy), (dst[0], ty)
        g1, g2 = src_column[1] + gutter / 2, dst_column[0] - gutter / 2
    else:
        start, end = (src[0], sy), (dst[2], ty)
        g1, g2 = src_column[0] - gutter / 2, dst_column[1] + gutter / 2
    if abs(g1 - g2) < 1:
        return [start, (g1, sy), (g1, ty), end]
    return [start, (g1, sy), (g1, detour_y), (g2, detour_y), (g2, ty), end]


# --- placing marks ----------------------------------------------------------------------------

Box = tuple[float, float, float, float]


def _hits(box: Box, others: list[Box], margin: float = 2) -> bool:
    return any(box[0] < o[2] + margin and o[0] < box[2] + margin and box[1] < o[3] + margin and o[1] < box[3] + margin
               for o in others)


def _positions(size: tuple[float, float], rect: Box, start_y: float) -> list[tuple[float, float]]:
    """Top-left corners for a `size` box inside `rect`: centred, from `start_y` down, then up (and
    against either side when the rect is wide); empty if it does not fit."""
    w, h = size
    if w > rect[2] - rect[0] or h > rect[3] - rect[1]:
        return []
    step = max(6.0, h / 3)
    lo, hi = rect[1] + 1, rect[3] - h - 1
    start = min(max(start_y, lo), hi)
    ys = [start]
    k = 1
    while start + k * step <= hi or start - k * step >= lo:
        ys += [y for y in (start + k * step, start - k * step) if lo <= y <= hi]
        k += 1
    xs = [(rect[0] + rect[2] - w) / 2]
    if rect[2] - rect[0] >= 2.2 * w:
        xs += [rect[0] + 3, rect[2] - w - 3]
    return [(x, y) for y in ys for x in xs]


Badge = tuple[str, str]  # (kind, gid)


def _search(rect: Box, badges: list[Badge], size_of: Callable, start_y: float | None, avoid: list[Box]) -> tuple[Box, bool] | None:
    """The first spot inside `rect` clear of `avoid` for the full stack, else for its compact form."""
    for small in (False, True):
        w, h = size_of(badges, small)
        start = start_y + 3 if start_y is not None else (rect[1] + rect[3] - h) / 2
        for x, y in _positions((w, h), rect, start):
            box = (x, y, x + w, y + h)
            if not _hits(box, avoid):
                return box, small
    return None


def _around(rect: Box, size: tuple[float, float], page: tuple[float, float], rings: int = 4) -> tuple[list[Box], list[Box]]:
    """Spots for a compact stack at a small rect, kept on the page: (near, far). Near: on its top
    edge's corners, inside (left, then right-aligned, spilling out of the other side if wider), then
    straddling its top and bottom edges, then just above and below it. Far: just right of and left of
    it, then all of those a stack's height further out each time (`rings` in all)."""
    w, h = size
    x0, y0, x1, y1 = rect
    near = [(x0 + 1, y0 + 1), (x1 - w - 1, y0 + 1)]
    for x in (x0 + 1, x1 - w - 1):
        near += [(x, y0 - h / 2), (x, y1 - h / 2)]
    near += [(x0, y0 - h - 1), (x0, y1 + 1), (x1 - w, y0 - h - 1), (x1 - w, y1 + 1)]
    far = [(x1 + 1, y0), (x0 - w - 1, y0)]
    for k in range(1, rings):
        step = k * (h + 2)
        far += [(x0, y0 - h - 1 - step), (x0, y1 + 1 + step), (x1 + 1, y0 + step), (x0 - w - 1, y0 + step),
                (x1 - w, y0 - h - 1 - step), (x1 - w, y1 + 1 + step)]

    def on_page(spots):
        boxes = []
        for x, y in spots:
            x, y = min(max(x, 0), page[0] - w), min(max(y, 0), page[1] - h)
            boxes.append((x, y, x + w, y + h))
        return boxes

    return on_page(near), on_page(far)


def _fallback(rect: Box, size: tuple[float, float], page: tuple[float, float], labels: list[Box],
              placed: list[Box]) -> Box:
    """Where a compact badge goes that fits nowhere inside its rect (`_around`'s spots): near the
    rect clear of labels and stacks; else near, over the rect's own label (one centred in it) but
    clear of the others and of stacks; else further out clear of both; else anywhere near clear of
    stacks, then further out; else at the corner."""
    near, far = _around(rect, size, page)
    centre_in = lambda lb: rect[0] <= (lb[0] + lb[2]) / 2 <= rect[2] and rect[1] <= (lb[1] + lb[3]) / 2 <= rect[3]  # noqa: E731
    others = [lb for lb in labels if not centre_in(lb)]
    for candidates, avoid in ((near, labels + placed), (near, others + placed), (far, labels + placed), (near, placed),
                              (far, placed)):
        box = next((b for b in candidates if not _hits(b, avoid)), None)
        if box is not None:
            return box
    return near[0]


def place_stacks(stacks: list[tuple[Box, list[Badge]]], size_of: Callable[[list[Badge], bool], tuple[float, float]],
                 labels: list[Box], page: tuple[float, float]) -> list[list[tuple[Box, bool, list[Badge]]]]:
    """Where to draw each stack of badges (kind, gid), in input order: a list of (box, whether in
    compact form, the badges drawn there), one entry for a stack placed whole, one per badge for a
    stack placed badge by badge, none when nothing of it needs drawing. Rects, labels and `page`
    (width, height) are in page pixels; `size_of(badges, compact)` gives a stack's size.

    First every stack is looked for a spot inside its rect: just below the label there (a label
    whose centre lies in the rect), then further down or up, clear of `labels` and of the stacks
    placed before; failing that, the same for its compact form. The stacks that found none come
    after (most important first, then top to bottom), left with only the badges no other stack on the page shows (a group's mark already shown
    on another of its boxes is not repeated where it does not fit; a badge without an id, gid "",
    is one AS article's own mark and always kept): tried inside again whole, then
    badge by badge, and a badge that still does not fit goes compact at or around the rect's
    top-left (`_fallback`).
    """
    placed: list[Box] = []
    shown: set[Badge] = set()
    out: list[list[tuple[Box, bool, list[Badge]]]] = [[] for _ in stacks]

    def label_bottom(rect: Box) -> float | None:
        return max((lb[3] for lb in labels if rect[0] <= (lb[0] + lb[2]) / 2 <= rect[2]
                    and rect[1] <= (lb[1] + lb[3]) / 2 <= rect[3]), default=None)

    def put(i: int, box: Box, small: bool, badges: list[Badge]) -> None:
        placed.append(box)
        shown.update(badges)
        out[i].append((box, small, badges))

    later = []
    for i, (rect, badges) in enumerate(stacks):
        found = _search(rect, badges, size_of, label_bottom(rect), labels + placed)
        if found:
            put(i, *found, badges)
        else:
            later.append(i)
    # the rest top to bottom, each badge's spot near its rect's top-left leaves the ones below theirs
    later.sort(key=lambda i: (_priority(stacks[i][1][0][0]) if stacks[i][1] else 0, stacks[i][0][1], stacks[i][0][0]))
    for i in later:
        rect, badges = stacks[i]
        left = [b for b in badges if not b[1] or b not in shown]  # an unlabelled mark is one article's own
        if not left:
            continue
        found = _search(rect, left, size_of, label_bottom(rect), labels + placed)
        if found is not None:
            put(i, *found, left)
            continue
        for badge in left:
            found = _search(rect, [badge], size_of, label_bottom(rect), labels + placed)
            if found is None:
                found = (_fallback(rect, size_of([badge], True), page, labels, placed), True)
            put(i, *found, [badge])
    return out


# --- drawing ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scale:
    """Mark sizes in px for pages `width` px wide."""

    icon: int
    gid_font: int
    tag_font: int

    @classmethod
    def for_width(cls, width: int) -> "Scale":
        return cls(max(22, round(width * 0.05)), max(10, round(width * 0.019)), max(10, round(width * 0.018)))

    def compact(self) -> "Scale":
        return Scale(max(11, round(self.icon * 0.62)), max(9, round(self.gid_font * 0.8)), max(8, round(self.tag_font * 0.75)))


def draw_icon(draw: ImageDraw.ImageDraw, kind: str, box: Box) -> None:
    """An icon mark in `box` (a square): a disc on a white halo with its shape on it. "=" is the
    quiet one (white disc, dark ring and bars); the verdicts are filled discs (white check, white
    cross, dark question mark on amber)."""
    x0, y0, x1, y1 = box
    d = x1 - x0
    color = MARK_COLORS[kind]
    halo = max(2, round(d / 12))
    draw.ellipse(box, fill=HALO)
    inner = (x0 + halo, y0 + halo, x1 - halo, y1 - halo)
    stroke = max(2, round(d / 7))
    if kind == EQUAL:
        draw.ellipse(inner, fill=HALO, outline=color, width=max(2, round(d / 12)))
    else:
        draw.ellipse(inner, fill=color)
    u = lambda fx, fy: (x0 + fx * d, y0 + fy * d)  # noqa: E731
    if kind == EQUAL:
        for fy in (0.40, 0.60):
            draw.line([u(0.30, fy), u(0.70, fy)], fill=color, width=max(2, round(d / 9)))
    elif kind == RIGHT:
        draw.line([u(0.28, 0.52), u(0.44, 0.68), u(0.73, 0.34)], fill=HALO, width=stroke, joint="curve")
    elif kind == WRONG:
        draw.line([u(0.33, 0.33), u(0.67, 0.67)], fill=HALO, width=stroke)
        draw.line([u(0.67, 0.33), u(0.33, 0.67)], fill=HALO, width=stroke)
    elif kind == UNSURE:
        font = _font(max(9, round(d * 0.72)))
        draw.text(u(0.5, 0.53), "?", fill=LABEL_INK, font=font, anchor="mm", stroke_width=1, stroke_fill=LABEL_INK)


def _tag_size(kind: str, font_px: int) -> tuple[int, int]:
    font = _font(font_px)
    left, top, right, bottom = font.getbbox(kind, anchor="lm")
    return round(right - left) + 2 * max(4, font_px // 3) + 4, round(bottom - top) + 2 * max(2, font_px // 4) + 4


def draw_tag(draw: ImageDraw.ImageDraw, kind: str, at: tuple[float, float], font_px: int) -> None:
    """A tag mark with its top-left at `at`: the word in white on a rounded rect of its
    colour inside a 2 px white halo (sized by `_tag_size`)."""
    w, h = _tag_size(kind, font_px)
    x, y = at
    draw.rounded_rectangle((x, y, x + w, y + h), radius=max(3, h // 4), fill=HALO)
    draw.rounded_rectangle((x + 2, y + 2, x + w - 2, y + h - 2), radius=max(2, h // 4 - 1), fill=MARK_COLORS[base_kind(kind)])
    draw.text((x + w / 2, y + h / 2), kind, fill=HALO, font=_font(font_px), anchor="mm")


def _symbol_size(kind: str, scale: Scale) -> tuple[int, int]:
    if kind in ICONS:
        return scale.icon, scale.icon
    return _tag_size(kind, scale.tag_font if kind != DUP else max(8, round(scale.tag_font * 0.85)))


def _draw_symbol(draw: ImageDraw.ImageDraw, kind: str, at: tuple[float, float], scale: Scale) -> None:
    if kind in ICONS:
        draw_icon(draw, kind, (at[0], at[1], at[0] + scale.icon, at[1] + scale.icon))
    else:
        draw_tag(draw, kind, at, scale.tag_font if kind != DUP else max(8, round(scale.tag_font * 0.85)))


def _gid_size(gid: str, font_px: int) -> tuple[int, int]:
    left, top, right, bottom = _font(font_px).getbbox(gid, anchor="lm")
    return round(right - left) + 6, round(bottom - top) + 4


def _kinds(kind: str | tuple[str, ...]) -> tuple[str, ...]:
    return kind if isinstance(kind, tuple) else (kind,)


def _symbols_size(kind: str | tuple[str, ...], scale: Scale) -> tuple[int, int]:
    """One mark, or several of one group side by side (2 px apart)."""
    sizes = [_symbol_size(k, scale) for k in _kinds(kind)]
    return sum(w for w, _ in sizes) + 2 * (len(sizes) - 1), max(h for _, h in sizes)


def badge_size(kind: str | tuple[str, ...], gid: str, scale: Scale) -> tuple[int, int]:
    """A mark (or a group's marks, `kind` a tuple) with its id to its right (none for gid ""): (width, height)."""
    sw, sh = _symbols_size(kind, scale)
    if not gid:
        return sw, sh
    gw, gh = _gid_size(gid, scale.gid_font)
    return sw + 2 + gw, max(sh, gh)


def draw_badge(draw: ImageDraw.ImageDraw, kind: str | tuple[str, ...], gid: str, at: tuple[float, float],
               scale: Scale) -> None:
    """The mark (or marks, side by side), then its group id in small dark type on a white backing,
    centred on the marks' height."""
    sw, sh = _symbols_size(kind, scale)
    w, h = badge_size(kind, gid, scale)
    x, y = at
    for k in _kinds(kind):
        kw, kh = _symbol_size(k, scale)
        _draw_symbol(draw, k, (x, y + (h - kh) / 2), scale)
        x += kw + 2
    if not gid:
        return
    gw, gh = _gid_size(gid, scale.gid_font)
    gx, gy = at[0] + sw + 2, y + (h - gh) / 2
    draw.rounded_rectangle((gx, gy, gx + gw, gy + gh), radius=3, fill=HALO)
    draw.text((gx + 3, gy + gh / 2), gid, fill=LABEL_INK, font=_font(scale.gid_font), anchor="lm")


def stack_size(badges: list[tuple[str, str]], scale: Scale) -> tuple[int, int]:
    sizes = [badge_size(k, g, scale) for k, g in badges]
    return max(w for w, _ in sizes), sum(h for _, h in sizes) + 2 * (len(sizes) - 1)


def draw_stack(draw: ImageDraw.ImageDraw, badges: list[tuple[str, str]], box: Box, scale: Scale) -> None:
    """The badges one under another, each centred in `box`'s width."""
    y = box[1]
    for kind, gid in badges:
        w, h = badge_size(kind, gid, scale)
        draw_badge(draw, kind, gid, (box[0] + (box[2] - box[0] - w) / 2, y), scale)
        y += h + 2


def _priority(kind: str | tuple[str, ...]) -> int:
    return min(_rank(k) for k in _kinds(kind))


def _stacks(marks: list[Mark], width: int, height: int) -> list[tuple[Box, list[tuple]]]:
    """The marks on one row of one page as stacks, one per rect (in page pixels), a badge per group
    (its marks there side by side, most important first, then its id once; marks not `labelled`
    together in a badge without an id, gid ""), in placing order: most
    important mark first (PRIORITY), then smaller rects first (they have fewer places to go)."""
    by_rect: dict[Rect, dict[str, list[str]]] = {}
    for m in marks:
        kinds = by_rect.setdefault(m.rect, {}).setdefault(m.gid if m.labelled else "", [])
        if m.kind not in kinds:
            kinds.append(m.kind)
    stacks = []
    for rect, groups in by_rect.items():
        badges = [(tuple(sorted(kinds, key=_rank)) if len(kinds) > 1 else kinds[0], gid)
                  for gid, kinds in groups.items()]
        badges.sort(key=lambda b: _priority(b[0]))
        px = (rect[0] * width, rect[1] * height, rect[2] * width, rect[3] * height)
        stacks.append((px, badges))
    stacks.sort(key=lambda s: (_priority(s[1][0][0]), (s[0][2] - s[0][0]) * (s[0][3] - s[0][1])))
    return stacks


def _grey_out(image: Image.Image, font_px: int) -> Image.Image:
    """The page washed out to pale grey, with OTHER_SCAN_NOTE across its middle."""
    grey = ImageOps.grayscale(image).convert("RGB")
    out = Image.blend(grey, Image.new("RGB", image.size, (235, 235, 235)), 0.6)
    draw = ImageDraw.Draw(out)
    font = _font(font_px)
    cx, cy = image.width / 2, image.height / 2
    left, top, right, bottom = font.getbbox(OTHER_SCAN_NOTE, anchor="mm")
    draw.rounded_rectangle((cx + left - 10, cy + top - 8, cx + right + 10, cy + bottom + 8), radius=6, fill=HALO,
                           outline=MARK_COLORS[NOT_SERVED], width=2)
    draw.text((cx, cy), OTHER_SCAN_NOTE, fill=LABEL_INK, font=font, anchor="mm")
    return out


def _polyline(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: tuple, width: int, *,
              head: bool, end_dot: bool = False) -> None:
    """A solid line through `points` on a white halo, with a dot at its start, and at its end an
    arrow head if `head`, else a dot if `end_dot`."""
    (ax, ay), (bx, by) = points[-2], points[-1]
    length = math.dist((ax, ay), (bx, by)) or 1
    ux, uy = (bx - ax) / length, (by - ay) / length
    r = width + 1
    dots = [points[0]] + ([points[-1]] if end_dot and not head else [])
    for fill, w, grow in ((HALO, width + 4, 2), (color, width, 0)):
        draw.line(points, fill=fill, width=w, joint="curve")
        for sx, sy in dots:
            draw.ellipse((sx - r - grow, sy - r - grow, sx + r + grow, sy + r + grow), fill=fill)
        if head:
            draw.polygon(arrow_head(points[-1], ux, uy, width), fill=fill, outline=fill, width=1 + grow)


def _tally_items(counts: dict[str, tuple[int, int]], kinds: tuple[str, ...]) -> list[tuple[str, int, int]]:
    return [(k, *counts.get(k, (0, 0))) for k in kinds]


def tally_text(count: int, decisions: int) -> str:
    """A tally's number: " 15", or " 15 (1 decision)" when fewer decisions are behind it."""
    if count == decisions:
        return f" {count}"
    return f" {count} ({decisions} decision{'' if decisions == 1 else 's'})"


def _draw_tally(draw: ImageDraw.ImageDraw, at: tuple[float, float], page: int,
                items: list[tuple[str, int, int]] | None, max_width: float, font_px: int) -> None:
    """"page N" and the tally, each kind's mark (small, no id) then its count, " · " between; shrunk to fit."""
    for size in range(font_px, 8, -1):
        scale = Scale(size + 4, size, size - 2)
        font = _font(size)
        parts: list[tuple[str, object]] = [("text", f"page {page}   ")]
        if items is None:
            parts.append(("text", OTHER_SCAN_NOTE))
        for i, (kind, n, decisions) in enumerate(items or []):
            if i:
                parts.append(("text", " · "))
            parts += [("mark", kind), ("text", tally_text(n, decisions))]
        widths = [draw.textlength(v, font=font) if t == "text" else _symbol_size(v, scale)[0] for t, v in parts]
        if sum(widths) <= max_width or size == 9:
            break
    x, y = at
    h = max(scale.icon, _tag_size(SNEW_ONLY, scale.tag_font)[1])
    for (t, v), w in zip(parts, widths):
        if t == "text":
            draw.text((x, y + h / 2), v, fill=(0, 0, 0), font=font, anchor="lm")
        else:
            sh = _symbol_size(v, scale)[1]
            _draw_symbol(draw, v, (x, y + (h - sh) / 2), scale)
        x += w


def _n(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def scorecard_sections(verdicts: list[GroupVerdict], seg: IssueSegmentation | None = None, *,
                       usage: bool = True) -> list[str]:
    """The issue's scorecard (adjudicate.scorecard) as short sections, zero counts left out: the
    decisions, then (given the segmentation) what they judged, then (unless `usage` is false, as for
    published results) the questions and their tokens and cost."""
    card = scorecard(verdicts, seg)

    def section(name: str, counts: dict, outcomes: tuple) -> str:
        found = [f"{OUTCOME_LABELS.get(o, o)} {counts[o]}" for o in outcomes if counts.get(o)]
        return f"{name} {sum(counts.values())}: {', '.join(found)}" if found else ""

    judged, errors = card["judged"], card.get("as_errors")
    sections = [
        section("our components", card["components"], COMPONENT_OUTCOMES),
        section("N ours : 1 AS", card["merges"], DISPUTE_OUTCOMES),
        section("AS only", card["as_only"], AS_ONLY_OUTCOMES),
        section("attached outside", card["attached_outside"], ATTACHED_OUTCOMES),
        section("lines in another AS article", card["nibbles"], ATTACHED_OUTCOMES),
        f"AS duplicated text {card['as_defects'][AS_DUPLICATED]}" if card["as_defects"][AS_DUPLICATED] else "",
        (f"judged wrong: {_n(judged['as_articles']['wrong'], 'AS article')}, "
         f"{_n(judged['components']['wrong'], 'snewpapers component')} (right: {judged['as_articles']['right']} AS, "
         f"{judged['components']['right']} snewpapers)" if judged else ""),
        (f"AS errors (decided for snewpapers): {_n(errors[ITEMS_MERGED], 'item')} AS merged "
         f"({_n(errors['merge_decisions'], 'decision')}), {_n(errors[FRAGMENT_ARTICLES], 'fragment article')} "
         f"({_n(errors['split_decisions'], 'decision')})" if errors else ""),
        f"{card['questions']} questions ({card['cached']} cached), {usage_text(card)}" if usage else "",
    ]
    return [s for s in sections if s]


def scorecard_line(verdicts: list[GroupVerdict], seg: IssueSegmentation | None = None, *, usage: bool = True) -> str:
    """The issue's scorecard in one line (`scorecard_sections`)."""
    return "Scorecard  ·  " + "  ·  ".join(scorecard_sections(verdicts, seg, usage=usage))


LEGEND_MARKS = [
    (EQUAL, "same article on both sides"),
    (RIGHT, "judged right"),
    (WRONG, "judged wrong"),
    (items_tag(7), "AS has 7 of our items as 1 article"),
    (UNSURE, "unstable / no verdict"),
    (SNEW_ONLY, "AS has no box here"),
    (AS_PARTIAL_TAG, "AS has text for under 90% of it"),
    (AS_ILLEGIBLE_TAG, "AS lacks its text because it marked it illegible"),
    (AS_ONLY_TAG, "AS text snewpapers does not serve"),
    (NOT_SERVED, "ad, terms or furniture: left out by policy"),
    (DUP, "AS has this text twice"),
    (ILLEGIBLE_TAG, "AS marked it illegible and read nothing; snewpapers read it"),
    (NO_TEXT_TAG, "AS article with no text, outside our components"),
]


# --- the header: units of the legend flowed into lines ------------------------------------------


@dataclass(frozen=True)
class Unit:
    """One piece of the header: `width` x `height` px, drawn by `paint(draw, x, y_middle)`, `gap` px
    after the piece before it on its line."""

    width: float
    height: float
    paint: Callable
    gap: float = 22


def wrap(widths: list[float], gaps: list[float], max_width: float, indent: float = 0) -> list[list[int]]:
    """The items on each line, filled greedily: item i takes `widths[i]` after `gaps[i]` (none at a
    line's start); lines after the first start `indent` in. An item wider than a line has one of its own."""
    lines: list[list[int]] = []
    x = 0.0
    for i, w in enumerate(widths):
        if lines and x + gaps[i] + w <= max_width:
            lines[-1].append(i)
            x += gaps[i] + w
        else:
            lines.append([i])
            x = (indent if len(lines) > 1 else 0) + w
    return lines


def _text_unit(text: str, font, gap: float = 22, fill=(0, 0, 0)) -> Unit:
    return Unit(font.getlength(text), font.size + 6,
                lambda d, x, y: d.text((x, y), text, fill=fill, font=font, anchor="lm"), gap)


def _text_units(text: str, font, max_width: float, gap: float = 22) -> list[Unit]:
    """`text` as one unit, or word by word when it is wider than `max_width`."""
    if font.getlength(text) <= max_width:
        return [_text_unit(text, font, gap)]
    words = text.split()
    space = font.getlength(" ")
    return [_text_unit(w, font, gap if k == 0 else space) for k, w in enumerate(words)]


def _symbol_unit(kind: str, text: str, scale: Scale, font, gap: float = 22) -> Unit:
    sw, sh = _symbol_size(kind, scale)

    def paint(d, x, y):
        _draw_symbol(d, kind, (x, y - sh / 2), scale)
        d.text((x + sw + 6, y), text, fill=(0, 0, 0), font=font, anchor="lm")

    return Unit(sw + 6 + font.getlength(text), max(sh, font.size + 6), paint, gap)


def _drawn_unit(sample_w: float, sample, text: str, font, gap: float = 22) -> Unit:
    """A sample `sample_w` px wide drawn by `sample(draw, x, y_middle)`, then `text`."""
    def paint(d, x, y):
        sample(d, x, y)
        d.text((x + sample_w + 6, y), text, fill=(0, 0, 0), font=font, anchor="lm")

    return Unit(sample_w + 6 + font.getlength(text), font.size + 6, paint, gap)


def _header_sections(title: str, score: list[str], scale: Scale, max_width: float) -> list[list[Unit]]:
    """The header, one list of units per section (each starts a line): the title, the scorecard,
    the marks, the tallies, the lines, and the fills of each row."""
    big, small = _font(24), _font(17)
    space = big.getlength(" ")
    title_units = [_text_unit(w, big, space) for w in title.split()]
    score_units = [_text_unit("Scorecard:", small)]
    for k, text in enumerate(score):
        score_units += _text_units(text + (" ·" if k < len(score) - 1 else ""), small, max_width, 12)

    marks = [_text_unit("Marks:", small)] + [_symbol_unit(kind, text, scale, small) for kind, text in LEGEND_MARKS]
    gw, gh = _gid_size("p3-14", scale.gid_font)

    def gid_sample(d, x, y):
        d.rounded_rectangle((x, y - gh / 2, x + gw, y + gh / 2), radius=3, fill=HALO, outline=(200, 200, 200))
        d.text((x + 3, y), "p3-14", fill=LABEL_INK, font=_font(scale.gid_font), anchor="lm")

    marks.append(_drawn_unit(gw, gid_sample, "the group's id in the report (page 3, 14th group)", small))

    def outline_sample(d, x, y):
        d.rectangle((x, y - 9, x + 30, y + 9), fill=(214, 228, 245), outline=MARK_COLORS[WRONG], width=4)

    marks.append(_drawn_unit(30, outline_sample, "thick outline: a box of the marked group (a check, cross or ? "
                             "on every AS article judged, the other marks on the group's largest box; its id "
                             "once per page)", small))

    tally = [_text_unit("Tallies over each page:", small)]
    pair = lambda d, x, y: (_draw_symbol(d, RIGHT, (x, y - scale.icon / 2), scale),  # noqa: E731
                            _draw_symbol(d, WRONG, (x + scale.icon + 4, y - scale.icon / 2), scale))
    tally.append(_drawn_unit(2 * scale.icon + 4, pair, "count the AS articles (row 1) and snewpapers components "
                             "(row 2) judged so,", small))
    tally += _text_units("with the decisions behind them when fewer, e.g. 15 (1 decision);", small, max_width, 8)
    tally += _text_units("the other marks count groups", small, max_width, 8)

    def dashes(d, x, y):
        for k in range(3):
            d.line([(x + 14 * k, y), (x + 14 * k + 9, y)], fill=(214, 39, 40), width=ORDER_LINE_WIDTH)

    lines = [_text_unit("Lines:", small),
             _drawn_unit(38, dashes, "reading order within a page (each article / component)", small),
             _drawn_unit(44, lambda d, x, y: _polyline(d, [(x + 2, y), (x + 42, y)], LINK_INK, ARROW_WIDTH, head=True),
                         "snewpapers component continued on another page (\"continues p. 3\" ... \"from p. 2\"): "
                         "one comparison across its pages, decided once", small),
             _drawn_unit(44, lambda d, x, y: _polyline(d, [(x + 4, y), (x + 40, y)], ATTACHED_INK, ATTACHED_WIDTH,
                                                       head=False, end_dot=True),
                         "AS text outside our component, to the AS article it is attached to", small),
             _drawn_unit(26, lambda d, x, y: d.rectangle((x, y - 8, x + 26, y + 8), fill=(215, 215, 215),
                                                         outline=(150, 150, 150)),
                         f"page greyed out: {OTHER_SCAN_NOTE}", small)]

    def swatches(heading: str, colors: dict) -> list[Unit]:
        units = [_text_unit(heading, small)]
        for name, color in colors.items():
            units.append(_drawn_unit(14, lambda d, x, y, c=color: d.rectangle((x, y - 7, x + 14, y + 7), outline=c, width=3),
                                     name, small, 16))
        return units

    return [title_units, score_units, marks, tally, lines,
            swatches("Row 1, American Stories (dashes: one colour per article):", AS_COLORS),
            swatches("Row 2, snewpapers:", FAMILY_COLORS)]


def flow(sections: list[list[Unit]], left: float, right: float, top: float, *, indent: float = 24,
         spacing: float = 8) -> tuple[list[tuple[Unit, float, float]], float]:
    """(each unit with the x and middle y it is drawn at, the y below the last line): every section
    on lines of its own, wrapped between `left` and `right`."""
    placed = []
    y = top
    for units in sections:
        for k, line in enumerate(wrap([u.width for u in units], [u.gap for u in units], right - left, indent)):
            h = max(units[i].height for i in line)
            x = left + (indent if k else 0)
            for n, i in enumerate(line):
                x += units[i].gap if n else 0
                placed.append((units[i], x, y + h / 2))
                x += units[i].width
            y += h + spacing
    return placed, y


def verdict_overlay(pair, seg: IssueSegmentation, verdicts: list[GroupVerdict] | None, *, width: int = 700,
                    load_pages: Callable = load_issue_pages, out_dir: Path = OVERLAY_DIR,
                    quality: int = 80, usage: bool = True) -> tuple[Image.Image, Path]:
    """Draw the verdict overlay (see the module docstring) and save it as
    `out_dir/{lccn}_{date}_ed-{n}_verdicts.webp`; returns (the sheet, its path).

    `load_pages(pair, width)` gives page -> (scan image, AS page data, real scan size), by default
    the issue's JP2s (downloaded once into data/loc).
    """
    issue = pair.issue
    loaded = load_pages(pair, width)
    if not loaded:
        raise ValueError("No page scans could be located for this issue (no LoC link on its components).")
    other_scan = set(seg.pages_other_scan)
    boxes = {page: as_boxes(data, page, size) for page, (_img, data, size) in loaded.items() if page not in other_scan}
    marks = issue_marks(seg, verdicts, boxes)
    counts = tallies(marks, issue_judgements(seg, verdicts or []))
    shown, outlines = reduce_marks(marks)
    scale = Scale.for_width(width)
    font_px = max(11, width // 45)

    top, bottom = {}, {}
    for page, (image, data, size) in loaded.items():
        if page in other_scan:
            top[page] = bottom[page] = _grey_out(image, font_px + 2)
            continue
        top[page] = draw_american_stories_page(image, data, scan_size=size, line_width=ORDER_LINE_WIDTH, label_once=True)
        bottom[page] = draw_snewpapers_page(image, pair.components, page, line_width=ORDER_LINE_WIDTH, label_once=True)

    title = (f"{issue.lccn} {issue.date} ed-{issue.edition}: where American Stories and snewpapers agree, "
             "and who was judged right where they differ")
    sheet_w = SheetLayout.fit(top, bottom, header_h=0).size[0]
    sections = _header_sections(title, scorecard_sections(verdicts or [], seg, usage=usage), scale, sheet_w - 2 * GUTTER)
    header, header_bottom = flow(sections, GUTTER, sheet_w - GUTTER, 10)
    layout = SheetLayout.fit(top, bottom, header_h=round(header_bottom) + 6, label_h=max(30, scale.icon + 8))
    sheet = two_row_canvas(top, bottom, layout)
    draw = ImageDraw.Draw(sheet)
    for unit, x, y in header:
        unit.paint(draw, x, y)

    for row, side in enumerate(ROWS):
        for page in layout.numbers:
            x, y = layout.left(page), layout.caption_top(row) + 3
            items = None if page in other_scan else _tally_items(counts.get((side, page), {}), ROW_KINDS[side])
            _draw_tally(draw, (x, y), page, items, layout.cell_w, font_px)

    def to_sheet(row: int, page: int, rect: Rect) -> Box:
        ox, oy = layout.origin(row, page)
        w, h = (top if row == 0 else bottom)[page].size
        return ox + rect[0] * w, oy + rect[1] * h, ox + rect[2] * w, oy + rect[3] * h

    # outlines, then lines, then marks on top
    outline_w = max(3, round(width / 175))
    kinds_at: dict[tuple, list[str]] = {}
    for m in outlines:
        kinds_at.setdefault((m.side, m.page, m.rect), []).append(m.kind)
    for (side, page, rect), kinds in kinds_at.items():
        if page in other_scan or page not in (top if side == AMERICAN_STORIES else bottom):
            continue
        x0, y0, x1, y1 = to_sheet(ROWS.index(side), page, rect)
        for k, kind in enumerate(sorted(kinds, key=OUTLINED.index)):  # nested when a box has several
            inset = k * outline_w
            if x1 - x0 > 2 * inset and y1 - y0 > 2 * inset:
                draw.rectangle((x0 + inset, y0 + inset, x1 - inset, y1 - inset), outline=MARK_COLORS[kind], width=outline_w)

    for group, gid in zip(seg.groups, group_ids(seg)):
        for page, source, target, _gid in attached_links(group, gid):
            if page in top and page not in other_scan:
                s, t = to_sheet(0, page, source), to_sheet(0, page, target)
                a = _centre(s)
                _polyline(draw, [a, _nearest_point(a, t)], ATTACHED_INK, ATTACHED_WIDTH, head=False, end_dot=True)
    link_font = _font(font_px)
    link_boxes: dict[int, list[Box]] = {}  # page -> its link labels on our row, in page pixels (marks keep clear)
    for link in cross_page_links(pair.components):
        if link.from_page in bottom and link.to_page in bottom and not {link.from_page, link.to_page} & other_scan:
            columns = [(layout.left(p), layout.left(p) + layout.cell_w) for p in (link.from_page, link.to_page)]
            route = arrow_route(to_sheet(1, link.from_page, link.from_rect), to_sheet(1, link.to_page, link.to_rect),
                                columns[0], columns[1], layout.origin(1, link.to_page)[1] - 4)
            _polyline(draw, route, LINK_INK, ARROW_WIDTH, head=True)
            for (text, at, anchor), page in zip(link_labels(link, route), (link.from_page, link.to_page)):
                box = _draw_link_label(draw, text, at, anchor, link_font)
                ox, oy = layout.origin(1, page)
                link_boxes.setdefault(page, []).append((box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy))

    for row, side in enumerate(ROWS):
        for page in layout.numbers:
            if page in other_scan or page not in (top if row == 0 else bottom):
                continue
            image = (top if row == 0 else bottom)[page]
            w, h = image.size
            stacks = _stacks([m for m in shown if m.side == side and m.page == page], w, h)
            labels = _page_labels(pair, loaded, side, page, w, h) + (link_boxes.get(page, []) if row == 1 else [])
            size_of = lambda badges, small: stack_size(badges, scale.compact() if small else scale)  # noqa: E731
            ox, oy = layout.origin(row, page)
            for spots in place_stacks(stacks, size_of, labels, (w, h)):
                for box, small, badges in spots:
                    draw_stack(draw, badges, (box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy),
                               scale.compact() if small else scale)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{issue.lccn}_{issue.date}_ed-{issue.edition}_verdicts.webp"
    sheet.save(path, "WEBP", quality=quality)
    return sheet, path


def _draw_link_label(draw: ImageDraw.ImageDraw, text: str, at: tuple[float, float], anchor: str, font) -> Box:
    """A cross-page arrow's label: blue bold-ish text on a white rounded backing with a blue border,
    placed by `anchor` at `at`; returns its backing box."""
    left, top, right, bottom = draw.textbbox(at, text, font=font, anchor=anchor)
    box = (left - 5, top - 3, right + 5, bottom + 3)
    draw.rounded_rectangle(box, radius=4, fill=HALO, outline=LINK_INK, width=2)
    draw.text(at, text, fill=LINK_INK, font=font, anchor=anchor, stroke_width=0)
    return box


def _page_labels(pair, loaded: dict, side: str, page: int, w: int, h: int) -> list[Box]:
    """Where the draw_* functions put their labels (once per component, AS article and class) on
    this row's page, in page pixels."""
    image, data, (scan_w, scan_h) = loaded[page]
    if side == AMERICAN_STORIES:
        sx, sy = w / scan_w, h / scan_h
        rect = lambda b: (b["bbox"]["x0"] * sx, b["bbox"]["y0"] * sy, b["bbox"]["x1"] * sx, b["bbox"]["y1"] * sy)  # noqa: E731
        sets = american_stories_label_sets(data, rect)
    else:
        sets = snewpapers_label_sets(pair.components, page, w, h)
    return [label_backing(*label) for label in queue_labels(sets, w, once=True)]
