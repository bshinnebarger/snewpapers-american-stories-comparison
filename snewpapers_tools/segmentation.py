"""Region by region: how snewpapers' segmentation of an issue lines up with American Stories'.

snewpapers' components are taken to be correct, so every comparison is anchored on one of them.
Both sides are put in page fractions (0-1) on the same scan. A snewpapers component's boxes on one
page are an `OurPiece`; on the American Stories (AS) side the unit is the box (`AsBox`): each has
its OCR text and belongs to at most one AS article (an orphan box belongs to none).

AS boxes are often whole columns that straddle several of our pieces, so a box is sliced, not
assigned whole. Per page:
- a piece claims a slice of a box when their overlap is at least 15% of the box's area or at least
  half of the piece's own (`box_claims`); the slice's text is the box text cut at the overlap's
  vertical extent (AS text runs top to bottom), refined by finding the piece's opening and closing
  words there (`slice_box_text`);
- the gaps the cuts leave are repaired by text (`_repair_gaps`): a run of a box's text between two
  slices goes to the neighbour whose piece text shares more of its 4-word phrases
  (`phrase_containment`), if either shares any; a run at the box's start or end goes to whichever
  of the slice next to it and the pieces overlapping the box without a claim shares most of its
  phrases, if at least 30% (such a piece gets it as a slice); a tiny run otherwise unowned (a word
  the cuts left) joins the next slice;
- each piece is one case: no slices, `ours_only`; slices from one AS article (an orphan box counts
  as its own article), `match`; from two or more, `as_split`, the AS side being each article's
  slices, in position order. First, an article's part that duplicates another's (the shorter
  sharing at least 50% of its phrases with the longer) is left out and the group flagged
  `as_duplicated`: one part left is a `match`;
- each AS article (or orphan box) that two or more of our pieces belong to is an `as_merge`: a piece
  belongs to it when at least 80% of its area lies inside the article's region (the union of its
  boxes), or when it claims a slice of the article and at least 50% of its area lies inside
  (`_merges`). The merge is asked about besides whatever case those pieces are in, with one
  exception: a piece that would be a `match` with the merged article is `in_as_merge` instead. Its
  slices are cut along our boundaries, not AS's, so it is no agreement; its text metrics stay, as
  the text agreement inside AS's merged article, and the merge (`Group.merge`) decides it. A shared
  article no merge comes of (one piece mostly inside, the others marginal) leaves the cases as they are;
- box text no slice claimed (a whole unclaimed box, or what the gap repair left between or beyond
  the slices) is then placed by evidence (`_place_leftovers`): a short run (under 600 characters,
  not tiny) whose text is found in one of our pieces' (`fuzz.partial_ratio` at least 80, the run the
  shorter) is attached to the best such piece as a slice, as the gap repair would; a run of an AS
  article whose other text exactly one of our pieces claimed is "attached outside" that piece
  (`Group.as_attached_outside`): AS draws the article's boundary wider than our component, a
  boundary question, not text snewpapers left out. It is flagged `as_duplicated` too when another
  AS article on the page holds the same text and has a box next to it (`attached_duplicates`,
  `adjacent`): a section heading repeated in another column is no duplicate;
- what remains, grouped by AS article (orphans one by one), is `as_only`, or `as_only_tiny` when it
  is too short to be worth asking about.

A component of ours running over two or more pages is one comparison across them (`link_pages`),
since American Stories never links an article across pages: its per-page cases are gathered, and
when the AS text inside it on all its pages comes from two or more AS articles in all (one on each
of two pages is two articles for one story), it is one `as_split` across its pages, listed on its
first page: its AS parts are every page's, page by page, each in position order (`AsPart.page`
tells them apart), and its question is asked once. On each later page its case becomes `continued`,
a pointer to that group (`Group.continues`): it keeps that page's AS parts, duplicates and attached
text (the boundary question stays per page) but asks nothing of its own. A cross-page component is
never a per-page match (nor `in_as_merge`); the as_merges it is in stay per page. Its AS text all
one article's (only one page holding AS text at all) leaves the per-page cases as they are.

A page whose AS scan is not the LoC scan the API's boxes are on (American Stories segmented an older
digitisation, `pages.is_same_scan`) is left out: boxes on two different scans cannot be compared.

No I/O here: scan sizes come in as an argument (the caller reads the JP2 headers).
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
from rapidfuzz import fuzz

from snewpapers_tools.pages import as_page_numbers, is_same_scan, linked_stems, page_by_seq

Rect = tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions of the page
Region = tuple[Rect, ...]

GRID_WIDTH = 1000  # areas are counted on a grid this many cells wide, as tall as the page's aspect
CLAIM_BOX_SHARE = 0.15  # a piece claims a slice of a box overlapping at least this share of the box...
CLAIM_PIECE_SHARE = 0.5  # ...or at least this share of the piece itself
SLICE_MARGIN = 0.25  # the piece's edges are looked for this share of the rough slice beyond each end...
SLICE_MIN_MARGIN = 100  # ...or at least this many characters
EDGE_NEEDLE_WORDS = 10  # words of the piece's opening and closing looked for in the box text
BAND_ROW_SHARE = 0.25  # a row of a box counts towards a piece's band when the piece covers this share of it
ALIGN_SCORE = 70  # partial_ratio an edge must reach to move the cut...
ALIGN_LENGTH = 0.8  # ...over at least this share of its length (not cut short by the end of the text searched)
MERGE_INSIDE = 0.8  # a piece with at least this share of its area inside an AS article counts towards a merge...
MERGE_CLAIM_INSIDE = 0.5  # ...and so does one claiming a slice of the article with at least this share inside...
MERGE_SLICE_SHARE = 0.25  # ...or claiming one and covering this share of the article (a long story's end)
TINY_CHARS = 20  # left-over AS text shorter than this (stripped) is not asked about
PHRASE_WORDS = 4  # texts are compared by the word phrases of this length they share (phrase_containment)
EDGE_JOIN_SHARE = 30  # an unclaimed run at a box's start or end joins the slice next to it when sharing this much
DUPLICATE_SHARE = 50  # two AS parts in one piece are duplicates when the shorter shares this much with the longer...
DUPLICATE_RATIO = 80  # ...or their normalised texts have at least this fuzz.ratio (one item OCR'd twice, noisily)
FRAGMENT_CHARS = 600  # a left-over run shorter than this (and not tiny) is looked for in our pieces' text...
FRAGMENT_SCORE = 80  # ...and attached to the piece where fuzz.partial_ratio finds it at least this well
ATTACHED_DUPLICATE_SHARE = 70  # attached-outside text another AS article holds this much of is duplicated
COLUMN_GAP = 0.04  # regions whose centres are closer than this (page widths) share a column
ADJACENT_OVERLAP = 0.5  # two boxes are adjacent when they overlap horizontally by this share of the narrower...
ADJACENT_GAP = 1.5  # ...and the vertical gap between them is at most this many heights of the first (or none)
NIBBLE_WORDS = 12  # an AS article's part of our piece with at most this many words...
NIBBLE_SHARE = 0.05  # ...and at most this share of the piece's area is a nibble, when another article holds the rest
NESTED_SHARE = 0.8  # an AS text box with at least this share of its area inside another article's box is nested in it
MATCH_COVER = 0.9
EDGE_SLACK = 0.004  # page heights (about a third of a text line): box edges this close count as the same  # one AS article is a match when AS text covers at least this share of our piece's area

MATCH = "match"
AS_SPLIT = "as_split"  # our one component holds boxes of several AS articles
AS_MERGE = "as_merge"  # several of our components sit inside one AS article
IN_AS_MERGE = "in_as_merge"  # one of those, whose AS text is all that article's: decided by the merge
CONTINUED = "continued"  # a later page of a cross-page as_split: decided by that group, on its first page
OURS_ONLY = "ours_only"
AS_ONLY = "as_only"
AS_ONLY_TINY = "as_only_tiny"
AS_ONLY_EMPTY = "as_only_empty"  # an AS article outside our components with no text at all (AS read nothing)
AS_PARTIAL = "as_partial"  # one AS article, but AS has text for under MATCH_COVER of our piece
AS_WIDER = "as_wider"  # one AS article, but our piece covers under MATCH_COVER of it: decided by its attached text
CASES = (MATCH, AS_SPLIT, AS_MERGE, IN_AS_MERGE, CONTINUED, OURS_ONLY, AS_ONLY, AS_ONLY_TINY, AS_ONLY_EMPTY, AS_PARTIAL,
         AS_WIDER)
ANCHORED = (MATCH, AS_SPLIT, IN_AS_MERGE, CONTINUED, OURS_ONLY, AS_PARTIAL, AS_WIDER)  # one case per snewpapers piece

# Stable machine keys (above) vs. the neutral wording a report shows: named from the counts each side has.
CASE_LABELS = {
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


@dataclass(frozen=True)
class OurPiece:
    """A snewpapers component's boxes on one page. `text` is the whole component's (it may span pages)."""

    uuid: str
    label: str | None
    page: int
    rects: Region
    text: str
    pages: tuple[int, ...]


@dataclass(frozen=True)
class AsBox:
    """One American Stories box, or a slice of one: `rect` and `text` are then the slice's,
    `start` is where its text begins in the whole box's and `whole` is the whole box's rect.
    `order` is the box's place in its article's `object_ids` (AS reading order)."""

    page: int
    box_id: int
    article_id: int | None  # None: an orphan box, in no article
    rect: Rect
    text: str
    cls: str
    legibility: str = ""
    order: int = 0
    start: int = 0
    whole: Rect | None = None  # a slice's box; None for a whole box

    @property
    def box_rect(self) -> Rect:
        """The real American Stories box: the whole box a slice was cut from, or the box itself.
        Slices are cut along our boundaries, so only this is ever drawn as an AS box."""
        return self.whole if self.whole is not None else self.rect


@dataclass(frozen=True)
class AsPart:
    """Some boxes (or slices of boxes) of one AS article, or of one orphan box, on a page, in AS reading order."""

    article_id: int | None
    boxes: tuple[AsBox, ...]

    @property
    def page(self) -> int:
        return self.boxes[0].page

    @property
    def rects(self) -> Region:
        return tuple(b.rect for b in self.boxes)

    @property
    def box_rects(self) -> Region:
        """The real AS boxes the part's boxes or slices are of, each once, in order."""
        return tuple(dict.fromkeys(b.box_rect for b in self.boxes))

    @property
    def text(self) -> str:
        return "\n\n".join(t for t in (b.text.strip() for b in self.boxes) if t)

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(b.cls for b in self.boxes)

    @property
    def name(self) -> str:
        return f"AS #{self.article_id}" if self.article_id is not None else f"AS orphan {'/'.join(self.classes)}"

    @property
    def short_name(self) -> str:
        """`#{article id}`, or `orphan box {id}`."""
        return f"#{self.article_id}" if self.article_id is not None else f"orphan box {self.boxes[0].box_id}"


@dataclass
class Group:
    """One comparison on a page.

    match / as_split / in_as_merge / ours_only: one piece, and the slices of AS boxes it claimed as
    one part per AS article (position order); an in_as_merge's `merge` is the as_merge group of
    that article, which decides it. as_merge: the pieces belonging to the AS article (position
    order) and that article's whole boxes as one part. as_only / as_only_tiny: no piece, one AS
    article's unclaimed box text as one part.

    `duplicates` (anchored cases): AS parts left out because they duplicate a longer one, as
    (the kept part's short name, then each duplicate's), one tuple per kept part.

    `as_attached_outside` (match / as_split): AS text of the piece's own AS articles that the piece
    does not cover (one part per article), left out of the comparison and asked about on its own;
    `attached_duplicates`: (the other AS article holding that text, the attached part), by short name.

    `as_nibbles` (anchored cases): parts of AS articles holding a line or two of the piece (at most
    NIBBLE_WORDS words and NIBBLE_SHARE of its area) while another article holds the rest, left out
    of the case and asked about on their own (does the line belong to our component?).

    A cross-page as_split (`link_pages`) has one piece per page of our component and every page's AS
    parts, page by page; `page` is its first, `continued` the `continued` groups of its later pages.
    Those keep their page's piece, parts, duplicates and attached text, `continues` pointing back.
    Duplicates and attached text are always the group's own page's. Geometry (`our_rects`,
    `as_rects`, `as_box_rects`, `rects`, `bbox`) is the group's own page's; `pieces_on` and
    `parts_on` give another's. Texts are the whole group's, our component's text once.
    """

    page: int
    case: str
    pieces: list[OurPiece]
    parts: list[AsPart]
    metrics: dict[str, float] = field(default_factory=dict)
    other_pages: list[int] = field(default_factory=list)
    duplicates: list[tuple[str, ...]] = field(default_factory=list)
    as_attached_outside: list[AsPart] = field(default_factory=list)
    attached_duplicates: list[tuple[str, str]] = field(default_factory=list)
    as_nibbles: list[AsPart] = field(default_factory=list)
    page_cover: dict[int, tuple[float, float]] = field(default_factory=dict)  # cross-page: page -> (as_cover, illegible_cover)
    merge: Group | None = field(default=None, repr=False, compare=False)  # in_as_merge: its as_merge group
    continues: Group | None = field(default=None, repr=False, compare=False)  # continued: its cross-page as_split
    continued: list[Group] = field(default_factory=list, repr=False, compare=False)  # a cross-page as_split's pointers

    @property
    def pages(self) -> tuple[int, ...]:
        """The pages the group compares across: its own, or a cross-page as_split's every page."""
        return tuple(sorted({self.page} | {p.page for p in self.pieces} | {part.page for part in self.parts}))

    @property
    def cross_page(self) -> bool:
        return len(self.pages) > 1

    def pieces_on(self, page: int) -> list[OurPiece]:
        return [p for p in self.pieces if p.page == page]

    def parts_on(self, page: int) -> list[AsPart]:
        return [part for part in self.parts if part.page == page]

    @property
    def as_duplicated(self) -> bool:
        """American Stories has this text twice (or more): in different articles inside our piece, or
        the text it attaches outside our piece also in another article."""
        return bool(self.duplicates or self.attached_duplicates)

    @property
    def illegible_part(self) -> bool:
        """AS marked at least ILLEGIBLE_PART of our component illegible (reading nothing), on any of its pages."""
        covers = self.page_cover or {self.page: (self.metrics.get("as_cover", 0.0), self.metrics.get("illegible_cover", 0.0))}
        return any(illegible >= ILLEGIBLE_PART for _cover, illegible in covers.values())

    @property
    def missing_pages(self) -> list[int]:
        """A cross-page as_split's pages AS has text for under MATCH_COVER of our component on."""
        return sorted(page for page, (cover, _illegible) in self.page_cover.items() if cover < MATCH_COVER)

    @property
    def as_illegible(self) -> bool:
        """AS lacks text for this component (ours_only, or as_partial) mostly because it marked the boxes
        there illegible and read nothing (metrics `illegible_cover`), not because it drew no box there."""
        if self.case not in (OURS_ONLY, AS_PARTIAL):
            return False
        missing = 1.0 - self.metrics.get("as_cover", 0.0)
        return missing > 0 and self.metrics.get("illegible_cover", 0.0) >= missing / 2

    @property
    def attached_text(self) -> str:
        return "\n\n".join(part.text for part in self.as_attached_outside)

    @property
    def ours_text(self) -> str:
        """Our components' texts, each once (a cross-page component has a piece per page, all its text)."""
        return "\n\n".join({p.uuid: p.text for p in self.pieces}.values())

    @property
    def as_text(self) -> str:
        return "\n\n".join(part.text for part in self.parts)

    @property
    def our_rects(self) -> Region:
        return tuple(r for p in self.pieces_on(self.page) for r in p.rects)

    @property
    def as_rects(self) -> Region:
        return tuple(r for part in self.parts_on(self.page) for r in part.rects)

    @property
    def as_box_rects(self) -> Region:
        """The real AS boxes of the group's parts on its page (whole boxes, never slices), each once."""
        return self.as_box_rects_on(self.page)

    def as_box_rects_on(self, page: int) -> Region:
        return tuple(dict.fromkeys(r for part in self.parts_on(page) for r in part.box_rects))

    @property
    def rects(self) -> Region:
        """The area the group is about: our piece for the cases anchored on one, else the AS boxes."""
        return self.our_rects if self.case in ANCHORED else self.as_rects

    def bbox(self) -> Rect:
        rects = self.rects
        return (min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects))


@dataclass(frozen=True)
class PageCoverage:
    """Grid-cell areas on one page: each side's union, and how much of it the other side covers."""

    ours_area: int
    ours_covered: int
    as_area: int
    as_covered: int


@dataclass
class IssueSegmentation:
    groups: list[Group]
    coverage: dict[int, PageCoverage]
    pages_without_as: list[int]  # pages our components sit on that American Stories has no scan for
    same_scan: tuple[int, int] | None = None  # (AS scans that are the API's LoC scan, all AS scans), see pages.same_scan
    pages_other_scan: list[int] = field(default_factory=list)  # pages left out: AS segmented another digitisation
    as_nested: list[tuple[int, str, str]] = field(default_factory=list)  # AS boxes inside another's (nested_boxes)
    illegible_read: list[tuple[int, int, str, tuple[str, ...]]] = field(default_factory=list)  # see illegible_read
    as_illegible_area: tuple[int, int] = (0, 0)  # (AS boxed area marked illegible and unread, all AS boxed area), grid cells

    def summary(self) -> dict:
        cases = {case: sum(g.case == case for g in self.groups) for case in CASES}
        total = {f: sum(getattr(c, f) for c in self.coverage.values()) for f in PageCoverage.__dataclass_fields__}
        ratios = [g.metrics["ratio"] for g in self.groups if g.case == MATCH and "ratio" in g.metrics]
        return {
            "groups": len(self.groups),
            "cases": cases,
            "ours_covered_by_as": _share(total["ours_covered"], total["ours_area"]),
            "as_covered_by_ours": _share(total["as_covered"], total["as_area"]),
            "median_match_ratio": statistics.median(ratios) if ratios else None,
            "as_duplicated": sum(g.as_duplicated for g in self.groups),
            "attached_outside": sum(bool(g.as_attached_outside) for g in self.groups),
            "pages": sorted(self.coverage),
            "pages_without_as": self.pages_without_as,
            "pages_other_scan": self.pages_other_scan,
            "same_scan": self.same_scan,
        }


def _share(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


# --- geometry ---------------------------------------------------------------------------------


def grid_shape(aspect: float) -> tuple[int, int]:
    """(rows, columns) of the counting grid for a page `aspect` (height / width) times as tall as wide."""
    return round(GRID_WIDTH * aspect), GRID_WIDTH


def _cells(rect: Rect, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """The rect as grid (row0, row1, col0, col1), clipped to the page."""
    rows, cols = shape
    x0, y0, x1, y1 = rect
    clip = lambda v, n: min(max(round(v * n), 0), n)  # noqa: E731
    return clip(y0, rows), clip(y1, rows), clip(x0, cols), clip(x1, cols)


def _window(rects: Region, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    cells = [_cells(r, shape) for r in rects]
    return min(c[0] for c in cells), max(c[1] for c in cells), min(c[2] for c in cells), max(c[3] for c in cells)


def _mask(rects: Region, shape: tuple[int, int], window: tuple[int, int, int, int]) -> np.ndarray:
    """The union of `rects` as booleans over `window` (row0, row1, col0, col1) of the grid."""
    r0, r1, c0, c1 = window
    mask = np.zeros((max(r1 - r0, 0), max(c1 - c0, 0)), dtype=bool)
    for rect in rects:
        a0, a1, b0, b1 = _cells(rect, shape)
        mask[max(a0 - r0, 0):max(a1 - r0, 0), max(b0 - c0, 0):max(b1 - c0, 0)] = True
    return mask


def region_area(rects: Region, aspect: float) -> int:
    """Grid cells covered by the union of `rects`."""
    if not rects:
        return 0
    shape = grid_shape(aspect)
    return int(_mask(rects, shape, _window(rects, shape)).sum())


def intersection_area(a: Region, b: Region, aspect: float) -> int:
    """Grid cells in both unions; 0 straight away when their bounding windows miss each other."""
    if not a or not b:
        return 0
    shape = grid_shape(aspect)
    wa, wb = _window(a, shape), _window(b, shape)
    window = (max(wa[0], wb[0]), min(wa[1], wb[1]), max(wa[2], wb[2]), min(wa[3], wb[3]))
    if window[0] >= window[1] or window[2] >= window[3]:
        return 0
    return int((_mask(a, shape, window) & _mask(b, shape, window)).sum())


def _union_area(regions: list[Region], shape: tuple[int, int]) -> np.ndarray:
    """The union of every region as a full-page mask."""
    rects = tuple(r for region in regions for r in region)
    return _mask(rects, shape, (0, shape[0], 0, shape[1]))


def reading_order(regions: list[Region]) -> list[int]:
    """Indices of `regions` column by column, top to bottom within a column.

    Each region is placed by its leftmost rect (its first column): regions whose rect centres are
    within COLUMN_GAP of the next share a column, so a centred headline stays with its body.
    """
    anchors = [min(region, key=lambda r: (r[0], r[1])) for region in regions]
    by_x = sorted(range(len(regions)), key=lambda k: (anchors[k][0] + anchors[k][2]) / 2)
    column, previous, columns = 0, None, {}
    for k in by_x:
        centre = (anchors[k][0] + anchors[k][2]) / 2
        if previous is not None and centre - previous > COLUMN_GAP:
            column += 1
        columns[k], previous = column, centre
    return sorted(range(len(regions)), key=lambda k: (columns[k], anchors[k][1], anchors[k][0]))


# --- slicing boxes among pieces ---------------------------------------------------------------


def _overlap_band(box: Rect, piece: Region, aspect: float) -> tuple[float, float] | None:
    """The vertical extent of `piece`'s overlap with `box`, as fractions of the box's height: the
    first and last grid rows of the box where the overlap spans BAND_ROW_SHARE of its width (any
    overlap at all, if no row does), so a neighbouring column's sliver does not stretch it."""
    shape = grid_shape(aspect)
    window = _cells(box, shape)
    rows, cols = window[1] - window[0], window[3] - window[2]
    if rows <= 0 or cols <= 0:
        return None
    covered = _mask(piece, shape, window).sum(axis=1) / cols
    found = np.flatnonzero(covered >= BAND_ROW_SHARE)
    if not found.size:
        found = np.flatnonzero(covered > 0)
    if not found.size:
        return None
    return float(found[0] / rows), float((found[-1] + 1) / rows)


def box_claims(box: AsBox, pieces: list[OurPiece], aspect: float,
               piece_areas: list[int] | None = None) -> list[tuple[int, float, float]]:
    """(piece index, top, bottom) for each piece claiming a slice of `box`: its overlap is at least
    CLAIM_BOX_SHARE of the box or CLAIM_PIECE_SHARE of the piece; top and bottom are the overlap's
    vertical extent as fractions of the box's height."""
    area = region_area((box.rect,), aspect)
    if not area:
        return []
    if piece_areas is None:
        piece_areas = [region_area(p.rects, aspect) for p in pieces]
    found = []
    for i, (piece, piece_area) in enumerate(zip(pieces, piece_areas)):
        overlap = intersection_area((box.rect,), piece.rects, aspect)
        if overlap and (overlap >= CLAIM_BOX_SHARE * area or overlap >= CLAIM_PIECE_SHARE * piece_area):
            band = _overlap_band(box.rect, piece.rects, aspect)
            if band is not None:
                found.append((i, *band))
    return found


def article_key(box: AsBox) -> tuple:
    """What groups boxes into AS articles: the article id, or the box itself for an orphan."""
    return ("article", box.article_id) if box.article_id is not None else ("orphan", box.box_id)


def as_parts(boxes: list[AsBox]) -> list[AsPart]:
    """`boxes` grouped by AS article (orphans one by one), each in AS reading order, parts in position order."""
    grouped: dict[tuple, list[AsBox]] = {}
    for box in boxes:
        grouped.setdefault(article_key(box), []).append(box)
    parts = [AsPart(members[0].article_id, tuple(sorted(members, key=lambda b: (b.order, b.box_id, b.start))))
             for members in grouped.values()]
    return [parts[k] for k in reading_order([p.rects for p in parts])] if parts else []


# --- text -------------------------------------------------------------------------------------


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


_WORD = re.compile(r"[a-z]+")


@lru_cache(maxsize=4096)
def _phrases(text: str, n: int) -> tuple[tuple[str, ...], frozenset]:
    """`text`'s words (lowercased runs of a-z) and the set of its word `n`-grams."""
    words = tuple(_WORD.findall(text.lower()))
    return words, frozenset(words[k:k + n] for k in range(len(words) - n + 1))


def phrase_containment(a: str, b: str, n: int = PHRASE_WORDS) -> float:
    """The share (0-100) of `a`'s word `n`-grams that occur anywhere in `b`; for an `a` of fewer
    than `n` words, the share of its words that occur in `b`.

    Words are lowercased runs of a-z, so OCR punctuation and digits do not count. Unlike a longest
    common subsequence, a phrase must occur whole, so a long `b` does not "contain" an unrelated `a`
    by scattering its letters across itself.
    """
    words, _ = _phrases(a, n)
    if not words:
        return 0.0
    other_words, other = _phrases(b, n)
    if len(words) < n:
        present = set(other_words)
        return 100 * sum(w in present for w in words) / len(words)
    grams = [words[k:k + n] for k in range(len(words) - n + 1)]
    return 100 * sum(g in other for g in grams) / len(grams)


def text_metrics(ours: str, theirs: str, *, partial: bool) -> dict[str, float]:
    """fuzz.ratio and token_set_ratio on normalised text; with `partial` (one side may hold more),
    `shared_phrases` too: the phrase_containment of the side with fewer words in the other."""
    a, b = normalise(ours), normalise(theirs)
    metrics = {"ratio": round(fuzz.ratio(a, b), 1), "token_set_ratio": round(fuzz.token_set_ratio(a, b), 1)}
    if partial:
        short, long = sorted((a, b), key=lambda t: len(_phrases(t, PHRASE_WORDS)[0]))
        metrics["shared_phrases"] = round(phrase_containment(short, long), 1)
    return metrics


def is_tiny(text: str) -> bool:
    return len(text.strip()) < TINY_CHARS


def _word_start(text: str, k: int) -> int:
    while 0 < k < len(text) and not text[k - 1].isspace() and not text[k].isspace():
        k -= 1
    return k


def _word_end(text: str, k: int) -> int:
    while 0 < k < len(text) and not text[k - 1].isspace() and not text[k].isspace():
        k += 1
    return k


def _nearest_boundary(text: str, k: int) -> int:
    """`k`, or the nearer end of the word it falls inside."""
    a, b = _word_start(text, k), _word_end(text, k)
    return a if k - a <= b - k else b


def _comparable(text: str) -> str:
    """Lower case, every space a plain space, the length (and so every position) kept."""
    return "".join(" " if c.isspace() else c.lower() if len(c.lower()) == 1 else c for c in text)


def _align(needle: str, haystack: str) -> tuple[int, int] | None:
    """Where `needle` best sits in `haystack` (start, end), if well enough (ALIGN_SCORE) and nearly
    whole (ALIGN_LENGTH: an alignment against the haystack's very end matches only part of it)."""
    if not needle or len(haystack) < len(needle):
        return None
    found = fuzz.partial_ratio_alignment(needle, haystack)
    if found is None or found.score < ALIGN_SCORE or found.dest_end - found.dest_start < ALIGN_LENGTH * len(needle):
        return None
    return found.dest_start, found.dest_end


def slice_span(text: str, top_frac: float, bottom_frac: float, piece_text: str) -> tuple[int, int]:
    """Where in a box's `text` the part from `top_frac` to `bottom_frac` of the box's height lies,
    for a piece whose text is `piece_text`: (start, end) character positions.

    AS text runs top to bottom, so the rough cut is proportional by character position, each end
    pushed outward to a word boundary. Within that cut plus SLICE_MARGIN of its length (at least
    SLICE_MIN_MARGIN characters) either side, the piece's opening and closing EDGE_NEEDLE_WORDS
    words are looked for; an edge found well enough (`_align`) moves that end of the cut to the
    nearest word boundary there, else the proportional end stands.
    """
    n = len(text)
    top, bottom = sorted((min(max(top_frac, 0.0), 1.0), min(max(bottom_frac, 0.0), 1.0)))
    start, end = _word_start(text, round(top * n)), _word_end(text, round(bottom * n))
    margin = max(round(SLICE_MARGIN * (end - start)), SLICE_MIN_MARGIN)
    lo, hi = max(start - margin, 0), min(end + margin, n)
    window = _comparable(text[lo:hi])
    words = piece_text.split()
    opening = _align(_comparable(" ".join(words[:EDGE_NEEDLE_WORDS])), window)
    closing = _align(_comparable(" ".join(words[-EDGE_NEEDLE_WORDS:])), window)
    new_start = _nearest_boundary(text, lo + opening[0]) if opening else start
    new_end = _nearest_boundary(text, lo + closing[1]) if closing else end
    if new_start < new_end:
        return new_start, new_end
    return start, end


def slice_box_text(text: str, top_frac: float, bottom_frac: float, piece_text: str) -> str:
    """The text of a box's slice from `top_frac` to `bottom_frac` of its height, for a piece whose
    text is `piece_text` (see `slice_span`)."""
    start, end = slice_span(text, top_frac, bottom_frac, piece_text)
    return text[start:end]


# --- one page ---------------------------------------------------------------------------------


def _coverage(pieces: list[OurPiece], boxes: list[AsBox], aspect: float) -> PageCoverage:
    shape = grid_shape(aspect)
    ours_mask = _union_area([p.rects for p in pieces], shape)
    as_mask = _union_area([(b.rect,) for b in boxes], shape)
    return PageCoverage(
        int(ours_mask.sum()), int((ours_mask & as_mask).sum()),
        int(as_mask.sum()), int((as_mask & ours_mask).sum()),
    )


def collapse_duplicates(parts: list[AsPart]) -> tuple[list[AsPart], list[tuple[str, ...]]]:
    """`parts` without those that duplicate another, and the duplicates found.

    Two parts duplicate each other when the shorter (in characters) has at least PHRASE_WORDS words
    and phrase_containment of it in the longer is at least DUPLICATE_SHARE, or the two normalised
    texts' fuzz.ratio is at least DUPLICATE_RATIO (the same item OCR'd twice too noisily to share
    half its phrases; the ratio counts the length difference too); the longer is kept.
    Returns the kept parts in their order and, per kept part with duplicates, (its short name, the
    duplicates' short names).
    """
    kept: list[AsPart] = []
    dropped: dict[int, list[AsPart]] = {}
    for part in sorted(parts, key=lambda p: -len(p.text)):
        enough = len(_phrases(part.text, PHRASE_WORDS)[0]) >= PHRASE_WORDS
        twin = next((k for k, other in enumerate(kept)
                     if enough and (phrase_containment(part.text, other.text) >= DUPLICATE_SHARE
                                    or fuzz.ratio(normalise(part.text), normalise(other.text)) >= DUPLICATE_RATIO)), None)
        if twin is None:
            kept.append(part)
        else:
            dropped.setdefault(twin, []).append(part)
    duplicates = [(kept[k].short_name, *(p.short_name for p in found)) for k, found in dropped.items()]
    return [p for p in parts if any(p is k for k in kept)], duplicates


_TOKEN = re.compile(r"[a-z0-9]+")


def holds_text(longer: str, text: str) -> bool:
    """Whether `longer` holds `text`: phrase_containment of `text` in it at least
    ATTACHED_DUPLICATE_SHARE; for a `text` of fewer than PHRASE_WORDS words, its lowercased letter
    and digit runs found as a whole run of `longer`'s (an exact substring, punctuation aside)."""
    if len(_phrases(text, PHRASE_WORDS)[0]) >= PHRASE_WORDS:
        return phrase_containment(text, longer) >= ATTACHED_DUPLICATE_SHARE
    needle = " ".join(_TOKEN.findall(text.lower()))
    return bool(needle) and f" {needle} " in f" {' '.join(_TOKEN.findall(longer.lower()))} "


def adjacent(box: Rect, other: Rect) -> bool:
    """Whether `other` sits right next to `box`: overlapping it horizontally by at least
    ADJACENT_OVERLAP of the narrower width, and vertically touching, overlapping or apart by at most
    ADJACENT_GAP times `box`'s height."""
    overlap = min(box[2], other[2]) - max(box[0], other[0])
    narrower = min(box[2] - box[0], other[2] - other[0])
    if narrower <= 0 or overlap < ADJACENT_OVERLAP * narrower:
        return False
    gap = max(other[1] - box[3], box[1] - other[3], 0.0)
    return gap <= ADJACENT_GAP * (box[3] - box[1])


def attached_duplicates(attached: list[AsPart], articles: list[AsPart]) -> list[tuple[str, str]]:
    """(the other article's short name, the attached part's) for each attached part whose text
    another of the page's whole `articles` holds (`holds_text`) with a box `adjacent` to one of the
    part's boxes, the first such article. The same text in an article elsewhere on the page (a
    section heading repeated in another column) is no duplicate."""
    found = []
    for part in attached:
        key = article_key(part.boxes[0])
        near = lambda a: any(adjacent(b.rect, o.rect) for b in part.boxes for o in a.boxes)  # noqa: E731
        other = next((a for a in articles if article_key(a.boxes[0]) != key and near(a) and holds_text(a.text, part.text)),
                     None)
        if other is not None:
            found.append((other.short_name, part.short_name))
    return found


def row_coverage(ours: Region, theirs: Region, slack: float = EDGE_SLACK) -> float:
    """How much of `ours` AS boxes cover, by rows: in each of our rects, the share of its height that
    boxes overlapping it side to side cover (a centred headline narrower than the column covers its
    row), the rects weighted by their area. `theirs` reach `slack` further up and down: two systems'
    edges on the same line of print differ by a few pixels, which on a one-line item is a lot."""
    theirs = tuple((b[0], b[1] - slack, b[2], b[3] + slack) for b in theirs)
    total = covered = 0.0
    for x0, y0, x1, y1 in ours:
        area = (x1 - x0) * (y1 - y0)
        if area <= 0:
            continue
        spans = sorted((max(b[1], y0), min(b[3], y1)) for b in theirs if b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0)
        height, top = 0.0, y0
        for a, b in spans:
            a = max(a, top)
            if b > a:
                height += b - a
                top = b
        total += area
        covered += area * height / (y1 - y0)
    return covered / total if total else 0.0


def _scraps(piece: OurPiece, parts: list[AsPart], aspect: float) -> list[bool]:
    """Per part: a scrap of at most NIBBLE_WORDS words covering at most NIBBLE_SHARE of the piece's area."""
    area = region_area(piece.rects, aspect)
    return [len(part.text.split()) <= NIBBLE_WORDS and area > 0
            and intersection_area(part.rects, piece.rects, aspect) <= NIBBLE_SHARE * area for part in parts]


def nibbles(piece: OurPiece, parts: list[AsPart], aspect: float) -> tuple[list[AsPart], list[AsPart]]:
    """`parts` split into those kept and the nibbles: scraps (`_scraps`), while at least one part is none."""
    small = _scraps(piece, parts, aspect)
    if all(small):
        return parts, []
    return [p for p, s in zip(parts, small) if not s], [p for p, s in zip(parts, small) if s]


def _anchored(page: int, piece: OurPiece, slices: list[AsBox], outside: list[AsBox] = (),
              articles: list[AsPart] = (), aspect: float = 1.0) -> Group:
    """The case for one of our pieces, from the slices of AS boxes it claimed: slices without text
    left out (an empty AS box is an artifact), AS parts that duplicate another left out
    (`collapse_duplicates`), nibbles set aside (`nibbles`). One part is a match when AS text covers
    at least MATCH_COVER of the piece (metrics `as_cover`, the share, and `iou`, of the piece and
    the article's whole boxes), else `as_partial`; and when our piece covers at least MATCH_COVER of
    the article's text that goes with it, its slices on the piece (a run placed by its text from
    elsewhere is ours) and the text it attaches outside (metrics
    `ours_cover`), else `as_wider`: AS's article runs on beyond our piece, and the question about its
    attached text decides it. Coverage is by rows (`row_coverage`), of the
    slices with text and the empty slices of those slices' articles (AS boxed it but read nothing). `outside` is its AS articles' text attached
    outside it, checked against the page's whole `articles` for duplicated text."""
    texts = [s for s in slices if s.text.strip()]
    parts, duplicates = collapse_duplicates(as_parts(texts))
    kept, nibbled = nibbles(piece, parts, aspect)
    other_pages = sorted(set(piece.pages) - {page})
    case = OURS_ONLY if not kept else MATCH if len(kept) == 1 else AS_SPLIT
    attached = as_parts(list(outside))
    group = Group(page, case, [piece], kept, other_pages=other_pages, duplicates=duplicates,
                  as_attached_outside=attached, attached_duplicates=attached_duplicates(attached, list(articles)),
                  as_nibbles=nibbled)
    if kept:
        group.metrics = text_metrics(group.ours_text, group.as_text, partial=bool(other_pages))
    illegible = tuple(s.rect for s in slices if s.legibility == "Illegible" and not s.text.strip())
    if illegible:  # AS boxed it, marked it illegible and read nothing
        group.metrics["illegible_cover"] = round(row_coverage(piece.rects, illegible), 3)
    if kept:
        area = region_area(piece.rects, aspect)
        if area:
            articles = {s.article_id for s in texts if s.article_id is not None}
            as_region = tuple(s.rect for s in slices
                              if s.text.strip() or (s.article_id in articles and s.cls == "headline"))
            group.metrics["as_cover"] = round(row_coverage(piece.rects, as_region), 3)
        if case == MATCH and area:
            boxes = kept[0].box_rects
            both = intersection_area(piece.rects, boxes, aspect)
            group.metrics["iou"] = round(both / (area + region_area(boxes, aspect) - both), 3)
            on_piece = tuple(r for r in kept[0].rects if intersection_area((r,), piece.rects, aspect))
            article = on_piece + tuple(r for part in attached for r in part.rects)
            group.metrics["ours_cover"] = round(row_coverage(article, piece.rects), 3)
            if group.metrics["as_cover"] < MATCH_COVER:
                group.case = AS_PARTIAL
            elif group.metrics["ours_cover"] < MATCH_COVER:
                group.case = AS_WIDER
        if case == AS_SPLIT and group.metrics.get("as_cover", 1.0) < MATCH_COVER and all(_scraps(piece, kept, aspect)):
            group.case = AS_PARTIAL  # only scraps of AS text over it: nothing to split, AS lacks the text
    return group


def merge_members(pieces: list[OurPiece], members: list[AsBox], claimants: set[int], aspect: float,
                  piece_areas: list[int]) -> list[int]:
    """The indices of the pieces that belong to one AS article (or orphan box) of `members`, its
    boxes: at least MERGE_INSIDE of the piece's area inside the article's region, or, for a piece
    in `claimants` (it claimed a slice of the article), at least MERGE_CLAIM_INSIDE, or covering at
    least MERGE_SLICE_SHARE of the article's region (the end of a story running over several
    columns, most of it elsewhere)."""
    region = tuple(b.rect for b in members)
    whole = region_area(region, aspect)
    found = []
    for i, (piece, area) in enumerate(zip(pieces, piece_areas)):
        share = intersection_area(piece.rects, region, aspect) / area if area else 0.0
        filled = intersection_area(piece.rects, region, aspect) / whole if whole else 0.0
        if share >= MERGE_INSIDE or (i in claimants and (share >= MERGE_CLAIM_INSIDE or filled >= MERGE_SLICE_SHARE)):
            found.append(i)
    return found


def _merges(page: int, pieces: list[OurPiece], boxes: list[AsBox], aspect: float,
            slices: list[list[AsBox]] = ()) -> list[Group]:
    """An as_merge for each AS article (or orphan box) two or more pieces belong to (`merge_members`;
    `slices` are each piece's, telling which pieces claimed a slice of which article)."""
    claimants: dict[tuple, set[int]] = {}
    for i, own in enumerate(slices):
        for b in own:
            claimants.setdefault(article_key(b), set()).add(i)
    by_article: dict[tuple, list[AsBox]] = {}
    for box in boxes:
        by_article.setdefault(article_key(box), []).append(box)
    found = []
    piece_areas = [region_area(p.rects, aspect) for p in pieces]
    for key, members in by_article.items():
        [part] = as_parts(members)
        inside = [pieces[i] for i in merge_members(pieces, members, claimants.get(key, set()), aspect, piece_areas)]
        if len(inside) >= 2:
            inside = [inside[k] for k in reading_order([p.rects for p in inside])]
            other_pages = sorted({n for p in inside for n in p.pages} - {page})
            group = Group(page, AS_MERGE, inside, [part], other_pages=other_pages)
            group.metrics = text_metrics(group.ours_text, group.as_text, partial=True)
            found.append(group)
    return found


def mark_in_merges(anchored: list[Group], merges: list[Group]) -> None:
    """Make each `match` whose AS article is an as_merge it belongs to `in_as_merge`, in place,
    pointing at that merge: its slices are cut along our boundaries, not AS's, so it is no agreement."""
    by_article = {article_key(m.parts[0].boxes[0]): m for m in merges}
    for group in anchored:
        if group.case not in (MATCH, AS_PARTIAL, AS_WIDER):
            continue
        merge = by_article.get(article_key(group.parts[0].boxes[0]))
        if merge is not None and any(p is group.pieces[0] for p in merge.pieces):
            group.case, group.merge = IN_AS_MERGE, merge


def _leftovers(page: int, fragments: list[AsBox]) -> list[Group]:
    """One as_only (or as_only_tiny) per AS article among the unclaimed box text, orphans one by one."""
    def case(part: AsPart) -> str:
        return AS_ONLY_EMPTY if not part.text.strip() else AS_ONLY_TINY if is_tiny(part.text) else AS_ONLY

    return [Group(page, case(part), [], [part]) for part in as_parts(fragments)]


def fragment_owner(text: str, pieces: list[OurPiece]) -> int | None:
    """The piece a short left-over run's text is found in: the index of the piece with the best
    fuzz.partial_ratio of the run (normalised) in its text, if at least FRAGMENT_SCORE. Only a run
    under FRAGMENT_CHARS and not tiny is looked for, and only in texts at least as long (else
    partial_ratio would look for the piece in the run)."""
    if is_tiny(text) or len(text.strip()) >= FRAGMENT_CHARS:
        return None
    needle = normalise(text)
    scores = [(fuzz.partial_ratio(needle, hay), k) for k, hay in enumerate(normalise(p.text) for p in pieces)
              if len(hay) >= len(needle)]
    if not scores:
        return None
    score, k = max(scores, key=lambda found: found[0])  # the first of equals
    return k if score >= FRAGMENT_SCORE else None


def inside_owner(frag: AsBox, pieces: list[OurPiece], slices: list[list[AsBox]], aspect: float) -> int | None:
    """The piece a run of a box's text lies inside (at least half its band's area) when that piece
    holds a slice of the same box: printed inside our component, it is its text, however it reads."""
    area = region_area((frag.rect,), aspect)
    if not area:
        return None
    for i, piece in enumerate(pieces):
        if (any(b.box_id == frag.box_id and b.page == frag.page for b in slices[i])
                and intersection_area((frag.rect,), piece.rects, aspect) >= area / 2):
            return i
    return None


def _place_leftovers(pieces: list[OurPiece], slices: list[list[AsBox]],
                     fragments: list[AsBox], aspect: float = 1.0) -> tuple[list[list[AsBox]], list[AsBox]]:
    """Place the box text nobody claimed: a run lying inside a piece that holds a slice of its box
    (`inside_owner`), or else found in a piece's text (`fragment_owner`), joins that piece's slices
    (in place); then a run of an AS article whose other text exactly one piece
    claimed is attached outside that piece, and so is a run of an article several pieces claimed that
    directly follows or precedes exactly one of their slices of its box. Returns (each piece's attached-outside runs, what remains)."""
    remaining = []
    for frag in fragments:
        owner = inside_owner(frag, pieces, slices, aspect) if frag.whole is not None else None
        owner = owner if owner is not None else fragment_owner(frag.text, pieces)
        if owner is None:
            remaining.append(frag)
        else:
            slices[owner].append(frag)
    claimants: dict[int, set[int]] = {}
    for i, own in enumerate(slices):
        for b in own:
            if b.article_id is not None:
                claimants.setdefault(b.article_id, set()).add(i)
    outside: list[list[AsBox]] = [[] for _ in pieces]
    left = []
    for frag in remaining:
        owners = claimants.get(frag.article_id, set()) if frag.article_id is not None else set()
        if len(owners) > 1:  # the one whose slice of the same box the run directly follows or precedes
            owners = {i for i in owners if any(
                b.box_id == frag.box_id and b.page == frag.page
                and (b.start + len(b.text) == frag.start or frag.start + len(frag.text) == b.start) for b in slices[i])}
        if len(owners) == 1 and any(c.isalnum() for c in frag.text):
            outside[next(iter(owners))].append(frag)
        else:
            left.append(frag)
    return outside, left


def _band_rect(rect: Rect, top: float, bottom: float) -> Rect:
    """The horizontal band of `rect` from `top` to `bottom` (fractions of its height)."""
    x0, y0, x1, y1 = rect
    return x0, y0 + top * (y1 - y0), x1, y0 + bottom * (y1 - y0)


def _gap_owner(run: str, before: int | None, after: int | None, piece_texts: list[str],
               other_texts: list[str]) -> tuple[str, int] | None:
    """Who takes an unclaimed `run` of a box's text, by the phrases it shares (see `_repair_gaps`):
    ("span", k) for the span of `piece_texts[k]`, ("other", k) for the piece of `other_texts[k]`
    (it overlaps the box without claiming a slice), None for nobody. `before` and `after` are the
    spans either side of the run (None at the box's start or end)."""
    if before is not None and after is not None:
        shares = [phrase_containment(run, piece_texts[k]) for k in (before, after)]
        if max(shares) > 0:
            return ("span", before) if shares[0] >= shares[1] else ("span", after)
    else:
        edge = after if before is None else before
        candidates = [("span", edge, piece_texts[edge])] + [("other", k, t) for k, t in enumerate(other_texts)]
        share, kind, k = max(((phrase_containment(run, t), kind, k) for kind, k, t in candidates),
                             key=lambda found: found[0])  # the first of equals: the adjacent slice
        if share >= EDGE_JOIN_SHARE:
            return kind, k
    if is_tiny(run):  # a word the cuts left: no evidence needed
        return "span", after if after is not None else before
    return None


def _repair_gaps(text: str, spans: list[list[int]], piece_texts: list[str],
                 other_texts: list[str] = ()) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    """The runs of a box's `text` no span covers, once the slicing gaps are repaired. `spans` are
    the claimed slices ([start, end] lists, piece k's with `piece_texts[k]`), widened here for the
    runs they take; `other_texts` are the texts of pieces that overlap the box without claiming it.

    A run holding a letter or digit between two spans goes to the neighbour whose piece text shares
    more of its phrases (phrase_containment; a tie to the span before), if either shares any. A run
    at the box's start or end goes to whichever of the span next to it and the overlapping pieces
    shares most of its phrases (a tie to the span), when that is at least EDGE_JOIN_SHARE. A run
    that is neither, but tiny (a word the cuts left), joins the next span, or at the end the span
    reaching furthest. Returns (what remains, as (start, end); the runs the overlapping pieces
    take, as (k in `other_texts`, start, end)).
    """
    n, k, before, left, taken = len(text), 0, None, [], []
    for after in sorted(range(len(spans)), key=lambda j: spans[j]) + [None]:
        stop = spans[after][0] if after is not None else n
        run = text[k:stop]
        if stop > k and any(c.isalnum() for c in run):
            owner = _gap_owner(run, before, after, piece_texts, list(other_texts))
            if owner is None:
                left.append((k, stop))
            elif owner[0] == "other":
                taken.append((owner[1], k, stop))
            elif owner[1] == after:
                spans[after][0] = k
            else:
                spans[before][1] = stop
        if after is not None and (before is None or spans[after][1] > spans[before][1]):
            before = after
        k = max(k, spans[before][1]) if before is not None else k
    return left, taken


def slice_boxes(pieces: list[OurPiece], boxes: list[AsBox], aspect: float) -> tuple[list[list[AsBox]], list[AsBox]]:
    """Each piece's slices of the AS boxes, and the box text nobody claimed, as boxes of their own.

    A slice keeps its box's ids and order, with the slice's text, the band of the box the piece
    overlaps as its rect, its text's place in the box as `start` and the box's rect as `whole`. The runs of a claimed box's
    text outside every slice are repaired by the text they share (`_repair_gaps`): a run a piece
    overlapping the box without a claim takes becomes a slice of that piece, drawn as the run's
    proportional band of the box. Unclaimed text is what remains, drawn the same way, and every
    box nobody claims, whole.
    """
    piece_areas = [region_area(p.rects, aspect) for p in pieces]
    slices: list[list[AsBox]] = [[] for _ in pieces]
    unclaimed = []
    for box in boxes:
        claims = box_claims(box, pieces, aspect, piece_areas)
        if not claims:
            unclaimed.append(box)
            continue
        claimed = {i for i, _, _ in claims}
        others = [i for i, p in enumerate(pieces) if i not in claimed and intersection_area((box.rect,), p.rects, aspect)]
        spans = [list(slice_span(box.text, top, bottom, pieces[i].text)) for i, top, bottom in claims]
        n = len(box.text)
        band = lambda a, b: replace(box, rect=_band_rect(box.rect, a / n, b / n), text=box.text[a:b], start=a,  # noqa: E731
                                    whole=box.rect)
        left, taken = _repair_gaps(box.text, spans, [pieces[i].text for i, _, _ in claims], [pieces[i].text for i in others])
        unclaimed += [band(a, b) for a, b in left]
        for k, a, b in taken:
            slices[others[k]].append(band(a, b))
        for (i, top, bottom), (a, b) in zip(claims, spans):
            slices[i].append(replace(box, rect=_band_rect(box.rect, top, bottom), text=box.text[a:b], start=a,
                                     whole=box.rect))
    return slices, unclaimed


def segment_page(pieces: list[OurPiece], boxes: list[AsBox], aspect: float) -> tuple[list[Group], PageCoverage]:
    """The page's groups in position order, and its coverage each way."""
    page = (pieces or boxes)[0].page if (pieces or boxes) else 0
    slices, unclaimed = slice_boxes(pieces, boxes, aspect)
    outside, unclaimed = _place_leftovers(pieces, slices, unclaimed, aspect)
    articles = as_parts(boxes)
    groups = [_anchored(page, piece, own, out, articles, aspect) for piece, own, out in zip(pieces, slices, outside)]
    merges = _merges(page, pieces, boxes, aspect, slices)
    mark_in_merges(groups, merges)
    groups += merges
    groups += _leftovers(page, unclaimed)
    groups = [groups[k] for k in reading_order([g.rects for g in groups])] if groups else []
    return groups, _coverage(pieces, boxes, aspect)


# --- across pages -----------------------------------------------------------------------------


def link_pages(groups: list[Group]) -> None:
    """Make each component of ours anchored on two or more pages one comparison across them, in
    place (see the module docstring): when its per-page cases' AS parts number two or more in all,
    its first page's group becomes the cross-page `as_split` (every page's piece and parts, page by
    page) and each later page's `continued`, pointing to it. Groups keep their places."""
    by_uuid: dict[str, list[Group]] = {}
    for g in groups:
        if g.case in (MATCH, AS_PARTIAL, AS_WIDER, AS_SPLIT, IN_AS_MERGE, OURS_ONLY):
            by_uuid.setdefault(g.pieces[0].uuid, []).append(g)
    for found in by_uuid.values():
        if len({g.page for g in found}) < 2:
            continue
        found.sort(key=lambda g: g.page)
        parts = [part for g in found for part in g.parts]
        if len(parts) < 2:
            continue
        first, later = found[0], found[1:]
        first.page_cover = {g.page: (g.metrics.get("as_cover", 0.0), g.metrics.get("illegible_cover", 0.0)) for g in found}
        pages = {g.page for g in found}
        first.case, first.merge = AS_SPLIT, None
        first.pieces = [g.pieces[0] for g in found]
        first.parts = parts
        first.other_pages = sorted(set(first.pieces[0].pages) - pages)
        first.metrics = text_metrics(first.ours_text, first.as_text, partial=bool(first.other_pages))
        first.continued = later
        for g in later:
            g.case, g.merge, g.continues, g.metrics = CONTINUED, None, first, {}


# --- inputs -----------------------------------------------------------------------------------


def our_pieces(components: list[dict]) -> list[OurPiece]:
    """One piece per component per page it has boxes on (components without boxes are left out)."""
    return [
        OurPiece(comp["uuid"], comp.get("label"), entry["page"], tuple(tuple(r) for r in entry["rects"]),
                 comp.get("content") or "", tuple(comp.get("pages") or [entry["page"]]))
        for comp in components
        for entry in comp.get("boxes") or []
    ]


def off_page(rect: Rect) -> bool:
    """Most of the box lies outside the scan (American Stories sometimes boxes the margin beyond a tall
    page's right edge: an empty, illegible "article" at x 1.0-1.4)."""
    x0, y0, x1, y1 = rect
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    inside = max(0.0, min(x1, 1.0) - max(x0, 0.0)) * max(0.0, min(y1, 1.0) - max(y0, 0.0))
    return area == 0 or inside < area / 2


def as_boxes(data: dict, page: int, scan_size: tuple[int, int]) -> list[AsBox]:
    """The AS page's boxes as fractions of the real scan, each with its article (from the articles'
    `object_ids`; the first article to list a box has it) and its place in that article's order.

    Boxes mostly off the scan with no text are American Stories artifacts and are left out."""
    width, height = scan_size
    owner: dict[int, tuple[int, int]] = {}
    for article in data["full articles"]:
        for k, box_id in enumerate(article["object_ids"]):
            owner.setdefault(box_id, (article["full_article_id"], k))
    found = []
    for box in data["bboxes"]:
        b = box["bbox"]
        rect = (b["x0"] / width, b["y0"] / height, b["x1"] / width, b["y1"] / height)
        text = box.get("raw_text") or ""
        if off_page(rect) and not text.strip():
            continue
        # a box in no listed article keeps its own full_article_id (AS leaves an article whose body read as
        # nothing out of "full articles" but its boxes still say whose they are), ordered by reading_order_id
        article_id, order = owner.get(box["id"], (box.get("full_article_id"), box.get("reading_order_id") or 0))
        found.append(AsBox(page, box["id"], article_id, rect, text, box["class"], box.get("legibility") or "", order))
    return found


def nested_boxes(boxes: list[AsBox]) -> list[tuple[int, str, str]]:
    """(page, inner article, outer article) by short name for each AS box with text lying at least
    NESTED_SHARE inside a larger text box of another article: American Stories' layout detection
    boxed the same print twice (a column and an item in it, say), and both were OCR'd and published
    as articles. Nothing in AS's data marks or explains it; this is found from the geometry alone."""
    def short(b: AsBox) -> str:
        return AsPart(b.article_id, (b,)).short_name

    def area(r: Rect) -> float:
        return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])

    texts = [b for b in boxes if b.text.strip()]
    found = []
    for inner in texts:
        own = area(inner.rect)
        for outer in texts:
            if outer is inner or (inner.article_id is not None and outer.article_id == inner.article_id):
                continue
            if area(outer.rect) > own > 0:
                x0, y0, x1, y1 = (max(inner.rect[0], outer.rect[0]), max(inner.rect[1], outer.rect[1]),
                                  min(inner.rect[2], outer.rect[2]), min(inner.rect[3], outer.rect[3]))
                if area((x0, y0, x1, y1)) >= NESTED_SHARE * own:
                    found.append((inner.page, short(inner), short(outer)))
                    break
    return found


ILLEGIBLE_PART = 0.1  # AS marked at least this share of our component illegible: its pieces do not show the text
ILLEGIBLE_INSIDE = 0.5  # an empty Illegible AS box with at least this share of its area inside our components


def illegible_read(pieces: list[OurPiece], boxes: list[AsBox], aspect: float) -> list[tuple[int, int, str, tuple[str, ...]]]:
    """(page, AS box id, its article's short name, the uuids of our pieces over it) for each AS box
    American Stories marked "Illegible" and read no text from, at least ILLEGIBLE_INSIDE of it inside
    our components: print AS's pipeline judged unreadable and left out, that snewpapers read."""
    found = []
    for b in boxes:
        if b.legibility != "Illegible" or b.text.strip():
            continue
        area = region_area((b.rect,), aspect)
        over = [p for p in pieces if intersection_area((b.rect,), p.rects, aspect)]
        if area and over and intersection_area((b.rect,), tuple(r for p in over for r in p.rects), aspect) >= ILLEGIBLE_INSIDE * area:
            found.append((b.page, b.box_id, AsPart(b.article_id, (b,)).short_name, tuple(dict.fromkeys(p.uuid for p in over))))
    return found


def illegible_area(boxes: list[AsBox], aspect: float) -> tuple[int, int]:
    """(grid cells of AS boxes marked "Illegible" with no text, grid cells of all AS boxes), each a union
    (overlaps counted once): how much of what American Stories boxed on the page it did not read."""
    unread = tuple(b.rect for b in boxes if b.legibility == "Illegible" and not b.text.strip())
    return region_area(unread, aspect), region_area(tuple(b.rect for b in boxes), aspect)


def segment_issue(pair, scan_sizes: dict[int, tuple[int, int]]) -> IssueSegmentation:
    """Every page American Stories has a scan for, compared, except where that scan is not the one
    the API's boxes refer to (pages.is_same_scan: American Stories segmented an older LoC
    digitisation); those pages get no groups and are listed in `pages_other_scan`. A component of
    ours on two or more compared pages is then compared across them (`link_pages`).

    `scan_sizes` is page -> the real (width, height) of the scan, read from the JP2 (overlay's
    issue_scan_sizes); a page missing from it falls back to the size the AS JSON states.
    """
    pieces = our_pieces(pair.components)
    by_seq = page_by_seq(pair.components)
    stems = linked_stems(pair.components)
    pages, _ = as_page_numbers([scan for scan, _ in pair.pages], by_seq)
    groups, coverage, other_scan, same, nested, illegible = [], {}, [], 0, [], []
    unread_area = boxed_area = 0
    for scan, data in pair.pages:
        page = pages.get(scan.seq)
        if not is_same_scan(scan, stems):
            if page is not None:
                other_scan.append(page)
            continue
        same += 1
        if page is None:
            continue
        size = scan_sizes.get(page) or (data["scan"]["width"], data["scan"]["height"])
        boxes = as_boxes(data, page, size)
        page_groups, coverage[page] = segment_page([p for p in pieces if p.page == page], boxes, size[1] / size[0])
        nested += nested_boxes(boxes)
        illegible += illegible_read([p for p in pieces if p.page == page], boxes, size[1] / size[0])
        unread, boxed = illegible_area(boxes, size[1] / size[0])
        unread_area, boxed_area = unread_area + unread, boxed_area + boxed
        groups += page_groups
    link_pages(groups)
    pages_without_as = sorted({p.page for p in pieces} - set(coverage) - set(other_scan))
    return IssueSegmentation(groups, coverage, pages_without_as, (same, len(pair.pages)), sorted(other_scan),
                             sorted(nested), illegible, (unread_area, boxed_area))
