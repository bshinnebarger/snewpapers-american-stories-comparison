"""Layouts drawn on page scans: American Stories' boxes and snewpapers' boxes, side by side."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from snewpapers_tools.american_stories import ScanName
from snewpapers_tools.loc import LOC_DIR, download_jp2, load_page, page_jp2_urls
from snewpapers_tools.pages import as_page_numbers, page_by_seq
from snewpapers_tools.paths import DATA_DIR

OVERLAY_DIR = DATA_DIR / "overlays"
# American Stories classes and snewpapers label families share one palette (dataviz-validated).
AS_COLORS = {"article": (42, 120, 214), "headline": (235, 104, 52), "author": (232, 123, 164)}
FAMILY_COLORS = {
    "news": (42, 120, 214),
    "opinion": (235, 104, 52),
    "notices": (27, 175, 122),
    "commerce": (237, 161, 0),
    "people": (232, 123, 164),
    "literary": (0, 131, 0),
    "listings": (74, 58, 167),
    "other": (138, 138, 138),  # any label the table below does not name
}
# snewpapers' 32 labels (GET /labels), grouped into the families above.
LABEL_FAMILY = {
    "domestic_news": "news", "foreign_news": "news", "local_news": "news",
    "sports_report": "news", "weather_report": "news",
    "editorial": "opinion", "letter_to_editor": "opinion", "review": "opinion",
    "notice": "notices", "auction": "notices", "reward": "notices", "runaway": "notices", "errata": "notices",
    "commercial": "commerce", "cost_of_goods": "commerce", "personal_ad": "commerce", "shipping": "commerce",
    "birth_announcement": "people", "obituary": "people", "wedding": "people", "society": "people",
    "literary": "literary", "story": "literary", "poem": "literary", "epigraph": "literary",
    "filler": "literary", "recipe": "literary", "crossword_puzzle": "literary",
    "radio_listings": "listings", "tv_listings": "listings", "theater_listings": "listings",
    "train_schedules": "listings",
}
# Reading-order lines cycle through these so neighbouring articles differ.
ORDER_COLORS = [(214, 39, 40), (44, 160, 44), (148, 103, 189), (140, 86, 75), (23, 190, 207), (0, 0, 0)]
GUTTER = 12

# Box interiors are filled with their colour at this alpha (of 255) so overlapping newsprint still reads.
FILL_ALPHA = 56
# Centred labels: dark ink on a near-opaque white backing, sized relative to the page and shrunk to fit.
LABEL_INK = (0x1F, 0x1F, 0x1F)
LABEL_BG_ALPHA = 204  # ~80% of 255
LABEL_MIN_SIZE = 9
LABEL_PAD = (4, 3)  # x, y padding around the text inside its backing box


def family_of(label: str | None) -> str:
    """The label's family name, as in the table above ("other" when the label is unknown or None)."""
    return LABEL_FAMILY.get(label, "other")


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


HALO = (255, 255, 255)


def arrow_head(b: tuple, ux: float, uy: float, width: int = 1) -> list[tuple[float, float]]:
    """The triangle of an arrow head pointing along (ux, uy) with its tip at `b`, sized for a line `width` px wide."""
    length, half = 10 + 3 * (width - 1), 5 + 2 * (width - 1)
    return [b, (b[0] - length * ux + half * uy, b[1] - length * uy - half * ux),
            (b[0] - length * ux - half * uy, b[1] - length * uy + half * ux)]


def _dashed(draw: ImageDraw.ImageDraw, a: tuple, b: tuple, color: tuple, *, dash: int | None = None,
            gap: int | None = None, width: int = 1, halo: bool = False) -> None:
    """A dashed line from `a` to `b` with an arrow head at `b`; `halo` draws it on a white underlay 2 px wider.
    Dashes are 8 px with 6 px gaps, longer for heavy lines (3 and 2 widths) so they still read as dashes."""
    dash = dash if dash is not None else max(8, 3 * width)
    gap = gap if gap is not None else max(6, 2 * width)
    length = math.dist(a, b)
    if length < 1:
        return
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    passes = [(HALO, width + 2, 1), (color, width, 0)] if halo else [(color, width, 0)]
    for fill, w, grow in passes:
        t = 0.0
        while t < length:
            start, end = max(t - grow, 0), min(t + dash + grow, length)
            draw.line([(a[0] + ux * start, a[1] + uy * start), (a[0] + ux * end, a[1] + uy * end)], fill=fill, width=w)
            t += dash + gap
        head = arrow_head(b, ux, uy, width)
        draw.polygon(head, fill=fill, outline=fill if grow else None, width=2 if grow else 1)


def _composite_fills(image: Image.Image, fills: list[tuple[tuple, tuple]]) -> Image.Image:
    """`image` as RGBA, with each (rect, colour) filled at FILL_ALPHA and composited on in one pass.

    Fills are drawn onto one transparent layer in the order given, so where boxes overlap the later
    one simply sits on top; the layer is then alpha-composited onto the (opaque) page once.
    """
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for rect, color in fills:
        draw.rectangle(rect, fill=(*color, FILL_ALPHA))
    return Image.alpha_composite(base, overlay)


def _fit_label_font(text: str, box_w: float, box_h: float, start_size: int) -> ImageFont.ImageFont | None:
    """The largest font from `start_size` down to LABEL_MIN_SIZE that fits `text` plus padding in
    the box; None if even the minimum size does not fit (the label should then be skipped)."""
    pad_x, pad_y = LABEL_PAD
    for size in range(start_size, LABEL_MIN_SIZE - 1, -1):
        font = _font(size)
        left, top, right, bottom = font.getbbox(text, anchor="mm")
        if (right - left) + 2 * pad_x <= box_w and (bottom - top) + 2 * pad_y <= box_h:
            return font
    return None


def _queue_label(labels: list, rect: tuple, text: str, img_width: int) -> None:
    """Append (centre_x, centre_y, font, text) to `labels` for a centred label in `rect`, sized
    relative to `img_width` and shrunk to fit; appends nothing if `text` is empty or never fits."""
    if not text:
        return
    start_size = max(LABEL_MIN_SIZE, img_width // 55)
    font = _fit_label_font(text, rect[2] - rect[0], rect[3] - rect[1], start_size)
    if font is not None:
        labels.append((((rect[0] + rect[2]) / 2), ((rect[1] + rect[3]) / 2), font, text))


def label_backing(cx: float, cy: float, font: ImageFont.ImageFont, text: str) -> tuple[float, float, float, float]:
    """The box of a queued label's white backing: its text centred on (cx, cy), plus LABEL_PAD."""
    left, top, right, bottom = font.getbbox(text, anchor="mm")
    pad_x, pad_y = LABEL_PAD
    return (cx + left - pad_x, cy + top - pad_y, cx + right + pad_x, cy + bottom + pad_y)


def label_boxes(rects_texts: list[tuple[tuple, str]], img_width: int) -> list[tuple[float, float, float, float]]:
    """Where the labels the draw_* functions put in these (rect, text) pairs end up (their backings),
    for drawing something else around them; a label too big for its rect is skipped, as when drawn."""
    queued: list = []
    for rect, text in rects_texts:
        _queue_label(queued, rect, text, img_width)
    return [label_backing(*label) for label in queued]


def _area(rect: tuple) -> float:
    return (rect[2] - rect[0]) * (rect[3] - rect[1])


def queue_labels(label_sets: list[list[tuple[tuple, str]]], img_width: int, *, once: bool = False) -> list:
    """The labels for sets of (rect, text) pairs (one set per component or AS article): every rect's,
    or with `once` only one per set, in the largest of its rects the label fits."""
    queued: list = []
    for pairs in label_sets:
        if not once:
            for rect, text in pairs:
                _queue_label(queued, rect, text, img_width)
            continue
        for rect, text in sorted(pairs, key=lambda p: _area(p[0]), reverse=True):
            before = len(queued)
            _queue_label(queued, rect, text, img_width)
            if len(queued) > before:
                break
    return queued


def american_stories_label_sets(page: dict, rect: Callable) -> list[list[tuple[tuple, str]]]:
    """(rect, class) of each box, one set per article and class (a box in no article on its own);
    `rect(box)` gives a box's rect in image pixels."""
    sets: dict = {}
    owner = {i: a.get("full_article_id", k) for k, a in enumerate(page.get("full articles") or []) for i in a["object_ids"]}
    for box in page["bboxes"]:
        key = (owner[box["id"]], box["class"]) if box["id"] in owner else ("box", box["id"])
        sets.setdefault(key, []).append((rect(box), box["class"]))
    return list(sets.values())


def snewpapers_label_sets(components: list[dict], page: int, width: int, height: int) -> list[list[tuple[tuple, str]]]:
    """(rect in image pixels, label with underscores as spaces) of each rect on `page`, one set per component."""
    sets = []
    for comp in components:
        entry = next((b for b in comp.get("boxes") or [] if b["page"] == page), None)
        if entry is None:
            continue
        text = (comp.get("label") or "").replace("_", " ")
        sets.append([((x0 * width, y0 * height, x1 * width, y1 * height), text) for x0, y0, x1, y1 in entry["rects"]])
    return sets


def _draw_labels(page: Image.Image, labels: list[tuple[float, float, ImageFont.ImageFont, str]]) -> Image.Image:
    """`page` (RGBA) with each label's translucent white backing composited on, then its ink text
    drawn on top, centred on the label's point. A no-op (returns `page` unchanged) when `labels` is empty."""
    if not labels:
        return page
    overlay = Image.new("RGBA", page.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for cx, cy, font, text in labels:
        draw.rounded_rectangle(label_backing(cx, cy, font, text), radius=3, fill=(255, 255, 255, LABEL_BG_ALPHA))
    page = Image.alpha_composite(page, overlay)
    draw = ImageDraw.Draw(page)
    for cx, cy, font, text in labels:
        draw.text((cx, cy), text, fill=LABEL_INK, font=font, anchor="mm")
    return page


def draw_american_stories_page(
    image: Image.Image, page: dict, *, scan_size: tuple[int, int] | None = None, labels: bool = True,
    line_width: int = 1, label_once: bool = False,
) -> Image.Image:
    """A copy of `image` with the page's American Stories boxes drawn at its scale.

    Boxes are in the pixels of the full scan. `scan_size` is that scan's real (width, height),
    read from the JP2; it defaults to the page's own `scan.width/height`, which is exact on most
    pages but only approximate where the JSON says `dimensions_approx` (older pages: the stated
    size is smaller than the scan, so scaling by it stretches every box).

    Each box's interior is filled with its class colour at FILL_ALPHA and outlined solidly in the
    same colour; a dashed line runs under the labels through each article's boxes in reading order.
    Each box is then labelled with its class name, centred, shrunk to fit or skipped if too small
    (unless `labels` is False: close-up crops leave them off so they hide no text); `label_once`
    labels each article's boxes of a class once, in the largest. A `line_width` above 1 draws the
    reading-order lines that heavy, on a white halo.
    """
    scan_w, scan_h = scan_size or (page["scan"]["width"], page["scan"]["height"])
    sx, sy = image.width / scan_w, image.height / scan_h

    def rect(box):
        b = box["bbox"]
        return (b["x0"] * sx, b["y0"] * sy, b["x1"] * sx, b["y1"] * sy)

    def color_of(box):
        return AS_COLORS.get(box["class"], (128, 128, 128))

    out = _composite_fills(image, [(rect(box), color_of(box)) for box in page["bboxes"]])
    draw = ImageDraw.Draw(out)
    for box in page["bboxes"]:
        draw.rectangle(rect(box), outline=color_of(box), width=max(2, image.width // 400))

    boxes = {b["id"]: b for b in page["bboxes"]}
    for k, article in enumerate(page["full articles"]):
        ordered = sorted((boxes[i] for i in article["object_ids"] if i in boxes), key=lambda b: b["reading_order_id"])
        centres = [((r[0] + r[2]) / 2, (r[1] + r[3]) / 2) for r in map(rect, ordered)]
        for a, b in zip(centres, centres[1:]):
            _dashed(draw, a, b, ORDER_COLORS[k % len(ORDER_COLORS)], width=line_width, halo=line_width > 1)

    queued = queue_labels(american_stories_label_sets(page, rect), image.width, once=label_once) if labels else []
    out = _draw_labels(out, queued)
    return out.convert("RGB")


def draw_snewpapers_page(image: Image.Image, components: list[dict], page: int, *, labels: bool = True,
                         line_width: int = 1, label_once: bool = False) -> Image.Image:
    """A copy of `image` with snewpapers' boxes for `page` drawn at its scale.

    Each component's `boxes` entry for this page (if any) holds its rects as fractions (0-1) of the
    page image, in reading order (the API lists them so). Rects are filled and outlined in the component's label family
    colour; every rect is labelled with the component's label (underscores shown as spaces),
    centred, shrunk to fit or skipped if too small. A dashed line runs through the rect centres, in
    reading order, when a component has more than one rect on this page (a story in two columns,
    say). Components with no boxes entry for this page are skipped. `labels=False` leaves the labels off,
    `label_once` puts each component's in its largest rect only; a `line_width` above 1 draws the
    dashed lines that heavy, on a white halo.
    """
    rects_colors = []  # (rect, color) per rect, for the fill layer
    groups = []  # (rects, color) per component, for outlines, labels and the dashed line
    for comp in components:
        entry = next((b for b in comp.get("boxes") or [] if b["page"] == page), None)
        if entry is None:
            continue
        color = FAMILY_COLORS[family_of(comp.get("label"))]
        rects = [(x0 * image.width, y0 * image.height, x1 * image.width, y1 * image.height)
                 for x0, y0, x1, y1 in entry["rects"]]
        rects_colors.extend((r, color) for r in rects)
        groups.append((rects, color, (comp.get("label") or "").replace("_", " ")))

    out = _composite_fills(image, rects_colors)
    draw = ImageDraw.Draw(out)
    for rects, color, _label in groups:
        for r in rects:
            draw.rectangle(r, outline=color, width=max(2, image.width // 400))
        centres = [((r[0] + r[2]) / 2, (r[1] + r[3]) / 2) for r in rects]
        for a, b in zip(centres, centres[1:]):
            _dashed(draw, a, b, color, width=line_width, halo=line_width > 1)

    label_sets = snewpapers_label_sets(components, page, image.width, image.height) if labels else []
    out = _draw_labels(out, queue_labels(label_sets, image.width, once=label_once))
    return out.convert("RGB")


def tile_pages(pages: dict[int, Image.Image], *, columns: int = 4, title: str = "") -> Image.Image:
    """Pages left to right, `columns` per row, each labelled, under a title and a colour key."""
    numbers = sorted(pages)
    cell_w = max(p.width for p in pages.values())
    cell_h = max(p.height for p in pages.values())
    rows = math.ceil(len(numbers) / columns)
    cols = min(columns, len(numbers))
    label_h, header_h = 28, 56
    sheet = Image.new("RGB", (cols * cell_w + (cols + 1) * GUTTER, header_h + rows * (cell_h + label_h + GUTTER) + GUTTER), "white")
    draw = ImageDraw.Draw(sheet)
    font, small = _font(22), _font(16)
    draw.text((GUTTER, 8), title, fill=(0, 0, 0), font=font)
    x = GUTTER
    for name, color in AS_COLORS.items():
        draw.rectangle((x, 38, x + 14, 52), outline=color, width=3)
        draw.text((x + 20, 36), name, fill=(0, 0, 0), font=small)
        x += 30 + int(draw.textlength(name, font=small))
    for i, number in enumerate(numbers):
        row, col = divmod(i, columns)
        left = GUTTER + col * (cell_w + GUTTER)
        top = header_h + row * (cell_h + label_h + GUTTER)
        draw.text((left, top + 4), f"page {number}", fill=(0, 0, 0), font=small)
        sheet.paste(pages[number], (left, top + label_h))
    return sheet


def _swatch_legend(draw: ImageDraw.ImageDraw, at: tuple[int, int], heading: str, colors: dict, font) -> int:
    """One legend line: the heading, then an outlined swatch and name per colour; returns the x it ends at."""
    x, y = at
    draw.text((x, y - 2), heading, fill=(0, 0, 0), font=font)
    x += int(draw.textlength(heading, font=font)) + 10
    for name, color in colors.items():
        draw.rectangle((x, y, x + 14, y + 14), outline=color, width=3)
        draw.text((x + 20, y - 2), name, fill=(0, 0, 0), font=font)
        x += 34 + int(draw.textlength(name, font=font))
    return x


@dataclass(frozen=True)
class SheetLayout:
    """Where a two-row sheet puts things: page columns left to right (GUTTER apart), a header on top,
    then two rows, each a caption strip `label_h` tall over the page images."""

    numbers: tuple[int, ...]
    cell_w: int
    cell_h: int
    header_h: int
    label_h: int = 28

    @property
    def row_h(self) -> int:
        return self.label_h + self.cell_h + GUTTER

    @property
    def size(self) -> tuple[int, int]:
        n = len(self.numbers)
        return n * self.cell_w + (n + 1) * GUTTER, self.header_h + 2 * self.row_h + GUTTER

    def left(self, number: int) -> int:
        """The x of page `number`'s column."""
        return GUTTER + self.numbers.index(number) * (self.cell_w + GUTTER)

    def caption_top(self, row: int) -> int:
        """The y of row `row`'s caption strip (row 0 on top)."""
        return self.header_h + row * self.row_h

    def origin(self, row: int, number: int) -> tuple[int, int]:
        """The top-left corner of page `number`'s image in row `row`."""
        return self.left(number), self.caption_top(row) + self.label_h

    @classmethod
    def fit(cls, top: dict[int, Image.Image], bottom: dict[int, Image.Image], *, header_h: int,
            label_h: int = 28) -> "SheetLayout":
        """A layout with a column per page number in either row, each cell as big as the biggest image."""
        images = [*top.values(), *bottom.values()]
        return cls(tuple(sorted(set(top) | set(bottom))), max(img.width for img in images),
                   max(img.height for img in images), header_h, label_h)


def two_row_canvas(top: dict[int, Image.Image], bottom: dict[int, Image.Image], layout: SheetLayout) -> Image.Image:
    """A white sheet of `layout`'s size with `top`'s pages pasted in row 0 and `bottom`'s in row 1
    (a page missing from a row leaves that cell blank); header and captions are left to the caller."""
    sheet = Image.new("RGB", layout.size, "white")
    for row, pages in enumerate((top, bottom)):
        for number, image in pages.items():
            sheet.paste(image, layout.origin(row, number))
    return sheet


def two_row_sheet(top: dict[int, Image.Image], bottom: dict[int, Image.Image], *, title: str) -> Image.Image:
    """Page columns aligned left to right: `top`'s pages, then `bottom`'s, under a title and two legend lines.

    A page number missing from one row (no scan, or no boxes to overlay) leaves that cell blank.
    """
    legend_h = 24
    layout = SheetLayout.fit(top, bottom, header_h=36 + 2 * legend_h)
    sheet = two_row_canvas(top, bottom, layout)
    draw = ImageDraw.Draw(sheet)
    font, small = _font(22), _font(16)
    draw.text((GUTTER, 8), title, fill=(0, 0, 0), font=font)
    _swatch_legend(draw, (GUTTER, 8 + legend_h + 4), "American Stories (row 1):", AS_COLORS, small)
    _swatch_legend(draw, (GUTTER, 8 + 2 * legend_h + 4), "snewpapers (row 2):", FAMILY_COLORS, small)
    for row in (0, 1):
        for number in layout.numbers:
            draw.text((layout.left(number), layout.caption_top(row) + 4), f"page {number}", fill=(0, 0, 0), font=small)
    return sheet


def components_by_page(components: list[dict]) -> pd.DataFrame:
    """Our components listed under every page they sit on, in issue order (until the API serves boxes)."""
    rows = []
    for comp in components:
        pages = comp.get("pages") or [None]
        span = str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"
        opening = " ".join((comp.get("content") or "").split())[:80]
        for page in pages:
            rows.append({"page": page, "label": comp.get("label"), "chars": comp.get("char_count"),
                         "spans_pages": span, "opening": opening})
    table = pd.DataFrame(rows, columns=["page", "label", "chars", "spans_pages", "opening"])
    return table.sort_values("page", kind="stable", key=lambda s: s.fillna(10**6)).reset_index(drop=True)


def issue_jp2s(pair, *, download: Callable = download_jp2) -> dict[int, tuple[Path, dict]]:
    """Download the issue's JP2s (once): page -> (JP2 path, American Stories page data).

    Pages are numbered by sequence (see pages.py), so scans whose names all say p1 keep their own pages.
    """
    issue = pair.issue
    urls = page_jp2_urls(pair.components, [scan for scan, _ in pair.pages])
    folder = LOC_DIR / issue.lccn / issue.date / f"ed-{issue.edition}"
    by_seq = page_by_seq(pair.components)
    pages, _ = as_page_numbers([scan for scan, _ in pair.pages], by_seq)
    found = {}
    for scan, data in pair.pages:
        page = pages.get(scan.seq)
        if page in urls:
            found[page] = (download(urls[page], folder), data)
    return found


def jp2_size(path: Path) -> tuple[int, int]:
    """The scan's real (width, height), from its header only."""
    with Image.open(path) as header:
        return header.size


def issue_scan_sizes(pair, *, download: Callable = download_jp2) -> dict[int, tuple[int, int]]:
    """page -> the real (width, height) of its scan, downloading the JP2s once but decoding none."""
    return {page: jp2_size(jp2) for page, (jp2, _data) in issue_jp2s(pair, download=download).items()}


def load_issue_pages(pair, width: int, *, download: Callable = download_jp2) -> dict[int, tuple[Image.Image, dict, tuple[int, int]]]:
    """Download the issue's JP2s (once) and load each: page -> (image at `width`, AS page data, real scan size)."""
    return {
        page: (load_page(jp2, width=width), data, jp2_size(jp2))
        for page, (jp2, data) in issue_jp2s(pair, download=download).items()
    }


def american_stories_overlay(pair, *, width: int = 700, columns: int = 4) -> tuple[Image.Image, Path]:
    """Download the issue's JP2s (once), draw American Stories' layout on them, save a WebP; returns both."""
    issue = pair.issue
    loaded = load_issue_pages(pair, width)
    if not loaded:
        raise ValueError("No page scans could be located for this issue (no LoC link on its components).")
    drawn = {page: draw_american_stories_page(image, data, scan_size=scan_size)
             for page, (image, data, scan_size) in loaded.items()}
    title = f"American Stories layout: {issue.lccn} {issue.date} ed-{issue.edition}"
    sheet = tile_pages(drawn, columns=columns, title=title)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    path = OVERLAY_DIR / f"{issue.lccn}_{issue.date}_ed-{issue.edition}_american_stories.webp"
    sheet.save(path, "WEBP", quality=80)
    return sheet, path


def issue_overlay(pair, *, width: int = 700) -> tuple[Image.Image, Path]:
    """Both layouts on the same scans, row 1 American Stories and row 2 snewpapers; save a WebP.

    Reuses the same loaded page images as `american_stories_overlay`, so rows 1 and 2 are the same
    pages at the same scale. snewpapers' boxes come straight off each component (fetch_both already
    carried them over from the browse hit); a component with no boxes for a page is left out of it.
    """
    issue = pair.issue
    loaded = load_issue_pages(pair, width)
    if not loaded:
        raise ValueError("No page scans could be located for this issue (no LoC link on its components).")
    top = {page: draw_american_stories_page(image, data, scan_size=scan_size)
           for page, (image, data, scan_size) in loaded.items()}
    bottom = {page: draw_snewpapers_page(image, pair.components, page)
              for page, (image, _data, _scan_size) in loaded.items()}
    title = f"{issue.lccn} {issue.date} ed-{issue.edition}: American Stories vs snewpapers"
    sheet = two_row_sheet(top, bottom, title=title)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    path = OVERLAY_DIR / f"{issue.lccn}_{issue.date}_ed-{issue.edition}_compare.webp"
    sheet.save(path, "WEBP", quality=80)
    return sheet, path
