"""One HTML report per issue: snewpapers' segmentation against American Stories', group by group.

Each group (see segmentation.py) is shown as the same crop of the page scan twice, left with
snewpapers' boxes and right with American Stories', the group's own boxes outlined thicker: on the
American Stories side always AS's real boxes, never the slices our boundaries cut them into. The
crop is our component's area for the cases anchored on one, the AS article's for an as_merge, and
the left-over boxes' for AS only. Under an as_merge, the slices of the merged article that its
pieces hold are listed with their text metrics, as the text agreement inside AS's merged article;
each such piece's own group (`in_as_merge`) points to the merge that decides it. Beneath them both texts with a word diff (for an as_split the AS side
is only the text of the slices of AS boxes inside our component) and, for a dispute, the numbered pieces as
the model is shown them. Text American Stories attaches outside one of our components (an AS
article of the component's running beyond it) is shown with the group, outside the comparison:
"American Stories also attaches: «…» (outside this component)". Given the LLM's verdicts (see
adjudicate.py), a scorecard goes at the top and each group shows its verdict and the model's reasons.

A component of ours running over pages is one group across them (segmentation's `link_pages`),
listed on its first page with a crop pair per page and its AS pieces labelled by page; each later
page lists a `continued` group pointing to it ("continues p2-3"), with that page's crops.

By default crops are written as sidecar WebP files next to the HTML and referenced by relative path,
lazy-loaded, so the page itself stays small; pass `embed=True` for the old one-file-stands-alone
behaviour (crops inlined as base64 WebP), handy for somewhere like Jupyter's output area that only
shows the one file.
"""
from __future__ import annotations

import base64
import difflib
import html
import io
import re
import statistics
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

from snewpapers_tools.adjudicate import (
    AMERICAN_STORIES,
    AS_DUPLICATED,
    COMPONENT_OUTCOMES,
    DISPUTED,
    EXCLUDED,
    FRAGMENT_ARTICLES,
    FURNITURE,
    ILLEGIBLE_UNSURE,
    ITEMS_MERGED,
    LINES_ELSEWHERE,
    MISSED,
    PARTIAL_ARTICLES,
    WIDER_ARTICLES,
    MISSING_COMPONENTS,
    MISSING_PAGES,
    OUTCOME_LABELS,
    NEITHER,
    NO_VERDICT,
    NOT_CAPTURED,
    PARTLY_CAPTURED,
    TINY,
    EMPTY,
    UNSTABLE,
    SIDES,
    SNEWPAPERS,
    GroupVerdict,
    edges,
    many_pieces,
    scorecard,
    usage_text,
)
from snewpapers_tools.american_stories import REVISION, REPO_ID
from snewpapers_tools.overlay import (
    AS_COLORS,
    FAMILY_COLORS,
    draw_american_stories_page,
    draw_snewpapers_page,
    load_issue_pages,
)
from snewpapers_tools.paths import DATA_DIR
from snewpapers_tools.text_quality import real_word_rate, word_counts
from snewpapers_tools.segmentation import (
    ANCHORED,
    AS_MERGE,
    AS_ONLY,
    AS_ONLY_TINY,
    AS_ONLY_EMPTY,
    AS_PARTIAL,
    AS_SPLIT,
    AS_WIDER,
    CASE_LABELS,
    CASES,
    IN_AS_MERGE,
    CONTINUED,
    MATCH,
    OURS_ONLY,
    Group,
    IssueSegmentation,
    Rect,
)

REPORTS_DIR = DATA_DIR / "reports"
PAGE_WIDTH = 2400  # px the scans are loaded at before cropping: a column is ~350 px wide at this size
CROP_MARGIN = 0.04  # of the page, around the group's area
MAX_CROP_HEIGHT = 1400  # px; taller crops are scaled down to this
WEBP_QUALITY = 70
HIGHLIGHT = (214, 39, 40)
HIGHLIGHT_WIDTH = 6  # px at PAGE_WIDTH

CASE_HELP = {
    MATCH: "the AS text inside our component (boxes, or the slices of boxes it overlaps) all belongs to one AS article "
           "(or to several that duplicate one another: American Stories duplicated the text)",
    AS_SPLIT: "the AS text inside our component belongs to several AS articles: one component or several?",
    AS_MERGE: "two or more of our components lie inside one AS article (80% or more of each, or 50% or more with "
              "a slice of the article): one component or several?",
    IN_AS_MERGE: "all the AS text inside our component is a slice of an AS article that holds other components of ours "
                 "too: no agreement (the slice is cut along our boundaries, not AS's); decided by that N ours : 1 AS",
    CONTINUED: "a later page of a component of ours running over pages: compared, asked and decided once, across "
               "its pages, in the group on its first page",
    OURS_ONLY: "no AS box overlaps our component enough to give it a slice: not captured by American Stories",
    AS_WIDER: "one AS article holds our component, but runs on beyond it into other text (our component covers under "
              "90% of the article's text, measured line by line): the text it runs into is asked about, and decides it",
    AS_PARTIAL: "the AS text inside our component all belongs to one AS article, but AS has text for under 90% of "
                "our component, measured line by line (the rest is in no AS box, or AS marked it illegible): partly "
                "captured by American Stories, not asked; its AS article counts as an AS error",
    AS_ONLY: "AS text inside none of our components (one AS article's): left out on purpose, or missed?",
    AS_ONLY_TINY: "AS text inside none of our components, under 20 characters: not asked",
    AS_ONLY_EMPTY: "an AS article inside none of our components with no text at all (AS boxed it but read nothing, "
                   "most often marking it illegible): nothing to ask, counted for neither side",
}
SUMMARY_NAMES = {
    "ours_covered_by_as": "snewpapers covered by AS (area)",
    "as_covered_by_ours": "AS covered by ours (area)",
    "median_match_ratio": "median text ratio over matches",
}

CSS = """
.seg-report { font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: #1f1f1f;
  background: #fafaf7; max-width: 1200px; margin: 0 auto; padding: 16px; }
.seg-report h1 { font-size: 1.5em; margin: 0 0 .3em; }
.seg-report h2 { font-size: 1.25em; margin: 1.6em 0 .6em; padding-bottom: .2em; border-bottom: 2px solid #ddd; }
.seg-report a { color: #2a5db0; }
.seg-report table { border-collapse: collapse; margin: .6em 0; }
.seg-report td, .seg-report th { padding: 2px 12px 2px 0; text-align: left; vertical-align: top; }
.seg-report th { font-weight: 600; color: #555; }
.seg-report .legend { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: .4em 0; font-size: .9em; }
.seg-report .swatch { display: inline-block; width: 12px; height: 12px; border: 3px solid; margin-right: 4px; vertical-align: -2px; }
.seg-report details.group { background: #fff; border: 1px solid #ddd; border-radius: 6px; margin: 10px 0; padding: 6px 12px; }
.seg-report details.group > summary { cursor: pointer; padding: 4px 0; }
.seg-report .badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: .85em; font-weight: 600; margin-right: 8px; color: #fff; }
.seg-report .badge.match { background: #2e7d32; }
.seg-report .badge.as-split { background: #6a3fb5; }
.seg-report .badge.as-merge { background: #b0306a; }
.seg-report .badge.in-as-merge { background: #d58bb0; }
.seg-report .badge.continued { background: #2a78d6; }
.seg-report .inside-merge { font-size: .9em; margin: 8px 0 4px; padding: 4px 8px; border-left: 3px solid #b0306a; background: #fbf3f7; }
.seg-report .inside-merge .texts { margin: 4px 0 10px; }
.seg-report .badge.ours-only { background: #c62828; }
.seg-report .badge.as-only { background: #757575; }
.seg-report .badge.as-only-tiny { background: #a8a8a8; }
.seg-report .badge.as-only-empty { background: #c4c4c4; }
.seg-report .badge.as-partial { background: #2a78d6; }
.seg-report .badge.as-wider { background: #8a5a00; }
.seg-report .meta { color: #555; font-size: .9em; }
.seg-report .gid { font-weight: 700; font-family: ui-monospace, monospace; font-size: .9em; }
.seg-report .crops { display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0; }
.seg-report figure { margin: 0; }
.seg-report figure img { display: block; max-width: 100%; border: 1px solid #ccc; }
.seg-report figcaption { font-size: .85em; color: #555; }
.seg-report .texts { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.seg-report .text { font-size: .9em; max-height: 22em; overflow: auto; background: #f6f6f2; padding: 6px 8px; border-radius: 4px; }
.seg-report .text h4 { margin: 0 0 4px; font-size: .95em; color: #555; }
.seg-report .only-ours { background: #ffe08a; }
.seg-report .only-as { background: #b9dcff; }
.seg-report ul.evidence { font-size: .88em; color: #333; margin: 8px 0 4px; padding-left: 1.2em; }
.seg-report .note { color: #8a4b00; }
.seg-report .attached { font-size: .9em; margin: 8px 0 4px; padding: 4px 8px; border-left: 3px solid #2a5db0; background: #f3f7fd; }
.seg-report .verdict { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: .85em; font-weight: 600;
  margin-right: 8px; border: 2px solid #888; color: #333; background: #fff; }
.seg-report .verdict.snewpapers, .seg-report .verdict.correctly-not-served, .seg-report .verdict.match { border-color: #2e7d32; color: #2e7d32; }
.seg-report .verdict.american-stories, .seg-report .verdict.missed-by-snewpapers { border-color: #2a5db0; color: #2a5db0; }
.seg-report .verdict.unstable, .seg-report .verdict.no-verdict { border-style: dashed; color: #8a4b00; border-color: #8a4b00; }
.seg-report .scoreboard { display: flex; flex-wrap: wrap; gap: 8px 28px; }
.seg-report .score-block h3 { font-size: 1em; margin: .8em 0 .2em; }
.seg-report .score-block .meta { max-width: 36em; margin: .2em 0; }
.seg-report table.scorecard { background: #fff; border: 1px solid #ddd; }
.seg-report table.scorecard th, .seg-report table.scorecard td { padding: 4px 10px; border-bottom: 1px solid #eee; }
.seg-report table.scorecard td { text-align: right; font-variant-numeric: tabular-nums; }
.seg-report table.scorecard tr:first-child th { font-size: .85em; text-align: right; border-bottom: 2px solid #ddd; }
.seg-report details.method { font-size: .9em; color: #444; margin: .6em 0; }
.seg-report details.method summary { cursor: pointer; color: #2a5db0; }
.seg-report ul.reasons { font-size: .9em; margin: 8px 0 4px; padding-left: 1.2em; }
.seg-report ul.reasons pre { white-space: pre-wrap; font-size: .85em; background: #f6f6f2; padding: 4px 6px; max-height: 12em; overflow: auto; }
@media (max-width: 700px) { .seg-report .texts { grid-template-columns: 1fr; } }
"""

# A link to a group (#g-p2-4) opens it: groups start collapsed.
OPEN_LINKED = """<script>
function openLinked() {
  var target = location.hash && document.getElementById(location.hash.slice(1));
  if (target && target.tagName === "DETAILS") { target.open = true; target.scrollIntoView(); }
}
window.addEventListener("hashchange", openLinked);
openLinked();
</script>"""

# --- word diff --------------------------------------------------------------------------------


def _words(text: str) -> list[tuple[str, bool]]:
    """(word, opens a new paragraph) for each whitespace-separated word."""
    words, previous_end = [], 0
    for m in re.finditer(r"\S+", text):
        words.append((m.group(), bool(re.search(r"\n\s*\n", text[previous_end:m.start()])) and bool(words)))
        previous_end = m.end()
    return words


def _key(word: str) -> str:
    """What two words are compared by: lowercase, punctuation dropped."""
    return re.sub(r"[^\w]", "", word.lower())


def _render(words: list[tuple[str, bool]], marked: set[int], css: str) -> str:
    """The words escaped and joined, each run of marked words in one span, paragraph breaks kept."""
    parts, k = [], 0
    while k < len(words):
        if k:
            parts.append("<br><br>" if words[k][1] else " ")
        if k not in marked:
            parts.append(html.escape(words[k][0], quote=False))
            k += 1
            continue
        run = [html.escape(words[k][0], quote=False)]
        while k + 1 < len(words) and k + 1 in marked and not words[k + 1][1]:
            k += 1
            run.append(html.escape(words[k][0], quote=False))
        parts.append(f'<span class="{css}">{" ".join(run)}</span>')
        k += 1
    return "".join(parts)


def word_diff_html(ours: str, theirs: str) -> tuple[str, str]:
    """Both texts as HTML, words only in ours in <span class="only-ours">, only in AS in <span class="only-as">.

    Words are compared lowercase without punctuation, aligned with difflib; a replaced run is marked
    on both sides. With one side empty the other is left unmarked (there is nothing to compare).
    """
    a, b = _words(ours), _words(theirs)
    if not a or not b:
        return _render(a, set(), "only-ours"), _render(b, set(), "only-as")
    matcher = difflib.SequenceMatcher(None, [_key(w) for w, _ in a], [_key(w) for w, _ in b], autojunk=False)
    only_a, only_b = set(), set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            only_a.update(range(i1, i2))
        if tag in ("insert", "replace"):
            only_b.update(range(j1, j2))
    return _render(a, only_a, "only-ours"), _render(b, only_b, "only-as")


# --- crops ------------------------------------------------------------------------------------


def crop_box(bbox: Rect, margin: float = CROP_MARGIN) -> Rect:
    """`bbox` grown by `margin` (page fractions) on every side, clipped to the page."""
    x0, y0, x1, y1 = bbox
    clip = lambda v: round(min(max(v, 0.0), 1.0), 6)  # noqa: E731
    return clip(x0 - margin), clip(y0 - margin), clip(x1 + margin), clip(y1 + margin)


def crop_group(drawn: Image.Image, box: Rect, own: tuple[Rect, ...], crop_width: int) -> Image.Image:
    """The `box` of a drawn page with the group's `own` rects outlined thicker, scaled to fit
    `crop_width` and MAX_CROP_HEIGHT (never enlarged)."""
    w, h = drawn.size
    left, top = min(round(box[0] * w), w - 1), min(round(box[1] * h), h - 1)
    crop = drawn.crop((left, top, max(round(box[2] * w), left + 1), max(round(box[3] * h), top + 1)))  # never empty
    draw = ImageDraw.Draw(crop)
    width = max(2, round(HIGHLIGHT_WIDTH * w / PAGE_WIDTH))
    for x0, y0, x1, y1 in own:
        draw.rectangle((x0 * w - left, y0 * h - top, x1 * w - left, y1 * h - top), outline=HIGHLIGHT, width=width)
    scale = min(1.0, crop_width / crop.width, MAX_CROP_HEIGHT / crop.height)
    if scale < 1.0:
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
    return crop


def webp_data_uri(image: Image.Image, quality: int = WEBP_QUALITY) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "WEBP", quality=quality)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# --- HTML -------------------------------------------------------------------------------------


def _css_case(case: str) -> str:
    return case.replace("_", "-")


def _pages(pages) -> str:
    return "p. " + "–".join(map(str, (pages[0], pages[-1]))) if len(pages) > 1 else f"p. {pages[0]}"


METRIC_NAMES = {"ratio": "ratio", "token_set_ratio": "token set", "shared_phrases": "shared 4-word phrases"}


def _metric(key: str, value: float) -> str:
    """One text or area metric as shown: "ratio 86.7", "AS covers 85%", "IoU 0.95"."""
    if key == "as_cover":
        return f"AS covers {value:.0%}"
    if key == "illegible_cover":
        return f"AS marked {value:.0%} illegible"
    if key == "ours_cover":
        return f"ours covers {value:.0%} of AS's"
    if key == "iou":
        return f"IoU {value:.2f}"
    return f"{METRIC_NAMES[key]} {value:.1f}"


def _metrics_line(group: Group) -> str:
    parts = []
    if group.cross_page:
        parts.append(f"one component across {_pages(group.pages)}")
    if group.case in DISPUTED:
        parts.append(f"{len({p.uuid for p in group.pieces})} snewpapers : {len(group.parts)} AS")
    parts += [_metric(k, v) for k, v in group.metrics.items()]
    ours, theirs = real_word_rate(group.ours_text), real_word_rate(group.as_text)
    if group.case != CONTINUED and ours is not None and theirs is not None:
        parts.append(f"real words: snewpapers {ours:.0%}, AS {theirs:.0%}")
    if group.other_pages and group.case != CONTINUED:
        parts.append("continues on p. " + ", ".join(map(str, group.other_pages)))
    return " · ".join(parts)


def _and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def duplicated_note(group: Group) -> str:
    """"American Stories duplicated this text (articles #x and #y)", or "" when it did not (text it
    attaches outside our component and also has elsewhere is noted with the attached text)."""
    if not group.duplicates:
        return ""
    found = "; ".join(_and(list(names)) for names in group.duplicates)
    return f"American Stories duplicated this text (articles {found})"


def _part_name(group: Group, part) -> str:
    """An AS part's name, with its page in a group across pages ("AS #4 (p. 3)")."""
    return part.name + (f" (p. {part.page})" if group.cross_page else "")


def _who(group: Group, site_urls: dict[str, str]) -> str:
    ours = [
        f'<a href="{html.escape(site_urls[p.uuid])}">{html.escape(p.label or "?")}</a>' if p.uuid in site_urls
        else html.escape(p.label or "?")
        for p in {p.uuid: p for p in group.pieces}.values()
    ]
    theirs = [_part_name(group, part) for part in group.parts]
    return html.escape(", ".join(theirs)) if not ours else ", ".join(ours) + (
        " ↔ " + html.escape(", ".join(theirs)) if theirs else "")


def _crop_src(image: Image.Image, name: str, crops_dir: Path | None) -> str:
    """The crop's `src` attribute: a sidecar file written under `crops_dir` (relative path), or a
    base64 data URI when `crops_dir` is None (the self-contained `embed=True` report)."""
    if crops_dir is None:
        return webp_data_uri(image)
    image.save(crops_dir / name, "WEBP", quality=WEBP_QUALITY)
    return f"crops/{name}"


SIDE_NAMES = {SNEWPAPERS: "snewpapers", AMERICAN_STORIES: "American Stories"}
RAW_SHOWN = 2000  # characters of a no-verdict response shown in the report


def _missing_label(group: Group, label: str) -> str:
    """For a component AS lacks text for (as_partial, ours_only): what AS has and why the rest is missing."""
    why = "marked illegible" if group.as_illegible else "in no AS box"
    if group.case == OURS_ONLY:
        return f"{label} ({why})" if group.as_illegible else label
    if all(b.cls == "headline" for part in group.parts for b in part.boxes if b.text.strip()):
        return f"AS has the headline only (body {why})"
    return f"{label} (the rest {why})"


def _verdict_badge(verdict: GroupVerdict | None, group: Group | None = None) -> str:
    if verdict is None:
        return ""
    label = OUTCOME_LABELS[verdict.outcome]
    if group is not None and group.case in (AS_PARTIAL, OURS_ONLY):
        label = _missing_label(group, label)
    return f'<span class="verdict {_css_case(verdict.outcome).replace(" ", "-").lower()}">{html.escape(label)}</span>'


def _reasons_html(verdict: GroupVerdict | None) -> str:
    """Each question's answer: for a dispute, which belief was stated first; what it said, and its
    reason, or the raw response when there was no verdict."""
    if verdict is None or not verdict.asked:
        return ""
    items = []
    for q in verdict.asked:
        if q.one_side is not None:
            belief = "one component" if q.a_side == q.one_side else "several components"
            asked = f"stated first: {belief} ({SIDE_NAMES[q.a_side]}): "
        else:
            asked = ""
        parsed = q.answer.parsed
        if parsed is None:
            raw = q.answer.raw if len(q.answer.raw) <= RAW_SHOWN else q.answer.raw[:RAW_SHOWN] + " …"
            items.append(f"<li>{html.escape(asked)}<b>no verdict</b><pre>{html.escape(raw or '(empty response)')}</pre></li>")
            continue
        if verdict.case in DISPUTED and q.one_side is None:  # our component alone: AS could not read part
            asked = "asked whether snewpapers' component is one item (AS marked part of it illegible): "
            said = f"said {parsed['verdict']}"
        elif verdict.case in DISPUTED:
            said = OUTCOME_LABELS[q.verdict] + f" (said {parsed['verdict']})"
        else:
            said = {"advertisement": "advertisement", "terms": "subscription terms or colophon",
                    "furniture": "page furniture", "other": "other content"}[parsed["kind"]]
        items.append(f"<li>{html.escape(asked)}<b>{html.escape(said)}</b>: {html.escape(parsed['reason'])}</li>")
    return '<ul class="reasons">' + "".join(items) + "</ul>"


ATTACHED_BELIEFS = {AMERICAN_STORIES: "the component includes it", SNEWPAPERS: "the component does not include it"}
NIBBLE_BELIEFS = {SNEWPAPERS: "the component includes it", AMERICAN_STORIES: "the component does not include it"}


def _attached_html(group: Group, verdict: GroupVerdict | None) -> str:
    """The text American Stories attaches outside our component, the boundary question's verdict
    and each answer (which belief was stated first, what it said, its reason)."""
    if not group.as_attached_outside:
        return ""
    note = ""
    if group.attached_duplicates:
        found = "; ".join(f"{other} also holds the text of {part}" for other, part in group.attached_duplicates)
        note = f'<span class="note">American Stories duplicated it: {html.escape(found)}</span>'
    return _boundary_html("American Stories also attaches", "outside this component", group.as_attached_outside,
                          verdict.attached_outcome if verdict else None, verdict.attached_asked if verdict else [],
                          ATTACHED_BELIEFS, note)


def _nibble_html(group: Group, verdict: GroupVerdict | None) -> str:
    """A line or two of our component American Stories has in another article, the question's
    verdict and each answer."""
    if not group.as_nibbles:
        return ""
    return _boundary_html("American Stories puts part of this component in another article", "inside this component",
                          group.as_nibbles, verdict.nibble_outcome if verdict else None,
                          verdict.nibble_asked if verdict else [], NIBBLE_BELIEFS)


def _boundary_html(lead: str, where: str, parts: list, outcome: str | None, questions: list, beliefs: dict,
                   note: str = "") -> str:
    """A boundary question's text (`parts`), where it is, its verdict and each answer (which belief
    was stated first, what it said, its reason)."""
    names = _and([part.short_name for part in parts])
    text = "\n\n".join(part.text for part in parts)
    badge = ""
    if outcome is not None:
        badge = f' <span class="verdict {_css_case(outcome).replace(" ", "-").lower()}">{html.escape(OUTCOME_LABELS[outcome])}</span>'
    lines = [f"{html.escape(lead)}: «{html.escape(text.strip())}» ({where}; AS article {html.escape(names)}){badge}"]
    if note:
        lines.append(note)
    items = []
    for q in questions:
        asked = f"stated first: {beliefs[q.a_side]} ({SIDE_NAMES[q.a_side]}): "
        parsed = q.answer.parsed
        if parsed is None:
            raw = q.answer.raw if len(q.answer.raw) <= RAW_SHOWN else q.answer.raw[:RAW_SHOWN] + " …"
            items.append(f"<li>{html.escape(asked)}<b>no verdict</b><pre>{html.escape(raw or '(empty response)')}</pre></li>")
        else:
            said = OUTCOME_LABELS[q.verdict] + f" (said {parsed['verdict']})"
            items.append(f"<li>{html.escape(asked)}<b>{html.escape(said)}</b>: {html.escape(parsed['reason'])}</li>")
    reasons = '<ul class="reasons">' + "".join(items) + "</ul>" if items else ""
    return '<div class="attached">' + "<br>".join(lines) + reasons + "</div>"


def _inside_merge_html(group: Group, inside: list[tuple[str, Group]]) -> str:
    """Under an as_merge, the slices of the merged AS article its pieces hold (`inside`: each
    in_as_merge group with its id), each with its text metrics and word diff: the text agreement
    inside AS's merged article, which is no agreement on where it begins and ends."""
    if not inside:
        return ""
    items = []
    for gid, g in inside:
        metrics = " · ".join(_metric(k, v) for k, v in g.metrics.items())
        ours_html, as_html = word_diff_html(g.ours_text, g.as_text)
        items.append(f"<div><b>{html.escape(gid)}</b> {html.escape(g.pieces[0].label or '?')}"
                     f' <span class="meta">{html.escape(metrics)}</span>'
                     '<div class="texts">'
                     f'<div class="text"><h4>snewpapers</h4>{ours_html or "<em>nothing</em>"}</div>'
                     f'<div class="text"><h4>American Stories (slice of {html.escape(g.parts[0].short_name)})</h4>'
                     f'{as_html or "<em>nothing</em>"}</div></div></div>')
    return (f'<div class="inside-merge"><b>Text agreement inside AS\'s merged article</b> ({len(inside)} pieces): '
            "the merged article's text cut where our components begin and end; American Stories draws no "
            "boundary there, so these are no matches." + "".join(items) + "</div>")


def _crops_html(group: Group, page: int, drawn: tuple[Image.Image, Image.Image] | None, crop_width: int, *,
                name: str, crops_dir: Path | None, caption: str = "") -> str:
    """The crop pair of the group's boxes on `page` (its own, or another of a group across pages)."""
    ours_rects = tuple(r for p in group.pieces_on(page) for r in p.rects)
    as_rects = group.as_box_rects_on(page)
    if drawn is None:
        return f'<p class="note">no scan for {_pages([page])}, so no crops</p>'
    rects = ours_rects if group.case in ANCHORED else tuple(r for part in group.parts_on(page) for r in part.rects)
    if not rects:
        return ""
    x0, y0, x1, y1 = (min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects))
    if x0 >= 1 or y0 >= 1 or x1 <= 0 or y1 <= 0:
        return '<p class="note">the boxes lie outside the page scan (American Stories coordinates beyond it), so no crops</p>'
    attached = tuple(r for part in group.as_attached_outside for r in part.box_rects) if page == group.page else ()
    bbox = (x0, y0, x1, y1)
    if attached:  # the crop takes in the text AS attaches outside our component too
        bbox = (min([bbox[0]] + [r[0] for r in attached]), min([bbox[1]] + [r[1] for r in attached]),
                max([bbox[2]] + [r[2] for r in attached]), max([bbox[3]] + [r[3] for r in attached]))
    box = crop_box(bbox)
    ours_img = crop_group(drawn[0], box, ours_rects, crop_width)
    as_img = crop_group(drawn[1], box, tuple(dict.fromkeys(as_rects + attached)), crop_width)
    ours_src = _crop_src(ours_img, f"{name}_ours.webp", crops_dir)
    as_src = _crop_src(as_img, f"{name}_as.webp", crops_dir)
    lazy = ' loading="lazy"' if crops_dir is not None else ""
    return ('<div class="crops">'
            f'<figure><img src="{ours_src}" alt="snewpapers boxes"{lazy}><figcaption>snewpapers{caption}</figcaption></figure>'
            f'<figure><img src="{as_src}" alt="American Stories boxes"{lazy}><figcaption>American Stories{caption}</figcaption></figure>'
            '</div>')


def _group_html(group: Group, drawn: dict[int, tuple[Image.Image, Image.Image]], crop_width: int, site_urls: dict,
                 *, group_id: str, crops_dir: Path | None, verdict: GroupVerdict | None = None,
                 merge_id: str | None = None, inside: list[tuple[str, Group]] = (), continues_id: str | None = None,
                 continued_ids: list[str] = ()) -> str:
    """One group: its summary line, verdict and reasons, crops, texts and evidence. `drawn` is
    page -> its drawn pages. `merge_id` is an in_as_merge's merge's id; `inside` an as_merge's
    in_as_merge groups with their ids; `continues_id` a continued group's cross-page group's id,
    `continued_ids` a cross-page group's continued groups' ids."""
    case = _css_case(group.case)
    opened = ""  # every group starts collapsed: the page is an overview, opened one group at a time
    link = lambda gid: f'<a href="#g-{html.escape(gid)}">{html.escape(gid)}</a>'  # noqa: E731
    summary = (f'<summary><span class="badge {case}">{html.escape(CASE_LABELS[group.case])}</span>'
               f'{_verdict_badge(verdict, group)}{_who(group, site_urls)} <span class="gid">{html.escape(group_id)}</span>'
               f' <span class="meta">{html.escape(_metrics_line(group))}</span>'
               + (f' <span class="note">{html.escape(duplicated_note(group))}</span>' if group.duplicates else "")
               + (' <span class="note">AS attaches text outside it</span>' if group.as_attached_outside else "")
               + (' <span class="note">AS puts a line of it in another article</span>' if group.as_nibbles else "")
               + (f' <span class="note">inside AS\'s merged article, decided by {html.escape(merge_id)}</span>'
                  if merge_id else "")
               + (f' <span class="note">continues {link(continues_id)}: compared, asked and decided there</span>'
                  if continues_id else "")
               + (" <span class=\"note\">continued in " + ", ".join(link(c) for c in continued_ids) + "</span>"
                  if continued_ids else "")
               + '</summary>')
    if group.cross_page:
        crops = "".join(_crops_html(group, page, drawn.get(page), crop_width, name=f"{group_id}_p{page}",
                                    crops_dir=crops_dir, caption=f", p. {page}") for page in group.pages)
    else:
        crops = _crops_html(group, group.page, drawn.get(group.page), crop_width, name=group_id, crops_dir=crops_dir)
    if group.case == CONTINUED:  # its texts are compared in the group across pages
        return (f'<details class="group {case}" id="g-{html.escape(group_id)}"{opened}>{summary}'
                f'{_attached_html(group, verdict)}{_nibble_html(group, verdict)}{crops}</details>')
    ours_html, as_html = word_diff_html(group.ours_text, group.as_text)
    texts = ('<div class="texts">'
             f'<div class="text"><h4>snewpapers</h4>{ours_html or "<em>nothing</em>"}</div>'
             f'<div class="text"><h4>American Stories</h4>{as_html or "<em>nothing</em>"}</div>'
             '</div>')
    evidence = ""
    if group.case in DISPUTED:  # the pieces as the model is shown them
        names = ([_part_name(group, part) for part in group.parts] if group.case == AS_SPLIT
                 else [p.label or "?" for p in group.pieces])
        side = "American Stories" if group.case == AS_SPLIT else "snewpapers"
        evidence = (f'<ul class="evidence"><li>the pieces asked about ({side}):</li>' + "".join(
            f"<li>[{k}] {html.escape(name)}: {html.escape(edges(text))}</li>"
            for k, (name, text) in enumerate(zip(names, many_pieces(group)), start=1)) + "</ul>")
    return (f'<details class="group {case}" id="g-{html.escape(group_id)}"{opened}>{summary}{_reasons_html(verdict)}'
            f'{_attached_html(group, verdict)}{_nibble_html(group, verdict)}{crops}{texts}{evidence}{_inside_merge_html(group, list(inside))}</details>')


def _summary_html(summary: dict) -> str:
    rows = [f"<tr><th>groups</th><td>{summary['groups']}</td></tr>"]
    rows += [f'<tr><th><span class="badge {_css_case(c)}">{html.escape(CASE_LABELS[c])}</span></th><td>{n}</td></tr>' for c, n in summary["cases"].items()]
    for key, name in SUMMARY_NAMES.items():
        value = summary[key]
        shown = "n/a" if value is None else (f"{value:.1f}" if key == "median_match_ratio" else f"{value:.1%}")
        rows.append(f"<tr><th>{html.escape(name)}</th><td>{shown}</td></tr>")
    rows.append(f"<tr><th>AS duplicated text</th><td>{summary['as_duplicated']}</td></tr>")
    rows.append(f"<tr><th>AS text attached outside ours</th><td>{summary['attached_outside']}</td></tr>")
    if summary["pages_without_as"]:
        rows.append(f"<tr><th>pages American Stories lacks</th><td>{summary['pages_without_as']}</td></tr>")
    if summary["pages_other_scan"]:
        rows.append(f"<tr><th>pages left out: AS scan is another digitisation</th><td>{summary['pages_other_scan']}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def same_scan_html(seg: IssueSegmentation) -> str:
    """"Same LoC scan for N of M pages", and when not all, a warning naming the pages left out."""
    if seg.same_scan is None:
        return ""
    same, total = seg.same_scan
    line = f"Same LoC scan for {same} of {total} pages."
    if same == total:
        return f"<p>{line}</p>"
    pages = ", ".join(map(str, seg.pages_other_scan)) or "?"
    return (f'<p class="note"><b>{line}</b> American Stories segmented another LoC digitisation of p. {pages} '
            f"than the one snewpapers' boxes are on, so IoU on those pages is not meaningful; "
            "they are left out of the comparison.</p>")


def nested_note_html(seg: IssueSegmentation) -> str:
    """A note on the AS boxes lying inside another of its boxes (segmentation.nested_boxes), and how
    the comparison treats them; "" when there are none."""
    if not seg.as_nested:
        return ""
    by_outer: dict[tuple[int, str], list[str]] = {}
    for page, inner, outer in seg.as_nested:
        by_outer.setdefault((page, outer), []).append(inner)
    named = lambda inners: _and([n + (f" ({inners.count(n)} boxes)" if inners.count(n) > 1 else "")  # noqa: E731
                                 for n in dict.fromkeys(inners)])
    where = "; ".join(f"p. {page}: {named(inners)} inside {outer}" for (page, outer), inners in by_outer.items())
    n = len(seg.as_nested)
    return (f'<p class="note">{n} American Stories box{"es lie" if n > 1 else " lies"} inside another of its boxes '
            f"({html.escape(where)}). Nothing in AS's data marks or explains it: its layout detection boxed the same "
            "print twice (a column and an item in it, say) and both were OCR'd and published as articles. We compare "
            "every AS article as published; where both copies fall inside one of our components, the shorter is set "
            "aside and counted once as text AS has twice, not as another article.</p>")


def _row(label: str, *cells) -> str:
    return f"<tr><th>{html.escape(label)}</th>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def _table(title: str, head: tuple[str, ...], rows: list[str], note: str = "") -> str:
    heading = "<tr><th></th>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>"
    return (f'<div class="score-block"><h3>{html.escape(title)}</h3>'
            f'<table class="scorecard">{heading}{"".join(rows)}</table>{note}</div>')


def _decided(counts: dict, *, boundary: bool = False) -> tuple:
    """(times, snewpapers right, AS right, neither, unsure, page furniture) from a question's outcome
    counts: a dispute can be "neither", a boundary question "furniture"."""
    unsure = counts.get(UNSTABLE, 0) + counts.get(NO_VERDICT, 0) + counts.get(ILLEGIBLE_UNSURE, 0)
    return (sum(counts.values()), counts[SNEWPAPERS], counts[AMERICAN_STORIES],
            "–" if boundary else counts.get(NEITHER, 0), unsure, counts.get(FURNITURE, 0) if boundary else "–")


def _illegible_sentence(summary: dict | None) -> str:
    if not summary or not summary["boxes"]:
        return ""
    return (f' AS marked {summary["boxes"]} boxes illegible over {summary["components"]} of our components and read no '
            f'text from them; snewpapers\' text of those components is {_rate(summary["real_words"])} real words.')


def _quality_cells(counts: list[int]) -> tuple:
    return f"{counts[0]:,}", f"{counts[1]:,}", _rate(counts)


def _scorecard_html(card: dict, quality: dict | None = None, seg: IssueSegmentation | None = None, *,
                    usage: bool = True) -> str:
    """The verdicts counted, as tables: where the two segmentations agree or differ and who was
    judged right; where only one side has text; what the decisions judged; the American Stories
    errors they found; each side's real words (`quality`, from real_word_summary). How it was
    scored is folded away beneath."""
    dash = ("–",) * 5
    blocks = [_table("Where the two segmentations agree or differ",
                     ("times", "snewpapers right", "AS right", "neither", "unsure", "page furniture, not counted"), [
        _row("Same article on both sides (=)", card["matches"], *dash),
        _row("One snewpapers component, several AS articles (1 : N)", *_decided(card["splits"])),
        _row("Several snewpapers components, one AS article (N : 1)", *_decided(card["merges"])),
        _row("One snewpapers component, one AS article running on into other text", *_decided(card["wider"], boundary=True)),
        _row("AS text attached outside a snewpapers component", *_decided(card["attached_outside"], boundary=True)),
        _row("A line of a snewpapers component in another AS article", *_decided(card["nibbles"], boundary=True)),
    ], '<p class="meta">Unsure: the verdict changed when the two claims were stated in the other order, or no '
       "valid answer came back. Neither: the model found both segmentations wrong. Page furniture: the text was a "
       "line or two that could be a title, section title or the like, which one system leaves out by design, so it "
       "decides nothing.</p>")]
    as_only, comps = card["as_only"], card["components"]
    causes = {case: sum(g.case == case and g.as_illegible for g in (seg.groups if seg else [])) for case in (OURS_ONLY, AS_PARTIAL)}
    blocks.append(_table("Where only one side has text", ("times",), [
        _row("snewpapers component, AS has no text, no AS box there (an AS error)",
             comps[NOT_CAPTURED] - causes[OURS_ONLY]),
        _row("snewpapers component, AS has no text, AS marked it illegible (an AS error)", causes[OURS_ONLY]),
        _row("snewpapers component, AS has text for under 90% of it, the rest in no AS box (an AS error)",
             comps[PARTLY_CAPTURED] - causes[AS_PARTIAL]),
        _row("snewpapers component, AS has text for under 90% of it, the rest marked illegible (an AS error)",
             causes[AS_PARTIAL]),
        _row("AS text in no snewpapers component: ads, terms or page furniture (left out on purpose)", as_only[EXCLUDED]),
        _row("… of which ads (AS says it leaves ads out)", card["as_ads_kept"]),
        _row("AS text in no snewpapers component: other content snewpapers does not serve", as_only[MISSED]),
        _row("AS text in no snewpapers component: too short to ask", as_only[TINY]),
        _row("AS boxes in no snewpapers component with no text at all (AS read nothing)", as_only[EMPTY]),
        _row("AS text in no snewpapers component: no verdict", as_only[NO_VERDICT]),
    ]))
    judged = card.get("judged")
    if judged:
        blocks.append(_table("What the decisions judged", ("judged wrong", "judged right"), [
            _row("American Stories articles", judged["as_articles"]["wrong"], judged["as_articles"]["right"]),
            _row("snewpapers components", judged["components"]["wrong"], judged["components"]["right"]),
        ], '<p class="meta">Each article or component a decision was about, once per side: one story split into '
           "15 AS articles and decided for snewpapers is 15 AS articles judged wrong.</p>"))
    errors = card.get("as_errors")
    if errors:
        blocks.append(_table("American Stories errors found", ("errors", "decisions"), [
            _row("items of ours AS merged into one article", errors[ITEMS_MERGED], errors["merge_decisions"]),
            _row("fragment articles AS split our components into", errors[FRAGMENT_ARTICLES], errors["split_decisions"]),
            _row("lines of ours AS put in another article", errors[LINES_ELSEWHERE], errors["nibble_decisions"]),
            _row("AS articles running on into other items", errors[WIDER_ARTICLES], errors[WIDER_ARTICLES]),
            _row("AS articles missing part of a component (under 90%)", errors[PARTIAL_ARTICLES], "–"),
            _row("our components AS has no text for", errors[MISSING_COMPONENTS], "–"),
            _row("pages of our components AS lacks the text for", errors[MISSING_PAGES], "–"),
            _row("text AS has twice", card["as_defects"][AS_DUPLICATED], "–"),
            _row("ads AS kept (it says it leaves ads out)", card["as_ads_kept"], "–"),
            *([_row("AS boxes it marked illegible that snewpapers read", quality["illegible"]["boxes"], "–")]
              if quality and quality.get("illegible") else []),
        ]))
    if quality:
        more = quality["more_real"]
        blocks.append(_table("Text quality: words found in an English word list", ("real words", "words", "share"), [
            _row("snewpapers", *_quality_cells(quality[SNEWPAPERS])),
            _row("American Stories", *_quality_cells(quality[AMERICAN_STORIES])),
            *([_row("American Stories boxed area it marked illegible and did not read", "–", "–",
                    _rate(list(quality["illegible_area"])))] if quality.get("illegible_area", (0, 0))[1] else []),
        ], f'<p class="meta">Over the {quality["components"]} components of ours AS has text inside, the same text on '
           "both sides; words of 3 or more letters, a long s read as f allowed. Components where more of the text is "
           f"real words: snewpapers {more[SNEWPAPERS]} · AS {more[AMERICAN_STORIES]} · the same {more['same']}."
           + _illegible_sentence(quality.get("illegible")) + "</p>"))
    return (
        "<h2>Scorecard</h2>"
        '<p>Each comparison starts from a snewpapers component. "=" means both systems drew the same article. Where '
        "they differ, a language model, shown only the text, was asked which is right, twice, with the two claims "
        "stated in each order.</p>"
        '<div class="scoreboard">' + "".join(blocks) + "</div>"
        + (f'<p class="meta">{card["questions"]} questions, {card["cached"]} from the cache; '
           f'{html.escape(usage_text(card))}</p>' if usage else "") +
        '<details class="method"><summary>How this was scored</summary>'
        "<p>Every comparison is anchored on a snewpapers component and the American Stories boxes inside it. "
        "Where those boxes belong to several AS articles (or several of our components lie inside one AS article), "
        "the model was shown the passage as the numbered pieces and asked whether it is one component, several or mixed, "
        "twice, with the two beliefs stated, unattributed, in each order. "
        "A verdict that changed with the order is unstable and counts for neither side. "
        "AS text inside none of our components was asked whether it is advertisements, terms or page furniture, "
        "which snewpapers leaves out on purpose. "
        "Where AS attaches text outside one of our components (text of an AS article our component otherwise has), "
        "the model was shown our component's text and the attached text and asked whether the text belongs to the component, "
        "twice, the two beliefs stated in each order: \"separate\" means snewpapers is right, \"belongs\" American Stories. "
        "Where two AS articles inside one of our components hold the same text (the shorter sharing at least 50% "
        "of its 4-word phrases with the longer), the longer is kept and the text counted as duplicated by American Stories; "
        "so is attached text another AS article also holds. "
        "Where American Stories has a line or two of our component (at most 12 words and 5% of its area) in another "
        "AS article while one article holds the rest, the line is no split: the model was shown the rest of the AS text "
        "and the line and asked whether the line belongs to the component, both ways; \"belongs\" means snewpapers is "
        "right and judges only the article holding the line wrong (lines in another AS article). "
        "One AS article counts as a match only when the two cover each other, both ways: AS text covers at least 90% "
        "of our component, and our component at least 90% of that AS article's text. Coverage is measured line by line "
        "down each column (how much of its height the other side's boxes span, so a centred headline narrower than "
        "its column still counts). Under 90% the first way, the component is partly captured by AS (the rest in no AS "
        "box, or marked illegible): not asked, an AS error by the geometry, its AS article judged wrong (AS articles "
        "missing part of ours). Under 90% the second way, the AS article runs on beyond our component into other "
        "text: that text is asked about as text attached outside, and \"separate\" judges the AS article wrong "
        "(AS articles running on into other items). "
        "Decisions are the unit counted above; the last table counts what they judged: an as_split decided "
        "\"one\" is one decision and every AS article inside our component judged wrong, an N ours : 1 AS "
        "decided \"several\" one decision and each of our N components judged right "
        "(an article or component judged both ways by different decisions counts in both rows). "
        "AS errors count what the decisions for snewpapers found American Stories got wrong, next to those decisions: "
        "an N ours : 1 AS is N errors, the N items of ours AS failed to separate (items AS merged), a 1 ours : N AS "
        "N errors, the N AS fragment articles. A component inside an N ours : 1 AS (its AS text a slice of the "
        "merged article) is counted in the components table by that decision's outcome. "
        "A component running over pages is one comparison across them (American Stories never links an article "
        "across pages): its AS articles on all its pages, labelled by page, are the pieces asked about, once; "
        "it counts once among our components, and a decision for snewpapers judges every one of those AS "
        "articles wrong, each a fragment article.</p>"
        "</details>"
    )


def _legend_html() -> str:
    swatch = lambda name, rgb: f'<span><span class="swatch" style="border-color: rgb{rgb}"></span>{html.escape(name)}</span>'  # noqa: E731
    cases = "".join(f'<tr><th><span class="badge {_css_case(c)}">{html.escape(CASE_LABELS[c])}</span></th><td>{html.escape(CASE_HELP[c])}</td></tr>' for c in CASES)
    return (
        "<h2>How to read this</h2>"
        "<p>Each group is the same crop of the scan twice: snewpapers' boxes on the left, American Stories' on the right. "
        "The group's own boxes are outlined in red. Groups are in page position order (column by column), not reading order. "
        "A component can appear twice: in its own case and in an N ours : 1 AS group. "
        "Text of a component's own AS articles printed outside the component is left out of the comparison and shown "
        "beneath the group as \"American Stories also attaches: «…» (outside this component)\".</p>"
        f"<table>{cases}</table>"
        '<div class="legend"><b>snewpapers:</b>' + "".join(swatch(n, c) for n, c in FAMILY_COLORS.items()) + "</div>"
        '<div class="legend"><b>American Stories:</b>' + "".join(swatch(n, c) for n, c in AS_COLORS.items()) + "</div>"
        '<div class="legend"><span><span class="only-ours">only in snewpapers</span></span>'
        '<span><span class="only-as">only in American Stories</span></span>'
        "<span>(words compared lowercase, without punctuation)</span></div>"
    )


def _drawn_pages(pair, groups_by_page: dict, load_pages: Callable, page_width: int) -> dict[int, tuple[Image.Image, Image.Image]]:
    """page -> (snewpapers boxes drawn, American Stories boxes drawn), full page at `page_width`.

    Labels are left off: in a close-up they would sit on the very text being compared (the badge,
    the legend's colours and the captions say which side and kind each box is).
    """
    drawn = {}
    for page, (image, data, scan_size) in load_pages(pair, page_width).items():
        if page in groups_by_page:
            drawn[page] = (draw_snewpapers_page(image, pair.components, page, labels=False),
                           draw_american_stories_page(image, data, scan_size=scan_size, labels=False))
    return drawn


def segmentation_report(
    pair,
    seg: IssueSegmentation,
    *,
    crop_width: int = 520,
    load_pages: Callable = load_issue_pages,
    page_width: int = PAGE_WIDTH,
    reports_dir: Path = REPORTS_DIR,
    embed: bool = False,
    verdicts: list[GroupVerdict] | None = None,
    usage: bool = True,
) -> tuple[str, Path]:
    """Write the issue's segmentation report; returns (html, the path of the HTML file written).

    By default this writes `reports_dir/{lccn}_{date}_ed-{n}/index.html` plus one sidecar WebP per
    crop under its `crops/` folder, referenced by relative path and lazy-loaded, so opening the page
    does not pull every crop in at once. Pass `embed=True` for the old self-contained single file,
    `reports_dir/{lccn}_{date}_ed-{n}_segmentation.html`, with every crop inlined as a base64 WebP
    (handy somewhere like Jupyter's output area, which shows only the one file).

    `load_pages(pair, width)` gives page -> (scan image, AS page data, real scan size); by default
    the issue's JP2s are downloaded once into data/loc and loaded from there.

    `verdicts` (from adjudicate.adjudicate) add a scorecard at the top and each group's verdict;
    `usage=False` leaves the LLM's tokens and cost out of it (for published results: answers from the
    cache cost nothing, and the cost depends on the model anyone uses).
    """
    issue = pair.issue
    issue_name = f"{issue.lccn}_{issue.date}_ed-{issue.edition}"
    if embed:
        path, crops_dir = reports_dir / f"{issue_name}_segmentation.html", None
        reports_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = reports_dir / issue_name
        path, crops_dir = out_dir / "index.html", out_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
    groups_by_page: dict[int, list[Group]] = {}
    for group in seg.groups:
        groups_by_page.setdefault(group.page, []).append(group)
    drawn = _drawn_pages(pair, groups_by_page, load_pages, page_width)
    site_urls = {c["uuid"]: (c.get("links") or {}).get("site_component_url") for c in pair.components}
    site_urls = {u: url for u, url in site_urls.items() if url}
    loc_url = next(((c.get("links") or {}).get("loc_issue_url") for c in pair.components
                    if (c.get("links") or {}).get("loc_issue_url")), None)
    as_url = f"https://huggingface.co/datasets/{REPO_ID}/tree/{REVISION}"
    title = f"{issue.lccn} {issue.date} ed-{issue.edition}: segmentation, region by region"
    links = [f'<a href="{html.escape(loc_url)}">LoC issue</a>'] if loc_url else []
    links.append(f'<a href="{as_url}">American Stories revision {REVISION[:8]}</a>')
    by_id = {v.group_id: v for v in verdicts or []}  # ids are p{page}-{k} as here (adjudicate.group_ids)
    ids = {id(g): f"p{page}-{i + 1}" for page, groups in groups_by_page.items() for i, g in enumerate(groups)}
    inside: dict[int, list[tuple[str, Group]]] = {}  # an as_merge's in_as_merge groups, by the merge
    for g in seg.groups:
        if g.case == IN_AS_MERGE and g.merge is not None:
            inside.setdefault(id(g.merge), []).append((ids[id(g)], g))
    sections = []
    for page in sorted(groups_by_page):
        groups = groups_by_page[page]
        body = "".join(
            _group_html(g, drawn, crop_width, site_urls, group_id=ids[id(g)], crops_dir=crops_dir,
                        verdict=by_id.get(ids[id(g)]), merge_id=ids.get(id(g.merge)) if g.merge is not None else None,
                        inside=inside.get(id(g), []),
                        continues_id=ids.get(id(g.continues)) if g.continues is not None else None,
                        continued_ids=[ids[id(c)] for c in g.continued if id(c) in ids])
            for g in groups
        )
        sections.append(f'<section class="page" id="page-{page}"><h2>Page {page} '
                        f'<span class="meta">({len(groups)} groups)</span></h2>{body}</section>')
    document = (
        f"<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f'<body style="margin:0;background:#fafaf7"><div class="seg-report">'
        f"<h1>{html.escape(title)}</h1><p>{' · '.join(links)}</p>{same_scan_html(seg)}{nested_note_html(seg)}"
        f"{_scorecard_html(scorecard(verdicts, seg), real_word_summary(seg), seg, usage=usage) if verdicts is not None else ''}"
        f"{_summary_html(seg.summary())}{_legend_html()}{''.join(sections)}</div>{OPEN_LINKED}</body></html>\n"
    )
    path.write_text(document)
    return document, path


# --- the benchmark index ----------------------------------------------------------------------

BENCHMARK_COLUMNS = (  # (group heading, (row key, sub-key, column heading) ...)
    ("snewpapers components", tuple(("components", o, o) for o in COMPONENT_OUTCOMES)),
    ("N ours : 1 AS", (("merges", "snewpapers", "snewpapers right"), ("merges", "american_stories", "AS right"),
                       ("merges", "other", "other"))),
    ("attached outside", (("attached", "snewpapers", "snewpapers"), ("attached", "american_stories", "AS"),
                          ("attached", "other", "other"))),
    ("nibbles", (("nibbles", "snewpapers", "snewpapers"), ("nibbles", "american_stories", "AS"),
                 ("nibbles", "other", "other"))),
    ("AS only", (("as_only", EXCLUDED, EXCLUDED), ("as_only", MISSED, "missed"))),
    ("AS defects", (("as_defects", AS_DUPLICATED, "duplicated"), ("as_defects", "nested boxes", "nested boxes"),
                    ("as_defects", "ads kept", "ads kept"), ("as_defects", "illegible read", "illegible, we read"))),
    ("judged wrong", (("judged_wrong", "as_articles", "AS articles"), ("judged_wrong", "components", "snewpapers comps"))),
    ("AS errors (decided for snewpapers)", (("as_errors", "merge_decisions", "N:1 decisions"),
                                            ("as_errors", ITEMS_MERGED, ITEMS_MERGED),
                                            ("as_errors", "split_decisions", "1:N decisions"),
                                            ("as_errors", FRAGMENT_ARTICLES, FRAGMENT_ARTICLES),
                                            ("as_errors", LINES_ELSEWHERE, LINES_ELSEWHERE),
                                            ("as_errors", PARTIAL_ARTICLES, "missing part"),
                                            ("as_errors", WIDER_ARTICLES, "runs on"),
                                            ("as_errors", MISSING_COMPONENTS, "no text"))),
)
BENCHMARK_CSS = """
.seg-report .bench-wrap { overflow-x: auto; }
.seg-report table.bench.summary th { white-space: normal; vertical-align: bottom; max-width: 8.5em; }
.seg-report table.bench { font-size: .88em; background: #fff; border: 1px solid #ddd; }
.seg-report table.bench th, .seg-report table.bench td { padding: 3px 8px; border-bottom: 1px solid #eee; white-space: nowrap; }
.seg-report table.bench th.group { text-align: center; border-left: 1px solid #ddd; }
.seg-report table.bench td.n { text-align: right; }
.seg-report table.bench td.first, .seg-report table.bench th.first { border-left: 1px solid #ddd; }
.seg-report table.bench tr.total td { font-weight: 600; border-top: 2px solid #ccc; }
"""


QUALITY_CASES = (MATCH, AS_PARTIAL, AS_SPLIT, IN_AS_MERGE)  # one component of ours, with AS text inside it


def real_words(seg: IssueSegmentation) -> dict[str, list[int]]:
    """[real words, words] on each side, summed over the components of ours American Stories has
    text inside (QUALITY_CASES; a component across pages once), so both sides count the same text."""
    out = {SNEWPAPERS: [0, 0], AMERICAN_STORIES: [0, 0]}
    for g in seg.groups:
        if g.case in QUALITY_CASES and g.parts:
            for side, text in ((SNEWPAPERS, g.ours_text), (AMERICAN_STORIES, g.as_text)):
                real, total = word_counts(text)
                out[side][0] += real
                out[side][1] += total
    return out


def illegible_summary(seg: IssueSegmentation) -> dict:
    """The AS boxes American Stories marked illegible (reading no text) that lie inside our components
    (segmentation.illegible_read): how many, over how many of our components, and [real words,
    words] of those components' text: what AS's pipeline judged unreadable, as snewpapers read it."""
    texts = {p.uuid: p.text for g in seg.groups for p in g.pieces}
    uuids = list(dict.fromkeys(u for *_rest, over in seg.illegible_read for u in over))
    counts = [word_counts(texts.get(u, "")) for u in uuids]
    return {"boxes": len(seg.illegible_read), "components": len(uuids),
            "real_words": [sum(c[0] for c in counts), sum(c[1] for c in counts)]}


def real_word_summary(seg: IssueSegmentation) -> dict:
    """`real_words` and, per component of ours AS has text inside, which side's text has the larger
    share of real words ("more_real": snewpapers, american_stories, same), with how many there are."""
    more = {SNEWPAPERS: 0, AMERICAN_STORIES: 0, "same": 0}
    for g in seg.groups:
        if g.case in QUALITY_CASES and g.parts:
            ours, theirs = real_word_rate(g.ours_text) or 0.0, real_word_rate(g.as_text) or 0.0
            more[SNEWPAPERS if ours > theirs else AMERICAN_STORIES if theirs > ours else "same"] += 1
    return {**real_words(seg), "more_real": more, "components": sum(more.values()), "illegible": illegible_summary(seg),
            "illegible_area": seg.as_illegible_area}


def _rate(counts: list[int]) -> str:
    return f"{counts[0] / counts[1]:.1%}" if counts[1] else "n/a"


def benchmark_row(decade: str, paper: str, date: str, seg: IssueSegmentation, verdicts: list[GroupVerdict],
                  report: str) -> dict:
    """One issue's line of the benchmark index: its scorecard (adjudicate.scorecard) regrouped into
    the index's columns (what the decisions judged wrong: AS articles, snewpapers components; the AS
    errors the decisions for snewpapers found: items AS merged, fragment articles, with the
    decisions behind them), the same-scan count, the matches' text ratios and the issue report's path.

    Disputes and boundary questions keep the two sides and count everything else (neither,
    unstable, no verdict) as "other"; AS-only groups count only those decided either way.
    """
    card = scorecard(verdicts, seg)
    sides = lambda counts: {SNEWPAPERS: counts[SNEWPAPERS], AMERICAN_STORIES: counts[AMERICAN_STORIES],  # noqa: E731
                            "other": sum(n for o, n in counts.items() if o not in SIDES)}
    same, total = seg.same_scan or (0, 0)
    return {
        "decade": decade,
        "paper": paper,
        "date": date,
        "same_scan": [same, total],
        "components": card["components"],
        "merges": sides(card["merges"]),
        "attached": sides(card["attached_outside"]),
        "nibbles": sides(card["nibbles"]),
        "as_only": {EXCLUDED: card["as_only"][EXCLUDED], MISSED: card["as_only"][MISSED]},
        "as_defects": {**card["as_defects"], "nested boxes": len(seg.as_nested), "ads kept": card["as_ads_kept"],
                       "illegible read": len(seg.illegible_read)},
        "judged_wrong": {unit: card["judged"][unit]["wrong"] for unit in ("as_articles", "components")},
        "as_errors": card["as_errors"],
        "match_ratios": [g.metrics["ratio"] for g in seg.groups if g.case == MATCH and "ratio" in g.metrics],
        "text_median": seg.summary()["median_match_ratio"],
        "real_words": real_words(seg),
        "as_illegible_area": list(seg.as_illegible_area),
        "cost": card["cost"],
        "questions": card["questions"],
        "cached": card["cached"],
        "report": report,
        "scorecard": card,
    }


def benchmark_totals(rows: list[dict]) -> dict:
    """Every count summed over `rows`, the median text ratio over all their matches, the cost summed."""
    total: dict = {"decade": "total", "paper": f"{len(rows)} issues", "date": "", "report": None,
                   "same_scan": [sum(r["same_scan"][0] for r in rows), sum(r["same_scan"][1] for r in rows)]}
    for _heading, columns in BENCHMARK_COLUMNS:
        for key, sub, _name in columns:
            total.setdefault(key, {})[sub] = sum(r[key][sub] for r in rows)
    ratios = [x for r in rows for x in r["match_ratios"]]
    total["text_median"] = statistics.median(ratios) if ratios else None
    total["real_words"] = {side: [sum(r["real_words"][side][k] for r in rows) for k in (0, 1)] for side in SIDES}
    total["as_illegible_area"] = [sum(r.get("as_illegible_area", [0, 0])[k] for r in rows) for k in (0, 1)]
    total["cost"] = round(sum(r["cost"] for r in rows), 6)
    total["questions"] = sum(r["questions"] for r in rows)
    total["cached"] = sum(r["cached"] for r in rows)
    return total


def _bench_cells(row: dict, usage: bool = True) -> list[tuple[str, str]]:
    """(css class, escaped cell) for each column of a row."""
    same, total = row["same_scan"]
    cells = [("", html.escape(row["decade"])), ("", html.escape(row["paper"])), ("", html.escape(row["date"])),
             ("n first", f"{same} of {total}")]
    for _heading, columns in BENCHMARK_COLUMNS:
        cells += [("n first" if k == 0 else "n", str(row[key][sub])) for k, (key, sub, _name) in enumerate(columns)]
    median = row["text_median"]
    cells.append(("n first", "n/a" if median is None else f"{median:.1f}"))
    cells.append(("n first", _rate(row["real_words"][SNEWPAPERS])))
    cells.append(("n", _rate(row["real_words"][AMERICAN_STORIES])))
    cells.append(("n", _rate(row.get("as_illegible_area", [0, 0]))))
    if usage:
        cells.append(("n", f"${row['cost']:.4f}"))
    links = [f'<a href="{html.escape(row[key])}">{key}</a>' for key in ("report", "overlay") if row.get(key)]
    link = " · ".join(links)
    cells.append(("first", link))
    return cells


def _summary_table(rows: list[dict]) -> str:
    """The issues at a glance: each one's report and overlay links first, then its headline numbers."""
    head = ("decade", "paper", "date", "", "our components", "AS articles judged wrong",
            "snewpapers components judged wrong", "real words, snewpapers", "real words, AS")

    def line(row: dict, total: bool = False) -> str:
        links = " · ".join(f'<a href="{html.escape(row[key])}">{key}</a>' for key in ("report", "overlay") if row.get(key))
        cells = [html.escape(row["decade"]), html.escape(row["paper"]), html.escape(row["date"]), links,
                 f'{sum(row["components"].values()):,}', f'{row["judged_wrong"]["as_articles"]:,}',
                 f'{row["judged_wrong"]["components"]:,}', _rate(row["real_words"][SNEWPAPERS]),
                 _rate(row["real_words"][AMERICAN_STORIES])]
        return (f'<tr{" class=\"total\"" if total else ""}>'
                + "".join(f'<td{" class=\"n\"" if k > 3 else ""}>{c}</td>' for k, c in enumerate(cells)) + "</tr>")

    body = "".join(line(row) for row in rows) + line(benchmark_totals(rows), total=True)
    return (f'<div class="bench-wrap"><table class="bench summary"><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr>'
            f"{body}</table></div>")


def benchmark_index_html(rows: list[dict], *, title: str = "Benchmark: snewpapers vs American Stories",
                         intro: str | None = None, usage: bool = True) -> str:
    """The benchmark index: one row per issue (from `benchmark_row`), a totals row, each linking its report
    (and its verdict overlay, given an `overlay` path). Given an `intro` (HTML), a start page: the intro,
    the issues at a glance with their links first (`_summary_table`), then all the numbers. `usage=False`
    leaves out the LLM cost column (for published results)."""
    first = ' class="first"'
    top = ['<th colspan="3"></th><th class="group">pages</th>']
    sub = [f"<th>decade</th><th>paper</th><th>date</th><th{first}>same scan</th>"]
    for heading, columns in BENCHMARK_COLUMNS:
        top.append(f'<th class="group" colspan="{len(columns)}">{html.escape(heading)}</th>')
        sub += [f"<th{first if k == 0 else ''}>{html.escape(name)}</th>" for k, (_, _, name) in enumerate(columns)]
    top.append('<th class="group">text</th><th class="group" colspan="2">real words</th><th></th>'
               + ("<th></th>" if usage else "") + "<th></th>")
    sub.append('<th class="first">median ratio</th><th class="first">snewpapers</th><th>AS</th><th>AS area illegible</th>'
               + ("<th>LLM cost</th>" if usage else "") + '<th class="first"></th>')
    cells = lambda row: "".join(f'<td class="{c}">{v}</td>' for c, v in _bench_cells(row, usage))  # noqa: E731
    body = [f"<tr>{cells(row)}</tr>" for row in rows] + [f'<tr class="total">{cells(benchmark_totals(rows))}</tr>']
    table = (f'<div class="bench-wrap"><table class="bench"><tr>{"".join(top)}</tr><tr>{"".join(sub)}</tr>'
             f'{"".join(body)}</table></div>')
    note = ("<p>One row per benchmark issue. snewpapers components: every match, 1 ours : N AS and ours-only case, "
            "by its verdict (see each report's scorecard). N ours : 1 AS and attached-outside questions: which side the "
            "model found right in both orders; \"other\" is neither, unstable or no verdict. AS only: left-over AS text "
            "the model called advertisements, terms or page furniture (correctly not served) or not (missed). "
            "Judged wrong: what those decisions judged wrong, counted in AS articles and snewpapers components "
            "(an as_split of one component into 15 AS articles decided \"one\" is one decision and 15 AS articles). "
            "AS errors: what the decisions for snewpapers found American Stories got wrong, with the decisions behind "
            "them: an N ours : 1 AS is N items AS merged (seven items of ours in one AS article: 7), a 1 ours : N AS "
            "N fragment articles. "
            "Text: the median fuzz ratio over 1:1 matches (the totals row: over every match). "
            "Real words: the share of each side's words (3 or more letters) found in an English word list, a long s "
            "read as f allowed, over the components of ours AS has text inside, so both sides count the same text. "
            "Pages: how many American Stories scans are the LoC scan snewpapers' boxes are on.</p>")
    return (
        f"<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>{CSS}{BENCHMARK_CSS}</style></head>"
        f'<body style="margin:0;background:#fafaf7"><div class="seg-report" style="max-width:none">'
        f"<h1>{html.escape(title)}</h1>"
        + (f"{intro}{_summary_table(rows)}<h2>All the scorecard numbers</h2>" if intro is not None else "")
        + f"{note}{table}</div></body></html>\n"
    )


def benchmark_text(rows: list[dict]) -> str:
    """A compact plain-text table of the benchmark (stdout)."""
    head = f"{'decade':<7} {'date':<10} {'scan':>5}  {'comp ok/split/merge/nei/uns/nv/part/wide/nc':<43} {'merge s/a/o':<11} " \
           f"{'attach s/a/o':<12} {'asonly ok/miss':<14} {'dup':>3} {'wrong as/ours':>13} " \
           f"{'AS err N:1/items/1:N/frags/lines/part/wide/none':>47} {'text':>5} {'cost':>8}  paper"
    lines = [head]
    for row in rows + [benchmark_totals(rows)]:
        c, m, a, o, w = row["components"], row["merges"], row["attached"], row["as_only"], row["judged_wrong"]
        median = "n/a" if row["text_median"] is None else f"{row['text_median']:.1f}"
        wrong = f"{w['as_articles']}/{w['components']}"
        e = row["as_errors"]
        errors = "/".join(str(e[k]) for k in ("merge_decisions", ITEMS_MERGED, "split_decisions", FRAGMENT_ARTICLES,
                                              LINES_ELSEWHERE, PARTIAL_ARTICLES, WIDER_ARTICLES, MISSING_COMPONENTS))
        lines.append(
            f"{row['decade']:<7} {row['date']:<10} {row['same_scan'][0]:>2}/{row['same_scan'][1]:<2}  "
            f"{'/'.join(str(c[x]) for x in COMPONENT_OUTCOMES):<43} "
            f"{'/'.join(str(m[x]) for x in ('snewpapers', 'american_stories', 'other')):<11} "
            f"{'/'.join(str(a[x]) for x in ('snewpapers', 'american_stories', 'other')):<12} "
            f"{f'{o[EXCLUDED]}/{o[MISSED]}':<14} {row['as_defects'][AS_DUPLICATED]:>3} {wrong:>13} {errors:>47} "
            f"{median:>5} {row['cost']:>8.4f}  {row['paper']}")
    return "\n".join(lines)
