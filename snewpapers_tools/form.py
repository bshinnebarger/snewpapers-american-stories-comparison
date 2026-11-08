"""The notebook form: search, show the hits, pick one at random, estimate its issue, confirm."""
from __future__ import annotations

import html
import random
import string
from typing import Callable

import ipywidgets as w
import pandas as pd
from IPython.display import HTML, display

from snewpapers_tools.american_stories import ArchiveInfo, archive_info, day_covered, download_archive, issue_coverage
from snewpapers_tools.api import SnewpapersClient
from snewpapers_tools.selection import Estimate, estimate, hits_table, pick

# Years where both the API (from 1736) and American Stories (1770-1963) have issues.
FIRST_YEAR, LAST_YEAR = 1770, 1963
DENSE_PER_PAGE = 35  # components per page; about the corpus p90 (API per-decade stats); above this, suggest picking again
MIN_TEXT = 3  # the API rejects shorter keywords and topics


def render_pick(hit: dict, component: dict, est: Estimate) -> str:
    """The picked article and what comparing its issue would cost, as HTML."""
    e = html.escape
    hit_pages = ", ".join(str(p) for p in hit.get("pages") or [])
    newspaper = string.capwords(hit.get("newspaper") or "")
    parts = [
        f"<h4>{e(newspaper)}, {e(hit.get('issue_date') or '')}, page {e(hit_pages)}</h4>",
        f"<p><b>{e(hit.get('label') or '')}</b> ({e(hit.get('quality_tier') or '')}): {e(hit.get('summary') or '')}</p>",
    ]
    clips = (component.get("thumbs") or {}).get("clip_lg") or []
    if clips:
        parts.append(f'<img src="{e(clips[0]["url"])}" style="max-width:100%; max-height:420px">')
    citation = (component.get("citation") or {}).get("formatted", "")
    site_url = hit.get("site_url")
    link = f' <a href="{e(site_url)}" target="_blank">snewpapers.com</a>' if site_url else ""
    parts.append(f'<p><small>{e(citation)}</small>{link}</p>')

    issue, cost = est.issue, est.cost
    pages_phrase = f" on {est.issue_pages} pages" if est.issue_pages is not None else ""
    parts.append(
        f"<p>Issue {e(issue.lccn)} {e(issue.date)} ed-{issue.edition}: {est.total_components} components"
        f"{pages_phrase}. Comparing it costs up to <b>{cost['total']}</b> tokens ({cost['browse']} to browse, "
        f"{cost['text']} for text; texts you have read before are free). Your balance: {est.balance}.</p>"
    )

    as_line, as_warning = None, None
    if est.archive is None:
        as_warning = f"American Stories has no archive for {issue.year}."
    elif est.needs_download:
        gb = est.archive.size / 1e9
        as_line = (
            f"American Stories {est.archive.filename}: {gb:.1f} GB to download (free, once per year) "
            "before we can check it has this issue."
        )
    elif est.as_pages:
        as_line = f"American Stories has this issue: {len(est.as_pages)} pages ({est.archive.filename})."
    else:
        as_warning = "American Stories does not have this issue: pick another."
    if as_line:
        parts.append(f"<p>{e(as_line)}</p>")

    warnings = []
    page_count = est.issue_pages if est.issue_pages is not None else (len(est.as_pages) if est.as_pages else None)
    if page_count is not None and est.total_components > DENSE_PER_PAGE * page_count:
        warnings.append(f"Dense issue: {est.total_components} components on {page_count} pages: consider picking another.")
    if cost["total"] > est.balance:
        warnings.append("Not enough tokens for the whole issue.")
    if as_warning:
        warnings.append(as_warning)
    parts += [f'<p style="color:#b45309">{e(text)}</p>' for text in warnings]
    return "\n".join(parts)


class ComparisonForm:
    """Search, pick and estimate; `confirmed` holds the Estimate once the reviewer clicks Compare."""

    def __init__(
        self,
        client: SnewpapersClient,
        *,
        rng: random.Random | None = None,
        archive_lookup: Callable[[int], ArchiveInfo | None] = archive_info,
        coverage: Callable[[ArchiveInfo | None, object], list | None] = issue_coverage,
        download: Callable[[ArchiveInfo], object] = download_archive,
        day_coverage: Callable[[str, str], bool | None] = day_covered,
    ):
        self.client = client
        self.rng = rng or random.Random()
        self.archive_lookup = archive_lookup
        self.coverage = coverage
        self.download = download
        self.day_coverage = day_coverage
        self.hits: list[dict] = []
        self.covered: dict[str, bool | None] = {}
        self.picked: dict | None = None
        self.component: dict | None = None
        self.estimate: Estimate | None = None
        self.confirmed: Estimate | None = None

        self.start_year = w.BoundedIntText(value=1850, min=FIRST_YEAR, max=LAST_YEAR, description="From")
        self.end_year = w.BoundedIntText(value=1859, min=FIRST_YEAR, max=LAST_YEAR, description="To")
        self.keywords = w.Text(value="yellow fever epidemic quarantine", description="Keywords",
                               placeholder="words, any of them; more matches rank higher", layout=w.Layout(width="60%"))
        self.topic = w.Text(value="", description="Topic", placeholder="optional: one sentence, matched by meaning",
                            layout=w.Layout(width="60%"))
        self.search_button = w.Button(description="Search and pick (up to 3 tokens)", button_style="primary")
        self.repick_button = w.Button(description="Pick another (2 tokens)", disabled=True)
        self.confirm_button = w.Button(description="Compare this issue", button_style="success", disabled=True)
        self.download_button = w.Button(description="Download American Stories year (free)", disabled=True)
        self.message = w.HTML()
        self.results = w.Output()
        self.pick_view = w.Output()

        self.search_button.on_click(lambda _: self._guarded(self.search))
        self.repick_button.on_click(lambda _: self._guarded(self.pick_another))
        self.confirm_button.on_click(lambda _: self._guarded(self.confirm))
        self.download_button.on_click(lambda _: self._guarded(self.download_issue))
        self.widget = w.VBox([
            w.HBox([self.start_year, self.end_year]),
            self.keywords,
            self.topic,
            w.HBox([self.search_button, self.repick_button, self.confirm_button, self.download_button]),
            self.message,
            self.results,
            self.pick_view,
        ])

    def _guarded(self, action: Callable[[], None]) -> None:
        """Disable all four buttons while `action` runs, so a double-click cannot spend tokens twice.

        On any exception, clear the estimate and confirmation and show the error instead of raising it
        into ipywidgets. Either way, re-enable the search button and set the other three from current
        state, so button state lives in this one place rather than being repeated in every action.
        """
        self.search_button.disabled = self.repick_button.disabled = self.confirm_button.disabled = True
        self.download_button.disabled = True
        try:
            action()
        except Exception as err:
            self.estimate = None
            self.confirmed = None
            self._say(f"{type(err).__name__}: {err}")
        finally:
            self.search_button.disabled = False
            self.repick_button.disabled = not (self.picked and len(self.hits) > 1)
            self.confirm_button.disabled = not (self.estimate and self.estimate.ready)
            self.download_button.disabled = not (self.estimate and self.estimate.needs_download)

    def search(self) -> None:
        self._reset()
        start, end = self.start_year.value, self.end_year.value
        keywords, topic = self.keywords.value.strip(), self.topic.value.strip()
        if end < start:
            self._say("The end year is before the start year.")
            return
        if not (keywords or topic):
            self._say("Give some keywords or a topic.")
            return
        if any(0 < len(text) < MIN_TEXT for text in (keywords, topic)):
            self._say(f"The keywords and the topic need at least {MIN_TEXT} characters each.")
            return
        response = self.client.find_articles(start, end, keywords, topic=topic)
        self.hits = response["hits"]
        self.covered = {h["uuid"]: self.day_coverage(h.get("lccn"), h.get("issue_date")) for h in self.hits}
        with self.results:
            print(f"{response['total_components']} matching articles ({response['mode']}); the top {len(self.hits)}:")
            with pd.option_context("display.max_colwidth", 90, "display.max_columns", None, "display.width", None):
                display(hits_table(self.hits, covered=self.covered))
        if not self.hits:
            self._say("No hits: widen the years or change the words.")
            return
        self.pick_another()

    def pick_another(self) -> None:
        self.estimate = None
        self.confirmed = None
        self.pick_view.clear_output()
        candidates = [h for h in self.hits if self.covered.get(h["uuid"]) is not False] or self.hits
        self.picked = pick(candidates, self.rng, avoid=self.picked)
        self.component = self.client.component(self.picked["uuid"])
        self.estimate = estimate(self.client, self.component, archive_lookup=self.archive_lookup, coverage=self.coverage)
        page = render_pick(self.picked, self.component, self.estimate)  # built before `with`, so errors aren't swallowed
        self._say("")
        with self.pick_view:
            display(HTML(page))

    def confirm(self) -> None:
        self.confirmed = self.estimate
        self._say("Confirmed. Run the next cells to fetch and compare this issue.")

    def download_issue(self) -> None:
        """Download the pick's year archive (free, once), then re-estimate: no new pick, no new tokens."""
        archive = self.estimate.archive
        gb = archive.size / 1e9
        self._say(f"Downloading {archive.filename} ({gb:.1f} GB)... then indexing it once (about 20 s).")
        with self.pick_view:
            self.download(archive)
        self.estimate = estimate(self.client, self.component, archive_lookup=self.archive_lookup, coverage=self.coverage)
        page = render_pick(self.picked, self.component, self.estimate)  # built before `with`, so errors aren't swallowed
        self.pick_view.clear_output()
        with self.pick_view:
            display(HTML(page))

    def _reset(self) -> None:
        self.hits, self.covered = [], {}
        self.picked, self.component, self.estimate, self.confirmed = None, None, None, None
        self.repick_button.disabled = self.confirm_button.disabled = self.download_button.disabled = True
        self.results.clear_output()
        self.pick_view.clear_output()
        self._say("")

    def _say(self, text: str) -> None:
        self.message.value = f"<b>{html.escape(text)}</b>" if text else ""
