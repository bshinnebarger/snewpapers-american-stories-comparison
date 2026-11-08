"""Both versions of one issue: the API's components with full text, and American Stories' page scans."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from snewpapers_tools.american_stories import ScanName, download_archive, extract_issue, load_pages
from snewpapers_tools.api import SnewpapersClient
from snewpapers_tools.issues import IssueKey
from snewpapers_tools.pages import as_page_numbers, page_by_seq
from snewpapers_tools.selection import Estimate


@dataclass
class IssuePair:
    issue: IssueKey
    components: list[dict]  # API components in page order, each with content, pages, and boxes
    pages: list[tuple[ScanName, dict]]  # American Stories page scans in page order
    tokens_spent: int
    total_components: int  # from the browse: how many components the issue has
    missing: list[str]  # browse hit uuids that components_text did not return

    def per_page(self) -> pd.DataFrame:
        """Counts per page from both sides. A component spanning pages counts on each; its text on its first page.

        American Stories scans are put on their page by sequence (see pages.py), not by the page in their name.
        """
        api_n: dict = defaultdict(int)
        api_chars: dict = defaultdict(int)
        for comp in self.components:
            pages = comp.get("pages") or [None]
            for page in pages:
                api_n[page] += 1
            api_chars[pages[0]] += comp.get("char_count") or len(comp.get("content") or "")
        as_n: dict = defaultdict(int)
        as_chars: dict = defaultdict(int)
        as_illegible: dict = defaultdict(int)
        by_seq = page_by_seq(self.components)
        pages = as_page_numbers([scan for scan, _ in self.pages], by_seq)[0]
        for scan, data in self.pages:
            page = pages.get(scan.seq)
            as_n[page] += len(data["full articles"])
            as_chars[page] += sum(len(a.get("article") or "") for a in data["full articles"])
            as_illegible[page] += sum(b.get("legibility") == "Illegible" for b in data["bboxes"])
        pages = sorted(set(api_n) | set(as_n), key=lambda p: (p is None, p or 0))
        return pd.DataFrame(
            [
                {
                    "page": page,
                    "api_components": api_n[page],
                    "api_chars": api_chars[page],
                    "as_articles": as_n[page],
                    "as_chars": as_chars[page],
                    "as_illegible_boxes": as_illegible[page],
                }
                for page in pages
            ]
        )


def american_stories_pages(est: Estimate) -> list[tuple[ScanName, dict]]:
    """Download the year's archive if needed, extract the issue, load its pages."""
    return load_pages(extract_issue(download_archive(est.archive), est.issue))


def fetch_both(
    client: SnewpapersClient,
    est: Estimate | None,
    *,
    american_stories: Callable[[Estimate], list[tuple[ScanName, dict]]] = american_stories_pages,
) -> IssuePair:
    """Everything the comparison needs for a confirmed estimate. Spends at most est.cost tokens.

    Refuses before spending anything: when American Stories has no archive for the year, and when
    American Stories has no pages for this issue (the estimate may have been made before the year
    was downloaded).

    Each merged component keeps its browse hit's `boxes` (free, already fetched by browse_issue);
    text components from POST /components carry no boxes of their own.
    """
    if est is None:
        raise ValueError("Nothing confirmed yet: click 'Compare this issue' in the form first.")
    if est.archive is None:
        raise ValueError(f"American Stories has no archive for {est.issue.year}; nothing spent.")
    pages = american_stories(est)  # free, and the slow part the first time for a year
    if not pages:
        issue = est.issue
        raise ValueError(f"American Stories has no pages for {issue.lccn} {issue.date} ed-{issue.edition}; nothing spent.")
    before = client.tokens()["balance"]
    browsed = client.browse_issue(est.issue.issue_uuid)
    hits = browsed["hits"]
    texts = client.components_text([h["uuid"] for h in hits])
    hits_by_uuid = {h["uuid"]: h for h in hits}
    components = [
        c | {"pages": c.get("pages") or hits_by_uuid.get(c["uuid"], {}).get("pages"),
             "boxes": hits_by_uuid.get(c["uuid"], {}).get("boxes")}
        for c in texts
    ]
    spent = before - client.tokens()["balance"]
    found = {c["uuid"] for c in texts}
    missing = [h["uuid"] for h in hits if h["uuid"] not in found]
    return IssuePair(est.issue, components, pages, spent, browsed["total_components"], missing)
