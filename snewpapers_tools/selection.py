"""Choosing one article from search hits, and what comparing its issue would cost."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from snewpapers_tools.american_stories import ArchiveInfo, archive_info, issue_coverage
from snewpapers_tools.api import SnewpapersClient, issue_cost
from snewpapers_tools.issues import IssueKey, issue_key

HIT_COLUMNS = ["newspaper", "issue_date", "pages", "label", "quality_tier", "summary"]


def hits_table(hits: list[dict], *, covered: dict[str, bool | None] | None = None) -> pd.DataFrame:
    """One row per hit, with the columns a reader scans to choose an article.

    `covered`, keyed by hit uuid, says whether American Stories has that hit's day locally
    (True/False) or that is not known yet (None, the default for every hit without an entry).
    """
    covered = covered or {}
    columns = HIT_COLUMNS + ["in_american_stories"]
    rows = [{**{col: h.get(col) for col in HIT_COLUMNS}, "in_american_stories": covered.get(h.get("uuid"))} for h in hits]
    return pd.DataFrame(rows, columns=columns)


def pick(hits: list[dict], rng: random.Random, *, avoid: dict | None = None) -> dict:
    """A random hit from the page (the API has no random sort).

    Prefers a hit from a different issue than `avoid`, so a repick is worth the tokens; failing that,
    any other hit; failing that (avoid is everything there is), any hit at all.
    """
    if not hits:
        raise ValueError("no hits to pick from")
    if avoid is not None:
        other_issue = [h for h in hits if h.get("issue_uuid") != avoid.get("issue_uuid")]
        if other_issue:
            return rng.choice(other_issue)
    others = [h for h in hits if h is not avoid]
    return rng.choice(others or hits)


@dataclass(frozen=True)
class Estimate:
    issue: IssueKey
    newspaper: str
    total_components: int
    cost: dict  # issue_cost(): browse, text, total (an upper bound: texts already read are free)
    balance: int
    archive: ArchiveInfo | None  # None: American Stories has no archive for that year
    as_pages: list | None = None  # None: not known yet (archive not downloaded); []: AS lacks the issue
    issue_pages: int | None = None  # the issue's real page count, from browse page 1; None if not reported

    @property
    def ready(self) -> bool:
        """The balance covers the whole issue and American Stories is known to have the issue."""
        return self.archive is not None and bool(self.as_pages) and self.cost["total"] <= self.balance

    @property
    def needs_download(self) -> bool:
        """American Stories has a year archive, but it is not downloaded, so coverage is unknown."""
        return self.archive is not None and self.as_pages is None


def estimate(
    client: SnewpapersClient,
    component: dict,
    *,
    archive_lookup: Callable[[int], ArchiveInfo | None] = archive_info,
    coverage: Callable[[ArchiveInfo | None, IssueKey], list | None] = issue_coverage,
) -> Estimate:
    """Size the component's issue with browse page 1 (1 token, cached) and price comparing all of it."""
    key = issue_key(component)
    first = client.browse_issue_start(key.issue_uuid)
    total = first["total_components"]
    archive = archive_lookup(key.year)
    return Estimate(
        issue=key,
        newspaper=component["newspaper"],
        total_components=total,
        cost=issue_cost(total),
        balance=client.tokens()["balance"],
        archive=archive,
        as_pages=coverage(archive, key),
        issue_pages=first.get("issue_pages"),
    )
