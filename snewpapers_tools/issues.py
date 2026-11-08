"""One newspaper issue, identified the way both the API and American Stories can find it."""
from __future__ import annotations

import re
from dataclasses import dataclass

# links.loc_issue_url on a component: https://www.loc.gov/item/{lccn}/{YYYY-MM-DD}/ed-{N}/
_LOC_ISSUE = re.compile(r"/item/(?P<lccn>[^/]+)/(?P<date>\d{4}-\d{2}-\d{2})/ed-(?P<edition>\d+)/?$")


@dataclass(frozen=True)
class IssueKey:
    issue_uuid: str  # the API's id for the issue
    lccn: str
    date: str  # YYYY-MM-DD
    edition: int

    @property
    def year(self) -> int:
        return int(self.date[:4])


def issue_key(component: dict) -> IssueKey:
    """The issue a GET /components/{uuid} response belongs to (search hits carry no edition)."""
    url = component["links"]["loc_issue_url"]
    match = _LOC_ISSUE.search(url)
    if not match:
        raise ValueError(f"unexpected loc_issue_url: {url!r}")
    return IssueKey(component["issue_uuid"], match["lccn"], match["date"], int(match["edition"]))
