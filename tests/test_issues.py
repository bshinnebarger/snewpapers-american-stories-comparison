import pytest

from snewpapers_tools.issues import IssueKey, issue_key


def component(url):
    return {"issue_uuid": "iss-1", "links": {"loc_issue_url": url}}


def test_issue_key_reads_lccn_date_and_edition_from_the_loc_url():
    key = issue_key(component("https://www.loc.gov/item/sn84020713/1850-03-02/ed-1/"))
    assert key == IssueKey(issue_uuid="iss-1", lccn="sn84020713", date="1850-03-02", edition=1)
    assert key.year == 1850


def test_issue_key_accepts_a_url_without_trailing_slash():
    assert issue_key(component("https://www.loc.gov/item/sn84020713/1850-03-02/ed-2")).edition == 2


def test_issue_key_rejects_an_unexpected_url():
    with pytest.raises(ValueError, match="loc_issue_url"):
        issue_key(component("https://www.loc.gov/item/sn84020713/"))
