import io
import json
import tarfile
from types import SimpleNamespace

import pytest

from snewpapers_tools.american_stories import (
    ArchiveInfo,
    REPO_ID,
    REVISION,
    ScanName,
    archive_filename,
    archive_info,
    day_covered,
    download_archive,
    extract_issue,
    issue_coverage,
    load_pages,
    parse_scan_name,
    paper_day_covered,
    year_index,
)
from snewpapers_tools.issues import IssueKey


class FakeHub:
    """Stands in for HfApi: knows some file sizes, leaves unknown paths out like the real one."""

    def __init__(self, sizes):
        self.sizes = sizes
        self.calls = []

    def get_paths_info(self, repo_id, paths, *, repo_type, revision):
        self.calls.append((repo_id, paths, repo_type, revision))
        return [SimpleNamespace(path=p, size=self.sizes[p]) for p in paths if p in self.sizes]


def test_archive_filename():
    assert archive_filename(1853) == "faro_1853.tar.gz"


def test_archive_info_reports_size_and_where_it_will_live(tmp_path):
    hub = FakeHub({"faro_1853.tar.gz": 1_358_708_574})
    info = archive_info(1853, api=hub, archive_dir=tmp_path)
    assert (info.year, info.filename, info.size) == (1853, "faro_1853.tar.gz", 1_358_708_574)
    assert info.path == tmp_path / "faro_1853.tar.gz"
    assert hub.calls == [(REPO_ID, ["faro_1853.tar.gz"], "dataset", REVISION)]


def test_archive_info_is_local_once_the_file_exists(tmp_path):
    info = archive_info(1853, api=FakeHub({"faro_1853.tar.gz": 10}), archive_dir=tmp_path)
    assert not info.is_local
    info.path.write_bytes(b"x" * 10)
    assert info.is_local


def test_archive_info_is_none_for_a_year_the_dataset_lacks(tmp_path):
    assert archive_info(1775, api=FakeHub({}), archive_dir=tmp_path) is None


def test_is_local_is_false_for_a_partial_copy(tmp_path):
    info = archive_info(1853, api=FakeHub({"faro_1853.tar.gz": 10}), archive_dir=tmp_path)
    info.path.write_bytes(b"xxx")  # only 3 of the expected 10 bytes
    assert not info.is_local


def test_parse_scan_name_reads_both_ends():
    assert parse_scan_name("faro_1856/1856-04-22_p2_sn85025007_00280761023_1856042201_0412.json") == ScanName(
        date="1856-04-22", page=2, lccn="sn85025007", reel="00280761023", edition=1, seq="0412"
    )


def test_parse_scan_name_handles_odd_reels_and_pages():
    lettered = parse_scan_name("1856-09-11_p3_sn87065473_0029587756A_1856091101_0177.json")
    assert (lettered.reel, lettered.page) == ("0029587756A", 3)
    no_reel = parse_scan_name("1856-01-25_pNone_sn83025141_no_reel_1856012502_0402.json")
    assert (no_reel.reel, no_reel.page, no_reel.edition) == ("no_reel", None, 2)


def test_parse_scan_name_rejects_other_files():
    assert parse_scan_name("faro_1856") is None
    assert parse_scan_name("faro_1856/README.txt") is None
    assert parse_scan_name("faro_1856/not_a_scan.json") is None


ISSUE = IssueKey("iss-1", "sn85025007", "1856-04-22", 1)


def page_json(n):
    return {"lccn": {"lccn": "sn85025007"}, "edition": {"edition": "ed-01"}, "page_number": "na",
            "bboxes": [], "full articles": [{"full_article_id": 1, "headline": "", "article": f"page {n}", "byline": ""}]}


def make_archive(path, members):
    """A tiny faro_{year}.tar.gz: a directory entry plus JSON members."""
    with tarfile.open(path, "w:gz") as tar:
        folder = tarfile.TarInfo("faro_1856")
        folder.type = tarfile.DIRTYPE
        tar.addfile(folder)
        for name, payload in members.items():
            data = json.dumps(payload).encode()
            info = tarfile.TarInfo(f"faro_1856/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


ISSUE_MEMBERS = {
    "1856-04-22_p2_sn85025007_00280761023_1856042201_0412.json": page_json(2),
    "1856-04-22_p1_sn85025007_00280761023_1856042201_0411.json": page_json(1),
    "1856-04-23_p1_sn85025007_00280761023_1856042301_0413.json": page_json(9),  # next day
    "1856-04-22_p1_sn85025007_00280761023_1856042202_0414.json": page_json(9),  # second edition
    "1856-04-22_p1_sn84024738_00202190984_1856042201_0564.json": page_json(9),  # another paper
}


def test_extract_issue_copies_only_that_issue_in_page_order(tmp_path):
    archive = make_archive(tmp_path / "faro_1856.tar.gz", ISSUE_MEMBERS)
    paths = extract_issue(archive, ISSUE, issues_dir=tmp_path / "issues", index_dir=tmp_path / "index")
    assert [p.name for p in paths] == [
        "1856-04-22_p1_sn85025007_00280761023_1856042201_0411.json",
        "1856-04-22_p2_sn85025007_00280761023_1856042201_0412.json",
    ]
    assert paths[0].parent == tmp_path / "issues" / "sn85025007" / "1856-04-22" / "ed-1"
    assert [data["full articles"][0]["article"] for _, data in load_pages(paths)] == ["page 1", "page 2"]


def test_extract_issue_reads_the_archive_only_once(tmp_path):
    archive = make_archive(tmp_path / "faro_1856.tar.gz", ISSUE_MEMBERS)
    first = extract_issue(archive, ISSUE, issues_dir=tmp_path / "issues", index_dir=tmp_path / "index")
    archive.unlink()
    assert extract_issue(archive, ISSUE, issues_dir=tmp_path / "issues", index_dir=tmp_path / "index") == first


def test_extract_issue_of_an_issue_the_archive_lacks_is_empty_and_remembered(tmp_path):
    archive = make_archive(tmp_path / "faro_1856.tar.gz", ISSUE_MEMBERS)
    missing = IssueKey("iss-2", "sn99999999", "1856-04-22", 1)
    assert extract_issue(archive, missing, issues_dir=tmp_path / "issues", index_dir=tmp_path / "index") == []
    archive.unlink()
    assert extract_issue(archive, missing, issues_dir=tmp_path / "issues", index_dir=tmp_path / "index") == []


def test_extract_issue_redoes_an_interrupted_extraction(tmp_path):
    archive = make_archive(tmp_path / "faro_1856.tar.gz", ISSUE_MEMBERS)
    stale = tmp_path / "issues" / "sn85025007" / "1856-04-22" / "ed-1"
    stale.mkdir(parents=True)
    (stale / "leftover.json").write_text("{}")
    paths = extract_issue(archive, ISSUE, issues_dir=tmp_path / "issues", index_dir=tmp_path / "index")
    assert len(paths) == 2
    assert not (stale / "leftover.json").exists()


def test_year_index_groups_pages_by_issue_and_edition(tmp_path):
    archive = make_archive(tmp_path / "faro_1856.tar.gz", ISSUE_MEMBERS)
    index = year_index(archive, index_dir=tmp_path / "index")
    assert index == {
        "sn85025007/1856-04-22": {"1": [1, 2], "2": [1]},
        "sn85025007/1856-04-23": {"1": [1]},
        "sn84024738/1856-04-22": {"1": [1]},
    }
    cache_path = tmp_path / "index" / "faro_1856.tar.gz.json"
    cached = json.loads(cache_path.read_text())
    assert cached["revision"] == REVISION
    assert cached["rejected"] == 0
    assert cached["issues"] == index


def test_year_index_is_cached_after_the_first_pass(tmp_path):
    archive = make_archive(tmp_path / "faro_1856.tar.gz", ISSUE_MEMBERS)
    index_dir = tmp_path / "index"
    first = year_index(archive, index_dir=index_dir)
    archive.unlink()  # the cache must serve the second call without the archive
    assert year_index(archive, index_dir=index_dir) == first


def test_year_index_rebuilds_when_the_cached_revision_differs(tmp_path):
    archive = make_archive(tmp_path / "faro_1856.tar.gz", ISSUE_MEMBERS)
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    cache_path = index_dir / "faro_1856.tar.gz.json"
    cache_path.write_text(json.dumps({"revision": "stale", "rejected": 0, "issues": {"bogus/0000-00-00": {"9": [1]}}}))
    index = year_index(archive, index_dir=index_dir)
    assert "bogus/0000-00-00" not in index
    assert index["sn85025007/1856-04-22"]["1"] == [1, 2]
    assert json.loads(cache_path.read_text())["revision"] == REVISION


def test_year_index_counts_members_it_rejects(tmp_path):
    members = dict(ISSUE_MEMBERS)
    members["not_a_scan.json"] = {}
    archive = make_archive(tmp_path / "faro_1856.tar.gz", members)
    index_dir = tmp_path / "index"
    year_index(archive, index_dir=index_dir)
    cached = json.loads((index_dir / "faro_1856.tar.gz.json").read_text())
    assert cached["rejected"] == 1


def local_archive_info(path):
    return ArchiveInfo(1856, path.name, path.stat().st_size, path)


def test_issue_coverage_is_unknown_until_the_archive_is_downloaded(tmp_path):
    archive = make_archive(tmp_path / "faro_1856.tar.gz", ISSUE_MEMBERS)
    index_dir = tmp_path / "index"
    not_local = ArchiveInfo(1856, archive.name, archive.stat().st_size + 1, archive)
    assert issue_coverage(not_local, ISSUE, index_dir=index_dir) is None
    assert issue_coverage(None, ISSUE, index_dir=index_dir) is None


def test_issue_coverage_is_the_page_list_or_empty(tmp_path):
    archive = make_archive(tmp_path / "faro_1856.tar.gz", ISSUE_MEMBERS)
    info = local_archive_info(archive)
    index_dir = tmp_path / "index"
    assert issue_coverage(info, ISSUE, index_dir=index_dir) == [1, 2]
    absent = IssueKey("iss-2", "sn99999999", "1856-04-22", 1)
    assert issue_coverage(info, absent, index_dir=index_dir) == []


def test_paper_day_covered_is_true_for_any_edition():
    index = {"sn85025007/1856-04-22": {"1": [1, 2], "2": [1]}}
    assert paper_day_covered(index, "sn85025007", "1856-04-22")
    assert not paper_day_covered(index, "sn85025007", "1856-01-01")
    assert not paper_day_covered(index, "sn00000000", "1856-04-22")


def test_day_covered_looks_up_the_local_archive_without_a_network_call(tmp_path):
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    index_dir = tmp_path / "index"
    make_archive(archive_dir / "faro_1856.tar.gz", ISSUE_MEMBERS)
    assert day_covered("sn85025007", "1856-04-22", archive_dir=archive_dir, index_dir=index_dir) is True
    assert day_covered("sn00000000", "1856-04-22", archive_dir=archive_dir, index_dir=index_dir) is False
    assert day_covered("sn85025007", "1857-04-22", archive_dir=archive_dir, index_dir=index_dir) is None


def test_extract_issue_skips_the_pass_when_the_index_says_absent(tmp_path):
    archive = make_archive(tmp_path / "faro_1856.tar.gz", ISSUE_MEMBERS)
    index_dir = tmp_path / "index"
    year_index(archive, index_dir=index_dir)  # pre-build the index
    archive.unlink()  # the extraction pass, if attempted, would need this
    missing = IssueKey("iss-2", "sn99999999", "1856-04-22", 1)
    assert extract_issue(archive, missing, issues_dir=tmp_path / "issues", index_dir=index_dir) == []


def test_extract_issue_redoes_when_the_complete_marker_names_another_revision(tmp_path):
    archive = make_archive(tmp_path / "faro_1856.tar.gz", ISSUE_MEMBERS)
    folder = tmp_path / "issues" / "sn85025007" / "1856-04-22" / "ed-1"
    folder.mkdir(parents=True)
    (folder / "leftover.json").write_text("{}")
    (folder / "_complete").write_text("faro_1856.tar.gz @ some-older-revision\n")
    paths = extract_issue(archive, ISSUE, issues_dir=tmp_path / "issues", index_dir=tmp_path / "index")
    assert len(paths) == 2
    assert not (folder / "leftover.json").exists()


def test_extract_issue_raises_on_a_damaged_archive(tmp_path):
    path = tmp_path / "faro_1856.tar.gz"
    make_archive(path, ISSUE_MEMBERS)
    truncated = path.read_bytes()[:-150]
    path.write_bytes(truncated)
    with pytest.raises(RuntimeError, match="delete it"):
        extract_issue(path, ISSUE, issues_dir=tmp_path / "issues", index_dir=tmp_path / "index")


def test_extract_issue_sorts_pnone_pages_last(tmp_path):
    members = {
        "1856-04-22_p1_sn85025007_00280761023_1856042201_0411.json": page_json(1),
        "1856-04-22_pNone_sn85025007_00280761023_1856042201_0412.json": page_json(99),
        "1856-04-22_p2_sn85025007_00280761023_1856042201_0413.json": page_json(2),
    }
    archive = make_archive(tmp_path / "faro_1856.tar.gz", members)
    paths = extract_issue(archive, ISSUE, issues_dir=tmp_path / "issues", index_dir=tmp_path / "index")
    assert [p.name.split("_")[1] for p in paths] == ["p1", "p2", "pNone"]


def test_download_archive_fetches_the_pinned_file_once(tmp_path):
    calls = []

    def fake_download(repo_id, filename, *, repo_type, revision, local_dir):
        calls.append((repo_id, filename, repo_type, revision, local_dir))
        path = local_dir / filename
        path.write_bytes(b"gz")
        return str(path)

    info = ArchiveInfo(1856, "faro_1856.tar.gz", 2, tmp_path / "faro_1856.tar.gz")
    assert download_archive(info, download=fake_download) == info.path
    assert download_archive(info, download=fake_download) == info.path
    assert calls == [(REPO_ID, "faro_1856.tar.gz", "dataset", REVISION, tmp_path)]
