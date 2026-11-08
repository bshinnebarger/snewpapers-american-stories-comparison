"""American Stories (Dell et al., Harvard) on Hugging Face: one tar.gz of page-scan JSON per year."""
from __future__ import annotations

import json
import os
import shutil
import tarfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import HfApi, hf_hub_download

from snewpapers_tools.issues import IssueKey
from snewpapers_tools.paths import DATA_DIR

REPO_ID = "dell-research-harvard/AmericanStories"
# The dataset commit compared against (main as of 2026-10-23, the March 2025 update).
# Changing it changes the data, so a comparison names the revision it used.
REVISION = "77e27fa69c4788dfaad1c9efd8a226d5a32d3e9a"
ARCHIVE_DIR = DATA_DIR / "american_stories" / "archives"
ISSUES_DIR = DATA_DIR / "american_stories" / "issues"
INDEX_DIR = DATA_DIR / "american_stories" / "index"
_DONE = "_complete"  # written last in an issue folder: the extraction finished


@dataclass(frozen=True)
class ArchiveInfo:
    year: int
    filename: str
    size: int  # bytes
    path: Path  # where it is, or will be, downloaded

    @property
    def is_local(self) -> bool:
        """A full copy is on disk; a partial download (wrong size) does not count."""
        return self.path.exists() and self.path.stat().st_size == self.size


def archive_filename(year: int) -> str:
    return f"faro_{year}.tar.gz"


def archive_info(year: int, *, api: HfApi | None = None, archive_dir: Path = ARCHIVE_DIR) -> ArchiveInfo | None:
    """The year's archive on the Hub, from file metadata (nothing is downloaded); None if the dataset lacks that year."""
    filename = archive_filename(year)
    found = (api or HfApi()).get_paths_info(REPO_ID, [filename], repo_type="dataset", revision=REVISION)
    if not found:
        return None
    return ArchiveInfo(year, filename, found[0].size, archive_dir / filename)


@dataclass(frozen=True)
class ScanName:
    """One page scan, from its file name: {date}_p{page}_{lccn}_{reel}_{YYYYMMDD}{edition:02}_{seq}.json"""

    date: str  # YYYY-MM-DD
    page: int | None  # None when the name says pNone
    lccn: str
    reel: str  # may hold letters or "no_reel"
    edition: int
    seq: str


def parse_scan_name(name: str) -> ScanName | None:
    """Parse a member or file name; None for anything that is not a page scan."""
    base = name.rsplit("/", 1)[-1]
    if not base.endswith(".json"):
        return None
    fields = base[: -len(".json")].split("_")
    if len(fields) < 6 or not fields[1].startswith("p"):
        return None
    date, page, lccn, day_edition, seq = fields[0], fields[1][1:], fields[2], fields[-2], fields[-1]
    if len(day_edition) != 10 or not day_edition.isdigit() or day_edition[:8] != date.replace("-", ""):
        return None
    return ScanName(
        date=date,
        page=int(page) if page.isdigit() else None,
        lccn=lccn,
        reel="_".join(fields[3:-2]),
        edition=int(day_edition[8:]),
        seq=seq,
    )


def download_archive(info: ArchiveInfo, *, download: Callable = hf_hub_download) -> Path:
    """The year's archive on disk, downloading it (pinned revision) the first time. Progress shows in Jupyter."""
    if not info.is_local:
        info.path.parent.mkdir(parents=True, exist_ok=True)
        path = download(REPO_ID, info.filename, repo_type="dataset", revision=REVISION, local_dir=info.path.parent)
        return Path(path)
    return info.path


def issue_dir(key: IssueKey, issues_dir: Path = ISSUES_DIR) -> Path:
    return issues_dir / key.lccn / key.date / f"ed-{key.edition}"


def year_index(archive: Path, *, index_dir: Path = INDEX_DIR) -> dict[str, dict[str, list]]:
    """Which issues and editions the year archive has, and the pages of each: {"{lccn}/{date}": {"{edition}": [pages]}}.

    One streaming pass over the whole archive, file names only (no JSON parsing): about 20 seconds for a
    year, then served instantly from a cached JSON file until REVISION changes, when it is rebuilt.
    """
    cache_path = index_dir / f"{archive.name}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("revision") == REVISION:
            return cached["issues"]
    issues: dict[str, dict[str, list]] = {}
    rejected = 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            base = member.name.rsplit("/", 1)[-1]
            if not base.endswith(".json"):
                continue
            scan = parse_scan_name(member.name)
            if scan is None:
                rejected += 1
                continue
            by_edition = issues.setdefault(f"{scan.lccn}/{scan.date}", {})
            by_edition.setdefault(str(scan.edition), []).append(scan.page)
    for by_edition in issues.values():
        for edition, pages in by_edition.items():
            by_edition[edition] = sorted(pages, key=lambda p: (p is None, p or 0))
    index_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps({"revision": REVISION, "rejected": rejected, "issues": issues}))
    os.replace(tmp_path, cache_path)
    return issues


def issue_coverage(info: ArchiveInfo | None, key: IssueKey, *, index_dir: Path = INDEX_DIR) -> list | None:
    """The issue's page list in the year archive; None if that is not known yet (the archive isn't downloaded)."""
    if info is None or not info.is_local:
        return None
    by_edition = year_index(info.path, index_dir=index_dir).get(f"{key.lccn}/{key.date}", {})
    return by_edition.get(str(key.edition), [])


def paper_day_covered(index: dict, lccn: str, date: str) -> bool:
    """Whether the year index has any edition of this paper's day (search hits carry no edition)."""
    return bool(index.get(f"{lccn}/{date}"))


def day_covered(lccn: str, date: str, *, archive_dir: Path = ARCHIVE_DIR, index_dir: Path = INDEX_DIR) -> bool | None:
    """Whether American Stories' local year archive has this paper's day, without a network call.

    None if the year's archive has not been downloaded, so there is nothing local to check yet.
    """
    path = archive_dir / archive_filename(int(date[:4]))
    if not path.exists():
        return None
    return paper_day_covered(year_index(path, index_dir=index_dir), lccn, date)


def extract_issue(
    archive: Path, key: IssueKey, *, issues_dir: Path = ISSUES_DIR, index_dir: Path = INDEX_DIR
) -> list[Path]:
    """Copy the issue's page files out of the year archive, in page order.

    One streaming pass over the whole archive (its members are in no date order), matching on
    file names only. Done once per issue: the folder is marked complete, and an interrupted
    extraction (or one marked complete for a different REVISION) is started over. If a year index
    already says the issue is absent, the pass is skipped entirely.
    """
    folder = issue_dir(key, issues_dir)
    marker = f"{archive.name} @ {REVISION}\n"
    done = folder / _DONE
    if not (done.exists() and done.read_text() == marker):
        index_cache = index_dir / f"{archive.name}.json"
        known_absent = False
        if index_cache.exists():
            by_edition = year_index(archive, index_dir=index_dir).get(f"{key.lccn}/{key.date}", {})
            known_absent = str(key.edition) not in by_edition
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        if not known_absent:
            try:
                with tarfile.open(archive, "r|gz") as tar:
                    for member in tar:
                        scan = parse_scan_name(member.name) if member.isfile() else None
                        if scan and (scan.lccn, scan.date, scan.edition) == (key.lccn, key.date, key.edition):
                            (folder / member.name.rsplit("/", 1)[-1]).write_bytes(tar.extractfile(member).read())
            except (tarfile.ReadError, EOFError, zlib.error) as err:
                raise RuntimeError(f"{archive} looks damaged ({err}); delete it and run again to re-download")
        done.write_text(marker)
    scans = [(parse_scan_name(p.name), p) for p in folder.glob("*.json")]
    ordered = sorted(scans, key=lambda t: (t[0].page is None, t[0].page or 0, t[0].seq))
    return [p for _, p in ordered]


def load_pages(paths: list[Path]) -> list[tuple[ScanName, dict]]:
    """Each extracted page with its parsed name."""
    return [(parse_scan_name(p.name), json.loads(p.read_text())) for p in paths]
