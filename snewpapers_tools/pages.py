"""Which page of the issue an American Stories scan is.

American Stories' file names carry a page number, but it is unreliable: some issues name every
scan `p1` (the Martinsburgh Gazette 1814, The Independent 1897). The names also carry the LoC
sequence number (`seq`), and the API's `links.loc_pdf_pages` give page_number -> .../{seq}.pdf,
so scans are mapped to pages by sequence, falling back to the name only when no component links
that sequence.

American Stories sometimes segmented an older LoC digitisation of the issue than the one the API's
boxes refer to (another reel and sequence numbering): `same_scan` counts the AS scans whose
`.../{reel}/{YYYYMMDD}{ed:02}/{seq}` the components' links point at.
"""
from __future__ import annotations

from snewpapers_tools.american_stories import ScanName


def seq_of(url: str) -> str:
    """The LoC sequence number in a page URL: the file stem (".../0565.pdf" -> "0565")."""
    return url.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def page_by_seq(components: list[dict]) -> dict[str, int]:
    """seq -> page number, from every component's links.loc_pdf_pages."""
    return {
        seq_of(link["url"]): link["page_number"]
        for comp in components
        for link in (comp.get("links") or {}).get("loc_pdf_pages") or []
    }


def as_page_number(scan: ScanName, by_seq: dict[str, int]) -> int | None:
    """The scan's page in the issue: by its sequence when linked, else the page in its name (may be None)."""
    return by_seq.get(scan.seq, scan.page)


def _seq_int(seq: str) -> int | None:
    """`seq` as an int when it is purely digits; None for anything else (never raises)."""
    return int(seq) if seq.isdigit() else None


def as_page_numbers(scans: list[ScanName], by_seq: dict[str, int]) -> tuple[dict[str, int | None], list[str]]:
    """Every scan's page, keyed by its sequence.

    Linked scans get their page straight from `by_seq`. An unlinked scan is inferred from the
    nearest linked scan *on the same reel* (among `scans`), by sequence offset: if seq 0565 is
    linked to page 3, seq 0566 (one further along the same reel) is page 4. With no such anchor on
    its reel, it falls back to the page in its own name (which may be None).

    Two scans are never given the same page: once a page is taken (by a link, or by an earlier
    scan in this call), a later scan that would land on it is left `None` instead. Returns
    (seq -> page, the seqs left None because of such a collision), in scan order.
    """
    pages: dict[str, int | None] = {}
    taken: set[int] = set()
    for scan in scans:
        if scan.seq in by_seq:
            pages[scan.seq] = by_seq[scan.seq]
            taken.add(by_seq[scan.seq])

    anchors: dict[str, list[tuple[int, int]]] = {}
    for scan in scans:
        if scan.seq in by_seq:
            n = _seq_int(scan.seq)
            if n is not None:
                anchors.setdefault(scan.reel, []).append((n, by_seq[scan.seq]))

    collisions = []
    for scan in scans:
        if scan.seq in pages:
            continue
        n = _seq_int(scan.seq)
        candidates = anchors.get(scan.reel)
        page = None
        if n is not None and candidates:
            nearest_n, nearest_page = min(candidates, key=lambda a: abs(a[0] - n))
            page = nearest_page + (n - nearest_n)
        if page is None:
            page = scan.page
        if page is not None and page in taken:
            collisions.append(scan.seq)
            page = None
        pages[scan.seq] = page
        if page is not None:
            taken.add(page)
    return pages, collisions


def scan_path(scan: ScanName) -> str:
    """The scan's place in a LoC page URL: `/{reel}/{YYYYMMDD}{edition:02}/{seq}`."""
    return f"/{scan.reel}/{scan.date.replace('-', '')}{scan.edition:02d}/{scan.seq}"


def linked_stems(components: list[dict]) -> set[str]:
    """Every components' links.loc_pdf_pages URL without its file extension."""
    return {
        link["url"].rsplit(".", 1)[0] if "." in link["url"].rsplit("/", 1)[-1] else link["url"]
        for comp in components
        for link in (comp.get("links") or {}).get("loc_pdf_pages") or []
    }


def is_same_scan(scan: ScanName, stems: set[str]) -> bool:
    """Whether a link (one of `linked_stems`) points at exactly this scan: same reel, day, edition and seq."""
    path = scan_path(scan)
    return any(stem.endswith(path) for stem in stems)


def same_scan(pair) -> tuple[int, int]:
    """(American Stories scans the components' LoC links point at, all American Stories scans)."""
    stems = linked_stems(pair.components)
    return sum(is_same_scan(scan, stems) for scan, _ in pair.pages), len(pair.pages)
