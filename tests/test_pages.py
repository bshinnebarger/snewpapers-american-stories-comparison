from snewpapers_tools.american_stories import ScanName
from snewpapers_tools.fetching import IssuePair
from snewpapers_tools.issues import IssueKey
from snewpapers_tools.pages import as_page_number, as_page_numbers, is_same_scan, linked_stems, page_by_seq, same_scan, scan_path, seq_of

DIR = "https://tile.loc.gov/storage-services/service/ndnp/wvu/batch_wvu_x/data/sn84026815/00000000001/1814010101"


def scan(page, seq):
    return ScanName("1814-01-01", page, "sn84026815", "00000000001", 1, seq)


def linked(*pairs):
    """One component per (page_number, seq) link."""
    return [{"uuid": f"c{n}", "links": {"loc_pdf_pages": [{"page_number": n, "url": f"{DIR}/{seq}.pdf"}]}}
            for n, seq in pairs]


def test_seq_of_is_the_file_stem_of_the_url():
    assert seq_of(f"{DIR}/0565.pdf") == "0565"


def test_page_by_seq_collects_every_components_pdf_links():
    comps = linked((1, "0001"), (2, "0002")) + [
        {"uuid": "x", "links": {"loc_pdf_pages": [{"page_number": 3, "url": f"{DIR}/0003.pdf"},
                                                   {"page_number": 4, "url": f"{DIR}/0004.pdf"}]}},
        {"uuid": "y", "links": {}},
        {"uuid": "z"},
    ]
    assert page_by_seq(comps) == {"0001": 1, "0002": 2, "0003": 3, "0004": 4}


def test_duplicate_p1_scan_names_map_to_their_pages_by_sequence():
    by_seq = page_by_seq(linked((1, "0001"), (2, "0002"), (3, "0003"), (4, "0004")))
    scans = [scan(1, s) for s in ("0001", "0002", "0003", "0004")]  # every scan named p1
    assert [as_page_number(s, by_seq) for s in scans] == [1, 2, 3, 4]


def test_an_unlinked_scan_falls_back_to_the_page_in_its_name():
    by_seq = page_by_seq(linked((1, "0001")))
    assert as_page_number(scan(3, "0003"), by_seq) == 3
    assert as_page_number(scan(None, "0009"), by_seq) is None


# --- as_page_numbers: batch inference with collision avoidance --------------------------------


def test_as_page_numbers_infers_unlinked_scans_from_the_nearest_linked_scan_by_sequence_offset():
    # Only 0565 is linked (to page 3); every scan's name says p1 (the unreliable case this exists for).
    by_seq = page_by_seq(linked((3, "0565")))
    scans = [scan(1, "0563"), scan(1, "0564"), scan(1, "0565"), scan(1, "0566")]
    pages, collisions = as_page_numbers(scans, by_seq)
    assert pages == {"0563": 1, "0564": 2, "0565": 3, "0566": 4}
    assert collisions == []


def test_as_page_numbers_falls_back_to_the_name_when_its_reel_has_no_linked_scan():
    by_seq = page_by_seq(linked((1, "0001")))  # linked, but on a different scan batch
    other_reel = ScanName("1814-01-01", 5, "sn84026815", "another_reel", 1, "0009")
    pages, collisions = as_page_numbers([other_reel], by_seq)
    assert pages == {"0009": 5}
    assert collisions == []


def test_as_page_numbers_never_assigns_two_scans_the_same_page():
    # Neither scan is linked, and both names say p1 (the duplicate-name bug): without an anchor on
    # the reel, both would fall back to the same page; the second is left None and reported instead.
    scans = [scan(1, "0001"), scan(1, "0002")]
    pages, collisions = as_page_numbers(scans, {})
    assert pages == {"0001": 1, "0002": None}
    assert collisions == ["0002"]


def test_as_page_numbers_avoids_a_collision_between_an_inferred_page_and_a_name_fallback():
    # 0565 is linked to page 3. 0566 has no page in its name, but seq-offset inference from 0565
    # gives it page 4. A third scan on another reel (no anchor there) is *named* p4 too: its
    # name-fallback page collides with the inferred one, so it is left None and reported.
    by_seq = page_by_seq(linked((3, "0565")))
    other_reel = ScanName("1814-01-01", 4, "sn84026815", "another_reel", 1, "9999")
    scans = [scan(3, "0565"), scan(None, "0566"), other_reel]
    pages, collisions = as_page_numbers(scans, by_seq)
    assert pages == {"0565": 3, "0566": 4, "9999": None}
    assert collisions == ["9999"]


def test_scan_path_is_reel_day_edition_and_seq():
    assert scan_path(scan(1, "0565")) == "/00000000001/1814010101/0565"


def test_same_scan_counts_the_as_scans_the_components_link_to():
    """The Patowmac Guardian 1791: AS has reel 01000718192 seq 0007-0010, the API links
    .../sn84038410/print/1791071101/0133.pdf and on: another digitisation, so none is the same scan."""
    api = "https://tile.loc.gov/storage-services/service/ndnp/vi/batch_vi_x/data/sn84038410/print/1791071101"
    comps = [{"uuid": f"c{n}", "links": {"loc_pdf_pages": [{"page_number": n, "url": f"{api}/{133 + n - 1:04d}.pdf"}]}}
             for n in (1, 2, 3, 4)]
    scans = [ScanName("1791-07-11", n, "sn84038410", "01000718192", 1, f"{6 + n:04d}") for n in (1, 2, 3, 4)]
    pair = IssuePair(IssueKey("i", "sn84038410", "1791-07-11", 1), comps, [(s, {}) for s in scans], 0, 4, [])
    assert same_scan(pair) == (0, 4)
    as_pages = [(scan(n, f"000{n}"), {}) for n in (1, 2, 3)]
    partly = IssuePair(pair.issue, linked((1, "0001"), (2, "0002")), as_pages, 0, 2, [])
    assert same_scan(partly) == (2, 3)  # an AS scan no component links is not known to be the same


def test_is_same_scan_needs_the_whole_path_not_just_the_seq():
    stems = linked_stems(linked((1, "0001")))
    assert stems == {f"{DIR}/0001"}
    assert is_same_scan(scan(1, "0001"), stems)
    assert not is_same_scan(ScanName("1814-01-01", 1, "sn84026815", "00000000002", 1, "0001"), stems)  # another reel
    assert not is_same_scan(ScanName("1814-01-01", 1, "sn84026815", "00000000001", 2, "0001"), stems)  # another edition
    assert not is_same_scan(scan(1, "00001"), stems)
