import httpx
import pytest
from PIL import Image

from snewpapers_tools.american_stories import ScanName
from snewpapers_tools.loc import download_jp2, load_page, page_jp2_urls

DIR = "https://tile.loc.gov/storage-services/service/ndnp/ncu/batch_ncu_lumber_ver01/data/sn84026536/00295879087/1856080801"


def scan(page, seq, reel="00295879087"):
    return ScanName("1856-08-08", page, "sn84026536", reel, 1, seq)


COMPONENTS = [
    {"uuid": "a", "links": {"loc_pdf_pages": [{"page_number": 3, "url": f"{DIR}/0565.pdf"}]}},
    {"uuid": "b", "links": {"loc_pdf_pages": []}},
]


def test_page_jp2_urls_use_the_component_directory_and_the_scan_sequence():
    urls = page_jp2_urls(COMPONENTS, [scan(1, "0563"), scan(3, "0565")])
    assert urls == {1: f"{DIR}/0563.jp2", 3: f"{DIR}/0565.jp2"}


def test_page_jp2_urls_skip_pages_on_another_reel_or_without_a_page_number():
    urls = page_jp2_urls(COMPONENTS, [scan(2, "0564", reel="00295879999"), scan(None, "0570")])
    assert urls == {}


def test_page_jp2_urls_without_any_pdf_link_is_empty():
    assert page_jp2_urls([COMPONENTS[1]], [scan(1, "0563")]) == {}


def test_download_jp2_fetches_once_with_a_user_agent(tmp_path):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=b"jp2-bytes")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    path = download_jp2(f"{DIR}/0565.jp2", tmp_path, http=client)
    assert path == tmp_path / "0565.jp2" and path.read_bytes() == b"jp2-bytes"
    download_jp2(f"{DIR}/0565.jp2", tmp_path, http=client)
    assert len(seen) == 1
    assert "snewpapers-public-tools" in seen[0].headers["User-Agent"]


def test_download_jp2_raises_on_http_errors_and_leaves_nothing(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(httpx.HTTPStatusError):
        download_jp2(f"{DIR}/0565.jp2", tmp_path, http=client)
    assert list(tmp_path.iterdir()) == []


def test_download_jp2_retries_a_dropped_connection_and_a_503(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("peer closed connection", request=request)
        if len(calls) == 2:
            return httpx.Response(503)
        return httpx.Response(200, content=b"jp2-bytes")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    path = download_jp2(f"{DIR}/0565.jp2", tmp_path, http=client, backoff=0)
    assert path.read_bytes() == b"jp2-bytes" and len(calls) == 3
    assert not list(tmp_path.glob("*.part"))


def test_download_jp2_gives_up_after_its_attempts(tmp_path):
    def handler(request):
        raise httpx.RemoteProtocolError("peer closed connection", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.RemoteProtocolError):
        download_jp2(f"{DIR}/0565.jp2", tmp_path, http=client, attempts=2, backoff=0)
    assert list(tmp_path.iterdir()) == []


def test_load_page_scales_to_the_requested_width(tmp_path):
    path = tmp_path / "page.jp2"
    Image.new("L", (800, 1000), 200).save(path)  # Pillow writes JPEG 2000 from the .jp2 suffix
    page = load_page(path, width=200)
    assert page.size == (200, 250)
    assert page.mode == "RGB"


def test_page_jp2_urls_key_scans_by_sequence_when_every_name_says_p1():
    comps = [{"uuid": u, "links": {"loc_pdf_pages": [{"page_number": n, "url": f"{DIR}/{seq}.pdf"}]}}
             for u, n, seq in (("a", 1, "0563"), ("b", 2, "0564"))]
    urls = page_jp2_urls(comps, [scan(1, "0563"), scan(1, "0564")])
    assert urls == {1: f"{DIR}/0563.jp2", 2: f"{DIR}/0564.jp2"}
