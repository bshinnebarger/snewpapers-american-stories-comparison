"""Page scans (JP2) from the Library of Congress, for drawing layouts on.

www.loc.gov's JSON is behind a bot challenge, but tile.loc.gov serves the JP2s directly. A
component's links.loc_pdf_pages give the issue's directory on tile.loc.gov; each page is
{seq}.jp2 in it, and American Stories' scan names carry that seq.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
from PIL import Image

from snewpapers_tools.american_stories import ScanName
from snewpapers_tools.pages import as_page_numbers, page_by_seq
from snewpapers_tools.paths import DATA_DIR

LOC_DIR = DATA_DIR / "loc"
USER_AGENT = "snewpapers-public-tools (research notebook; https://snewpapers.com)"

Image.MAX_IMAGE_PIXELS = 200_000_000  # newspaper scans are ~75 MP; keep Pillow's bomb check for anything wilder


def page_jp2_urls(components: list[dict], scans: list[ScanName]) -> dict[int, str]:
    """page number -> JP2 URL, for every American Stories page whose reel matches the API's directory.

    Pages are numbered by sequence (see pages.py), so scans whose names all say p1 still get their own pages.
    """
    directory = next(
        (link["url"].rsplit("/", 1)[0] for c in components for link in (c.get("links") or {}).get("loc_pdf_pages") or []),
        None,
    )
    if directory is None:
        return {}
    by_seq = page_by_seq(components)
    pages, _ = as_page_numbers(scans, by_seq)
    urls = {}
    for scan in scans:
        day_edition = f"{scan.date.replace('-', '')}{scan.edition:02d}"
        page = pages.get(scan.seq)
        if page is not None and directory.endswith(f"/{scan.reel}/{day_edition}"):
            urls[page] = f"{directory}/{scan.seq}.jp2"
    return urls


def download_jp2(
    url: str, folder: Path, *, http: httpx.Client | None = None, attempts: int = 4, backoff: float = 5.0
) -> Path:
    """The JP2 on disk under `folder`, downloaded the first time (about 10 MB a page).

    tile.loc.gov sometimes drops a connection mid-file or answers 429 or a 5xx (including Cloudflare's 52x): those are retried
    (`attempts` in all, waiting `backoff`, then twice as long each time). A 404 fails at once.
    """
    path = folder / url.rsplit("/", 1)[-1]
    if path.exists():
        return path
    folder.mkdir(parents=True, exist_ok=True)
    http = http or httpx.Client(timeout=120.0, follow_redirects=True)
    tmp = path.with_suffix(".part")
    for attempt in range(1, attempts + 1):
        try:
            with http.stream("GET", url, headers={"User-Agent": USER_AGENT}) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as out:
                    for chunk in resp.iter_bytes():
                        out.write(chunk)
            os.replace(tmp, path)
            return path
        except (httpx.TransportError, httpx.HTTPStatusError) as err:
            retryable = not isinstance(err, httpx.HTTPStatusError) or (err.response.status_code == 429 or err.response.status_code >= 500)
            if not retryable or attempt == attempts:
                raise
            time.sleep(backoff * 2 ** (attempt - 1))
        finally:
            tmp.unlink(missing_ok=True)
    return path


def load_page(path: Path, *, width: int = 1200) -> Image.Image:
    """The page as RGB at `width` px wide. Decodes at half resolution when it can (twice as fast)."""
    image = Image.open(path)
    image.reduce = 1
    try:
        image.load()
    except OSError:
        image = Image.open(path)
        image.load()
    height = round(image.height * width / image.width)
    return image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
