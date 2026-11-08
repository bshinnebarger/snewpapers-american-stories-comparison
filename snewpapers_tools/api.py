"""Client for the snewpapers public API.

Base https://api.snewpapers.com, OpenAPI at /openapi.json, the method guide at
GET /guide (free). Prices in tokens: a search page 1, a component 1, a
component's full text 3 (once per account and component, re-reads free), the
balance 0. A short balance is a 402 and nothing is charged.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Callable

import httpx
from dotenv import load_dotenv

from snewpapers_tools.paths import DATA_DIR

BASE_URL = "https://api.snewpapers.com"
PAGE_SIZE = 25  # hits per search page, and ids per POST /components
TEXT_PRICE = 3  # tokens per component full text, first read only
CURSOR_TTL = 9 * 60  # a search cursor is valid 10 minutes; refetch page 1 before then


class InsufficientTokens(Exception):
    """402: the balance does not cover the call. Nothing was charged."""

    def __init__(self, detail: str, price: int | None, tokens_remaining: int | None):
        super().__init__(f"{detail} (price {price}, balance {tokens_remaining})")
        self.detail = detail
        self.price = price
        self.tokens_remaining = tokens_remaining


def issue_cost(total_components: int) -> dict:
    """Tokens to browse a whole issue and read every component's text.

    An upper bound: texts this account has already paid for are free.
    """
    browse = math.ceil(total_components / PAGE_SIZE)
    text = TEXT_PRICE * total_components
    return {"browse": browse, "text": text, "total": browse + text}


class SnewpapersClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        cache_dir: Path | None = DATA_DIR / "snewpapers_api",
        transport: httpx.BaseTransport | None = None,
    ):
        if api_key is None:
            load_dotenv()
            api_key = os.environ.get("SNEWPAPERS_API_KEY")
        if not api_key:
            raise RuntimeError("SNEWPAPERS_API_KEY is not set: copy .env.example to .env and put your key in it")
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
            transport=transport,
        )
        self.cache_dir = cache_dir

    def tokens(self) -> dict:
        """GET /account/tokens: balance, grant, purchased tokens, recent charges (free, never cached)."""
        return self._request("GET", "/account/tokens")

    def search(
        self,
        filters: dict,
        text: dict | None = None,
        *,
        sort: str = "relevance",
        page: int = 1,
        cursor: str | None = None,
        guarded: bool = False,
        include_boxes: bool = False,
    ) -> dict:
        """POST /search, one page of 25 full-shape hits (1 token). Not cached: pages > 1 need a live cursor.

        guarded=False also returns caution hits (about 4%) with their summaries, as printed.
        include_boxes=True adds each hit's `boxes` (free): rectangles per page, as fractions of the
        page image. Left out of the body when false, so an ordinary search's wire shape (and any
        cache key built from it) is unchanged.
        """
        body: dict = {"filters": filters, "sort": sort, "page": page, "guarded": guarded}
        if text:
            body["text"] = text
        if cursor:
            body["cursor"] = cursor
        if include_boxes:
            body["include_boxes"] = True
        return self._request("POST", "/search", body)

    def find_articles(self, start_year: int, end_year: int, keywords: str = "", *, topic: str = "",
                      refresh: bool = False) -> dict:
        """Page 1 of a search over whole years (cached). The keywords are matched as OR over summary,
        full text and keywords, hits ranked by how many match; a topic (one sentence) is matched by
        meaning over the narrative genres' summaries and widens the keyword match."""
        filters = {"start_date": f"{start_year}-01-01", "end_date": f"{end_year}-12-31"}
        text = {}
        if keywords.strip():
            text["should_have_keywords"] = keywords.strip()
        if topic.strip():
            text["semantic_text"] = topic.strip()
        key = _body_key({"filters": filters, "text": text, "guarded": False})
        return self._cached("search", key, lambda: self.search(filters, text or None), refresh=refresh)

    def component(self, uuid: str, *, refresh: bool = False) -> dict:
        """GET /components/{uuid}: links, citation, thumbs; no text (1 token, cached)."""
        return self._cached("component", uuid, lambda: self._request("GET", f"/components/{uuid}"), refresh=refresh)

    def browse_issue_start(self, issue_uuid: str, *, refresh: bool = False) -> dict:
        """Page 1 of the issue in page order, with boxes: total_components, issue_pages, and the
        cursor for the rest (1 token, cached). include_boxes is free, so every browsed hit carries
        its own `boxes` too.

        total_components is what the cost estimate needs before anything else is fetched. The cache
        key carries "_boxes" so an entry cached before this field existed is never served stale.
        """
        filters = {"issue_uuid": issue_uuid}
        return self._cached(
            "browse",
            f"{issue_uuid}_p1_boxes",
            lambda: self.search(filters, sort="pages:asc", include_boxes=True),
            refresh=refresh,
        )

    def browse_issue(self, issue_uuid: str, *, refresh: bool = False) -> dict:
        """Every component of the issue in page order, with boxes, without text (1 token per 25, cached whole).

        Raises instead of caching if the server returns fewer hits than total_components. The cache
        key carries "_boxes" so an entry cached before boxes existed is never served stale.
        """

        def fetch() -> dict:
            first = self.browse_issue_start(issue_uuid, refresh=refresh)
            total = first["total_components"]
            if total > len(first["hits"]) and self._cache_age("browse", f"{issue_uuid}_p1_boxes") > CURSOR_TTL:
                first = self.browse_issue_start(issue_uuid, refresh=True)
            hits = list(first["hits"])
            page, cursor = 1, first["cursor"]
            while len(hits) < total:
                page += 1
                resp = self.search(
                    {"issue_uuid": issue_uuid}, sort="pages:asc", page=page, cursor=cursor, include_boxes=True
                )
                if not resp["hits"]:
                    break
                hits.extend(resp["hits"])
                cursor = resp.get("cursor") or cursor
            if len(hits) < total:
                raise RuntimeError(f"issue {issue_uuid}: browse returned {len(hits)} of {total} components")
            return {"issue_uuid": issue_uuid, "total_components": total, "hits": hits}

        return self._cached("issue", f"{issue_uuid}_boxes", fetch, refresh=refresh)

    def components_text(self, ids: list[str]) -> list[dict]:
        """Full text for each id, in the order given; unknown or unserved ids are dropped.

        Cached texts are read locally; the rest go 25 per POST /components, each batch priced
        whole (3 tokens per text this account has not read before) and refused whole on 402.
        guarded=false returns caution components (about 4%) as printed.

        ids must be bare component uuids, not `{uuid}_evt_N` event ids: the API returns
        bare uuids, so an event id would be charged for but dropped from the result.
        """
        found: dict[str, dict] = {}
        todo = []
        for uuid in ids:
            hit = self._cache_get("text", uuid)
            if hit is None:
                todo.append(uuid)
            else:
                found[uuid] = hit
        for i in range(0, len(todo), PAGE_SIZE):
            body = {"ids": todo[i : i + PAGE_SIZE], "include_text": True, "guarded": False}
            for comp in self._request("POST", "/components", body)["components"]:
                self._cache_put("text", comp["uuid"], comp)
                found[comp["uuid"]] = comp
        return [found[uuid] for uuid in ids if uuid in found]

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        resp = self._http.request(method, path, json=body)
        if resp.status_code == 402:
            try:
                payload = resp.json()
            except ValueError:
                payload = {}
            info = payload["detail"] if isinstance(payload.get("detail"), dict) else payload
            raise InsufficientTokens(
                str(info.get("detail", "insufficient tokens")), info.get("price"), info.get("tokens_remaining")
            )
        if resp.is_error:
            raise httpx.HTTPStatusError(
                f"{resp.status_code} {method} {path}: {resp.text}", request=resp.request, response=resp
            )
        return resp.json()

    # -- local cache: data/snewpapers_api/{kind}/{key}.json

    def _cache_path(self, kind: str, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / kind / f"{key}.json"

    def _cache_get(self, kind: str, key: str) -> dict | None:
        path = self._cache_path(kind, key)
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text())

    def _cache_put(self, kind: str, key: str, value: dict) -> None:
        path = self._cache_path(kind, key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".tmp").write_text(json.dumps(value))
        os.replace(path.with_suffix(".tmp"), path)

    def _cached(self, kind: str, key: str, fetch: Callable[[], dict], *, refresh: bool = False) -> dict:
        if not refresh and (hit := self._cache_get(kind, key)) is not None:
            return hit
        value = fetch()
        self._cache_put(kind, key, value)
        return value

    def _cache_age(self, kind: str, key: str) -> float:
        path = self._cache_path(kind, key)
        return time.time() - path.stat().st_mtime if path is not None and path.exists() else 0.0


def _body_key(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
