import json
import os
import time

import httpx
import pytest

from snewpapers_tools import api
from snewpapers_tools.api import InsufficientTokens, SnewpapersClient, issue_cost


def make_client(handler, tmp_path):
    return SnewpapersClient("test-key", cache_dir=tmp_path, transport=httpx.MockTransport(handler))


def test_tokens_sends_the_bearer_key(tmp_path):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"balance": 10})

    assert make_client(handler, tmp_path).tokens() == {"balance": 10}
    assert seen[0].headers["Authorization"] == "Bearer test-key"
    assert (seen[0].method, seen[0].url.path) == ("GET", "/account/tokens")


def test_402_raises_with_price_and_balance(tmp_path):
    def handler(request):
        return httpx.Response(402, json={"detail": "not enough tokens", "tokens_remaining": 2, "price": 75})

    with pytest.raises(InsufficientTokens) as err:
        make_client(handler, tmp_path).tokens()
    assert (err.value.detail, err.value.price, err.value.tokens_remaining) == ("not enough tokens", 75, 2)


def test_402_with_nested_fastapi_detail_shape(tmp_path):
    def handler(request):
        return httpx.Response(402, json={"detail": {"detail": "not enough tokens", "price": 75,
                                                      "tokens_remaining": 2}})

    with pytest.raises(InsufficientTokens) as err:
        make_client(handler, tmp_path).tokens()
    assert (err.value.detail, err.value.price, err.value.tokens_remaining) == ("not enough tokens", 75, 2)


def test_402_with_non_json_body_has_price_none(tmp_path):
    def handler(request):
        return httpx.Response(402, content=b"service unavailable")

    with pytest.raises(InsufficientTokens) as err:
        make_client(handler, tmp_path).tokens()
    assert err.value.price is None
    assert err.value.tokens_remaining is None


def test_other_errors_raise_http_status_error(tmp_path):
    def handler(request):
        return httpx.Response(422, json={"detail": "bad date"})

    with pytest.raises(httpx.HTTPStatusError) as err:
        make_client(handler, tmp_path).tokens()
    assert "bad date" in str(err.value)


def test_missing_key_explains_where_it_goes(tmp_path, monkeypatch):
    monkeypatch.delenv("SNEWPAPERS_API_KEY", raising=False)
    monkeypatch.setattr(api, "load_dotenv", lambda: None)
    with pytest.raises(RuntimeError, match=".env"):
        SnewpapersClient(cache_dir=tmp_path)


def test_issue_cost_counts_browse_pages_and_text():
    assert issue_cost(30) == {"browse": 2, "text": 90, "total": 92}
    assert issue_cost(25) == {"browse": 1, "text": 75, "total": 76}


def search_response(hits, total, cursor="cur-1"):
    return {"mode": "keywords", "total_components": total, "page": 1, "cursor": cursor,
            "hits": hits, "tokens_charged": 1, "tokens_remaining": 99}


def test_find_articles_builds_the_search_and_caches_it(tmp_path):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=search_response([{"uuid": "a"}], 1))

    client = make_client(handler, tmp_path)
    first = client.find_articles(1850, 1859, "yellow fever epidemic quarantine")
    again = client.find_articles(1850, 1859, "yellow fever epidemic quarantine")
    assert first == again
    assert len(bodies) == 1
    assert bodies[0] == {
        "filters": {"start_date": "1850-01-01", "end_date": "1859-12-31"},
        "text": {"should_have_keywords": "yellow fever epidemic quarantine"},
        "sort": "relevance",
        "page": 1,
        "guarded": False,
    }


def test_find_articles_sends_a_topic_as_semantic_text(tmp_path):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=search_response([{"uuid": "a"}], 1))

    client = make_client(handler, tmp_path)
    client.find_articles(1850, 1859, "quarantine", topic=" a yellow fever epidemic in a port city ")
    client.find_articles(1850, 1859, "", topic="a yellow fever epidemic in a port city")
    assert bodies[0]["text"] == {"should_have_keywords": "quarantine",
                                 "semantic_text": "a yellow fever epidemic in a port city"}
    assert bodies[1]["text"] == {"semantic_text": "a yellow fever epidemic in a port city"}


def test_find_articles_leaves_out_empty_text_and_refresh_refetches(tmp_path):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=search_response([], 0))

    client = make_client(handler, tmp_path)
    client.find_articles(1850, 1850, " ")
    client.find_articles(1850, 1850, " ", refresh=True)
    assert len(bodies) == 2
    assert "text" not in bodies[0]


def test_search_sends_guarded_false_by_default_and_true_when_asked(tmp_path):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=search_response([], 0))

    client = make_client(handler, tmp_path)
    client.search({"issue_uuid": "iss-1"})
    client.search({"issue_uuid": "iss-1"}, guarded=True)
    assert [b["guarded"] for b in bodies] == [False, True]


def test_search_sends_include_boxes_only_when_true(tmp_path):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=search_response([], 0))

    client = make_client(handler, tmp_path)
    client.search({"issue_uuid": "iss-1"})
    client.search({"issue_uuid": "iss-1"}, include_boxes=True)
    assert "include_boxes" not in bodies[0]
    assert bodies[1]["include_boxes"] is True


def test_find_articles_does_not_send_or_key_on_include_boxes(tmp_path):
    """An ordinary search's cache key must not change: ordinary searches never request boxes."""
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=search_response([{"uuid": "a"}], 1))

    client = make_client(handler, tmp_path)
    client.find_articles(1850, 1859, "yellow fever epidemic quarantine")
    assert "include_boxes" not in bodies[0]


def test_component_is_fetched_once(tmp_path):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={"uuid": "abc", "tokens_charged": 1})

    client = make_client(handler, tmp_path)
    assert client.component("abc")["uuid"] == "abc"
    client.component("abc")
    assert paths == ["/components/abc"]


def browse_handler(total, bodies):
    """Serves an issue of `total` components, 25 per page, cursor cur-N on page N."""

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        start = (body["page"] - 1) * 25
        hits = [{"uuid": f"c{i}", "pages": [1]} for i in range(start, min(start + 25, total))]
        return httpx.Response(200, json=search_response(hits, total, cursor=f"cur-{body['page']}"))

    return handler


def test_browse_issue_start_is_one_cached_page(tmp_path):
    bodies = []
    client = make_client(browse_handler(30, bodies), tmp_path)
    first = client.browse_issue_start("iss-1")
    client.browse_issue_start("iss-1")
    assert first["total_components"] == 30
    assert bodies == [{"filters": {"issue_uuid": "iss-1"}, "sort": "pages:asc", "page": 1, "guarded": False,
                        "include_boxes": True}]
    assert (tmp_path / "browse" / "iss-1_p1_boxes.json").exists()


def test_browse_issue_start_ignores_an_old_box_less_cache_entry(tmp_path):
    bodies = []
    (tmp_path / "browse").mkdir()
    (tmp_path / "browse" / "iss-1_p1.json").write_text('{"total_components": 999, "hits": [], "cursor": "stale"}')
    client = make_client(browse_handler(30, bodies), tmp_path)
    first = client.browse_issue_start("iss-1")
    assert first["total_components"] == 30  # not the stale old-key value
    assert len(bodies) == 1  # a real request was made, the old-key file was not read


def test_browse_issue_follows_the_cursor_and_caches_the_whole_issue(tmp_path):
    bodies = []
    client = make_client(browse_handler(30, bodies), tmp_path)
    issue = client.browse_issue("iss-1")
    assert [h["uuid"] for h in issue["hits"]] == [f"c{i}" for i in range(30)]
    assert issue["total_components"] == 30
    assert [(b["page"], b.get("cursor"), b["include_boxes"]) for b in bodies] == [
        (1, None, True), (2, "cur-1", True)
    ]
    assert (tmp_path / "issue" / "iss-1_boxes.json").exists()
    client.browse_issue("iss-1")
    assert len(bodies) == 2


def test_browse_issue_refetches_page_one_when_its_cursor_is_stale(tmp_path):
    bodies = []
    client = make_client(browse_handler(30, bodies), tmp_path)
    client.browse_issue_start("iss-1")
    old = time.time() - 15 * 60
    os.utime(tmp_path / "browse" / "iss-1_p1_boxes.json", (old, old))
    client.browse_issue("iss-1")
    assert [b["page"] for b in bodies] == [1, 1, 2]


def test_browse_issue_of_one_page_does_not_refetch(tmp_path):
    bodies = []
    client = make_client(browse_handler(10, bodies), tmp_path)
    client.browse_issue_start("iss-1")
    old = time.time() - 15 * 60
    os.utime(tmp_path / "browse" / "iss-1_p1_boxes.json", (old, old))
    assert len(client.browse_issue("iss-1")["hits"]) == 10
    assert len(bodies) == 1


def test_browse_issue_raises_and_does_not_cache_a_partial_result(tmp_path):
    def handler(request):
        body = json.loads(request.content)
        if body["page"] == 1:
            return httpx.Response(200, json=search_response([{"uuid": "c0"}], 30, cursor="cur-1"))
        return httpx.Response(200, json=search_response([], 30, cursor="cur-1"))

    client = make_client(handler, tmp_path)
    with pytest.raises(RuntimeError, match="1 of 30 components"):
        client.browse_issue("iss-1")
    assert not (tmp_path / "issue" / "iss-1_boxes.json").exists()


def test_browse_issue_refresh_refetches_page_one_and_page_two(tmp_path):
    bodies = []
    client = make_client(browse_handler(30, bodies), tmp_path)
    client.browse_issue("iss-1")
    bodies.clear()
    client.browse_issue("iss-1", refresh=True)
    assert [b["page"] for b in bodies] == [1, 2]


def test_browse_issue_cursor_advances_through_each_page(tmp_path):
    bodies = []
    client = make_client(browse_handler(60, bodies), tmp_path)
    client.browse_issue("iss-1")
    assert [(b["page"], b.get("cursor")) for b in bodies] == [(1, None), (2, "cur-1"), (3, "cur-2")]


def test_components_text_batches_by_25_and_skips_cached(tmp_path):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        comps = [{"uuid": u, "content": f"text of {u}"} for u in body["ids"]]
        return httpx.Response(200, json={"components": comps, "missing": [],
                                         "tokens_charged": 3 * len(comps), "tokens_remaining": 500})

    client = make_client(handler, tmp_path)
    ids = [f"c{i}" for i in range(30)]
    client.components_text(ids[:2])
    bodies.clear()

    result = client.components_text(ids)
    assert [c["uuid"] for c in result] == ids
    assert [len(b["ids"]) for b in bodies] == [25, 3]
    assert all(b["include_text"] is True and b["guarded"] is False for b in bodies)
    assert "c0" not in bodies[0]["ids"]


def test_components_text_drops_missing_ids(tmp_path):
    def handler(request):
        return httpx.Response(200, json={"components": [{"uuid": "a", "content": "x"}], "missing": ["gone"],
                                         "tokens_charged": 3, "tokens_remaining": 1})

    assert [c["uuid"] for c in make_client(handler, tmp_path).components_text(["a", "gone"])] == ["a"]
