import asyncio
import json
import re
from types import SimpleNamespace

import pytest
from openai.types import CompletionUsage

from snewpapers_tools import adjudicate as adj
from snewpapers_tools.adjudicate import (
    AMERICAN_STORIES,
    AS_DUPLICATED,
    AS_MERGE_RIGHT,
    AS_ONLY_SCHEMA,
    AS_SPLIT_RIGHT,
    AS_WIDER_RIGHT,
    FRAGMENT_ARTICLES,
    FURNITURE,
    ITEMS_MERGED,
    ILLEGIBLE_UNSURE,
    LINES_ELSEWHERE,
    MISSING_COMPONENTS,
    MISSING_PAGES,
    ONE_COMPONENT,
    NIBBLE,
    PARTIAL,
    PARTIAL_ARTICLES,
    PARTLY_CAPTURED,
    WIDER_ARTICLES,
    ATTACHED,
    ATTACHED_SCHEMA,
    CORRECT,
    EMPTY,
    MISSED,
    EXCLUDED,
    MATCHED,
    NEITHER,
    NO_VERDICT,
    NOT_CAPTURED,
    ONE_VS_MANY_SCHEMA,
    PROMPT_FILES,
    SNEWPAPERS,
    TINY,
    UNSTABLE,
    TIMEOUT,
    Adjudicator,
    Asked,
    GroupVerdict,
    Judgement,
    PageContext,
    adjudicate,
    as_errors,
    as_only_messages,
    attached_messages,
    dispute_messages,
    edges,
    estimate_cost,
    estimate_questions,
    first_a_side,
    group_ids,
    judged_counts,
    judgements,
    one_vs_many_messages,
    parse,
    prompt,
    scorecard,
    validate,
)
from snewpapers_tools.issues import IssueKey
from snewpapers_tools.segmentation import (
    CONTINUED,
    AS_MERGE,
    AS_ONLY,
    AS_ONLY_TINY,
    AS_ONLY_EMPTY,
    AS_PARTIAL,
    AS_SPLIT,
    AS_WIDER,
    IN_AS_MERGE,
    MATCH,
    OURS_ONLY,
    AsBox,
    AsPart,
    Group,
    IssueSegmentation,
    OurPiece,
)

FRAMING = "We are comparing two systems that each try to segment historical newspaper pages into components."
CONTEXT = PageContext("Wilmington Journal", "1856-08-08", 2)
BOX = (0.1, 0.1, 0.3, 0.3)


# --- fixtures ---------------------------------------------------------------------------------


def piece(text, page=2):
    return OurPiece(f"u-{text[:8]}", "notice", page, (BOX,), text, (page,))


def part(text, article_id=1, page=2, headline=""):
    boxes = [AsBox(page, 100 * article_id, article_id, BOX, headline, "headline", "Legible", 0)] if headline else []
    boxes.append(AsBox(page, 100 * article_id + 1, article_id, BOX, text, "article", "Legible", 1))
    return AsPart(article_id, tuple(boxes))


def group(case, ours=(), theirs=(), page=2):
    return Group(page, case, [piece(t, page) for t in ours], [part(t, k, page) for k, t in enumerate(theirs, start=1)])


def pair():
    return SimpleNamespace(issue=IssueKey("i", "sn84026536", "1856-08-08", 1),
                           components=[{"uuid": "x", "newspaper": "wilmington journal"}])


def segmentation():
    """One group of every case on page 2, in this order."""
    return IssueSegmentation([
        group(MATCH, ["OURS match"], ["AS match"]),
        group(AS_SPLIT, ["OURS whole"], ["AS part one", "AS part two"]),
        group(AS_MERGE, ["OURS part one", "OURS part two"], ["AS whole"]),
        group(OURS_ONLY, ["OURS alone"], []),
        group(AS_ONLY, [], ["AS ad: just received, for sale cheap"]),
        group(AS_ONLY_TINY, [], ["NOTICES."]),
    ], {}, [])


def response(content, finish_reason="stop", cost=0.0002):
    usage = CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, estimated_cost=cost)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)],
                           usage=usage)


class FakeClient:
    """Stands in for openai.AsyncOpenAI: `reply(request kwargs)` gives the response; calls are recorded."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply(kwargs)


def pieces_block(request) -> str:
    """A one-vs-many user message from the first numbered piece on: the passage shown."""
    user = request["messages"][1]["content"]
    return user[re.search(r"^\[1\]( \(p\. \d+\))?\n", user, re.M).start():]


def judge(prefers):
    """A fake model that answers on the content: "several" when the pieces are `prefers`' ("OURS" or
    "AS"), else "one"; and calls AS-only text excluded."""
    def reply(request):
        schema = request["response_format"]["json_schema"]["schema"]
        if schema is AS_ONLY_SCHEMA:
            return response(json.dumps({"kind": "advertisement", "reason": "an advertisement"}))
        verdict = "several" if prefers in pieces_block(request) else "one"
        return response(json.dumps({"verdict": verdict, "reason": f"{prefers} keeps the item whole"}))
    return reply


def adjudicator(client, tmp_path, **kwargs):
    return Adjudicator(client, model="m", base_url="https://llm.example/v1", cache_dir=tmp_path / "adj", **kwargs)


def run(coro):
    return asyncio.run(coro)


# --- prompts and messages ---------------------------------------------------------------------


@pytest.mark.parametrize("case, schema", [(AS_SPLIT, ONE_VS_MANY_SCHEMA), (AS_MERGE, ONE_VS_MANY_SCHEMA),
                                          (AS_ONLY, AS_ONLY_SCHEMA), (ATTACHED, ATTACHED_SCHEMA)])
def test_every_prompt_opens_with_the_framing_and_ends_with_its_schema(case, schema):
    text = prompt(case)
    assert text.startswith(FRAMING)
    assert "One component is one printed item" in text
    assert "short reason" in text
    assert json.loads(text.strip().splitlines()[-1]) == schema


def test_only_the_shipped_prompts_ship():
    from importlib.resources import files

    shipped = {f.name for f in files("snewpapers_tools").joinpath("prompts").iterdir() if f.name.endswith(".md")}
    assert shipped == set(PROMPT_FILES.values()) == {"one_vs_many.md", "as_only.md", "attached_outside.md",
                                                            "one_component.md"}


def test_one_vs_many_prompt_asks_about_the_passage_not_a_side():
    text = prompt(AS_SPLIT)
    assert text == prompt(AS_MERGE)
    assert "System A" not in text and '"A"' not in text
    assert 'Answer "one"' in text and '"several"' in text and '"mixed"' in text
    assert ("A piece that begins or ends mid-sentence is almost always a continuation of a neighbour, "
            "not a separate component.") in text


def test_edges_quote_the_first_and_last_fifteen_words_with_whitespace_normalised():
    text = " ".join(f"w{k}" for k in range(40)).replace("w3 ", "w3\n\n  ")
    assert edges(text) == ('begins: "' + " ".join(f"w{k}" for k in range(15)) + '" … ends: "'
                           + " ".join(f"w{k}" for k in range(25, 40)) + '"')
    assert edges("  short\n piece ") == 'begins: "short piece" … ends: "short piece"'


def test_one_vs_many_message_shows_the_pieces_with_their_edges_and_states_both_beliefs_unlabelled():
    system, user = one_vs_many_messages(CONTEXT, ["first", "second", "third"], one_first=False)
    assert system == {"role": "system", "content": prompt(AS_SPLIT)}
    content = user["content"]
    assert content == (
        "Newspaper: Wilmington Journal\nDate: 1856-08-08\nPage: 2\n\n"
        "One system believes this passage is 3 components, shown below as [1] to [3]; "
        "the other system believes the whole passage is one component.\n\n"
        '[1]\nbegins: "first" … ends: "first"\n"""\nfirst\n"""\n\n'
        '[2]\nbegins: "second" … ends: "second"\n"""\nsecond\n"""\n\n'
        '[3]\nbegins: "third" … ends: "third"\n"""\nthird\n"""\n\n'
        "Is it one component or several?")
    _, user = one_vs_many_messages(CONTEXT, ["first", "second"], one_first=True)
    content = user["content"]
    assert ("One system believes this passage is one component; the other system believes it is "
            "2 components, shown below as [1] and [2].") in content
    assert content.index("first") < content.index("second")  # the pieces keep their position order
    assert "System" not in content.replace("One system", "").replace("other system", "")
    with pytest.raises(ValueError):
        one_vs_many_messages(CONTEXT, ["a"], one_first=True)


def test_every_piece_is_headed_by_a_begins_ends_line():
    long = "the end of the editorial, which runs on from the column before and closes here with a flourish of words."
    _, user = one_vs_many_messages(CONTEXT, [long, "A CANINE NOBLEMAN. We take the following."], one_first=True)
    assert f"[1]\n{edges(long)}\n\"\"\"\n{long}" in user["content"]
    assert '[2]\nbegins: "A CANINE NOBLEMAN. We take the following." … ends: "A CANINE NOBLEMAN. We take the following."' in user["content"]


def test_as_only_message_carries_the_one_text_and_the_question():
    system, user = as_only_messages(CONTEXT, "  RUNAWAY! Ten dollars reward.  ")
    assert system["content"] == prompt(AS_ONLY)
    assert "The text:\n\"\"\"\nRUNAWAY! Ten dollars reward.\n\"\"\"" in user["content"]
    assert "System A" not in user["content"]
    assert user["content"].endswith("page furniture such as a masthead or section heading?")


@pytest.mark.parametrize("case, ours, theirs, pieces, single", [
    (AS_MERGE, ["OURS one", "OURS two"], ["AS whole"], ["OURS one", "OURS two"], "AS whole"),
    (AS_SPLIT, ["OURS whole"], ["AS one", "AS two"], ["AS one", "AS two"], "OURS whole"),
])
def test_dispute_messages_show_the_many_side_and_state_the_first_sides_belief_first(case, ours, theirs, pieces, single):
    g = group(case, ours, theirs)
    many_side = SNEWPAPERS if case == AS_MERGE else AMERICAN_STORIES
    one_side = AMERICAN_STORIES if case == AS_MERGE else SNEWPAPERS
    many_first = dispute_messages(g, CONTEXT, many_side)
    one_first = dispute_messages(g, CONTEXT, one_side)
    assert many_first == one_vs_many_messages(CONTEXT, pieces, one_first=False)
    assert one_first == one_vs_many_messages(CONTEXT, pieces, one_first=True)
    for messages in (many_first, one_first):
        assert single not in messages[1]["content"]  # the single side's text is the same passage: not repeated
        assert messages[0]["content"] == prompt(case)


def test_as_split_pieces_include_the_headline_boxes():
    g = Group(2, AS_SPLIT, [piece("ours")], [part("body", 1, headline="HEAD"), part("other", 2)])
    assert '"""\nHEAD\n\nbody\n"""' in dispute_messages(g, CONTEXT, AMERICAN_STORIES)[1]["content"]


def test_only_disputes_have_a_question():
    with pytest.raises(ValueError):
        dispute_messages(group(MATCH, ["a"], ["a"]), CONTEXT, SNEWPAPERS)


# --- validation -------------------------------------------------------------------------------


def test_one_vs_many_schema_takes_one_several_or_mixed_only():
    for verdict in ("one", "several", "mixed"):
        assert validate({"verdict": verdict, "reason": "r"}, ONE_VS_MANY_SCHEMA) is not None
    for verdict in ("A", "B", "both", "neither", "One"):
        assert validate({"verdict": verdict, "reason": "r"}, ONE_VS_MANY_SCHEMA) is None


@pytest.mark.parametrize("answer", [
    {"verdict": "one"},  # missing field
    {"verdict": "one", "reason": "r", "confidence": 1},  # extra field
    {"verdict": "A", "reason": "r"},  # not in the enum
    {"verdict": "One", "reason": "r"},  # case matters
    {"verdict": "one", "reason": 3},  # wrong type
    ["A", "r"],
])
def test_validate_rejects_anything_off_the_schema(answer):
    assert validate(answer, ONE_VS_MANY_SCHEMA) is None


def test_validate_accepts_the_schema_and_checks_booleans():
    assert validate({"verdict": "mixed", "reason": "r"}, ONE_VS_MANY_SCHEMA) == {"verdict": "mixed", "reason": "r"}
    assert validate({"kind": "other", "reason": "r"}, AS_ONLY_SCHEMA) is not None
    assert validate({"kind": "ad", "reason": "r"}, AS_ONLY_SCHEMA) is None


def test_parse_is_no_verdict_for_bad_json_empty_content_or_running_out_of_tokens():
    good = json.dumps({"verdict": "several", "reason": "r"})
    assert parse(good, "stop", ONE_VS_MANY_SCHEMA) == {"verdict": "several", "reason": "r"}
    assert parse(good, "length", ONE_VS_MANY_SCHEMA) is None
    assert parse('Sure! {"verdict": "several", "reason": "r"}', "stop", ONE_VS_MANY_SCHEMA) is None  # no scraping
    assert parse("", "stop", ONE_VS_MANY_SCHEMA) is None
    assert parse(None, "stop", ONE_VS_MANY_SCHEMA) is None


# --- asking -----------------------------------------------------------------------------------

MESSAGES = one_vs_many_messages(CONTEXT, ["a", "b"], one_first=True)


def test_ask_sends_a_strict_json_schema_and_reads_the_billed_cost(tmp_path):
    client = FakeClient(lambda r: response(json.dumps({"verdict": "one", "reason": "r"}), cost=0.00031))
    answer = run(adjudicator(client, tmp_path).ask(MESSAGES, ONE_VS_MANY_SCHEMA))
    assert answer.parsed == {"verdict": "one", "reason": "r"}
    assert answer.cost == 0.00031 and not answer.cached
    (request,) = client.calls
    assert request["model"] == "m" and request["messages"] == MESSAGES and request["max_tokens"] == 16000
    assert request["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "verdict_reason", "strict": True, "schema": ONE_VS_MANY_SCHEMA}}
    assert "temperature" not in request and "service_tier" not in request


def test_ask_reads_the_token_usage_and_a_cached_answer_counts_none(tmp_path):
    client = FakeClient(lambda r: response(json.dumps({"verdict": "one", "reason": "r"})))
    first = run(adjudicator(client, tmp_path).ask(MESSAGES, ONE_VS_MANY_SCHEMA))
    assert (first.input_tokens, first.output_tokens) == (100, 50)
    again = run(adjudicator(client, tmp_path).ask(MESSAGES, ONE_VS_MANY_SCHEMA))
    assert again.cached and (again.input_tokens, again.output_tokens) == (0, 0)


def test_ask_without_a_cost_in_the_usage_gives_none(tmp_path):
    def reply(_request):
        resp = response(json.dumps({"verdict": "one", "reason": "r"}))
        resp.usage = CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return resp
    assert run(adjudicator(FakeClient(reply), tmp_path).ask(MESSAGES, ONE_VS_MANY_SCHEMA)).cost is None


@pytest.mark.parametrize("content, finish", [
    (json.dumps({"verdict": "maybe", "reason": "r"}), "stop"),
    (json.dumps({"verdict": "one", "reason": "r"}), "length"),
    ("not json", "stop"),
])
def test_ask_keeps_the_raw_content_of_a_no_verdict_and_does_not_cache_it(tmp_path, content, finish):
    client = FakeClient(lambda r: response(content, finish))
    judge_ = adjudicator(client, tmp_path)
    answer = run(judge_.ask(MESSAGES, ONE_VS_MANY_SCHEMA))
    assert answer.parsed is None and answer.raw == content and answer.cost == 0.0002
    run(judge_.ask(MESSAGES, ONE_VS_MANY_SCHEMA))
    assert len(client.calls) == 2  # asked again: a no verdict is never cached


def test_ask_answers_again_from_the_cache_for_free(tmp_path):
    client = FakeClient(lambda r: response(json.dumps({"verdict": "several", "reason": "r"})))
    judge_ = adjudicator(client, tmp_path)
    first = run(judge_.ask(MESSAGES, ONE_VS_MANY_SCHEMA))
    again = run(adjudicator(FakeClient(lambda r: pytest.fail("asked the client")), tmp_path).ask(MESSAGES, ONE_VS_MANY_SCHEMA))
    assert again.parsed == first.parsed and again.cost == 0.0 and again.cached
    assert len(list((tmp_path / "adj").glob("*.json"))) == 1


def test_cache_key_depends_on_model_service_and_messages(tmp_path):
    base = adjudicator(FakeClient(None), tmp_path)
    keys = {
        base.cache_key(MESSAGES),
        Adjudicator(FakeClient(None), model="other", base_url=base.base_url).cache_key(MESSAGES),
        Adjudicator(FakeClient(None), model="m", base_url="https://other.example").cache_key(MESSAGES),
        base.cache_key(one_vs_many_messages(CONTEXT, ["a", "c"], one_first=True)),
    }
    assert len(keys) == 4
    assert base.cache_key(MESSAGES) == adjudicator(FakeClient(None), tmp_path).cache_key(list(MESSAGES))


class SlowAtFirst(FakeClient):
    """Hangs on the first `slow` calls (past any short deadline), then answers."""

    def __init__(self, slow):
        super().__init__(lambda r: response(json.dumps({"verdict": "one", "reason": "r"})))
        self.slow = slow

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.slow:
            await asyncio.sleep(5)
        return self.reply(kwargs)


def test_the_default_deadline_is_twenty_minutes_a_call():
    assert TIMEOUT == 1200 and Adjudicator(FakeClient(None), model="m", base_url="https://llm.example/v1").timeout == 1200


def test_a_call_that_times_out_is_made_once_more(tmp_path):
    client = SlowAtFirst(slow=1)
    answer = run(adjudicator(client, tmp_path, timeout=0.01).ask(MESSAGES, ONE_VS_MANY_SCHEMA))
    assert answer.parsed == {"verdict": "one", "reason": "r"} and not answer.error
    assert len(client.calls) == 2 and client.calls[0] == client.calls[1]


def test_a_call_that_times_out_twice_is_no_verdict(tmp_path):
    client = SlowAtFirst(slow=2)
    answer = run(adjudicator(client, tmp_path, timeout=0.01).ask(MESSAGES, ONE_VS_MANY_SCHEMA))
    assert answer.parsed is None and answer.error and "Timeout" in answer.raw
    assert len(client.calls) == 2
    assert not (tmp_path / "adj").exists()  # nothing cached


def test_ask_that_fails_is_no_verdict_without_another_try(tmp_path):
    def boom(_request):
        raise RuntimeError("503 overloaded")
    client = FakeClient(boom)
    answer = run(adjudicator(client, tmp_path).ask(MESSAGES, ONE_VS_MANY_SCHEMA))
    assert answer.parsed is None and answer.error and "503 overloaded" in answer.raw
    assert len(client.calls) == 1  # the SDK already retried what is worth retrying


def test_default_client_is_async_openai_from_the_env_with_retries_raised(monkeypatch):
    import openai

    made = {}
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: made.update(kwargs) or "client")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "some/model")
    judge_ = Adjudicator(api_key="k")
    assert judge_.client == "client"
    assert made == {"api_key": "k", "base_url": "https://llm.example/v1", "max_retries": 6}
    assert judge_.model == "some/model"


@pytest.mark.parametrize("missing", ["LLM_BASE_URL", "LLM_MODEL"])
def test_the_service_and_model_must_be_set(monkeypatch, missing):
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: "client")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "some/model")
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match=missing):
        Adjudicator(api_key="k")


# --- groups ---------------------------------------------------------------------------------


def test_group_ids_number_groups_within_their_page():
    seg = IssueSegmentation([group(MATCH, ["a"], ["a"], page=1), group(MATCH, ["b"], ["b"], page=1),
                             group(MATCH, ["c"], ["c"], page=2)], {}, [])
    assert group_ids(seg) == ["p1-1", "p1-2", "p2-1"]


def test_first_a_side_is_random_per_group_but_reproducible():
    sides = [first_a_side(f"issue/p1-{k}") for k in range(40)]
    assert sides == [first_a_side(f"issue/p1-{k}") for k in range(40)]
    assert set(sides) == {SNEWPAPERS, AMERICAN_STORIES}


def test_estimates_count_two_questions_per_dispute_and_one_per_as_only_group():
    assert estimate_questions(segmentation()) == 2 * 2 + 1  # tiny AS-only text is not asked
    assert estimate_cost(segmentation(), 0.001) == pytest.approx(0.005)


def test_adjudicate_gives_every_group_a_verdict_and_asks_only_disputes_and_as_only(tmp_path):
    client = FakeClient(judge("OURS"))
    verdicts = run(adjudicate(segmentation(), pair(), adjudicator=adjudicator(client, tmp_path)))
    assert [v.group_id for v in verdicts] == [f"p2-{k}" for k in range(1, 7)]
    by_case = {v.case: v for v in verdicts}
    assert len(client.calls) == 2 * 2 + 1
    for case in (AS_SPLIT, AS_MERGE):
        v = by_case[case]
        assert v.outcome == SNEWPAPERS
        assert {q.a_side for q in v.asked} == {SNEWPAPERS, AMERICAN_STORIES}  # both beliefs stated first once
        assert v.reasons == ["OURS keeps the item whole"] * 2
    assert [q.answer.parsed["verdict"] for q in by_case[AS_SPLIT].asked] == ["one", "one"]  # ours is the one
    assert [q.answer.parsed["verdict"] for q in by_case[AS_MERGE].asked] == ["several", "several"]  # ours are the several
    assert by_case[AS_ONLY].outcome == "correctly not served"
    assert (by_case[MATCH].outcome, by_case[MATCH].asked) == (MATCHED, [])
    assert (by_case[OURS_ONLY].outcome, by_case[OURS_ONLY].asked) == (NOT_CAPTURED, [])
    assert (by_case[AS_ONLY_TINY].outcome, by_case[AS_ONLY_TINY].asked) == (TINY, [])
    assert "Newspaper: Wilmington Journal" in client.calls[0]["messages"][1]["content"]


def test_adjudicate_finds_american_stories_right_when_the_judge_prefers_it(tmp_path):
    verdicts = run(adjudicate(segmentation(), pair(), adjudicator=adjudicator(FakeClient(judge("AS")), tmp_path)))
    assert {v.outcome for v in verdicts if v.case in (AS_SPLIT, AS_MERGE)} == {AMERICAN_STORIES}


# The mapping: as_split "one" -> snewpapers, "several" -> AS; as_merge the other way round; "mixed" -> neither.
MAPPING = [
    (AS_SPLIT, "one", SNEWPAPERS), (AS_SPLIT, "several", AMERICAN_STORIES), (AS_SPLIT, "mixed", NEITHER),
    (AS_MERGE, "one", AMERICAN_STORIES), (AS_MERGE, "several", SNEWPAPERS), (AS_MERGE, "mixed", NEITHER),
]


@pytest.mark.parametrize("case, answer, expected", MAPPING)
def test_one_several_and_mixed_map_to_sides(tmp_path, case, answer, expected):
    ours, theirs = (["OURS whole"], ["AS one", "AS two"]) if case == AS_SPLIT else (["OURS one", "OURS two"], ["AS whole"])
    client = FakeClient(lambda r: response(json.dumps({"verdict": answer, "reason": "r"})))
    (v,) = run(adjudicate(IssueSegmentation([group(case, ours, theirs)], {}, []), pair(),
                          adjudicator=adjudicator(client, tmp_path)))
    assert v.outcome == expected
    assert [q.verdict for q in v.asked] == [expected, expected]
    # both orders asked: once with the one-component belief stated first, once with the N-components belief
    users = [r["messages"][1]["content"] for r in client.calls]
    assert sorted("believes this passage is one component" in u for u in users) == [False, True]


def test_asked_maps_verdicts_whoever_was_stated_first():
    for first in (SNEWPAPERS, AMERICAN_STORIES):
        said = lambda v: Asked(first, adj.Answer({"verdict": v, "reason": "r"}, 0.0, ""), AMERICAN_STORIES).verdict  # noqa: E731
        assert (said("one"), said("several"), said("mixed")) == (AMERICAN_STORIES, SNEWPAPERS, NEITHER)
    assert Asked(SNEWPAPERS, adj.Answer(None, 0.0, "junk"), SNEWPAPERS).verdict is None


def test_a_judge_that_follows_the_first_belief_is_unstable(tmp_path):
    def first_belief(request):
        one_first = "believes this passage is one component" in request["messages"][1]["content"]
        return response(json.dumps({"verdict": "one" if one_first else "several", "reason": "r"}))
    seg = IssueSegmentation([group(AS_SPLIT, ["OURS whole"], ["AS one", "AS two"])], {}, [])
    (v,) = run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(first_belief), tmp_path)))
    assert v.outcome == UNSTABLE


def test_one_bad_answer_is_no_verdict(tmp_path):
    def half_bad(request):
        if "believes this passage is one component" in request["messages"][1]["content"]:
            return response(json.dumps({"verdict": "one", "reason": "r"}))
        return response(json.dumps({"verdict": "one", "reason": "r"}), finish_reason="length")
    seg = IssueSegmentation([group(AS_MERGE, ["OURS one", "OURS two"], ["AS whole"])], {}, [])
    (v,) = run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(half_bad), tmp_path)))
    assert v.outcome == NO_VERDICT


def test_as_only_false_is_missed_by_snewpapers(tmp_path):
    reply = lambda r: response(json.dumps({"kind": "other", "reason": "a news item"}))  # noqa: E731
    seg = IssueSegmentation([group(AS_ONLY, [], ["news of the war in Kansas"])], {}, [])
    (v,) = run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(reply), tmp_path)))
    assert v.outcome == "missed by snewpapers" and v.reasons == ["a news item"]


def test_scorecard_counts_components_merges_as_only_questions_and_cost(tmp_path):
    def mixed(request):
        if request["response_format"]["json_schema"]["schema"] is AS_ONLY_SCHEMA:
            return response(json.dumps({"kind": "other", "reason": "r"}), cost=0.001)
        user = request["messages"][1]["content"]
        if "split two" in user:  # follows the stated order: unstable
            one_first = "believes this passage is one component" in user
            return response(json.dumps({"verdict": "one" if one_first else "several", "reason": "r"}), cost=0.001)
        if "split three" in user:
            return response(json.dumps({"verdict": "several", "reason": "r"}), cost=0.001)
        return response(json.dumps({"verdict": "mixed", "reason": "r"}), cost=0.001)
    seg = IssueSegmentation([
        group(MATCH, ["OURS match"], ["AS match"]),
        group(MATCH, ["OURS match 2"], ["AS match 2"]),
        group(AS_SPLIT, ["OURS split one"], ["AS split two a", "AS split two b"]),  # unstable
        group(AS_SPLIT, ["OURS whole"], ["AS split three a", "AS split three b"]),  # several: AS right
        group(AS_MERGE, ["OURS part one", "OURS part two"], ["AS whole"]),  # mixed: neither
        group(OURS_ONLY, ["OURS alone"], []),
        group(AS_ONLY, [], ["news of the war in Kansas"]),
        group(AS_ONLY_TINY, [], ["NOTICES."]),
    ], {}, [])
    card = scorecard(run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(mixed), tmp_path))))
    assert card["components"] == {CORRECT: 2, AS_SPLIT_RIGHT: 1, AS_MERGE_RIGHT: 0, NEITHER: 0, UNSTABLE: 1,
                                  NO_VERDICT: 0, PARTLY_CAPTURED: 0, AS_WIDER_RIGHT: 0, NOT_CAPTURED: 1}
    assert card["merges"] == {SNEWPAPERS: 0, AMERICAN_STORIES: 0, NEITHER: 1, UNSTABLE: 0, ILLEGIBLE_UNSURE: 0, NO_VERDICT: 0}
    assert card["as_only"] == {"correctly not served": 0, "missed by snewpapers": 1, NO_VERDICT: 0, TINY: 1, EMPTY: 0}
    assert card["as_defects"] == {AS_DUPLICATED: 0}
    assert card["questions"] == 7 and card["cached"] == 0 and card["uncosted"] == 0
    assert card["cost"] == pytest.approx(0.007)
    assert (card["input_tokens"], card["output_tokens"]) == (700, 350)
    again = scorecard(run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(mixed), tmp_path))))
    assert again["cost"] == 0 and again["cached"] == 7 and again["components"] == card["components"]
    assert (again["input_tokens"], again["output_tokens"]) == (0, 0)


@pytest.mark.parametrize("cost, uncosted, text", [
    (0.006, 0, "1,500 input + 250 output tokens, $0.0060"),
    (0.0, 5, "1,500 input + 250 output tokens, cost not reported by the service"),
    (0.001, 2, "1,500 input + 250 output tokens, $0.0010 (2 answers without a cost)"),
])
def test_usage_text_gives_the_tokens_and_the_cost_only_as_reported(cost, uncosted, text):
    from snewpapers_tools.adjudicate import usage_text

    assert usage_text({"input_tokens": 1500, "output_tokens": 250, "cost": cost, "uncosted": uncosted}) == text


def test_an_as_split_decided_one_counts_as_correct_as_is():
    from snewpapers_tools.adjudicate import GroupVerdict, component_outcome

    assert component_outcome(GroupVerdict("p1-1", AS_SPLIT, SNEWPAPERS)) == CORRECT
    assert component_outcome(GroupVerdict("p1-1", AS_SPLIT, AMERICAN_STORIES)) == AS_SPLIT_RIGHT
    assert component_outcome(GroupVerdict("p1-1", MATCH, MATCHED)) == CORRECT
    assert component_outcome(GroupVerdict("p1-1", AS_MERGE, SNEWPAPERS)) is None
    assert component_outcome(GroupVerdict("p1-1", IN_AS_MERGE, SNEWPAPERS)) == CORRECT  # its merge's outcome
    assert component_outcome(GroupVerdict("p1-1", IN_AS_MERGE, AMERICAN_STORIES)) == AS_MERGE_RIGHT
    assert component_outcome(GroupVerdict("p1-1", IN_AS_MERGE, UNSTABLE)) == UNSTABLE


def test_scorecard_counts_the_text_american_stories_duplicated_as_its_defect(tmp_path):
    duplicated = group(MATCH, ["OURS fleas"], ["AS fleas"])
    duplicated.duplicates = [("#5", "#83")]
    seg = IssueSegmentation([duplicated, group(MATCH, ["OURS match"], ["AS match"])], {}, [])
    verdicts = run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(judge("OURS")), tmp_path)))
    assert [v.as_duplicated for v in verdicts] == [True, False]
    card = scorecard(verdicts)
    assert card["as_defects"] == {AS_DUPLICATED: 1}
    assert card["components"][CORRECT] == 2 and card["questions"] == 0  # still a match, nothing asked


# --- AS text attached outside our component ---------------------------------------------------

NOTICE = "NOTICE. THE CO-PARTNERSHIP heretofore existing is this day dissolved."
DATE = "Dec. Is, 1854."


def attached_group(case=MATCH, theirs=("AS notice",), attached=DATE):
    g = group(case, [NOTICE], list(theirs))
    g.as_attached_outside = [part(attached, 7)]
    return g


def test_attached_message_shows_the_component_then_the_attached_text_with_both_beliefs_unlabelled():
    belongs = attached_messages(CONTEXT, NOTICE, DATE, belongs_first=True)
    separate = attached_messages(CONTEXT, NOTICE, DATE, belongs_first=False)
    assert belongs[0]["content"] == prompt(ATTACHED) == separate[0]["content"]
    user = belongs[1]["content"]
    assert user.startswith(str(CONTEXT) + "\n\nThe component:\n")
    assert user.index(NOTICE) < user.index(f"«{DATE}»")
    assert ("One system believes this component also includes the following text, printed next to it on the page:\n"
            f"«{DATE}»\nThe other system believes it does not.") in user
    assert user.endswith("Does the text belong to this component?")
    assert ("One system believes this component does not include the following text, printed next to it on the page:\n"
            f"«{DATE}»\nThe other system believes it does.") in separate[1]["content"]
    assert "snewpapers" not in user.lower() and "american stories" not in user.lower()


def test_attached_prompt_asks_belongs_or_separate():
    text = prompt(ATTACHED)
    assert 'Answer "belongs"' in text and '"separate"' in text
    assert validate({"verdict": "belongs", "reason": "r"}, ATTACHED_SCHEMA)
    assert validate({"verdict": "one", "reason": "r"}, ATTACHED_SCHEMA) is None


def test_estimates_add_two_questions_per_group_with_text_attached_outside():
    seg = IssueSegmentation([attached_group(), group(MATCH, ["OURS"], ["AS"])], {}, [])
    assert estimate_questions(seg) == 2


@pytest.mark.parametrize("answer, expected", [("separate", SNEWPAPERS), ("belongs", AMERICAN_STORIES)])
def test_attached_outside_is_asked_both_ways_and_separate_means_snewpapers_is_right(tmp_path, answer, expected):
    client = FakeClient(lambda r: response(json.dumps({"verdict": answer, "reason": "the date ends the notice above"})))
    (v,) = run(adjudicate(IssueSegmentation([attached_group()], {}, []), pair(), adjudicator=adjudicator(client, tmp_path)))
    assert (v.case, v.outcome, v.asked) == (MATCH, MATCHED, [])  # the match itself is still not asked
    assert v.attached_outcome == expected
    assert {q.a_side for q in v.attached_asked} == {SNEWPAPERS, AMERICAN_STORIES}
    assert [q.verdict for q in v.attached_asked] == [expected, expected]
    users = [r["messages"][1]["content"] for r in client.calls]
    assert sorted("also includes the following text" in u for u in users) == [False, True]
    assert all(r["response_format"]["json_schema"]["schema"] is ATTACHED_SCHEMA for r in client.calls)


def test_an_attached_verdict_that_follows_the_first_belief_is_unstable_and_a_bad_one_no_verdict(tmp_path):
    def first_belief(request):
        includes_first = "also includes the following text" in request["messages"][1]["content"]
        return response(json.dumps({"verdict": "belongs" if includes_first else "separate", "reason": "r"}))
    (v,) = run(adjudicate(IssueSegmentation([attached_group()], {}, []), pair(),
                          adjudicator=adjudicator(FakeClient(first_belief), tmp_path / "a")))
    assert v.attached_outcome == UNSTABLE
    bad = FakeClient(lambda r: response(json.dumps({"verdict": "maybe", "reason": "r"})))
    (v,) = run(adjudicate(IssueSegmentation([attached_group()], {}, []), pair(), adjudicator=adjudicator(bad, tmp_path / "b")))
    assert v.attached_outcome == NO_VERDICT


def test_scorecard_has_an_attached_outside_row_and_counts_its_questions(tmp_path):
    def reply(request):
        schema = request["response_format"]["json_schema"]["schema"]
        if schema is ATTACHED_SCHEMA:
            return response(json.dumps({"verdict": "separate", "reason": "r"}), cost=0.001)
        return response(json.dumps({"verdict": "one", "reason": "r"}), cost=0.001)
    split = attached_group(AS_SPLIT, ["AS one", "AS two"], attached="General Notices.")
    dup = attached_group(attached="June 3, 1856.")
    dup.attached_duplicates = [("#31", "#7")]
    seg = IssueSegmentation([attached_group(), split, dup, group(MATCH, ["OURS"], ["AS"])], {}, [])
    verdicts = run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(reply), tmp_path)))
    assert [v.as_duplicated for v in verdicts] == [False, False, True, False]
    card = scorecard(verdicts)
    assert card["attached_outside"] == {SNEWPAPERS: 3, AMERICAN_STORIES: 0, FURNITURE: 0, UNSTABLE: 0, NO_VERDICT: 0}
    assert card["components"][CORRECT] == 4  # the boundary question does not change the component's case
    assert card["as_defects"] == {AS_DUPLICATED: 1}
    assert card["questions"] == 2 * 3 + 2 and card["cost"] == pytest.approx(0.008)
    assert verdicts[0].cost == pytest.approx(0.002)


# --- what decisions judge -----------------------------------------------------------------------


def test_an_as_split_decided_one_judges_our_component_right_and_every_as_article_inside_it_wrong():
    split = group(AS_SPLIT, ["OURS whole"], [f"AS {k}" for k in range(15)])
    found = judgements(split, GroupVerdict("p2-1", AS_SPLIT, SNEWPAPERS))
    assert [(j.side, j.judged) for j in found].count((AMERICAN_STORIES, "wrong")) == 15
    assert [(j.side, j.unit, j.judged) for j in found if j.side == SNEWPAPERS] == [(SNEWPAPERS, "u-OURS who", "right")]
    assert {j.decision for j in found} == {("p2-1", "case")}
    assert judged_counts(found) == {"as_articles": {"wrong": 15, "right": 0}, "components": {"wrong": 0, "right": 1}}


def test_an_as_merge_decided_several_judges_each_of_our_pieces_right_and_the_one_as_article_wrong():
    merge = group(AS_MERGE, ["first part", "second part", "third part"], ["AS whole"])
    counts = judged_counts(judgements(merge, GroupVerdict("p2-1", AS_MERGE, SNEWPAPERS)))
    assert counts == {"as_articles": {"wrong": 1, "right": 0}, "components": {"wrong": 0, "right": 3}}
    counts = judged_counts(judgements(merge, GroupVerdict("p2-1", AS_MERGE, AMERICAN_STORIES)))
    assert counts == {"as_articles": {"wrong": 0, "right": 1}, "components": {"wrong": 3, "right": 0}}
    counts = judged_counts(judgements(merge, GroupVerdict("p2-1", AS_MERGE, NEITHER)))
    assert counts == {"as_articles": {"wrong": 1, "right": 0}, "components": {"wrong": 3, "right": 0}}


def test_undecided_verdicts_matches_and_left_overs_judge_nothing():
    split = group(AS_SPLIT, ["OURS"], ["AS a", "AS b"])
    for outcome in (UNSTABLE, NO_VERDICT):
        assert judgements(split, GroupVerdict("p2-1", AS_SPLIT, outcome)) == []
    assert judgements(split, None) == []
    assert judgements(group(MATCH, ["OURS"], ["AS"]), GroupVerdict("p2-1", MATCH, MATCHED)) == []
    assert judgements(group(AS_ONLY, [], ["AS"]), GroupVerdict("p2-1", AS_ONLY, "missed by snewpapers")) == []


def test_the_boundary_question_judges_our_component_and_the_attached_articles_as_a_decision_of_its_own():
    g = attached_group(AS_SPLIT, ["AS one", "AS two"])
    found = judgements(g, GroupVerdict("p2-1", AS_SPLIT, AMERICAN_STORIES, attached_outcome=SNEWPAPERS))
    assert {j.decision for j in found} == {("p2-1", "case"), ("p2-1", ATTACHED)}
    attached = [(j.side, j.unit, j.judged) for j in found if j.decision[1] == ATTACHED]
    assert attached == [(SNEWPAPERS, "u-NOTICE. ", "right"), (AMERICAN_STORIES, "#7", "wrong")]
    # our component judged wrong by the split and right by the boundary question counts both ways
    assert judged_counts(found)["components"] == {"wrong": 1, "right": 1}


def test_judged_counts_tell_as_articles_apart_by_page_and_our_components_across_pages():
    found = [Judgement(AMERICAN_STORIES, 1, "#3", "wrong", ("p1-1", "case")),
             Judgement(AMERICAN_STORIES, 2, "#3", "wrong", ("p2-1", "case")),
             Judgement(AMERICAN_STORIES, 2, "#3", "wrong", ("p2-1", ATTACHED)),
             Judgement(SNEWPAPERS, 1, "u", "wrong", ("p1-1", "case")),
             Judgement(SNEWPAPERS, 2, "u", "wrong", ("p2-1", "case"))]
    assert judged_counts(found) == {"as_articles": {"wrong": 2, "right": 0}, "components": {"wrong": 1, "right": 0}}


def test_scorecard_counts_what_was_judged_given_the_segmentation(tmp_path):
    seg = IssueSegmentation([group(AS_SPLIT, ["OURS whole"], ["AS a", "AS b", "AS c"]), group(MATCH, ["OURS"], ["AS"])],
                            {}, [])
    verdicts = run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(judge("OURS")), tmp_path)))
    assert verdicts[0].outcome == SNEWPAPERS
    assert scorecard(verdicts)["judged"] is None
    card = scorecard(verdicts, seg)
    assert card["judged"] == {"as_articles": {"wrong": 3, "right": 0}, "components": {"wrong": 0, "right": 1}}
    assert card["components"][CORRECT] == 2  # decisions stay the headline unit


# --- pieces inside an AS merge, and the AS errors decisions find -------------------------------


def merge_with_pieces(texts=("OURS first item", "OURS second item", "OURS third item")):
    """An as_merge of our items and, before it, each item's in_as_merge group (its slice of the AS article)."""
    merge = group(AS_MERGE, list(texts), ["AS whole"])
    inside = []
    for p in merge.pieces:
        g = Group(2, IN_AS_MERGE, [p], [part(p.text, 1)], metrics={"ratio": 99.0})
        g.merge = merge
        inside.append(g)
    return inside, merge


def test_pieces_inside_an_as_merge_are_not_asked_and_take_the_merges_outcome(tmp_path):
    inside, merge = merge_with_pieces()
    seg = IssueSegmentation(inside + [merge], {}, [])
    assert estimate_questions(seg) == 2  # the merge's question, both ways; nothing for its pieces
    client = FakeClient(judge("OURS"))
    verdicts = run(adjudicate(seg, pair(), adjudicator=adjudicator(client, tmp_path)))
    assert len(client.calls) == 2
    assert [(v.case, v.outcome, v.asked) for v in verdicts[:3]] == [(IN_AS_MERGE, SNEWPAPERS, [])] * 3
    card = scorecard(verdicts, seg)
    assert card["components"][CORRECT] == 3 and card["merges"][SNEWPAPERS] == 1
    # the merge judges them; their own groups judge nothing more
    assert card["judged"] == {"as_articles": {"wrong": 1, "right": 0}, "components": {"wrong": 0, "right": 3}}
    against = scorecard(run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(judge("AS")), tmp_path / "b"))))
    assert against["components"][AS_MERGE_RIGHT] == 3 and against["components"][CORRECT] == 0


def test_as_errors_count_items_as_merged_and_fragment_articles_per_decision_for_snewpapers(tmp_path):
    inside, merge = merge_with_pieces(tuple(f"{k} OURS item" for k in range(7)))
    split = group(AS_SPLIT, ["OURS whole"], ["AS a", "AS b", "AS c"])
    seg = IssueSegmentation(inside + [merge, split], {}, [])
    verdicts = run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(judge("OURS")), tmp_path)))
    card = scorecard(verdicts, seg)
    assert card["as_errors"] == {"merge_decisions": 1, ITEMS_MERGED: 7, "split_decisions": 1, FRAGMENT_ARTICLES: 3,
                                **{"nibble_decisions": 0, LINES_ELSEWHERE: 0, PARTIAL_ARTICLES: 0, WIDER_ARTICLES: 0, MISSING_COMPONENTS: 0, MISSING_PAGES: 0}}
    assert card["merges"][SNEWPAPERS] == 1  # decisions stay the headline unit
    assert scorecard(verdicts)["as_errors"] is None
    # decided for American Stories (or undecided), they find no AS errors
    verdicts = run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(judge("AS")), tmp_path / "b")))
    assert scorecard(verdicts, seg)["as_errors"] == {"merge_decisions": 0, ITEMS_MERGED: 0, "split_decisions": 0,
                                                     FRAGMENT_ARTICLES: 0, "nibble_decisions": 0, LINES_ELSEWHERE: 0, PARTIAL_ARTICLES: 0, WIDER_ARTICLES: 0, MISSING_COMPONENTS: 0, MISSING_PAGES: 0}


# --- a component across pages -------------------------------------------------------------------


def cross_page(parts_by_page=(("AS one", "AS two"), ("AS three",))):
    """Our component on pages 1-2 over AS #1, #2... on page 1 and the next ids on page 2 (as
    segmentation.link_pages leaves it): the as_split on page 1 and the continued group on page 2."""
    pieces = [OurPiece("u-story", "domestic_news", page, (BOX,), "OURS the whole story", (1, 2)) for page in (1, 2)]
    parts = [part(text, k, page) for page, texts in enumerate(parts_by_page, start=1)
             for k, text in enumerate(texts, start=1)]
    split = Group(1, AS_SPLIT, pieces, parts)
    pointer = Group(2, CONTINUED, [pieces[1]], [p for p in parts if p.page == 2], other_pages=[1])
    split.continued, pointer.continues = [pointer], split
    return IssueSegmentation([split, pointer], {}, [])


def test_a_component_across_pages_is_one_question_with_both_pages_pieces_labelled_by_page():
    split = cross_page().groups[0]
    content = dispute_messages(split, PageContext("Delaware Gazette", "1809-12-23", 1, (1, 2)), SNEWPAPERS)[1]["content"]
    assert content.startswith("Newspaper: Delaware Gazette\nDate: 1809-12-23\nPages: 1 and 2\n\n"
                              "One system believes this passage is one component running from page 1 to page 2; "
                              "the other system believes it is 3 components, shown below as [1] to [3].\n\n")
    assert '[1] (p. 1)\nbegins: "AS one"' in content and '[2] (p. 1)\n' in content and '[3] (p. 2)\nbegins: "AS three"' in content
    assert "OURS" not in content  # the passage is shown once, as the pieces
    many_first = dispute_messages(split, PageContext("Delaware Gazette", "1809-12-23", 1, (1, 2)), AMERICAN_STORIES)
    assert ("shown below as [1] to [3]; the other system believes the whole passage is one component running from "
            "page 1 to page 2.") in many_first[1]["content"]
    assert many_first[0]["content"] == prompt(AS_SPLIT)  # the system prompt is the same for every one-vs-many


def test_a_question_on_one_page_is_worded_as_before():
    _, user = dispute_messages(group(AS_SPLIT, ["OURS"], ["AS a", "AS b"]), CONTEXT, SNEWPAPERS)
    assert "(p. " not in user["content"] and "running from" not in user["content"]
    assert user["content"].startswith("Newspaper: Wilmington Journal\nDate: 1856-08-08\nPage: 2\n\n")


def test_a_component_across_pages_decided_one_judges_every_as_article_on_every_page_wrong(tmp_path):
    seg = cross_page()
    assert estimate_questions(seg) == 2
    client = FakeClient(judge("OURS"))
    verdicts = run(adjudicate(seg, pair(), adjudicator=adjudicator(client, tmp_path)))
    assert len(client.calls) == 2  # one question, both orders; nothing for the later page
    assert "Pages: 1 and 2" in client.calls[0]["messages"][1]["content"]
    assert [(v.group_id, v.case, v.outcome, len(v.asked)) for v in verdicts] == [
        ("p1-1", AS_SPLIT, SNEWPAPERS, 2), ("p2-1", CONTINUED, SNEWPAPERS, 0)]
    found = judgements(seg.groups[0], verdicts[0])
    assert sorted((j.page, j.unit) for j in found if j.side == AMERICAN_STORIES) == [(1, "#1"), (1, "#2"), (2, "#1")]
    assert sorted((j.page, j.judged) for j in found if j.side == SNEWPAPERS) == [(1, "right"), (2, "right")]
    assert judgements(seg.groups[1], verdicts[1]) == []  # the later page judges nothing again
    card = scorecard(verdicts, seg)
    assert card["judged"] == {"as_articles": {"wrong": 3, "right": 0}, "components": {"wrong": 0, "right": 1}}
    assert card["components"][CORRECT] == 1  # one component, counted once
    assert card["as_errors"] == {"merge_decisions": 0, ITEMS_MERGED: 0, "split_decisions": 1, FRAGMENT_ARTICLES: 3,
                                **{"nibble_decisions": 0, LINES_ELSEWHERE: 0, PARTIAL_ARTICLES: 0, WIDER_ARTICLES: 0, MISSING_COMPONENTS: 0, MISSING_PAGES: 0}}


def test_one_as_article_on_each_page_is_two_fragment_articles_though_their_ids_are_the_same(tmp_path):
    seg = cross_page((("AS one",), ("AS two",)))
    verdicts = run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(judge("OURS")), tmp_path)))
    assert verdicts[0].outcome == SNEWPAPERS
    assert as_errors(seg, verdicts)[FRAGMENT_ARTICLES] == 2  # "#1" on page 1 and "#1" on page 2
    assert scorecard(verdicts, seg)["judged"]["as_articles"] == {"wrong": 2, "right": 0}


# --- nibbles: a line of our component in another AS article ------------------------------------

TITLE = "To the CITIZENS of the United States."
BODY = "IF ever there was a time when national exertion, firmness and unanimity were necessary."


def nibble_group(case=MATCH):
    g = group(case, ["To the Citizens of the United States. " + BODY], [BODY])
    g.as_nibbles = [part(TITLE, 17)]
    return g


def test_a_nibble_is_asked_as_a_boundary_question_about_the_as_text_without_it(tmp_path):
    client = FakeClient(lambda r: response(json.dumps({"verdict": "belongs", "reason": "the address's title"})))
    (v,) = run(adjudicate(IssueSegmentation([nibble_group()], {}, []), pair(), adjudicator=adjudicator(client, tmp_path)))
    assert (v.outcome, v.asked, v.attached_outcome) == (MATCHED, [], None)
    assert v.nibble_outcome == SNEWPAPERS  # "belongs": our component holds the line, as snewpapers has it
    assert {q.a_side for q in v.nibble_asked} == {SNEWPAPERS, AMERICAN_STORIES}
    users = [r["messages"][1]["content"] for r in client.calls]
    assert len(users) == 2
    for user in users:  # the component shown is the AS text without the line, so neither side's text holds it
        component = user.split("The component:\n", 1)[1].split('"""')[1]
        assert BODY in component and "Citizens" not in component and f"«{TITLE}»" in user
    assert all(r["response_format"]["json_schema"]["schema"] is ATTACHED_SCHEMA for r in client.calls)


def test_a_nibble_decided_for_snewpapers_judges_only_the_nibbling_as_article_wrong():
    found = judgements(nibble_group(), GroupVerdict("p1-1", MATCH, MATCHED, nibble_outcome=SNEWPAPERS))
    assert [(j.side, j.unit, j.judged, j.decision[1]) for j in found] == [
        (SNEWPAPERS, "u-To the C", "right", NIBBLE), (AMERICAN_STORIES, "#17", "wrong", NIBBLE)]
    found = judgements(nibble_group(), GroupVerdict("p1-1", MATCH, MATCHED, nibble_outcome=AMERICAN_STORIES))
    assert [(j.side, j.judged) for j in found] == [(SNEWPAPERS, "wrong"), (AMERICAN_STORIES, "right")]


def test_estimates_and_scorecard_count_nibble_questions(tmp_path):
    seg = IssueSegmentation([nibble_group(), group(MATCH, ["OURS"], ["AS"])], {}, [])
    assert estimate_questions(seg) == 2
    client = FakeClient(lambda r: response(json.dumps({"verdict": "belongs", "reason": "r"})))
    verdicts = run(adjudicate(seg, pair(), adjudicator=adjudicator(client, tmp_path)))
    card = scorecard(verdicts, seg)
    assert card["nibbles"] == {SNEWPAPERS: 1, AMERICAN_STORIES: 0, FURNITURE: 0, UNSTABLE: 0, NO_VERDICT: 0}
    assert card["as_errors"][LINES_ELSEWHERE] == 1 and card["as_errors"]["nibble_decisions"] == 1
    assert card["judged"]["as_articles"] == {"wrong": 1, "right": 0}


# --- partial: one AS article holding under 90% of our component ------------------------------------


def test_a_partial_is_not_asked_and_counts_as_partly_captured(tmp_path):
    seg = IssueSegmentation([group(AS_PARTIAL, ["MEETINGS. The annual meeting."], ["MEETINGS."])], {}, [])
    client = FakeClient(lambda r: pytest.fail("nothing to ask"))
    (v,) = run(adjudicate(seg, pair(), adjudicator=adjudicator(client, tmp_path)))
    assert (v.case, v.outcome, v.asked) == (AS_PARTIAL, PARTLY_CAPTURED, [])
    assert scorecard([v])["components"][PARTLY_CAPTURED] == 1
    # AS missing part of our component is an AS error without asking: its article judged wrong, ours not judged
    assert judgements(seg.groups[0], v) == [Judgement(AMERICAN_STORIES, 2, "#1", "wrong", ("p2-1", PARTIAL))]
    card = scorecard([v], seg)
    assert card["as_errors"][PARTIAL_ARTICLES] == 1 and card["judged"]["components"] == {"wrong": 0, "right": 0}


def test_scorecard_counts_split_decisions_and_matches_on_their_own(tmp_path):
    seg = segmentation()
    card = scorecard(run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(judge("OURS")), tmp_path))))
    assert card["splits"] == {SNEWPAPERS: 1, AMERICAN_STORIES: 0, NEITHER: 0, UNSTABLE: 0, ILLEGIBLE_UNSURE: 0, NO_VERDICT: 0}
    assert card["matches"] == 1


# --- an AS article running on beyond our component ---------------------------------------------


def wider_group(attached="Just received and for sale, fresh lemons by the box. A. WILLIS."):
    g = group(AS_WIDER, ["Ten Dollars Reward. STRAYED, a hound dog. WILLIAM SPENCE."], ["Ten Dollars Reward. STRAYED"])
    if attached:
        g.as_attached_outside = [part(attached, 1)]
    return g


@pytest.mark.parametrize("answer, outcome, component", [("separate", SNEWPAPERS, CORRECT),
                                                         ("belongs", AMERICAN_STORIES, AS_WIDER_RIGHT)])
def test_as_wider_is_decided_by_its_attached_text(tmp_path, answer, outcome, component):
    seg = IssueSegmentation([wider_group()], {}, [])
    client = FakeClient(lambda r: response(json.dumps({"verdict": answer, "reason": "r"})))
    (v,) = run(adjudicate(seg, pair(), adjudicator=adjudicator(client, tmp_path)))
    assert (v.case, v.outcome, v.attached_outcome, v.asked) == (AS_WIDER, outcome, outcome, [])
    card = scorecard([v], seg)
    assert card["wider"][outcome] == 1 and sum(card["attached_outside"].values()) == 0  # counted once, as wider
    assert card["components"][component] == 1 and card["matches"] == 0
    assert card["as_errors"][WIDER_ARTICLES] == (1 if outcome == SNEWPAPERS else 0)
    assert [(j.side, j.judged) for j in judgements(seg.groups[0], v)] == [
        (SNEWPAPERS, "right" if outcome == SNEWPAPERS else "wrong"),
        (AMERICAN_STORIES, "wrong" if outcome == SNEWPAPERS else "right")]


def test_as_wider_without_attached_text_is_no_verdict(tmp_path):
    seg = IssueSegmentation([wider_group(attached=None)], {}, [])
    (v,) = run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(lambda r: pytest.fail("no question")), tmp_path)))
    assert v.outcome == NO_VERDICT



# --- a line that may be a section heading decides nothing ----------------------------------------


def test_the_boundary_prompt_says_one_system_leaves_page_furniture_out_without_naming_it():
    text = prompt(ATTACHED)
    assert "occasionally leave out what they judge to be page furniture" in text and "section titles" in text
    assert "not by tiny fragments" in text and 'answer "furniture"' in text
    assert "a jump line" in text
    assert "snewpapers" not in text.lower() and "american stories" not in text.lower()
    assert validate({"verdict": "furniture", "reason": "r"}, ATTACHED_SCHEMA)


@pytest.mark.parametrize("make", [lambda: attached_group(), lambda: nibble_group()])
def test_furniture_answered_both_ways_judges_neither_side(tmp_path, make):
    g = make()
    client = FakeClient(lambda r: response(json.dumps({"verdict": "furniture", "reason": "a section heading"})))
    (v,) = run(adjudicate(IssueSegmentation([g], {}, []), pair(), adjudicator=adjudicator(client, tmp_path)))
    outcome = v.attached_outcome if g.as_attached_outside else v.nibble_outcome
    assert outcome == FURNITURE and judgements(g, v) == []
    card = scorecard([v], IssueSegmentation([g], {}, []))
    assert card["attached_outside" if g.as_attached_outside else "nibbles"][FURNITURE] == 1
    assert card["as_errors"][LINES_ELSEWHERE] == 0


def test_as_wider_whose_extra_text_is_furniture_counts_our_component_correct(tmp_path):
    seg = IssueSegmentation([wider_group(attached="VIRGINIA LEGISLATURE.")], {}, [])
    client = FakeClient(lambda r: response(json.dumps({"verdict": "furniture", "reason": "r"})))
    (v,) = run(adjudicate(seg, pair(), adjudicator=adjudicator(client, tmp_path)))
    assert v.outcome == FURNITURE and scorecard([v], seg)["components"][CORRECT] == 1
    assert scorecard([v], seg)["as_errors"][WIDER_ARTICLES] == 0



def test_our_component_with_no_as_text_is_an_as_error_without_an_as_article_to_mark(tmp_path):
    seg = IssueSegmentation([group(OURS_ONLY, ["PROBATE NOTICE. The Probate Court for the District of Marlboro."], [])], {}, [])
    (v,) = run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(lambda r: pytest.fail("no question")), tmp_path)))
    card = scorecard([v], seg)
    assert card["components"][NOT_CAPTURED] == 1 and card["as_errors"][MISSING_COMPONENTS] == 1
    assert judgements(seg.groups[0], v) == []



@pytest.mark.parametrize("kind, outcome", [("advertisement", EXCLUDED), ("terms", EXCLUDED), ("furniture", EXCLUDED),
                                           ("other", MISSED)])
def test_as_only_text_is_asked_its_kind_and_ads_are_counted_as_kept_by_as(tmp_path, kind, outcome):
    seg = IssueSegmentation([group(AS_ONLY, [], ["Just received and for sale cheap, a fine lot of bacon."])], {}, [])
    client = FakeClient(lambda r: response(json.dumps({"kind": kind, "reason": "r"})))
    (v,) = run(adjudicate(seg, pair(), adjudicator=adjudicator(client, tmp_path)))
    assert v.outcome == outcome
    assert scorecard([v], seg)["as_ads_kept"] == (1 if kind == "advertisement" else 0)


def test_the_as_only_prompt_asks_for_the_kind():
    text = prompt(AS_ONLY)
    assert all(f'"{k}"' in text for k in ("advertisement", "terms", "furniture", "other"))



# --- AS marked part of our component illegible: is our component one item? ------------------------


def illegible_split():
    g = group(AS_SPLIT, ["A GALA LABOR DAY. Two monster meetings of workmen."], ["A GALA LABOR DAY.", "A GALA DAY FOR LABOR."])
    g.metrics = {"illegible_cover": 0.5}
    return g


@pytest.mark.parametrize("answer, outcome", [("one", SNEWPAPERS), ("several", ILLEGIBLE_UNSURE), ("mixed", ILLEGIBLE_UNSURE)])
def test_a_split_where_as_marked_part_illegible_asks_once_whether_our_component_is_one_item(tmp_path, answer, outcome):
    seg = IssueSegmentation([illegible_split()], {}, [])
    client = FakeClient(lambda r: response(json.dumps({"verdict": answer, "reason": "r"})))
    (v,) = run(adjudicate(seg, pair(), adjudicator=adjudicator(client, tmp_path)))
    assert v.outcome == outcome and len(client.calls) == 1
    request = client.calls[0]
    assert request["messages"][0]["content"] == prompt(ONE_COMPONENT)
    assert "A GALA LABOR DAY. Two monster meetings of workmen." in request["messages"][1]["content"]
    assert "A GALA DAY FOR LABOR." not in request["messages"][1]["content"]  # AS's pieces are not shown
    assert estimate_questions(seg) == 1
    judged = judgements(seg.groups[0], v)
    assert bool(judged) == (outcome == SNEWPAPERS)
    assert scorecard([v], seg)["splits"][outcome] == 1


def test_the_one_component_prompt_asks_about_one_passage_without_naming_a_system():
    text = prompt(ONE_COMPONENT)
    assert text.startswith(FRAMING) and '"one"' in text and '"several"' in text
    assert "snewpapers" not in text.lower() and "american stories" not in text.lower()


def test_pages_of_our_components_as_lacks_the_text_of_count_as_as_errors(tmp_path):
    g = illegible_split()
    g.page_cover = {2: (0.95, 0.0), 3: (0.02, 0.97)}
    seg = IssueSegmentation([g], {}, [])
    client = FakeClient(lambda r: response(json.dumps({"verdict": "one", "reason": "r"})))
    verdicts = run(adjudicate(seg, pair(), adjudicator=adjudicator(client, tmp_path)))
    assert scorecard(verdicts, seg)["as_errors"][MISSING_PAGES] == 1


@pytest.mark.parametrize("answers, outcome", [(("furniture", "separate"), FURNITURE), (("belongs", "furniture"), FURNITURE),
                                              (("furniture", "furniture"), FURNITURE), (("belongs", "separate"), UNSTABLE)])
def test_a_boundary_line_either_order_calls_furniture_decides_nothing(tmp_path, answers, outcome):
    """Workmen's Advocate 1886-04-25 p2-17: "WHAT OTHERS SAY." answered furniture one way, separate the other."""
    seen = []

    def reply(request):
        seen.append(request)
        return response(json.dumps({"verdict": answers[len(seen) - 1], "reason": "r"}))
    (v,) = run(adjudicate(IssueSegmentation([attached_group()], {}, []), pair(), adjudicator=adjudicator(FakeClient(reply), tmp_path)))
    assert v.attached_outcome == outcome



@pytest.mark.parametrize("case", [AS_SPLIT, ONE_COMPONENT])
def test_the_one_or_several_prompts_say_to_judge_by_the_overall_content_not_tiny_fragments(case):
    text = prompt(case)
    assert "occasionally leave out what they judge to be page furniture" in text and "not by tiny fragments" in text



def test_an_as_only_article_with_no_text_is_not_asked_and_counted_apart(tmp_path):
    seg = IssueSegmentation([group(AS_ONLY_EMPTY, [], [""])], {}, [])
    (v,) = run(adjudicate(seg, pair(), adjudicator=adjudicator(FakeClient(lambda r: pytest.fail("no question")), tmp_path)))
    assert v.outcome == EMPTY and scorecard([v])["as_only"][EMPTY] == 1 and estimate_questions(seg) == 0


def test_aclose_closes_the_client_when_it_can_and_is_harmless_when_it_cannot(tmp_path):
    closed = []

    class Closable(FakeClient):
        async def close(self):
            closed.append(True)
    run(adjudicator(Closable(lambda r: None), tmp_path).aclose())
    assert closed == [True]
    run(adjudicator(FakeClient(lambda r: None), tmp_path).aclose())  # no close(): nothing to do
