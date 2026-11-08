"""Ask an LLM which of two segmentations matches the printed page, case by case (see segmentation.py).

Two cases are disputes, both one component against several: `as_split` (our one component holds
boxes of N American Stories articles; the pieces are each article's text inside our component) and
`as_merge` (N of our components sit inside one AS article; the pieces are our components' texts).
Each is asked about the passage itself: it is shown once, as the N pieces, each with its first and
last words, and the model is told one system believes it is one component and the other that it
is N, then asked "one", "several" or "mixed". It is asked twice, the two beliefs stated in
opposite orders; a verdict that changes with the order is "unstable". A component whose AS text is
all the merged AS article's (segmentation's `in_as_merge`) is not asked on its own: the merge's
verdict is its outcome. A component of ours running over pages is one as_split across them
(segmentation's `link_pages`): its pieces are every page's AS articles, page by page, each labelled
"(p. N)", and the one-component belief says the passage runs from its first page to its last; it
is asked once, and its later pages (`continued`) take its outcome. An as_split where AS marked part of
our component illegible (segmentation's `Group.illegible_part`) is asked once, about our component's
text alone (its pieces do not show the text): "one" means snewpapers is right, otherwise it is unsure
(ILLEGIBLE_UNSURE), never AS's. Left-over AS text
(`as_only`) is asked once (it has one text, so nothing to blind): is it an advertisement,
subscription terms or page furniture, which snewpapers leaves out on purpose? (It answers which kind:
"advertisement", "terms", "furniture" or "other"; the ads among them are counted apart, since American
Stories says it leaves ads out.) Matches, our
components AS did not capture, and tiny AS left-overs are not asked. A component whose AS article
runs on beyond it into other text (segmentation's `as_wider`) takes the outcome of the question
about that attached text: "separate" means AS's article runs into another item (snewpapers right).

AS text attached outside one of our components (segmentation's `Group.as_attached_outside`: text of
the component's own AS articles printed beyond it) is a boundary question, asked besides the
group's case: the component's text is shown, then the attached text, with one system believing the
component includes it and the other not; "belongs" or "separate". It is asked in both orders of the
two beliefs, like a dispute: "separate" means snewpapers is right, "belongs" American Stories. The
prompt says one system leaves page furniture out by design (it does not say which): a line or two
that could plausibly be furniture is answered "furniture" and decides nothing either way.

The prompts are the files in prompts/, one per scenario, sent as the system message; the user
message carries the page (newspaper, date, page number) and the texts, nothing else. The answer's
shape is enforced with a strict JSON schema; an answer that does not parse or validate, or that
ran out of tokens, is "no verdict", never guessed. Valid answers are cached under
data/adjudications/, so asking again costs nothing.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

from snewpapers_tools.paths import DATA_DIR
from snewpapers_tools.segmentation import (
    AS_MERGE,
    AS_ONLY,
    AS_ONLY_TINY,
    AS_ONLY_EMPTY,
    AS_PARTIAL,
    AS_SPLIT,
    AS_WIDER,
    CONTINUED,
    IN_AS_MERGE,
    MATCH,
    OURS_ONLY,
    Group,
    IssueSegmentation,
)

CACHE_DIR = DATA_DIR / "adjudications"
MAX_TOKENS = 16_000  # the model reasons by default and reasoning counts as output
TIMEOUT = 1200.0  # seconds per call, the SDK's retries included; a call that times out is made once more
MAX_RETRIES = 6  # 429 / 529 are congestion: the SDK backs off and retries

SNEWPAPERS = "snewpapers"
AMERICAN_STORIES = "american_stories"
SIDES = (SNEWPAPERS, AMERICAN_STORIES)

# outcomes
NEITHER = "neither"
UNSTABLE = "unstable"
NO_VERDICT = "no verdict"
MATCHED = "match"  # a match: nothing to ask
EXCLUDED = "correctly not served"
MISSED = "missed by snewpapers"
TINY = "tiny"
EMPTY = "no text, not asked"  # an AS-only article with no text at all: AS read nothing there
NOT_CAPTURED = "not captured by AS"
ILLEGIBLE_UNSURE = "unsure: AS could not read part"  # a split AS read part of illegibly, our component not found one item
FURNITURE = "page furniture, not counted"  # a boundary question answered "furniture": decides nothing
PARTLY_CAPTURED = "partly captured by AS"  # an as_partial: AS text for under 90% of our component
DISPUTE_OUTCOMES = (SNEWPAPERS, AMERICAN_STORIES, NEITHER, UNSTABLE, ILLEGIBLE_UNSURE, NO_VERDICT)
AS_ONLY_OUTCOMES = (EXCLUDED, MISSED, NO_VERDICT, TINY, EMPTY)
OUTCOME_LABELS = {
    SNEWPAPERS: "snewpapers right",
    AMERICAN_STORIES: "AS right",
    NEITHER: "neither",
    UNSTABLE: "unstable",
    NO_VERDICT: "no verdict",
    MATCHED: MATCHED,
    EXCLUDED: EXCLUDED,
    MISSED: MISSED,
    TINY: "tiny, not asked",
    EMPTY: EMPTY,
    NOT_CAPTURED: NOT_CAPTURED,
    ILLEGIBLE_UNSURE: ILLEGIBLE_UNSURE,
    FURNITURE: FURNITURE,
    PARTLY_CAPTURED: PARTLY_CAPTURED,
}

# The scorecard's rows for our components (one per match, as_split, in_as_merge or ours_only case)
CORRECT = "correct as is"  # a match, an as_split decided "one", or in an as_merge decided "several"
AS_SPLIT_RIGHT = "AS split it correctly"  # an as_split decided "several"
AS_MERGE_RIGHT = "AS merged it correctly"  # in an as_merge decided "one"
AS_WIDER_RIGHT = "AS's wider article right"  # an as_wider whose attached text was decided "belongs"
COMPONENT_OUTCOMES = (CORRECT, AS_SPLIT_RIGHT, AS_MERGE_RIGHT, NEITHER, UNSTABLE, NO_VERDICT, PARTLY_CAPTURED,
                      AS_WIDER_RIGHT, NOT_CAPTURED)

# American Stories errors that decisions for snewpapers find, counted besides the decisions
ITEMS_MERGED = "items AS merged"  # an as_merge decided for snewpapers: the N items of ours AS failed to separate
FRAGMENT_ARTICLES = "fragment articles"  # an as_split decided for snewpapers: the N AS articles it broke ours into (on all its pages)
LINES_ELSEWHERE = "lines in another AS article"  # a nibble decided for snewpapers: a line of ours AS put in another article
PARTIAL_ARTICLES = "AS articles missing part of ours"  # an as_partial: AS has text for under 90% of our component
WIDER_ARTICLES = "AS articles running on into other items"  # an as_wider decided for snewpapers
MISSING_COMPONENTS = "our components AS has no text for"  # an ours_only: the item is not in AS's data at all
MISSING_PAGES = "pages of our components AS lacks the text for"  # a cross-page as_split's pages AS has under 90% of

# American Stories defects found without asking, counted in the scorecard
AS_DUPLICATED = "AS duplicated text"  # the same text in two (or more) AS articles inside one of our components
AS_DEFECTS = (AS_DUPLICATED,)

# Boundary questions: AS text attached outside one of our components (counted in their own row)
ATTACHED = "attached_outside"
ATTACHED_OUTCOMES = (SNEWPAPERS, AMERICAN_STORIES, FURNITURE, UNSTABLE, NO_VERDICT)
# Nibbles: a line or two of our component AS has in another article (segmentation's Group.as_nibbles)
NIBBLE = "nibble"
ONE_COMPONENT = "one_component"  # a split AS marked part of illegible: is our component one item?
PARTIAL = "partial"  # an as_partial's decision: made by the geometry (AS has no text for the rest), not asked

DISPUTED = (AS_SPLIT, AS_MERGE)
PROMPT_FILES = {
    AS_SPLIT: "one_vs_many.md",
    AS_MERGE: "one_vs_many.md",
    AS_ONLY: "as_only.md",
    ATTACHED: "attached_outside.md",
    ONE_COMPONENT: "one_component.md",
}
EDGE_WORDS = 15  # words of each piece's beginning and end quoted before it

ONE_VS_MANY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["one", "several", "mixed"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}
ATTACHED_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["belongs", "separate", "furniture"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}
AS_ONLY_SCHEMA = {
    "type": "object",
    "properties": {"kind": {"type": "string", "enum": ["advertisement", "terms", "furniture", "other"]},
                   "reason": {"type": "string"}},
    "required": ["kind", "reason"],
    "additionalProperties": False,
}


def prompt(case: str) -> str:
    """The system prompt for a case, or for ATTACHED (the packaged prompts/*.md file)."""
    return files("snewpapers_tools").joinpath("prompts", PROMPT_FILES[case]).read_text()


# --- messages ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class PageContext:
    newspaper: str
    date: str
    page: int
    pages: tuple[int, ...] = ()  # a passage across pages: all of them ("Pages: 2 and 3")

    def __str__(self) -> str:
        if len(self.pages) > 1:
            shown = ", ".join(map(str, self.pages[:-1])) + f" and {self.pages[-1]}"
            return f"Newspaper: {self.newspaper}\nDate: {self.date}\nPages: {shown}"
        return f"Newspaper: {self.newspaper}\nDate: {self.date}\nPage: {self.page}"


def page_context(pair, page: int, pages: tuple[int, ...] = ()) -> PageContext:
    """The newspaper (title-cased, from the components), the issue date and the page (or, for a
    passage across pages, the pages)."""
    name = next((c["newspaper"] for c in pair.components if c.get("newspaper")), pair.issue.lccn)
    return PageContext(name.title(), pair.issue.date, page, tuple(pages) if len(pages) > 1 else ())


def group_context(pair, group: Group) -> PageContext:
    """The context of a group's question: its page, or a cross-page as_split's pages."""
    return page_context(pair, group.page, group.pages)


def _quoted(text: str) -> str:
    return f'"""\n{text.strip()}\n"""'


def edges(text: str, words: int = EDGE_WORDS) -> str:
    """A piece's first and last `words` words, whitespace normalised:
    `begins: "..." … ends: "..."`."""
    split = text.split()
    return f'begins: "{" ".join(split[:words])}" … ends: "{" ".join(split[-words:])}"'


def _numbered(texts: list[str], pages: list[int] | None = None) -> str:
    """Each text as "[k]" (with "(p. N)" given its page), its edges and the text quoted."""
    heads = [f"[{k}]" + (f" (p. {pages[k - 1]})" if pages else "") for k in range(1, len(texts) + 1)]
    return "\n\n".join(f"{head}\n{edges(t)}\n{_quoted(t)}" for head, t in zip(heads, texts))


def _messages(case: str, user: str) -> list[dict]:
    return [{"role": "system", "content": prompt(case)}, {"role": "user", "content": user}]


def one_vs_many_messages(context: PageContext, pieces: list[str], *, one_first: bool,
                         pages: list[int] | None = None, running: tuple[int, int] | None = None) -> list[dict]:
    """The passage as the split side's numbered pieces (they cover the same text as the single side's
    one component), each headed by its first and last words, with both beliefs stated, unattributed:
    the one-component belief first when `one_first`, else the N-components belief first. No sides
    are named.

    A passage across pages gives each piece's page (`pages`: "[1] (p. 2)") and the pages the one
    component runs over (`running`: "one component running from page 2 to page 3")."""
    n = len(pieces)
    if n < 2:
        raise ValueError("one vs many needs at least two pieces")
    shown = "[1] and [2]" if n == 2 else f"[1] to [{n}]"
    span = f" running from page {running[0]} to page {running[1]}" if running else ""
    if one_first:
        beliefs = (f"One system believes this passage is one component{span}; the other system believes it is "
                   f"{n} components, shown below as {shown}.")
    else:
        beliefs = (f"One system believes this passage is {n} components, shown below as {shown}; "
                   f"the other system believes the whole passage is one component{span}.")
    user = f"{context}\n\n{beliefs}\n\n{_numbered(pieces, pages)}\n\nIs it one component or several?"
    return _messages(AS_SPLIT, user)


def one_component_messages(context: PageContext, text: str) -> list[dict]:
    """One passage, one system's single component, and the question whether it is one printed item."""
    user = f"{context}\n\nThe passage:\n{_quoted(text)}\n\nIs it one component or several?"
    return _messages(ONE_COMPONENT, user)


def as_only_messages(context: PageContext, text: str) -> list[dict]:
    user = (f"{context}\n\nThe text:\n{_quoted(text)}\n\n"
            "Is this the text of an advertisement (or several), subscription terms or a colophon, "
            "or page furniture such as a masthead or section heading?")
    return _messages(AS_ONLY, user)


def attached_messages(context: PageContext, component: str, attached: str, *, belongs_first: bool) -> list[dict]:
    """The component's text, then the text attached outside it, with both beliefs stated,
    unattributed: that the component includes the text first when `belongs_first`, else that it
    does not. No sides are named."""
    shown = f"«{attached.strip()}»"
    if belongs_first:
        beliefs = ("One system believes this component also includes the following text, printed next to it "
                   f"on the page:\n{shown}\nThe other system believes it does not.")
    else:
        beliefs = ("One system believes this component does not include the following text, printed next to it "
                   f"on the page:\n{shown}\nThe other system believes it does.")
    user = (f"{context}\n\nThe component:\n{_quoted(component)}\n\n{beliefs}\n\n"
            "Does the text belong to this component?")
    return _messages(ATTACHED, user)


def group_attached_messages(group: Group, context: PageContext, first: str) -> list[dict]:
    """The boundary question for a group's attached-outside text with the belief of `first` stated
    first: American Stories believes the component includes the text, snewpapers that it does not."""
    return attached_messages(context, group.ours_text, group.attached_text, belongs_first=first == AMERICAN_STORIES)


def group_nibble_messages(group: Group, context: PageContext, first: str) -> list[dict]:
    """The boundary question for a group's nibbles with the belief of `first` stated first:
    snewpapers believes the component includes the text, American Stories that it does not. The
    component shown is the AS text of the group's own articles, without the nibbles, so neither
    side's text already holds the line asked about."""
    nibbled = "\n\n".join(part.text for part in group.as_nibbles)
    return attached_messages(context, group.as_text, nibbled, belongs_first=first == SNEWPAPERS)


def one_side(group: Group) -> str:
    """In a dispute, the side that has the passage as one component: snewpapers for an as_split,
    American Stories for an as_merge."""
    if group.case not in DISPUTED:
        raise ValueError(f"not a disputed case: {group.case}")
    return SNEWPAPERS if group.case == AS_SPLIT else AMERICAN_STORIES


def many_pieces(group: Group) -> list[str]:
    """The dispute's N pieces in position order: for an as_split, each AS article's text inside our
    component (across pages: page by page); for an as_merge, our components' texts."""
    return [part.text for part in group.parts] if group.case == AS_SPLIT else [p.text for p in group.pieces]


def piece_pages(group: Group) -> list[int] | None:
    """Each of a cross-page as_split's pieces' page; None for a dispute on one page."""
    return [part.page for part in group.parts] if group.case == AS_SPLIT and group.cross_page else None


def dispute_messages(group: Group, context: PageContext, first: str) -> list[dict]:
    """The question for a dispute with the belief of `first` ("snewpapers" or "american_stories")
    stated first (sides are not named); a cross-page as_split's pieces labelled with their pages."""
    pages = piece_pages(group)
    running = (group.pages[0], group.pages[-1]) if pages else None
    return one_vs_many_messages(context, many_pieces(group), one_first=first == one_side(group),
                                pages=pages, running=running)


# --- validation -------------------------------------------------------------------------------


def validate(answer: object, schema: dict) -> dict | None:
    """`answer` when it has exactly the schema's fields, each of the right type and in its enum; else None."""
    if not isinstance(answer, dict) or set(answer) != set(schema["required"]):
        return None
    for name, spec in schema["properties"].items():
        value = answer[name]
        if spec["type"] == "string" and not isinstance(value, str):
            return None
        if spec["type"] == "boolean" and not isinstance(value, bool):
            return None
        if "enum" in spec and value not in spec["enum"]:
            return None
    return answer


def parse(content: str | None, finish_reason: str | None, schema: dict) -> dict | None:
    """The answer from a response's content: JSON, validated, and only from a response that finished."""
    if finish_reason == "length" or not content:
        return None
    try:
        return validate(json.loads(content), schema)
    except ValueError:
        return None


# --- asking -----------------------------------------------------------------------------------


@dataclass
class Answer:
    """One question asked: the parsed answer (None = no verdict), its cost and the raw content."""

    parsed: dict | None
    cost: float | None  # billed dollars; 0 from the cache, None if the service did not say
    raw: str  # the response's content as sent, or the error when there was no response
    cached: bool = False
    error: bool = False  # the question failed (timeout, API error): nothing came back
    input_tokens: int | None = None  # the usage the service reported; 0 from the cache
    output_tokens: int | None = None


def _usage(usage) -> dict:
    """The response's `usage` as a dict ({} if there is none)."""
    if usage is None:
        return {}
    return usage.model_dump() if hasattr(usage, "model_dump") else usage if isinstance(usage, dict) else vars(usage)


def _cost(usage) -> float | None:
    """The billed dollars, where the service reports them: `usage.estimated_cost` (DeepInfra) or
    `usage.cost` (OpenRouter), extension fields the SDK keeps as extra. None otherwise."""
    data = _usage(usage)
    cost = data.get("estimated_cost", data.get("cost"))
    return float(cost) if isinstance(cost, (int, float)) else None


def _tokens(usage, field: str) -> int | None:
    count = _usage(usage).get(field)
    return count if isinstance(count, int) else None


class Adjudicator:
    """Asks the questions, through an OpenAI-compatible async client, and caches valid answers.

    By default the client is `openai.AsyncOpenAI` built from LLM_API_KEY, LLM_BASE_URL and LLM_MODEL
    (read from .env), all three required; pass `client` to use another object with the same
    `chat.completions.create`.
    """

    def __init__(
        self,
        client=None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        cache_dir: Path | None = CACHE_DIR,
        timeout: float = TIMEOUT,
        max_tokens: int = MAX_TOKENS,
    ):
        if client is None and api_key is None:
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.environ.get("LLM_API_KEY")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL")
        self.model = model or os.environ.get("LLM_MODEL")
        for name, value in (("LLM_BASE_URL", self.base_url), ("LLM_MODEL", self.model)):
            if not value:
                raise RuntimeError(f"{name} is not set: put your OpenAI-compatible service's base URL and model in .env")
        if client is None:
            if not api_key:
                raise RuntimeError("LLM_API_KEY is not set: put a key for an OpenAI-compatible service in .env")
            import openai

            client = openai.AsyncOpenAI(api_key=api_key, base_url=self.base_url, max_retries=MAX_RETRIES)
        self.client = client
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_tokens = max_tokens

    async def aclose(self) -> None:
        """Close the client's connections, inside the event loop that used them (a client left to be
        garbage-collected after asyncio.run has closed its loop prints "Event loop is closed")."""
        close = getattr(self.client, "close", None)
        if close is not None:
            await close()

    def cache_key(self, messages: list[dict]) -> str:
        """sha256 of the model, the service, the prompt (system message) and the whole conversation."""
        prompt_text = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        body = json.dumps([self.model, self.base_url, prompt_text, messages], sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{key}.json"

    async def ask(self, messages: list[dict], schema: dict) -> Answer:
        """One question. A cached valid answer is returned free; otherwise the model is asked once
        (the SDK retries congestion) within `timeout` seconds, and once more if that times out.
        Only valid answers are cached."""
        path = self._cache_path(self.cache_key(messages))
        if path is not None and path.exists():
            hit = json.loads(path.read_text())
            parsed = parse(hit["content"], hit.get("finish_reason"), schema)
            if parsed is not None:
                return Answer(parsed, 0.0, hit["content"], cached=True, input_tokens=0, output_tokens=0)
        for attempt in range(2):  # a timeout is tried once more; anything else is final
            try:
                resp = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"name": "_".join(schema["required"]), "strict": True, "schema": schema},
                        },
                    ),
                    self.timeout,
                )
                break
            except TimeoutError:
                if attempt == 1:
                    return Answer(None, None, f"TimeoutError: no answer within {self.timeout:g} s, twice", error=True)
            except Exception as exc:  # API errors are "no verdict" for this one question
                return Answer(None, None, f"{type(exc).__name__}: {exc}", error=True)
        choice = resp.choices[0]
        content = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        cost = _cost(usage)
        tokens = {"input_tokens": _tokens(usage, "prompt_tokens"), "output_tokens": _tokens(usage, "completion_tokens")}
        parsed = parse(content, choice.finish_reason, schema)
        if parsed is not None and path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {"model": self.model, "base_url": self.base_url, "messages": messages,
                      "content": content, "finish_reason": choice.finish_reason, "cost": cost}
            path.with_suffix(".tmp").write_text(json.dumps(record))
            os.replace(path.with_suffix(".tmp"), path)
        return Answer(parsed, cost, content, **tokens)


# --- groups -----------------------------------------------------------------------------------


@dataclass
class Asked:
    """One question: whose belief was stated first (`a_side`), the answer, and its verdict mapped
    back. `one_side` is set for a dispute: the side that has the passage as one component; and for
    a boundary question: the side that has the attached text inside the component (American Stories)."""

    a_side: str
    answer: Answer
    one_side: str | None = None

    @property
    def verdict(self) -> str | None:
        """"snewpapers", "american_stories", "neither", or None (no verdict): "one" (or "belongs")
        is the side that has one component, "several" (or "separate") the other, "mixed" neither."""
        if self.answer.parsed is None or self.one_side is None:
            return None
        v = self.answer.parsed["verdict"]
        if v == "mixed":
            return NEITHER
        if v == "furniture":
            return FURNITURE
        return self.one_side if v in ("one", "belongs") else next(s for s in SIDES if s != self.one_side)


@dataclass
class GroupVerdict:
    group_id: str  # p{page}-{k}, k counting the page's groups from 1, as in the report
    case: str
    outcome: str  # DISPUTE_OUTCOMES (an in_as_merge: its merge's; continued: its group's), AS_ONLY_OUTCOMES, MATCHED or NOT_CAPTURED
    asked: list[Asked] = field(default_factory=list)
    as_duplicated: bool = False  # the group's AS text was there twice (segmentation's Group.as_duplicated)
    attached_outcome: str | None = None  # the boundary question's (ATTACHED_OUTCOMES), when AS attaches text outside
    attached_asked: list[Asked] = field(default_factory=list)
    nibble_outcome: str | None = None  # the nibble question's (ATTACHED_OUTCOMES), when AS has a line of ours elsewhere
    nibble_asked: list[Asked] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return sum(q.answer.cost or 0.0 for q in self.asked + self.attached_asked + self.nibble_asked)

    @property
    def reasons(self) -> list[str]:
        return [q.answer.parsed["reason"] for q in self.asked if q.answer.parsed is not None]


def group_ids(seg: IssueSegmentation) -> list[str]:
    """Each group's id, p{page}-{k}, numbered within its page in segmentation order."""
    seen: dict[int, int] = {}
    ids = []
    for group in seg.groups:
        seen[group.page] = seen.get(group.page, 0) + 1
        ids.append(f"p{group.page}-{seen[group.page]}")
    return ids


def first_a_side(seed: str) -> str:
    """Whose belief is stated first the first time a dispute is asked: random, but the same for the same seed."""
    return random.Random(seed).choice(SIDES)


def dispute_outcome(asked: list[Asked]) -> str:
    """Both orders agree (after mapping back): that verdict; they differ: unstable; either missing: no
    verdict. A boundary line either order calls page furniture decides nothing (FURNITURE): the prompt
    says a line that could plausibly be furniture should not decide which system is right."""
    verdicts = [q.verdict for q in asked]
    if not verdicts or any(v is None for v in verdicts):
        return NO_VERDICT
    if FURNITURE in verdicts:
        return FURNITURE
    return verdicts[0] if len(set(verdicts)) == 1 else UNSTABLE


def as_only_outcome(answer: Answer) -> str:
    if answer.parsed is None:
        return NO_VERDICT
    return MISSED if answer.parsed["kind"] == "other" else EXCLUDED


def estimate_questions(seg: IssueSegmentation) -> int:
    """Questions `adjudicate` will ask, cache aside: two per dispute, one per (not tiny) AS-only group,
    two per group with AS text attached outside it."""
    return sum((1 if g.case == AS_SPLIT and g.illegible_part else 2 if g.case in DISPUTED else 1 if g.case == AS_ONLY else 0) + (2 if g.as_attached_outside else 0)
               + (2 if g.as_nibbles else 0) for g in seg.groups)


def estimate_cost(seg: IssueSegmentation, per_question: float) -> float:
    """Dollars at `per_question` (measured on a real question), before the cache saves any."""
    return estimate_questions(seg) * per_question


async def adjudicate(seg: IssueSegmentation, pair, *, adjudicator: Adjudicator | None = None,
                     concurrency: int = 8) -> list[GroupVerdict]:
    """A verdict for every group, in segmentation order (only disputes, AS-only groups and AS text
    attached outside a component are asked; an in_as_merge takes its merge's outcome, a continued
    group its cross-page as_split's).

    In Jupyter: `verdicts = await adjudicate(seg, pair)`; in a script: `asyncio.run(adjudicate(...))`.
    """
    adjudicator = adjudicator or Adjudicator()
    issue = pair.issue
    issue_name = f"{issue.lccn}_{issue.date}_ed-{issue.edition}"
    gate = asyncio.Semaphore(concurrency)

    async def ask(messages: list[dict], schema: dict) -> Answer:
        async with gate:
            return await adjudicator.ask(messages, schema)

    def both_orders(seed: str) -> list[str]:
        first = first_a_side(seed)
        return [first, next(s for s in SIDES if s != first)]

    async def one(group: Group, gid: str) -> GroupVerdict:
        verdict, attached, nibbled = await asyncio.gather(case_verdict(group, gid), attached_asked(group, gid),
                                                          nibble_asked(group, gid))
        if attached:
            verdict.attached_asked, verdict.attached_outcome = attached, dispute_outcome(attached)
        if nibbled:
            verdict.nibble_asked, verdict.nibble_outcome = nibbled, dispute_outcome(nibbled)
        return verdict

    async def nibble_asked(group: Group, gid: str) -> list[Asked]:
        """The boundary question about the group's nibbles, both ways: "belongs" is snewpapers'."""
        if not group.as_nibbles:
            return []
        context = page_context(pair, group.page)
        sides = both_orders(f"{issue_name}/{gid}/nibble")
        answers = await asyncio.gather(*(ask(group_nibble_messages(group, context, s), ATTACHED_SCHEMA) for s in sides))
        return [Asked(s, a, SNEWPAPERS) for s, a in zip(sides, answers)]

    async def attached_asked(group: Group, gid: str) -> list[Asked]:
        """The boundary question about the AS text attached outside the group's component, both ways."""
        if not group.as_attached_outside:
            return []
        context = page_context(pair, group.page)
        sides = both_orders(f"{issue_name}/{gid}/attached")
        answers = await asyncio.gather(*(ask(group_attached_messages(group, context, s), ATTACHED_SCHEMA) for s in sides))
        return [Asked(s, a, AMERICAN_STORIES) for s, a in zip(sides, answers)]

    async def case_verdict(group: Group, gid: str) -> GroupVerdict:
        context = group_context(pair, group)
        if group.case == AS_SPLIT and group.illegible_part:  # AS's pieces do not show the text: ask about ours alone
            answer = await ask(one_component_messages(context, group.ours_text), ONE_VS_MANY_SCHEMA)
            outcome = (NO_VERDICT if answer.parsed is None
                       else SNEWPAPERS if answer.parsed["verdict"] == "one" else ILLEGIBLE_UNSURE)
            return GroupVerdict(gid, group.case, outcome, [Asked(SNEWPAPERS, answer)], group.as_duplicated)
        if group.case in DISPUTED:
            sides = both_orders(f"{issue_name}/{gid}")
            answers = await asyncio.gather(*(ask(dispute_messages(group, context, s), ONE_VS_MANY_SCHEMA) for s in sides))
            asked = [Asked(s, a, one_side(group)) for s, a in zip(sides, answers)]
            return GroupVerdict(gid, group.case, dispute_outcome(asked), asked, group.as_duplicated)
        if group.case == AS_ONLY:
            answer = await ask(as_only_messages(context, group.as_text), AS_ONLY_SCHEMA)
            return GroupVerdict(gid, group.case, as_only_outcome(answer), [Asked(AMERICAN_STORIES, answer)])
        outcome = {MATCH: MATCHED, AS_PARTIAL: PARTLY_CAPTURED, AS_WIDER: NO_VERDICT, OURS_ONLY: NOT_CAPTURED, AS_ONLY_TINY: TINY, AS_ONLY_EMPTY: EMPTY, IN_AS_MERGE: NO_VERDICT,
                   CONTINUED: NO_VERDICT}[group.case]
        return GroupVerdict(gid, group.case, outcome, as_duplicated=group.as_duplicated)

    verdicts = list(await asyncio.gather(*(one(g, gid) for g, gid in zip(seg.groups, group_ids(seg)))))
    position = {id(g): k for k, g in enumerate(seg.groups)}
    for group, verdict in zip(seg.groups, verdicts):
        if group.case == AS_WIDER:  # decided by the question about the text AS runs on into
            verdict.outcome = verdict.attached_outcome or NO_VERDICT
        decider = group.merge if group.case == IN_AS_MERGE else group.continues if group.case == CONTINUED else None
        if decider is not None and id(decider) in position:
            verdict.outcome = verdicts[position[id(decider)]].outcome
    return verdicts


def component_outcome(verdict: GroupVerdict) -> str | None:
    """Where a case anchored on one of our components counts in the scorecard (None for the others,
    and for continued: a cross-page component counts once, by its as_split)."""
    if verdict.case == MATCH:
        return CORRECT
    if verdict.case == OURS_ONLY:
        return NOT_CAPTURED
    if verdict.case == AS_PARTIAL:
        return PARTLY_CAPTURED
    if verdict.case == AS_WIDER:  # what AS runs on into only page furniture: as good as a match
        return {SNEWPAPERS: CORRECT, FURNITURE: CORRECT, AMERICAN_STORIES: AS_WIDER_RIGHT}.get(verdict.outcome, verdict.outcome)
    if verdict.case == AS_SPLIT:
        return {SNEWPAPERS: CORRECT, AMERICAN_STORIES: AS_SPLIT_RIGHT, ILLEGIBLE_UNSURE: UNSTABLE}.get(verdict.outcome,
                                                                                                    verdict.outcome)
    if verdict.case == IN_AS_MERGE:  # the merge's outcome
        return {SNEWPAPERS: CORRECT, AMERICAN_STORIES: AS_MERGE_RIGHT}.get(verdict.outcome, verdict.outcome)
    return None


# How a decision judges each snewpapers component and AS article it is about
JUDGED_RIGHT = "right"
JUDGED_WRONG = "wrong"
# A decided outcome -> (how our side is judged, how American Stories' is); other outcomes judge nothing.
JUDGED = {SNEWPAPERS: (JUDGED_RIGHT, JUDGED_WRONG), AMERICAN_STORIES: (JUDGED_WRONG, JUDGED_RIGHT),
          NEITHER: (JUDGED_WRONG, JUDGED_WRONG)}
CASE_QUESTION = "case"  # a decision's question: the group's case, or ATTACHED (its boundary question)


@dataclass(frozen=True)
class Judgement:
    """One unit judged by one decision: on `side`'s segmentation, a snewpapers component (`unit` is
    its uuid) or an American Stories article (its short name, "#24" or "orphan box 7"), on `page`,
    judged JUDGED_RIGHT or JUDGED_WRONG by the decision (group id, CASE_QUESTION or ATTACHED).
    A cross-page decision judges our component on each of its pages and each AS article on its own."""

    side: str
    page: int
    unit: str
    judged: str
    decision: tuple[str, str]


def judgements(group: Group, verdict: GroupVerdict | None) -> list[Judgement]:
    """The components and AS articles a group's decided questions judge, each once per question:

    - a dispute: our pieces' components and the AS articles of its parts (an as_split decided "one"
      judges our one component right and each of the N AS articles wrong; an as_merge decided
      "several" each of our N components right and the one AS article wrong; "mixed" all wrong);
    - the boundary question: our component, and the AS articles whose text is attached outside it.

    A cross-page as_split judges every AS article on every page (each on its page) and our component
    on each page it is on. Unstable, missing and undecided verdicts judge nothing; matches,
    in_as_merge (its merge judges it), continued (its as_split does), AS-only and ours-only cases
    neither, though a continued group's boundary question does."""
    if verdict is None:
        return []
    out: list[Judgement] = []

    def add(outcome: str | None, question: str, articles: list) -> None:
        if outcome not in JUDGED:
            return
        ours, theirs = JUDGED[outcome]
        decision = (verdict.group_id, question)
        out.extend(Judgement(SNEWPAPERS, page, u, ours, decision)
                   for page, u in dict.fromkeys((p.page, p.uuid) for p in group.pieces))
        out.extend(Judgement(AMERICAN_STORIES, page, a, theirs, decision)
                   for page, a in dict.fromkeys((part.page, part.short_name) for part in articles))

    if group.case in DISPUTED:
        add(verdict.outcome, CASE_QUESTION, group.parts)
    if group.case == AS_PARTIAL:  # AS lacks text for part of our component: its article wrong, ours not judged
        out.extend(Judgement(AMERICAN_STORIES, part.page, part.short_name, JUDGED_WRONG, (verdict.group_id, PARTIAL))
                   for part in dict.fromkeys(group.parts))
    if group.as_attached_outside:
        add(verdict.attached_outcome, ATTACHED, group.as_attached_outside)
    if group.as_nibbles:
        add(verdict.nibble_outcome, NIBBLE, group.as_nibbles)
    return out


def issue_judgements(seg: IssueSegmentation, verdicts: list[GroupVerdict]) -> list[Judgement]:
    """Every group's judgements, in segmentation order; verdicts are matched to groups by id."""
    by_id = {v.group_id: v for v in verdicts}
    return [j for g, gid in zip(seg.groups, group_ids(seg)) for j in judgements(g, by_id.get(gid))]


def judged_counts(found: list[Judgement]) -> dict:
    """{"as_articles": {"wrong": n, "right": n}, "components": {...}}: how many AS articles and
    snewpapers components at least one decision judged wrong, and right; a unit judged both ways
    by different decisions counts in both. AS articles are told apart by page (their ids are a
    page's), our components across the issue (one running over two pages is one component)."""
    def unit(j: Judgement):
        return (j.page, j.unit) if j.side == AMERICAN_STORIES else j.unit

    out = {}
    for side, name in ((AMERICAN_STORIES, "as_articles"), (SNEWPAPERS, "components")):
        mine = [j for j in found if j.side == side]
        out[name] = {k: len({unit(j) for j in mine if j.judged == k}) for k in (JUDGED_WRONG, JUDGED_RIGHT)}
    return out


def as_errors(seg: IssueSegmentation, verdicts: list[GroupVerdict]) -> dict:
    """The American Stories errors the decisions for snewpapers find, with the decisions behind them:
    an as_merge decided for snewpapers is N errors, the N items of ours AS failed to separate
    (ITEMS_MERGED: 7 for seven items AS has as one article); an as_split decided for snewpapers N
    errors, the N AS fragment articles it broke our component into, on all its pages
    (FRAGMENT_ARTICLES). Counted per decision: {"merge_decisions", ITEMS_MERGED, "split_decisions",
    FRAGMENT_ARTICLES}."""
    by_id = {v.group_id: v for v in verdicts}
    out = {"merge_decisions": 0, ITEMS_MERGED: 0, "split_decisions": 0, FRAGMENT_ARTICLES: 0,
           "nibble_decisions": 0, LINES_ELSEWHERE: 0, PARTIAL_ARTICLES: 0, WIDER_ARTICLES: 0, MISSING_COMPONENTS: 0,
           MISSING_PAGES: 0}
    for group, gid in zip(seg.groups, group_ids(seg)):
        verdict = by_id.get(gid)
        if verdict is not None and group.case == AS_SPLIT:
            out[MISSING_PAGES] += len(group.missing_pages)
        if verdict is not None and group.case == OURS_ONLY:
            out[MISSING_COMPONENTS] += 1
        if verdict is not None and group.case == AS_WIDER and verdict.outcome == SNEWPAPERS:
            out[WIDER_ARTICLES] += len(dict.fromkeys(part.short_name for part in group.parts))
        if verdict is not None and group.case == AS_PARTIAL:
            out[PARTIAL_ARTICLES] += len(dict.fromkeys(part.short_name for part in group.parts))
        if verdict is not None and group.as_nibbles and verdict.nibble_outcome == SNEWPAPERS:
            out["nibble_decisions"] += 1
            out[LINES_ELSEWHERE] += len(dict.fromkeys(part.short_name for part in group.as_nibbles))
        if verdict is None or verdict.outcome != SNEWPAPERS:
            continue
        if group.case == AS_MERGE:
            out["merge_decisions"] += 1
            out[ITEMS_MERGED] += len(dict.fromkeys(p.uuid for p in group.pieces))
        elif group.case == AS_SPLIT:
            out["split_decisions"] += 1
            out[FRAGMENT_ARTICLES] += len(dict.fromkeys((part.page, part.short_name) for part in group.parts))
    return out


def scorecard(verdicts: list[GroupVerdict], seg: IssueSegmentation | None = None) -> dict:
    """Counts: our components (one per match, as_split, in_as_merge or ours_only case), the
    many-to-one decisions (as_merge), the AS-only groups, the boundary questions (AS text attached
    outside a component), the American Stories defects found without asking (duplicated text);
    questions, cache hits and cost.

    Decisions are the headline unit. Given the segmentation the verdicts are of, `judged` also counts
    what they judged (`judged_counts`): the AS articles and snewpapers components judged wrong and
    right (an as_split of one component into 15 AS articles, decided "one", is one decision and 15
    AS articles judged wrong); and `as_errors` the AS errors the decisions for snewpapers find
    (`as_errors`: an as_merge of 7 items is one decision and 7 items AS merged, an as_split into 15
    AS articles one decision and 15 fragment articles). Without it both are None."""
    count = lambda outcomes, found: {o: sum(f == o for f in found) for o in outcomes}  # noqa: E731
    answers = [q.answer for v in verdicts for q in v.asked + v.attached_asked + v.nibble_asked]
    return {
        "judged": judged_counts(issue_judgements(seg, verdicts)) if seg is not None else None,
        "as_errors": as_errors(seg, verdicts) if seg is not None else None,
        "components": count(COMPONENT_OUTCOMES, [component_outcome(v) for v in verdicts]),
        "matches": sum(v.case == MATCH for v in verdicts),
        "splits": count(DISPUTE_OUTCOMES, [v.outcome for v in verdicts if v.case == AS_SPLIT]),
        "merges": count(DISPUTE_OUTCOMES, [v.outcome for v in verdicts if v.case == AS_MERGE]),
        "as_only": count(AS_ONLY_OUTCOMES, [v.outcome for v in verdicts if v.case in (AS_ONLY, AS_ONLY_TINY, AS_ONLY_EMPTY)]),
        "wider": count(ATTACHED_OUTCOMES, [v.outcome for v in verdicts if v.case == AS_WIDER]),
        "attached_outside": count(ATTACHED_OUTCOMES, [v.attached_outcome for v in verdicts
                                                      if v.attached_outcome and v.case != AS_WIDER]),
        "nibbles": count(ATTACHED_OUTCOMES, [v.nibble_outcome for v in verdicts if v.nibble_outcome]),
        "as_defects": {AS_DUPLICATED: sum(v.as_duplicated for v in verdicts)},
        # AS-only articles the model called advertisements: AS says it leaves ads out (one per AS article)
        "as_ads_kept": sum(v.case == AS_ONLY and bool(v.asked) and v.asked[0].answer.parsed is not None
                           and v.asked[0].answer.parsed["kind"] == "advertisement" for v in verdicts),
        "questions": len(answers),
        "cached": sum(a.cached for a in answers),
        "cost": round(sum(a.cost or 0.0 for a in answers), 6),
        "uncosted": sum(a.cost is None and not a.error for a in answers),  # answered, but no cost given
        "input_tokens": sum(a.input_tokens or 0 for a in answers),
        "output_tokens": sum(a.output_tokens or 0 for a in answers),
    }


def usage_text(card: dict) -> str:
    """A scorecard's tokens and cost: the cost only as the service reported it (not every
    OpenAI-compatible service does)."""
    tokens = f"{card['input_tokens']:,} input + {card['output_tokens']:,} output tokens"
    if card["uncosted"] and not card["cost"]:
        return f"{tokens}, cost not reported by the service"
    missing = f" ({card['uncosted']} answers without a cost)" if card["uncosted"] else ""
    return f"{tokens}, ${card['cost']:.4f}{missing}"
