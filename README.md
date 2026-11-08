# SNEWPapers vs American Stories

A Jupyter notebook that uses the [snewpapers](https://snewpapers.com) public API to find a
newspaper article, rebuild the whole issue it appeared in, and compare that issue with the
same one in [American Stories](https://huggingface.co/datasets/dell-research-harvard/AmericanStories)
(Dell et al., Harvard). Both are built from Library of Congress Chronicling America scans with
different layout detection and OCR, so the comparison shows where they agree and differ.

## Setup

You need [pyenv](https://github.com/pyenv/pyenv) with the
[pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) plugin.

```sh
git clone https://github.com/bshinnebarger/snewpapers-american-stories-comparison.git
cd snewpapers-american-stories-comparison

# Python and a virtualenv for this repo; .python-version activates it whenever you cd here
pyenv install 3.12.12
pyenv virtualenv 3.12.12 snewpapers-public-tools
pyenv local snewpapers-public-tools
which python   # should be under ~/.pyenv/versions/snewpapers-public-tools

# pinned dependencies plus this repo's helpers (editable).
# `python -m` makes sure pip and jupyter run from this env, not a stale user install.
python -m pip install -r requirements.txt

# register the env as a Jupyter kernel; the notebook selects it by name
python -m ipykernel install --user --name snewpapers-public-tools --display-name "snewpapers-public-tools"

# your API key
cp .env.example .env   # then put your key after SNEWPAPERS_API_KEY=

python -m jupyter lab
```

Open `notebooks/compare.ipynb`. If Jupyter does not pick the kernel itself, choose
"snewpapers-public-tools" in the top right.

### Getting an API key

On [snewpapers.com](https://snewpapers.com), click the gear icon (top right) for Settings,
then API -> Research API key -> Generate. The key is shown only once, so save it somewhere
safe; generating a new one revokes the old one. Keys are valid for a year.

The key is a long token (about 440 characters) starting with `eyJ`. Paste all of it after
`SNEWPAPERS_API_KEY=` in `.env`, with no quotes. A partial key gets `401 malformed token`.

## Using the notebook

1. Run the first cells: they show your token balance.
2. Fill in the form: a year range (1770-1963, where both sources have issues), some keywords (any
   of them may match; articles matching more rank higher) and, optionally, a topic: one sentence
   matched by meaning, which widens the search to articles about it that use other words (news,
   editorials, letters, literary and other narrative pieces only). **Search** shows the top hits
   and picks one at random, with a crop of the article from the page scan.
3. Under the pick you see its issue (newspaper, date, edition), how many components it has, what
   comparing it costs in tokens, and how big the American Stories download for that year is.
   **Pick another** until the issue is small enough; **Compare this issue** confirms it.
4. **Fetch both versions**: downloads the year's American Stories archive (once), copies out the
   issue's pages, then browses the issue through the API and fetches every component's text. It
   stops before spending anything if American Stories lacks the issue.
5. **Page layouts**: both segmentations drawn on the Library of Congress scans, American Stories
   on top, snewpapers below.
6. **Compare the segmentations**: free and local; pairs every snewpapers component with the
   American Stories boxes inside it (see [How to read the results](#how-to-read-the-results)).
7. **Ask an LLM where they differ** (optional, needs an LLM service; see
   [Adjudication](#adjudication-optional)).
8. **The report and the verdict overlay**: writes the report under `data/reports/` and draws the
   verdicts on the scans (`data/overlays/`).

## What a comparison costs

The API is paid in tokens; `GET /account/tokens` (free) shows your balance.

| Step | Tokens |
|---|---|
| Search (one page of 25 hits) | 1 |
| The picked article's details (gives the issue's edition) | 1 |
| Browse the issue, per 25 components | 1 |
| Full text, per component | 3 (first read only) |

A whole issue of `n` components costs `ceil(n / 25) + 3n` on top of the search and the
article's details, so a 40-component issue is 2 + 120 = 122 tokens. The notebook shows this
before fetching anything and prefers small issues.

Searches, issue browses and article details are charged on every call; only full text is free
to re-read. The client therefore caches every response in `data/snewpapers_api/`, so re-running
a cell costs nothing. Cached answers do not change: if an issue is re-extracted later, pass
`refresh=True` or delete `data/snewpapers_api/` to see the new version.

American Stories is free but large: the notebook downloads one year archive (about 1 GB for
1850, up to 7.6 GB for 1910) into `data/american_stories/` and reuses it for every issue of
that year. The page scans come from the Library of Congress (about 10 MB a page, into `data/loc/`).

The LLM questions are paid to whichever service you use. An issue asks a few dozen; a question
averages about 1,000 input tokens (a long story up to about 10,000) and a short answer, plus what a
reasoning model spends thinking. With DeepSeek V4.1 Flash on DeepInfra an issue cost well under a cent.

## How to read the results

Every comparison is anchored on a snewpapers component: the comparison takes our segmentation as
the claim to test and looks at the American Stories (AS) boxes inside each component. The two are
put on the same page scan, and AS boxes that straddle several of our components (often whole
columns) are sliced along our boundaries and their text cut to match. Each pairing is a *group*
with a case:

| Case | Meaning |
|---|---|
| 1:1 match (=) | the AS text inside our component is one AS article's, and the two cover each other (90% or more both ways) |
| 1 ours : N AS | the AS text inside our component belongs to several AS articles: one item or several? |
| N ours : 1 AS | two or more of our components lie inside one AS article: one item or several? |
| in N ours : 1 AS | one of the components inside such an article; decided by that question |
| continued | a later page of a component of ours running over pages; compared and decided once, on its first page |
| 1:1, AS article wider | one AS article holds our component but runs on beyond it into other text; that text is asked about |
| 1:1, AS partial | one AS article, but AS has text for under 90% of our component (the rest in no AS box, or marked illegible); an AS error, not asked |
| ours only | AS has no text where our component is (no box, or a box marked illegible); an AS error, not asked |
| AS only | AS text inside none of our components: left out on purpose (ads, terms, page furniture) or missed? |
| AS only, tiny / no text | too short to ask about, or an AS box with no text at all; counted for neither side |

Besides its case, a group can carry a boundary question: AS text *attached outside* our component
(text of the same AS article printed beyond it), or a *line elsewhere* (a line or two of our
component that AS put in another article).

### The report

`data/reports/{lccn}_{date}_ed-{n}/index.html`. At the top, the scorecard:

- **Where the two segmentations agree or differ**: per kind of question, how many there were and
  how they were decided (snewpapers right, AS right, neither, unsure, page furniture). "Unsure"
  is an unstable verdict (the answer changed with the order the two claims were stated in);
  furniture is a line either order called page furniture. Neither counts for anyone.
- **Where only one side has text**: our components AS has no text for (or text for only part of),
  and AS text in none of our components, by what the model said it is.
- **What the decisions judged**: AS articles and snewpapers components judged right or wrong. A
  decision that our one component is one item judges every AS fragment inside it wrong; one that
  AS's one article holds several items judges each of our components inside it right.
- **American Stories errors found**: the same decisions counted as AS errors, by kind.
- **Text quality**: the share of words (3+ letters) found in an English word list
  ([wordfreq](https://github.com/rspeer/wordfreq)), for both sides' text of the same components,
  and how much of the area AS boxed it marked illegible and did not read.

"How this was scored" (collapsed) repeats the rules. Below it, every group page by page, collapsed:
its case badge, the verdict, both sides' crops from the scan, the texts with their differences
highlighted, and the model's answers. Links in the scorecard open the groups they count.

### The overlays

`data/overlays/{issue}_compare.webp` draws both layouts: AS on top (boxes coloured by class,
dashed line through each article in AS's reading order), snewpapers below (components coloured by
label family, dashed line in reading order, a blue arrow where a component continues on another
page).

`data/overlays/{issue}_verdicts.webp` adds the verdicts, each with its group's id as in the report:
"=" on matches; a green check on the side judged right and a red cross on the other (an amber "?"
on both when unsure); "N items in 1" on an AS article holding N of our components; tags for the
cases not asked (SNEW ONLY, AS PARTIAL, AS ILLEGIBLE, AS ONLY, NOT SERVED, NO TEXT, DUP for text AS
has twice); a purple line from text AS attached outside our component to the AS article it belongs
to. Above each page and row, a tally of its marks. The legend and scorecard are in the header.

### What to keep in mind

- The model judges text only, never the scan. When American Stories puts a line or two of our
  component in another article, the model is shown American Stories' OCR of that line and of the
  rest; where that OCR is badly garbled (a headline, say), it can misjudge the line.
- snewpapers is not perfect either. It sometimes attaches a fragment of a page-wide banner
  headline to the component printed below it; one of the 16 snewpapers components judged wrong
  in the benchmark (the 1910s Tonopah Daily Bonanza, page 1) is such a case.
- Some calls are conventions: dispatches under one heading, or a column of one-line personal
  notes, can fairly be read as one item or several. The prompts are kept general rather than tuned
  to snewpapers' conventions, so some AS wins are of this kind.
- Pages where American Stories segmented a different digitisation of the page than the one the
  API's boxes are on are left out (greyed in the overlays).

## Adjudication (optional)

The model gets only the texts, never the scans, and is never told which system is which. Prompts
are in `snewpapers_tools/prompts/`.

- **One or several** (1 ours : N AS, N ours : 1 AS): the passage is shown as numbered pieces, each
  headed by its first and last words; one system believes it is one item, the other several. The
  answer is "one", "several" or "mixed".
- **Boundary** (AS text attached outside our component, a line elsewhere, an AS article running on
  into other text): the component's text and the other text; one system believes it belongs, the
  other not. The answer is "belongs", "separate" or "furniture".
- **AS only**: is the text an advertisement, subscription terms, page furniture (all left out by
  snewpapers on purpose) or other content?
- **One item** (a 1 ours : N AS where AS marked 10% or more of our component illegible, so its
  pieces cannot show the whole text): is our component's text one item? "One" means snewpapers is
  right; anything else is unsure, never AS.

The first two are asked twice, with the two beliefs in opposite orders; a verdict that changes with
the order is unstable and counts for neither side. The prompts tell the model that segmentation
systems leave out page furniture (titles, datelines, section headings, jump lines) and to judge by
the overall content, not tiny fragments.

It needs an OpenAI-compatible service and a model of your choice, set in `.env`: `LLM_API_KEY`,
`LLM_BASE_URL` and `LLM_MODEL`. The model must support structured output (`response_format` with a
strict JSON schema): an answer that does not fit the schema is recorded as "no verdict", never
guessed, so a model without it leaves most questions undecided. The published benchmarks used
[DeepInfra](https://deepinfra.com) (`https://api.deepinfra.com/v1/openai`) with
`deepseek-ai/DeepSeek-V4.1-Flash-1017`; another model may judge some questions differently.

The scorecard shows the tokens used and, where the service reports it (DeepInfra and OpenRouter
do), the billed cost. Answers are cached in `data/adjudications/`, keyed on the service, the model
and the question, so re-running a report costs nothing and another model gets its own answers.

## Benchmarks

`benchmarks/benchmark_set.csv` lists the 18 hand-picked issues behind the published results, one
per decade from the 1770s to the 1950s (American Stories has nothing for the 1780s). To judge the
comparison on issues you choose yourself, use the notebook.

```sh
python scripts/fetch_benchmark.py   # fetches both versions (spends tokens; re-runs free)
python scripts/run_benchmark.py     # compares, asks, reports; fails if fetching charged tokens
```

Each issue's report goes to `data/reports/{issue}/`, with an index of all of them and their
scorecards in `data/reports/benchmark/`.

## Tests

```sh
python -m pytest
```

## Data and credits

American Stories: Dell et al., "American Stories: A Large-Scale Structured Text Dataset of
Historical U.S. Newspapers" (2023), arXiv:2308.12477, CC BY 4.0. Page scans: Library of
Congress, Chronicling America.
