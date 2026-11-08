"""Segment, adjudicate and report every issue in a benchmark list, then write the benchmark index.

Run scripts/fetch_benchmark.py first: this builds each issue from the caches under data/ and stops
with an error if fetching it charged any API tokens. The LLM questions are asked (and cached under
data/adjudications/); each issue's report goes to data/reports/{issue}/, the index and every
issue's scorecard to data/reports/benchmark/.

    python scripts/run_benchmark.py [benchmarks/benchmark_set.csv]
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

from snewpapers_tools.adjudicate import Adjudicator, adjudicate
from snewpapers_tools.american_stories import archive_info
from snewpapers_tools.api import SnewpapersClient, issue_cost
from snewpapers_tools.fetching import fetch_both
from snewpapers_tools.issues import IssueKey
from snewpapers_tools.overlay import issue_scan_sizes
from snewpapers_tools.paths import REPO_ROOT
from snewpapers_tools.report import (
    REPORTS_DIR, benchmark_index_html, benchmark_row, benchmark_text, benchmark_totals, segmentation_report,
)
from snewpapers_tools.segmentation import segment_issue
from snewpapers_tools.selection import Estimate
from snewpapers_tools.verdict_overlay import verdict_overlay

OUT_DIR = REPORTS_DIR / "benchmark"


async def _adjudicate(seg, pair):
    """The issue's verdicts, closing the LLM client in the same event loop."""
    adjudicator = Adjudicator()
    try:
        return await adjudicate(seg, pair, adjudicator=adjudicator, concurrency=8)
    finally:
        await adjudicator.aclose()


def main(csv_path: Path) -> None:
    out_dir = OUT_DIR if csv_path.stem == "benchmark_set" else REPORTS_DIR / csv_path.stem
    client = SnewpapersClient()
    start = client.tokens()["balance"]
    print(f"balance before: {start:,} tokens", flush=True)
    rows = []
    for row in csv.DictReader(csv_path.open()):
        t0 = time.time()
        key = IssueKey(row["issue_uuid"], row["lccn"], row["date"], int(row["edition"]))
        n = int(row["served_components"])
        before = client.tokens()["balance"]
        est = Estimate(issue=key, newspaper=row["paper"], total_components=n, cost=issue_cost(n), balance=before,
                       archive=archive_info(key.year), as_pages=row.get("as_pages", "").split() or None)
        try:
            pair = fetch_both(client, est)
        finally:  # the guard: everything should come from the caches
            charged = before - client.tokens()["balance"]
            if charged:
                sys.exit(f"{row['decade']} {row['paper']} {key.date}: fetching charged {charged:,} tokens; stopping"
                         " (run scripts/fetch_benchmark.py first)")
        seg = segment_issue(pair, issue_scan_sizes(pair))
        verdicts = asyncio.run(_adjudicate(seg, pair))
        # published results: no LLM tokens or cost (cached answers cost nothing; the cost depends on the model)
        _html, path = segmentation_report(pair, seg, verdicts=verdicts, usage=False)
        verdict_overlay(pair, seg, verdicts, usage=False)  # data/overlays/{issue}_verdicts.webp
        rows.append(benchmark_row(row["decade"], row["paper"], key.date, seg, verdicts, os.path.relpath(path, out_dir)))
        print(f"{row['decade']} {row['paper']} {key.date}: {len(seg.groups)} groups, ${rows[-1]['cost']:.4f},"
              f" {time.time() - t0:.0f}s", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(benchmark_index_html(rows, usage=False))
    (out_dir / "scorecards.json").write_text(json.dumps({"issues": rows, "totals": benchmark_totals(rows)}, indent=1))
    print(benchmark_text(rows))
    print(f"balance after: {client.tokens()['balance']:,} tokens; wrote {out_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "benchmarks" / "benchmark_set.csv")
