"""Fetch both versions of every issue in a benchmark list (spends API tokens; re-runs are free).

For each row of the CSV (default benchmarks/benchmark_set.csv): extract the issue's American
Stories pages, browse the whole issue through the API with boxes, fetch every component's text,
and download the issue's LoC page scans. Everything lands in the usual caches under data/.

    python scripts/fetch_benchmark.py [benchmarks/benchmark_set.csv]
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from snewpapers_tools.american_stories import archive_info
from snewpapers_tools.api import SnewpapersClient, issue_cost
from snewpapers_tools.fetching import fetch_both
from snewpapers_tools.issues import IssueKey
from snewpapers_tools.overlay import issue_jp2s
from snewpapers_tools.paths import REPO_ROOT
from snewpapers_tools.selection import Estimate


def main(csv_path: Path) -> None:
    client = SnewpapersClient()
    start = client.tokens()["balance"]
    print(f"balance {start:,} tokens", flush=True)
    for row in csv.DictReader(csv_path.open()):
        t0 = time.time()
        key = IssueKey(row["issue_uuid"], row["lccn"], row["date"], int(row["edition"]))
        n = int(row["served_components"])
        est = Estimate(
            issue=key,
            newspaper=row["paper"],
            total_components=n,
            cost=issue_cost(n),
            balance=start,
            archive=archive_info(key.year),
            as_pages=[p for p in row.get("as_pages", "").split()] or None,
        )
        try:
            pair = fetch_both(client, est)
            scans = issue_jp2s(pair)
        except Exception as err:  # one issue failing must not stop the rest; re-running retries it
            print(f"{row['decade']} {row['paper']} {key.date}: FAILED {type(err).__name__}: {err}", flush=True)
            continue
        print(f"{row['decade']} {row['paper']} {key.date}: {len(pair.components)} of {pair.total_components} components"
              f" ({len(pair.missing)} not returned), {len(pair.pages)} AS pages, {len(scans)} scans,"
              f" {pair.tokens_spent} tokens, {time.time() - t0:.0f}s", flush=True)
    print(f"spent {start - client.tokens()['balance']:,} tokens in total", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "benchmarks" / "benchmark_set.csv")
