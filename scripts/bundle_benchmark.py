"""Bundle the benchmark's reports into one folder you can open or host, and zip it.

Run scripts/run_benchmark.py first. This copies each issue's report (index.html and crops/) and its
verdict overlay, writes a start page (index.html: every issue's scorecard numbers, linking its report and
verdict overlay) and a README.txt, then zips the folder. Every link is relative, so the folder works
opened from disk or served as static files.

    python scripts/bundle_benchmark.py [out_dir]   # default data/bundle/
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from snewpapers_tools.overlay import OVERLAY_DIR
from snewpapers_tools.paths import DATA_DIR
from snewpapers_tools.report import REPORTS_DIR, benchmark_index_html

NAME = "snewpapers-vs-american-stories"
TITLE = "SNEWPapers vs American Stories: 18 issues, one per decade"
INTRO = (
    "<p>Eighteen newspaper issues, one per decade from the 1770s to the 1950s (American Stories has nothing for "
    "the 1780s), compared component by component. Each <b>report</b> has a scorecard, then every snewpapers "
    "component with the American Stories boxes inside it: crops of both from the page scan, both texts, and the "
    "verdicts of an LLM shown only the texts. Each <b>overlay</b> draws both segmentations on the page scans, "
    "American Stories on top and snewpapers below, with the verdicts marked.</p>"
    "<p><i>Judged wrong</i>: the American Stories articles and snewpapers components the LLM's decisions found "
    "wrongly segmented. <i>Real words</i>: the share of each side's words (3 or more letters) found in an English "
    "word list, over the same text.</p>"
)
README = """SNEWPapers vs American Stories: 18 newspaper issues, one per decade from the 1770s to the 1950s.

Open index.html in a browser. It lists every issue with its scorecard numbers and links to:
- the issue's report: a scorecard, then every snewpapers component compared with the American
  Stories boxes inside it, with crops of both, the texts and the LLM's verdicts;
- its verdict overlay: both segmentations drawn on the page scans with the verdicts marked.

The overlays and reports were made with the snewpapers public tools (see its README for how to read
them and how to compare issues of your own).
"""


def main(out: Path) -> None:
    rows = json.loads((REPORTS_DIR / "benchmark" / "scorecards.json").read_text())["issues"]
    folder = out / NAME
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "overlays").mkdir(parents=True)
    for row in rows:
        issue = Path(row["report"]).parent.name  # ../{lccn}_{date}_ed-{n}/index.html
        shutil.copytree(REPORTS_DIR / issue, folder / issue)
        overlay = OVERLAY_DIR / f"{issue}_verdicts.webp"  # both layouts, with the verdicts
        shutil.copy2(overlay, folder / "overlays" / overlay.name)
        row["report"] = f"{issue}/index.html"
        row["overlay"] = f"overlays/{issue}_verdicts.webp"
    (folder / "index.html").write_text(benchmark_index_html(rows, title=TITLE, intro=INTRO, usage=False))
    (folder / "README.txt").write_text(README)
    archive = shutil.make_archive(str(out / NAME), "zip", root_dir=out, base_dir=NAME)
    size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    print(f"{folder}: {len(rows)} issues, {size / 1e6:.0f} MB; {archive}: {Path(archive).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR / "bundle")
