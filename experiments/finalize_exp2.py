# SPDX-License-Identifier: Apache-2.0
"""Inject the exp-2 accuracy tables from summary_2.md into EXP2_OPENBOOKQA.md.

Replaces the ``<!-- RESULTS_PLACEHOLDER -->`` marker (or a previously injected
block delimited by the AUTO markers) with the body of summary_2.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "EXP2_OPENBOOKQA.md"
SUMMARY = REPO_ROOT / "experiments" / "results" / "exp2" / "summary_2.md"

BEGIN = "<!-- AUTO-RESULTS:BEGIN -->"
END = "<!-- AUTO-RESULTS:END -->"
PLACEHOLDER = "<!-- RESULTS_PLACEHOLDER -->"


def main() -> None:
    summary = SUMMARY.read_text(encoding="utf-8").strip()
    # Drop the summary's own top-level title; keep the tables/notes.
    lines = summary.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    body = "\n".join(lines).strip()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    block = (f"{BEGIN}\n_Generated {stamp} from "
             f"`experiments/results/exp2/summary_2.md`._\n\n{body}\n{END}")

    doc = DOC.read_text(encoding="utf-8")
    if BEGIN in doc and END in doc:
        pre = doc.split(BEGIN)[0]
        post = doc.split(END, 1)[1]
        doc = pre + block + post
    elif PLACEHOLDER in doc:
        doc = doc.replace(PLACEHOLDER, block)
    else:
        doc = doc.rstrip() + "\n\n" + block + "\n"
    DOC.write_text(doc, encoding="utf-8")
    print(f"Injected results into {DOC}")


if __name__ == "__main__":
    main()
