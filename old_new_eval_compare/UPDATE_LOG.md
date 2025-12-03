# Update Log

This project processes `20251203.xlsx` by merging its sheets and producing per-model diffs. Each entry notes what changed in the script.

- Added `merge_excel.py` to merge `Ground truth change`, `Old eval`, and `New eval`, treating all cells as text, joining on `PMID` + `QID`, prefixing non-key columns by sheet (`truth_`, `old_`, `new_`), and writing `20251203_merged.xlsx`.
- Added diff generation that pairs `old_* correct` with `new_* correct`, selects rows where the values differ or `truth_updated human answer?` is set, includes the question, and writes per-model sheets to `20251203_diffs.xlsx`.
- Included each model's answer column (the column before `correct`) in the per-model diff sheets.
- Included `truth_Human-Answer corrected` in all per-model diff sheets.
- Added `categorize_diffs.py` to classify rows in a diffs workbook (default `20251202_diffs.xlsx`) into change categories based on `truth_updated human answer?` and old/new `correct` values, writing categorized sheets plus a summary sheet. Accepts optional input/output paths via CLI.
- Added `summarize_diffs.py` to read a diffs workbook (default `20251202_diffs.xlsx`) and produce a summary-only workbook with counts for updated/no-change by ground truth vs algorithm, plus direction counts (false->true, true->false). Accepts optional input/output paths via CLI.
- Added `tag_diffs.py` to add `change_category` and `change_direction` columns to each sheet of a diffs workbook (default `20251203_diffs.xlsx`) following the ground-truth/algorithm and match/mismatch rules, writing a tagged workbook without a summary sheet. Accepts optional input/output paths via CLI.
- Added `summarize_tagged.py` to read a tagged diffs workbook (default `20251203_diffs_tagged.xlsx`) and produce a summary workbook with counts grouped by `change_category` and `change_direction` per sheet. Accepts optional input/output paths via CLI.

Usage: run `python merge_excel.py` to regenerate both `20251203_merged.xlsx` and `20251203_diffs.xlsx`.
