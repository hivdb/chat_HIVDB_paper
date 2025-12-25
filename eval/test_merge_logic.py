#!/usr/bin/env python3
"""Test the merge logic to see if QID remapping breaks alignment."""

import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.build_datasets import (
    get_canonical_qid_mapping,
    normalize_question,
    normalize_ids,
    load_model,
    merge_column,
)

print("="*80)
print("TESTING MERGE LOGIC")
print("="*80)

# Simulate what build_new30_human_rows does
print("\n1. Loading human answers and remapping to canonical QIDs...")
human_df = pd.read_excel(ROOT / "advanced-prompting/test/2025_new30.xlsx")
human_df = human_df.rename(columns={"Human answer": "Human Answer"})
human_df = normalize_ids(human_df)

canonical_map = get_canonical_qid_mapping()

# Remap human answer QIDs
human_remapped = []
for _, row in human_df.iterrows():
    q_norm = normalize_question(row["Question"])
    canonical_qid = canonical_map.get(q_norm)
    if canonical_qid:
        human_remapped.append({
            "PMID": str(row["PMID"]),
            "QID": canonical_qid,
            "Question": str(row["Question"]),
            "Human Answer": str(row.get("Human Answer", "")),
        })

human_remapped_df = pd.DataFrame(human_remapped)
human_remapped_df = normalize_ids(human_remapped_df)

print(f"   Human answers remapped: {len(human_remapped_df)} rows")

# Filter for test PMID
test_pmid = "40391923"
human_test = human_remapped_df[human_remapped_df["PMID"] == test_pmid].copy()
human_test = human_test.sort_values("QID")

print(f"\n   Human answers for PMID {test_pmid}:")
print(f"   QIDs: {list(human_test['QID'].astype(int).head(5))}")
for _, row in human_test.head(3).iterrows():
    print(f"     QID {int(row['QID']):2d}: {row['Question'][:50]}...")

# 2. Load GPT-4o base model responses WITH remapping
print("\n2. Loading GPT-4o base responses WITH remapping...")
gpt_base_df = load_model(
    ROOT / "eval/learning-curve/responses/base_new30_responses.csv",
    "GPT-4o base",
    "Answer",
    remap_by_question=True
)

print(f"   Model responses loaded: {len(gpt_base_df)} rows")

gpt_test = gpt_base_df[gpt_base_df["PMID"] == test_pmid].copy()
gpt_test = gpt_test.sort_values("QID")

print(f"\n   Model responses for PMID {test_pmid}:")
print(f"   QIDs: {list(gpt_test['QID'].astype(int).head(5))}")

# 3. Try to merge
print("\n3. Attempting merge on [PMID, QID]...")
merged = human_test.merge(
    gpt_test,
    on=["PMID", "QID"],
    how="left",
    suffixes=("", "_model")
)

print(f"   Merged rows: {len(merged)}")
print(f"\n   BEFORE FIX:")
print(f"   Merged QID column dtype: {merged['QID'].dtype}")
print(f"   All QIDs (sorted as current dtype): {sorted(merged['QID'].unique())}")

# Apply the fix: Convert QID to int
print(f"\n   APPLYING FIX: Converting QID to int...")
merged["QID"] = merged["QID"].astype(int)

print(f"\n   AFTER FIX:")
print(f"   Merged QID column dtype: {merged['QID'].dtype}")
print(f"   All QIDs (sorted numerically): {sorted(merged['QID'].unique())}")

print(f"\n   Sample of merged data (first 5 rows - should now be QID 1, 2, 3, 4, 5):")
print(f"   {'QID':<5} {'Question':<50} {'Has Answer'}")
print(f"   {'-'*70}")

# Sort by QID numerically and show first 5
merged_sorted = merged.sort_values("QID")
for _, row in merged_sorted.head(5).iterrows():
    has_answer = "✓" if pd.notna(row.get("GPT-4o base")) and row.get("GPT-4o base") != "" else "✗"
    print(f"   {int(row['QID']):<5} {row['Question'][:47]:<50} {has_answer}")

# Check alignment
print(f"\n4. Checking alignment after fix...")
aligned = 0
total = 0
misaligned_qids = []
for _, row in merged_sorted.head(16).iterrows():
    qid = int(row['QID'])
    total += 1
    has_answer = pd.notna(row.get("GPT-4o base")) and row.get("GPT-4o base") != ""
    if has_answer:
        aligned += 1
    else:
        misaligned_qids.append(qid)
        if len(misaligned_qids) <= 3:
            print(f"   QID {qid:2d}: No answer found - {row['Question'][:40]}...")

print(f"\n   Result: {aligned}/{total} rows have answers ({aligned/total*100:.1f}%)")

if aligned == total:
    print("   ✓ After converting QID to int, everything aligns correctly!")
    print("   ✓ The fix solves the sorting issue!")
else:
    print("   ✗ Still broken even after fix!")

# Verify the questions match S4Table canonical order
print(f"\n5. Verifying questions match S4Table canonical order...")
canonical_map = get_canonical_qid_mapping()
canonical_questions_ordered = sorted(canonical_map.items(), key=lambda x: x[1])

print(f"\n   First 5 questions in S4Table order:")
for q_text, qid in canonical_questions_ordered[:5]:
    print(f"     QID {qid:2d}: {q_text[:50]}...")

print(f"\n   First 5 questions in merged output (after QID int conversion):")
for _, row in merged_sorted.head(5).iterrows():
    q_text = normalize_question(row['Question'])
    print(f"     QID {int(row['QID']):2d}: {q_text[:50]}...")

# Check if they match
matches = 0
for i, (_, row) in enumerate(merged_sorted.head(5).iterrows()):
    expected_qid = i + 1
    actual_qid = int(row['QID'])
    if expected_qid == actual_qid:
        matches += 1

if matches == 5:
    print(f"\n   ✓ Perfect! QIDs are now 1, 2, 3, 4, 5 in the correct order!")
else:
    print(f"\n   ✗ QIDs still not in correct sequential order")

print("\n" + "="*80)
print("DONE")
print("="*80)
