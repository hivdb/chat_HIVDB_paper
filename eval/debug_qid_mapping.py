#!/usr/bin/env python3
"""Debug script to dump QID:Question mappings from all source files and test the remapping logic."""

import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# File paths
S4TABLE = ROOT / "advanced-prompting/csv/S4Table.xlsx"
NEW30_HUMAN = ROOT / "advanced-prompting/test/2025_new30.xlsx"
PV1_QUESTIONS = ROOT / "advanced-prompting/csv/gpt-4o-mini-2024-07-18_PV1_new30.xlsx"
BASE_NEW30 = ROOT / "eval/learning-curve/responses/base_new30_responses.csv"
FT_NEW30 = ROOT / "eval/learning-curve/responses/ft_new30_responses.csv"
LLAMA_70B_QSP = ROOT / "advanced-prompting/csv/llama-3.1-70B-PV1_new30_parsed.csv"
LLAMA_8B_QSP = ROOT / "advanced-prompting/csv/llama-3.1-8B-PV1_new30_parsed.csv"

def normalize_question(text):
    return " ".join(str(text or "").strip().lower().split())

def load_qid_mapping(path, name, pmid="40391923"):
    """Load QID mapping for a specific PMID from a file."""
    if not path.exists():
        print(f"\n{name}: FILE NOT FOUND - {path}")
        return

    loader = pd.read_excel if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv
    df = loader(path)

    # Show available PMIDs if this is S4Table
    if "S4Table" in name:
        print(f"\n{name}: Available PMIDs:")
        unique_pmids = df["PMID"].astype(str).unique()
        print(f"  {unique_pmids[:5]}")
        print(f"  Total unique PMIDs: {len(unique_pmids)}")

    # Convert PMID to string for consistent comparison
    df["PMID"] = df["PMID"].astype(str).str.strip()

    # Filter for the specific PMID
    pmid_df = df[df["PMID"] == pmid].copy()

    if pmid_df.empty:
        print(f"\n{name}: No data for PMID {pmid}")
        return

    # Sort by QID
    pmid_df = pmid_df.sort_values("QID")

    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"File: {path.name}")
    print(f"{'='*80}")
    print(f"{'QID':<5} {'Question':<75}")
    print(f"{'-'*80}")

    for _, row in pmid_df.head(16).iterrows():
        qid = row["QID"]
        question = str(row["Question"])[:70]
        print(f"{qid:<5} {question}")

    return pmid_df[["QID", "Question"]].head(16)

# Load all mappings
print("\nDEBUG: QID:Question Mappings")
print("="*80)

# Use an old PMID from S4Table to see the canonical ordering
s4_map = load_qid_mapping(S4TABLE, "S4Table (CANONICAL) - PMID 18715920", "18715920")

# Now load new30 papers with PMID 40391923
print("\n" + "="*80)
print("NEW 30 PAPERS - PMID 40391923")
print("="*80)
human_map = load_qid_mapping(NEW30_HUMAN, "2025_new30.xlsx (Human Answers)", "40391923")
pv1_map = load_qid_mapping(PV1_QUESTIONS, "PV1 Questions", "40391923")
base_map = load_qid_mapping(BASE_NEW30, "GPT-4o base responses", "40391923")
ft_map = load_qid_mapping(FT_NEW30, "GPT-4o FT responses", "40391923")
llama70b_map = load_qid_mapping(LLAMA_70B_QSP, "Llama 70B QSP responses", "40391923")
llama8b_map = load_qid_mapping(LLAMA_8B_QSP, "Llama 8B QSP responses", "40391923")

# Now check which files have matching QID orderings
print(f"\n{'='*80}")
print("QUESTION TEXT COMPARISON (normalized)")
print(f"{'='*80}")

s4_questions = {}
if s4_map is not None:
    print("\nS4Table questions (normalized):")
    s4_questions = {int(row["QID"]): normalize_question(row["Question"]) for _, row in s4_map.iterrows()}
    for qid in sorted(s4_questions.keys()):
        print(f"  QID {qid}: {s4_questions[qid][:60]}...")

if s4_questions:
    if human_map is not None:
        print("\n2025_new30.xlsx vs S4Table:")
        human_questions = {int(row["QID"]): normalize_question(row["Question"]) for _, row in human_map.iterrows()}
        for qid in sorted(human_questions.keys())[:16]:
            match = "✓" if s4_questions.get(qid) == human_questions.get(qid) else "✗"
            print(f"  QID {qid:2d}: {match} {human_questions[qid][:65]}")

    if base_map is not None:
        print("\nGPT-4o base vs S4Table:")
        base_questions = {int(row["QID"]): normalize_question(row["Question"]) for _, row in base_map.iterrows()}
        for qid in sorted(base_questions.keys())[:16]:
            match = "✓" if s4_questions.get(qid) == base_questions.get(qid) else "✗"
            print(f"  QID {qid:2d}: {match} {base_questions[qid][:65]}")

    if llama70b_map is not None:
        print("\nLlama 70B QSP vs S4Table:")
        llama70b_questions = {int(row["QID"]): normalize_question(row["Question"]) for _, row in llama70b_map.iterrows()}
        for qid in sorted(llama70b_questions.keys())[:16]:
            match = "✓" if s4_questions.get(qid) == llama70b_questions.get(qid) else "✗"
            print(f"  QID {qid:2d}: {match} {llama70b_questions[qid][:65]}")

print("\n" + "="*80)
print("TESTING build_datasets.py REMAPPING LOGIC")
print("="*80)

# Import the remapping function
from eval.build_datasets import get_canonical_qid_mapping, normalize_question

print("\nLoading canonical QID mapping from S4Table...")
canonical_map = get_canonical_qid_mapping()
print(f"Loaded {len(canonical_map)} question mappings")

print("\nCanonical QID mapping (first 16):")
for i, (q_text, qid) in enumerate(sorted(canonical_map.items(), key=lambda x: x[1])[:16], 1):
    print(f"  QID {qid:2d}: {q_text[:65]}")

# Test remapping for each source
def test_remapping(df, source_name):
    """Test if remapping would fix the QIDs."""
    print(f"\n{source_name}:")
    matches = 0
    mismatches = 0
    for _, row in df.iterrows():
        original_qid = int(row["QID"])
        q_norm = normalize_question(row["Question"])
        canonical_qid = canonical_map.get(q_norm)

        if canonical_qid is None:
            print(f"  WARNING: Question not in canonical map: {row['Question'][:50]}")
            continue

        if original_qid == canonical_qid:
            matches += 1
        else:
            mismatches += 1
            if mismatches <= 3:  # Show first 3 mismatches
                print(f"  QID {original_qid} -> {canonical_qid}: {row['Question'][:50]}")

    total = matches + mismatches
    if total > 0:
        accuracy = matches / total * 100
        status = "✓ CORRECT" if accuracy == 100 else f"✗ NEEDS REMAP ({mismatches} mismatches)"
        print(f"  Result: {matches}/{total} correct ({accuracy:.1f}%) - {status}")
    return matches, mismatches

print("\n" + "="*80)
print("REMAPPING TEST RESULTS")
print("="*80)

if human_map is not None:
    test_remapping(human_map, "2025_new30.xlsx (Human Answers)")

if base_map is not None:
    test_remapping(base_map, "GPT-4o base responses")

if ft_map is not None:
    test_remapping(ft_map, "GPT-4o FT responses")

if pv1_map is not None:
    test_remapping(pv1_map, "PV1 Questions")

if llama70b_map is not None:
    test_remapping(llama70b_map, "Llama 70B QSP responses")

if llama8b_map is not None:
    test_remapping(llama8b_map, "Llama 8B QSP responses")

print("\n" + "="*80)
print("TESTING ACTUAL REMAPPING LOGIC FROM build_datasets.py")
print("="*80)

# Test the actual load_model function with remapping
from eval.build_datasets import load_model

def test_actual_remapping(path, column, value_column, remap_flag, source_name, pmid="40391923"):
    """Test the actual load_model function with remapping."""
    print(f"\n{source_name} (remap_by_question={remap_flag}):")

    # Load original file to get question text
    loader = pd.read_excel if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv
    original_df = loader(path)
    original_df["PMID"] = original_df["PMID"].astype(str).str.strip()
    original_pmid = original_df[original_df["PMID"] == pmid].copy().sort_values("QID")

    if original_pmid.empty:
        print(f"  ERROR: No data for PMID {pmid} in original file")
        return

    # Call load_model
    result_df = load_model(path, column, value_column, remap_by_question=remap_flag)

    if result_df is None:
        print("  ERROR: load_model returned None")
        return

    # Filter for test PMID
    result_df["PMID"] = result_df["PMID"].astype(str).str.strip()
    result_pmid = result_df[result_df["PMID"] == pmid].copy().sort_values("QID")

    if result_pmid.empty:
        print(f"  ERROR: No data for PMID {pmid} in result")
        return

    print(f"\n  Original file QIDs (first 5): {list(original_pmid['QID'].astype(int).head(5))}")
    print(f"  Result QIDs (first 5):        {list(result_pmid['QID'].astype(int).head(5))}")

    # Check each question
    print(f"\n  Checking QID mapping (showing first 10):")
    matches = 0
    total = 0
    for i, (_, orig_row) in enumerate(original_pmid.head(10).iterrows()):
        orig_qid = int(orig_row["QID"])
        q_text = normalize_question(orig_row["Question"])
        canonical_qid = canonical_map.get(q_text)

        if canonical_qid is None:
            print(f"    QID {orig_qid}: Question not in S4Table map!")
            continue

        total += 1

        # Find what QID this question got in the result
        # Since load_model doesn't preserve Question, we need to match by position or check all QIDs
        result_qid_list = list(result_pmid["QID"].astype(int))

        # Check if canonical QID exists in results
        if canonical_qid in result_qid_list:
            status = "✓"
            matches += 1
        else:
            status = "✗"

        print(f"    Original QID {orig_qid:2d} -> Should be {canonical_qid:2d} {status} (Question: {orig_row['Question'][:40]}...)")

    if total > 0:
        accuracy = matches / total * 100
        status_str = "✓ CORRECT" if accuracy == 100 else f"✗ FAILED"
        print(f"\n  Result: {matches}/{total} questions have correct canonical QID ({accuracy:.1f}%) - {status_str}")

    return matches, total

print("\nTesting load_model function with remapping...")

# Test GPT-4o base (should be remapped)
test_actual_remapping(
    ROOT / "eval/learning-curve/responses/base_new30_responses.csv",
    "GPT-4o base",
    "Answer",
    True,  # remap_by_question=True
    "GPT-4o base (WITH remapping)"
)

# Test Llama 70B QSP (should NOT be remapped - already correct)
test_actual_remapping(
    ROOT / "advanced-prompting/csv/llama-3.1-70B-PV1_new30_parsed.csv",
    "Llama3.1-70B QSP",
    None,
    False,  # remap_by_question=False (already correct)
    "Llama 70B QSP (NO remapping)"
)

# Test with remapping enabled for Llama (should still be correct)
test_actual_remapping(
    ROOT / "advanced-prompting/csv/llama-3.1-70B-PV1_new30_parsed.csv",
    "Llama3.1-70B QSP",
    None,
    True,  # remap_by_question=True (should still work)
    "Llama 70B QSP (WITH remapping - should be no-op)"
)

print("\n" + "="*80)
print("DONE")
print("="*80)
