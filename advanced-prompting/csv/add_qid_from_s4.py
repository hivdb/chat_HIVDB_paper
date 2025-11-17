import pandas as pd
from pathlib import Path

BASE_CSV = Path('llama-3.1-70B-base.csv')
S4_TABLE = Path('S4Table.xlsx')
OUTPUT_CSV = Path('llama-3.1-70B-base_with_qid_from_s4.csv')


def main():
    # Load
    base = pd.read_csv(BASE_CSV, dtype={'PMID': str})
    s4 = pd.read_excel(S4_TABLE, dtype={'PMID': str})

    # Preserve any existing QID column for comparison
    base_existing_qid = None
    if 'QID' in base.columns:
        base_existing_qid = base['QID']
        base = base.drop(columns=['QID'])

    # Normalize question text for safer joins
    base['question_norm'] = base['question'].str.strip()
    s4['question_norm'] = s4['Question'].str.strip()

    # Merge on PMID + normalized question
    merged = base.merge(
        s4[['PMID', 'QID', 'question_norm']],
        how='left',
        on=['PMID', 'question_norm']
    )

    # Restore original base QID if it existed for reference
    if base_existing_qid is not None:
        merged.insert(1, 'QID_original', base_existing_qid)

    # Place the S4 QID right after PMID
    cols = merged.columns.tolist()
    if 'QID' in cols:
        cols.insert(1, cols.pop(cols.index('QID')))
    merged = merged[cols]

    merged = merged.drop(columns=['question_norm'])
    merged.to_csv(OUTPUT_CSV, index=False)

    missing = merged['QID'].isna().sum()
    total = len(merged)
    msg = f"Saved {OUTPUT_CSV} ({total} rows). Missing QID for {missing} rows."

    if base_existing_qid is not None:
        mismatched = (merged['QID'].notna()) & (merged['QID_original'] != merged['QID'])
        msg += f" Existing QID mismatches: {mismatched.sum()}."
    print(msg)


if __name__ == '__main__':
    main()
