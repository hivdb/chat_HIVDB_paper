import argparse
from pathlib import Path

import pandas as pd

QUESTION_TO_QID = {
    "Does the paper report HIV sequences from patient samples?": 1,
    "Does the paper report in vitro drug susceptibility data?": 2,
    "Were sequences from the paper made publicly available?": 3,
    "What were the GenBank accession numbers for sequenced HIV isolates?": 4,
    "How many individuals had samples obtained for HIV sequencing?": 5,
    "From which countries were the sequenced samples obtained?": 6,
    "From what years were the sequenced samples obtained?": 7,
    "Were samples cloned prior to sequencing?": 8,
    "Which HIV genes were reported to have been sequenced?": 9,
    "What method was used for sequencing?": 10,
    "What type of samples were sequenced?": 11,
    "Were any sequences obtained from individuals with virological failure on a treatment regimen?": 12,
    "Were the patients in the study in a clinical trial?": 13,
    "Does the paper report HIV sequences from individuals who had previously received ARV drugs?": 14,
    "Which drug classes were received by individuals in the study before sample sequencing?": 15,
    "Which drugs were received by individuals in the study before sample sequencing?": 16,
}

DEFAULT_INPUT = Path("csv/gpt-4o-base-new_30_parsed.csv")


def renumber_qid(input_path: Path, output_path: Path):
    df = pd.read_csv(input_path)

    if "Question" in df.columns:
        question_col = "Question"
    elif "question" in df.columns:
        question_col = "question"
    else:
        raise ValueError("Could not find a Question column in the CSV.")

    if "PMID" in df.columns:
        pmid_col = "PMID"
    elif "pmid" in df.columns:
        pmid_col = "pmid"
    else:
        raise ValueError("Could not find a PMID column in the CSV.")

    df["QID"] = df[question_col].str.strip().map(QUESTION_TO_QID)

    missing = df["QID"].isna()
    if missing.any():
        missing_questions = (
            df.loc[missing, question_col].drop_duplicates().to_list()
        )
        raise ValueError(
            "Some questions were not recognized. Please check spelling and capitalization: "
            f"{missing_questions}"
        )

    df = df.sort_values([pmid_col, "QID"])
    df.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with {len(df)} rows.")


def main():
    parser = argparse.ArgumentParser(
        description="Renumber QID based on exact Question text."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to input CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write updated CSV. Defaults to <input> with _renumbered suffix.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file instead of writing a new one.",
    )
    args = parser.parse_args()

    if args.inplace and args.output:
        parser.error("Use either --inplace or --output, not both.")

    output_path = args.output
    if output_path is None:
        output_path = (
            args.input
            if args.inplace
            else args.input.with_name(f"{args.input.stem}_renumbered{args.input.suffix}")
        )

    renumber_qid(args.input, output_path)


if __name__ == "__main__":
    main()
