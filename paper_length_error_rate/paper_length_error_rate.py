import pandas as pd


def main() -> None:
    study_df = pd.read_excel("study_length.xlsx")
    eval_df = pd.read_excel("detailed_evaluation_full150.xlsx")

    correct_cols = [col for col in eval_df.columns if col.endswith("Correct")]
    if not correct_cols:
        raise ValueError("No columns ending with 'Correct' found.")

    eval_df[correct_cols] = eval_df[correct_cols].apply(
        lambda s: pd.to_numeric(s, errors="coerce").fillna(0)
    )

    grouped = eval_df.groupby("PMID", as_index=False)[correct_cols].sum()
    grouped["All correct"] = grouped[correct_cols].sum(axis=1)

    merged = pd.merge(
        study_df[["PMID", "content_length"]],
        grouped,
        on="PMID",
        how="left",
    )

    output_path = "paper_length_error_rate.xlsx"
    merged.to_excel(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
