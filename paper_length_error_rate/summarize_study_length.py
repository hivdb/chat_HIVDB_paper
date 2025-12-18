import pandas as pd


def main() -> None:
    input_path = "study_length.xlsx"
    output_path = "study_length_summary.xlsx"

    df = pd.read_excel(input_path)

    numeric_cols = df.select_dtypes(include="number").columns
    if numeric_cols.empty:
        raise ValueError("No numeric columns found to summarize.")

    stats = {}
    for col in numeric_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        stats[col] = {
            "mean": series.mean(),
            "median": series.median(),
            "iqr25": q1,
            "iqr75": q3,
            "min": series.min(),
            "max": series.max(),
        }

    summary_df = pd.DataFrame.from_dict(stats, orient="index")
    summary_df.index.name = "column"

    summary_df.to_excel(output_path)
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
