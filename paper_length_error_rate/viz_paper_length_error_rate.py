import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_correct_figures(df: pd.DataFrame, output_dir: str) -> None:
    correct_cols = [col for col in df.columns if "Correct" in col]
    if not correct_cols:
        raise ValueError("No columns containing 'Correct' found.")

    df["content_length"] = pd.to_numeric(df["content_length"], errors="coerce")

    rng = np.random.default_rng(0)

    for idx, col in enumerate(correct_cols, start=1):
        y = pd.to_numeric(df[col], errors="coerce")
        x = df["content_length"]
        x_range = x.max() - x.min()
        x_jitter = rng.normal(0.0, 0.005 * x_range, size=len(x)) if x_range else 0.0

        plt.figure(figsize=(7, 5))
        low_mask = y <= 10
        colors = np.where(low_mask, "#d62728", "#1f77b4")
        plt.scatter(
            x + x_jitter,
            y,
            s=12,
            alpha=0.6,
            edgecolors="none",
            c=colors,
        )
        plt.title(f"{idx}. {col}")
        plt.xlabel("Paper length")
        plt.ylabel("Correct")
        if col != "All Correct":
            plt.yticks([0, 4, 8, 10, 12, 16])
            plt.axhline(10, color="#333333", linewidth=1.0, alpha=0.7)
        else:
            plt.yticks([100, 120, 140, 160, 180, 200])
            if y.notna().any():
                plt.axhline(y.max(), color="#333333", linewidth=1.0, alpha=0.7)
        plt.tight_layout()

        safe_name = f"{idx:02d}_{col}".replace(os.sep, "_")
        filename = os.path.join(output_dir, f"{safe_name}.png")
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"Wrote {filename}")


def main() -> None:
    df = pd.read_excel("paper_length_error_rate.xlsx")

    pmid_col = "PMID"
    if pmid_col not in df.columns:
        raise ValueError("No PMID column found.")

    pmid_str = df[pmid_col].astype(str).str.strip()
    is_2025 = pmid_str.str.startswith("4", na=False)

    output_dirs = {
        "All": df,
        "2025": df[is_2025],
        "before 2025": df[~is_2025],
    }

    for output_dir, subset in output_dirs.items():
        os.makedirs(output_dir, exist_ok=True)
        plot_correct_figures(subset.copy(), output_dir)


if __name__ == "__main__":
    main()
