import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def plot_correct_figures(df: pd.DataFrame, output_dir: str) -> None:
    correct_cols = [col for col in df.columns if "Correct" in col]
    if not correct_cols:
        raise ValueError("No columns containing 'Correct' found.")

    df["content_length"] = pd.to_numeric(df["content_length"], errors="coerce")

    rng = np.random.default_rng(0)
    for idx, col in enumerate(correct_cols, start=1):
        if "FT+QSP" in col:
            continue
        y = pd.to_numeric(df[col], errors="coerce")
        if col != "All Correct":
            y = y / 16
        x = df["content_length"]
        x_range = x.max() - x.min()
        x_jitter = rng.normal(0.0, 0.005 * x_range, size=len(x)) if x_range else 0.0

        fig, ax = plt.subplots(figsize=(7, 5))
        low_mask = y <= (10 / 16) if col != "All Correct" else y <= 10
        colors = np.where(low_mask, "#d62728", "#1f77b4")
        ax.set_title(f"{idx}. {col}")
        ax.set_xlabel("Paper length")
        ax.set_ylabel("Percent correct")

        valid_mask = x.notna() & y.notna()
        if valid_mask.sum() >= 2:
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            reg = stats.linregress(x_valid, y_valid)
            x_line = np.linspace(x_valid.min(), x_valid.max(), 200)
            y_line = reg.slope * x_line + reg.intercept
            r_squared = reg.rvalue**2
            stats_label = (
                f"slope={reg.slope:.3g}, R^2={r_squared:.3f}, p={reg.pvalue:.3g}"
            )
            if len(x_valid) >= 3:
                x_mean = x_valid.mean()
                sxx = ((x_valid - x_mean) ** 2).sum()
                if sxx > 0:
                    residuals = y_valid - (reg.slope * x_valid + reg.intercept)
                    s_err = np.sqrt((residuals**2).sum() / (len(x_valid) - 2))
                    se_fit = np.sqrt(
                        (1 / len(x_valid)) + ((x_line - x_mean) ** 2) / sxx
                    )
                    t_crit = stats.t.ppf(0.975, df=len(x_valid) - 2)
                    ci = t_crit * s_err * se_fit
                    ax.fill_between(
                        x_line,
                        y_line - ci,
                        y_line + ci,
                        color="#7f7f7f",
                        alpha=0.22,
                        linewidth=0,
                    )
            ax.plot(x_line, y_line, color="#2ca02c", linewidth=2, label=stats_label)
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                frameon=False,
            )
        if col != "All Correct":
            ax.set_ylim(bottom=0.5)
            ax.set_yticks([0.5, 0.75, 1.0])
            ax.set_yticklabels(["50%", "75%", "100%"])
        else:
            ax.set_ylim(bottom=100)
            ax.set_yticks([100, 150, 200])
        valid_x = x[x.notna()]
        if not valid_x.empty:
            y_min, y_max = ax.get_ylim()
            span = y_max - y_min
            rug_offset = 0.01 * span if span > 0 else 0.0
            rug_height = 0.12 * span if span > 0 else 0.0
            ax.vlines(
                valid_x,
                y_min + rug_offset,
                y_min + rug_offset + rug_height,
                color="black",
                alpha=0.35,
                linewidth=0.6,
            )
        fig.tight_layout(rect=(0, 0.08, 1, 1))

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
