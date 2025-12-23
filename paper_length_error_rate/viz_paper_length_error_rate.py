import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from scipy import stats


def plot_correct_figures(df: pd.DataFrame, output_dir: str) -> None:
    correct_cols = [col for col in df.columns if "Correct" in col]
    if not correct_cols:
        raise ValueError("No columns containing 'Correct' found.")

    title_fontsize = 16
    axis_label_fontsize = 14

    df["content_length"] = pd.to_numeric(df["content_length"], errors="coerce")

    rng = np.random.default_rng(0)

    def trim_correct_label(label: str) -> str:
        if label.endswith("Correct"):
            return label[: -len("Correct")].rstrip()
        return label

    def set_plot_title(ax: plt.Axes, title: str) -> None:
        ax.set_title(title, y=0.92, fontweight="bold", fontsize=title_fontsize)

    for idx, col in enumerate(correct_cols, start=1):
        if "FT+QSP" in col or col == "All Correct":
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
        set_plot_title(ax, trim_correct_label(col))
        ax.set_xlabel("# Characters", fontweight="bold", fontsize=axis_label_fontsize)
        ax.set_ylabel("Accuracy", fontweight="bold", fontsize=axis_label_fontsize)

        valid_mask = x.notna() & y.notna()
        if valid_mask.sum() >= 2:
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            reg = stats.linregress(x_valid, y_valid)
            x_line = np.linspace(x_valid.min(), x_valid.max(), 200)
            y_line = reg.slope * x_line + reg.intercept
            r_squared = reg.rvalue**2
            stats_label = f"$\\mathbf{{R}}^2$={r_squared:.1g}, p={reg.pvalue:.1g}"
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
            ax.plot(x_line, y_line, color="#2ca02c", linewidth=2)
            set_plot_title(ax, f"{trim_correct_label(col)} ({stats_label})")
        if col != "All Correct":
            ax.set_ylim(bottom=0.51)
            ax.set_yticks([0.51, 0.75, 1.0])
            ax.set_yticklabels(["51%", "75%", "100%"])
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
        fig.tight_layout(rect=(0, 0, 1, 1))

        safe_name = f"{idx:02d}_{col}".replace(os.sep, "_")
        filename = os.path.join(output_dir, f"{safe_name}.png")
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"Wrote {filename}")


def save_montage(grid_dir: str, output_path: str, nrows: int = 3, ncols: int = 3) -> None:
    image_paths = sorted(
        path
        for path in (os.path.join(grid_dir, name) for name in os.listdir(grid_dir))
        if path.lower().endswith(".png")
    )
    if not image_paths:
        raise ValueError(f"No PNG files found in {grid_dir}")

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.atleast_2d(axes)
    for idx, ax in enumerate(axes.flat):
        if idx < len(image_paths):
            img = mpimg.imread(image_paths[idx])
            ax.imshow(img)
        ax.axis("off")
    fig.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Wrote {output_path}")


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
        if output_dir == "All":
            save_montage(output_dir, "All.png")


if __name__ == "__main__":
    main()
