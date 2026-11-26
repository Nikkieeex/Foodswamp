"""
Data quality check for merged.csv
Writes a report to quality_report.md with plots in the plots/ folder

Usage:
    python quality_check.py merged.csv
    python quality_check.py merged.csv --id project_pseudo_id
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, plots are saved
import matplotlib.pyplot as plt
import pandas as pd


def md_tabel(df, max_rijen=50):
    """Convert a DataFrame to a markdown table (no extra packages)."""
    df = df.head(max_rijen)
    kolommen = [str(c) for c in df.columns]
    regels = ["| " + " | ".join(kolommen) + " |",
              "|" + "---|" * len(kolommen)]
    for _, rij in df.iterrows():
        regels.append("| " + " | ".join(str(v) for v in rij.values) + " |")
    return "\n".join(regels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="path to merged.csv")
    parser.add_argument("--id", default=None, help="name of the ID column (default first column)")
    parser.add_argument("--uit", default="quality_report.md", help="name of the report")
    args = parser.parse_args()

    plot_map = Path("plots")
    plot_map.mkdir(exist_ok=True)

    # ---------- Load ----------
    df = pd.read_csv(args.csv, low_memory=False)
    # Treat empty strings and whitespace as NA
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    r = []  # report lines
    r.append(f"# Data quality {args.csv}\n")

    # ---------- Overview ----------
    r.append("## Overview\n")
    r.append(f"- Rows: {df.shape[0]}")
    r.append(f"- Columns: {df.shape[1]}")
    r.append(f"- Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

    r.append("### First 5 rows\n")
    r.append(md_tabel(df.head()) + "\n")

    r.append("### Column types\n")
    types = df.dtypes.astype(str).value_counts().reset_index()
    types.columns = ["type", "n_columns"]
    r.append(md_tabel(types) + "\n")

    # ---------- Missing values ----------
    r.append("## Missing values (NA)\n")
    na = pd.DataFrame({
        "column": df.columns,
        "type": df.dtypes.astype(str).values,
        "n_na": df.isna().sum().values,
        "pct_na": (df.isna().mean() * 100).round(2).values,
        "n_unique": df.nunique(dropna=True).values,
    }).sort_values("pct_na", ascending=False)
    r.append(md_tabel(na, max_rijen=len(na)) + "\n")

    na_kol = na[na["n_na"] > 0]
    if len(na_kol) > 0:
        fig, ax = plt.subplots(figsize=(8, max(3, len(na_kol) * 0.25)))
        ax.barh(na_kol["column"][::-1], na_kol["pct_na"][::-1], color="steelblue")
        ax.set_xlabel("% missing")
        ax.set_title("Missing values per column")
        fig.tight_layout()
        fig.savefig(plot_map / "na_per_column.png", dpi=120)
        plt.close(fig)
        r.append("![NA per column](plots/na_per_column.png)\n")

    na_per_row = df.isna().sum(axis=1)
    r.append(f"- Rows without NA: {(na_per_row == 0).sum()}")
    r.append(f"- Completely empty rows: {(na_per_row == df.shape[1]).sum()}\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(na_per_row, bins=30, color="steelblue")
    ax.set_xlabel("Number of NA per row")
    ax.set_ylabel("Number of rows")
    fig.tight_layout()
    fig.savefig(plot_map / "na_per_row.png", dpi=120)
    plt.close(fig)
    r.append("![NA per row](plots/na_per_row.png)\n")

    # NA pattern (sample of max 1000 rows)
    sample = df.sample(min(1000, len(df)), random_state=1)
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1] * 0.2), 6))
    ax.imshow(sample.isna().values, aspect="auto", cmap="Reds", interpolation="nearest")
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=90, fontsize=6)
    ax.set_ylabel("Row (sample)")
    ax.set_title("NA pattern (red = NA)")
    fig.tight_layout()
    fig.savefig(plot_map / "na_pattern.png", dpi=120)
    plt.close(fig)
    r.append("![NA pattern](plots/na_pattern.png)\n")

    # ---------- Special missing codes ----------
    r.append("## Special missing codes\n")
    r.append("Lifelines sometimes uses codes like `$5` or `$7` instead of a real NA.\n")
    tekst = df.select_dtypes(exclude=["number", "bool", "datetime"])
    dollar = (tekst.stack()
              .loc[lambda s: s.astype(str).str.startswith("$")]
              .reset_index(level=1)
              .rename(columns={"level_1": "column", 0: "value"}))
    if len(dollar) > 0:
        tel = dollar.value_counts(["column", "value"]).reset_index(name="n")
        r.append(md_tabel(tel) + "\n")
    else:
        r.append("No values found starting with `$`.\n")

    getal = df.select_dtypes(include="number")
    codes = [-999, -99, -9, 999, 9999]
    code_tel = {k: getal[k].isin(codes).sum() for k in getal.columns if getal[k].isin(codes).any()}
    if code_tel:
        r.append(md_tabel(pd.DataFrame(code_tel.items(), columns=["column", "n_codes"])) + "\n")
    else:
        r.append("No numeric codes like -999 found.\n")

    # ---------- Duplicates ----------
    r.append("## Duplicates\n")
    id_kolom = args.id or df.columns[0]
    r.append(f"- Fully duplicated rows: {df.duplicated().sum()}")
    r.append(f"- Duplicate values in `{id_kolom}`: {df[id_kolom].duplicated().sum()}\n")

    # ---------- Constant columns ----------
    r.append("## Constant columns\n")
    const = na[na["n_unique"] <= 1]
    r.append(md_tabel(const) + "\n" if len(const) else "No constant columns.\n")

    # ---------- Numeric columns ----------
    if getal.shape[1] > 0:
        r.append("## Numeric columns\n")
        stats = getal.describe().T.round(2)
        q1, q3 = getal.quantile(0.25), getal.quantile(0.75)
        iqr = q3 - q1
        # outliers using the 1.5 x IQR rule
        stats["n_outlier"] = ((getal < q1 - 1.5 * iqr) | (getal > q3 + 1.5 * iqr)).sum()
        stats = stats.reset_index().rename(columns={"index": "column"})
        r.append(md_tabel(stats, max_rijen=len(stats)) + "\n")

        n = getal.shape[1]
        ncol = min(4, n)
        nrow = -(-n // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 2.5), squeeze=False)
        for ax, kol in zip(axes.flat, getal.columns):
            ax.boxplot(getal[kol].dropna())
            ax.set_title(kol, fontsize=8)
            ax.set_xticks([])
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(plot_map / "boxplots.png", dpi=120)
        plt.close(fig)
        r.append("![Boxplots](plots/boxplots.png)\n")

    # ---------- Categorical columns ----------
    if tekst.shape[1] > 0:
        r.append("## Categorical columns (top 5 values)\n")
        for kol in tekst.columns:
            top = df[kol].value_counts(dropna=False).head(5).reset_index()
            top.columns = ["value", "n"]
            r.append(f"### {kol}\n")
            r.append(md_tabel(top) + "\n")

    Path(args.uit).write_text("\n".join(r), encoding="utf-8")
    print(f"Done! Report saved as {args.uit} (plots in {plot_map}/)")
    print(na.head(15).to_string(index=False))


if __name__ == "__main__":
    main()