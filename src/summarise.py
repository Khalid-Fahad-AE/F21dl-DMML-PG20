"""Collect every model's results into the report tables and the headline comparison figure.

    python src/summarise.py
Writes results/tables/summary_test.md, summary_plantdoc.md and results/figures/model_comparison.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import ROOT

TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"


def md_table(df, cols, fmt):
    head = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    rows = ""
    for _, r in df.iterrows():
        rows += "| " + " | ".join(fmt.get(c, "{}").format(r[c]) if pd.notna(r[c]) else "–" for c in cols) + " |\n"
    return head + rows


def main():
    m = pd.read_csv(TAB / "metrics_all.csv")
    test = m[m.split == "test"].sort_values("macro_f1", ascending=False)
    pdoc = m[m.split == "plantdoc"].copy()
    pdoc["base"] = pdoc.model.str.replace("@PlantDoc", "", regex=False)
    fmt = {"accuracy": "{:.3f}", "macro_precision": "{:.3f}", "macro_recall": "{:.3f}", "macro_f1": "{:.3f}",
           "train_time_s": "{:.1f}", "infer_ms_per_img": "{:.3f}", "n_params": "{:,.0f}"}
    cols = ["model", "group", "accuracy", "macro_precision", "macro_recall", "macro_f1", "train_time_s",
            "infer_ms_per_img"]
    (TAB / "summary_test.md").write_text(md_table(test, cols, fmt))
    both = test.merge(pdoc[["base", "accuracy", "macro_f1"]].rename(
        columns={"base": "model", "accuracy": "plantdoc_acc", "macro_f1": "plantdoc_macro_f1"}), on="model", how="inner")
    both["f1_drop"] = both.macro_f1 - both.plantdoc_macro_f1
    (TAB / "summary_plantdoc.md").write_text(md_table(
        both, ["model", "macro_f1", "plantdoc_acc", "plantdoc_macro_f1", "f1_drop"],
        {"macro_f1": "{:.3f}", "plantdoc_acc": "{:.3f}", "plantdoc_macro_f1": "{:.3f}", "f1_drop": "{:.3f}"}))
    both.to_csv(TAB / "summary_plantdoc.csv", index=False)
    test.to_csv(TAB / "summary_test.csv", index=False)
    print(md_table(test, cols, fmt)); print(md_table(both, ["model", "macro_f1", "plantdoc_macro_f1", "f1_drop"], fmt))

    # headline figure: PlantVillage test vs PlantDoc field photos
    show = test[~test.model.str.contains("sameDims|_frozen|_generative")].copy()
    show = show.merge(both[["model", "plantdoc_macro_f1"]], on="model", how="left").sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(show) + 1))
    y = range(len(show))
    ax.barh([i + 0.2 for i in y], show.macro_f1, height=0.4, color="#3b6ea8", label="PlantVillage test")
    ax.barh([i - 0.2 for i in y], show.plantdoc_macro_f1.fillna(0), height=0.4, color="#d98f3b",
            label="PlantDoc field photos")
    for i, (a, b) in enumerate(zip(show.macro_f1, show.plantdoc_macro_f1)):
        ax.text(a + 0.005, i + 0.2, f"{a:.2f}", va="center", fontsize=6)
        if pd.notna(b):
            ax.text(b + 0.005, i - 0.2, f"{b:.2f}", va="center", fontsize=6)
    ax.set_yticks(list(y)); ax.set_yticklabels(show.model, fontsize=7)
    ax.set_xlabel("macro-F1"); ax.set_xlim(0, 1.05)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "model_comparison.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
