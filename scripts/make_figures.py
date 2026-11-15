"""Rebuild every table and figure in the README from the CSVs in results/.

No GPU or model download needed. Run from the repo root:
    python scripts/make_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iteration1_hidden_confidence as it1           # noqa: E402
import iteration2_stated_confidence_and_idk as it2   # noqa: E402

R1, R2, FIG = "results/iteration1", "results/iteration2", "figures"
os.makedirs(FIG, exist_ok=True)

# Order used in every table and figure
RUNS1 = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-3B",
    "Qwen2.5-3B-Instruct (plain prompt)",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
]
CHAT = [r for r in RUNS1 if r not in ("Qwen/Qwen2.5-3B", "Qwen2.5-3B-Instruct (plain prompt)")]


def overconf_table(summaries):
    rows = []
    for n, s in summaries.items():
        d = s["yor_minus_eng"]["overconfidence"]
        rows.append({"model": it1.short(n),
                     "overconf eng": s["eng"]["overconfidence"],
                     "overconf yor": s["yor"]["overconfidence"],
                     "gap [95% CI]": f"{d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"})
    return pd.DataFrame(rows)


def base_vs_chat(path):
    runs = [("Qwen2.5-3B (base)", "Qwen__Qwen2.5-3B.csv"),
            ("Qwen2.5-3B-Instruct (chat)", "Qwen__Qwen2.5-3B-Instruct.csv")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 11)
    for row, (title, f) in enumerate(runs):
        r = pd.read_csv(os.path.join(R1, f))
        for col, lang in enumerate(it1.LANGS):
            ax, g = axes[row, col], r[r["lang"] == lang]
            c, a, n = [], [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                k = (g["confidence"] > lo) & (g["confidence"] <= hi)
                if k.sum():
                    c.append((lo + hi) / 2)
                    a.append(g.loc[k, "correct"].mean())
                    n.append(int(k.sum()))
            ax.bar(c, a, width=0.09, color=it1.COLORS[lang])
            for x, y, k in zip(c, a, n):
                ax.text(x, y + 0.02, str(k), ha="center", fontsize=8, color="#555")
            ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.5)
            ax.set_title(f"{title}\n{'English' if lang == 'eng' else 'Yoruba'}: accuracy "
                         f"{g['correct'].mean():.0%}, mean confidence {g['confidence'].mean():.0%}",
                         fontsize=10, loc="left")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.25)
            if row == 1:
                ax.set_xlabel("confidence")
            if col == 0:
                ax.set_ylabel("accuracy")
    fig.suptitle("Qwen2.5-3B: base model vs chat version, English vs Yoruba\n"
                 "(bars = accuracy per confidence bin, numbers = questions, dashed = perfect calibration)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def signals_summary(path):
    S = {}
    for n in CHAT:
        s = it2.slug(n)
        v = pd.read_csv(f"{R2}/{s}_verbal.csv")
        a = pd.read_csv(f"{R2}/{s}_abstain.csv")
        h = pd.read_csv(f"{R1}/{s}.csv")
        S[n] = {}
        for lang in it2.LANGS:
            vv = v[(v["lang"] == lang) & (v["parsed"] == 1)]
            aa, hh = a[a["lang"] == lang], h[h["lang"] == lang]
            S[n][lang] = {"hidden": it2.auroc(hh["confidence"], hh["correct"]),
                          "stated": it2.auroc(vv["confidence"], vv["correct"]),
                          "idk": it2.auroc(1 - aa["p_idk"], aa["correct4"]),
                          "idk_rate": aa["abstained"].mean()}
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    it2.dumbbell(axes[0], S, "hidden", "AUROC", "A. Hidden probability (Iteration 1)", (0.3, 1), 0.5, "chance")
    it2.dumbbell(axes[1], S, "stated", "AUROC", "B. Stated confidence (0-100)", (0.3, 1), 0.5, "chance")
    it2.dumbbell(axes[2], S, "idk", "AUROC", "C. 1 - P('I don't know')", (0.3, 1), 0.5, "chance")
    it2.dumbbell(axes[3], S, "idk_rate", "share of questions", "D. How often it picks 'I don't know'", (0, 1))
    for ax in axes[1:]:
        ax.set_yticklabels([])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False)
    fig.suptitle("A-C: does each uncertainty signal separate right from wrong answers? "
                 "(AUROC, 0.5 = chance)    D: how often 'I don't know' is chosen",
                 x=0.01, ha="left", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return S


if __name__ == "__main__":
    pd.set_option("display.width", 250)

    # Iteration 1
    res1 = {n: pd.read_csv(os.path.join(R1, it1.slug(n) + ".csv")) for n in RUNS1}
    s1 = {n: it1.summarise(r, n) for n, r in res1.items()}
    with open("results/summary_iteration1.json", "w") as f:
        json.dump(s1, f, indent=2)
    t1 = it1.gap_table(s1)
    t1.to_csv("results/table_iteration1.csv", index=False)
    print("ITERATION 1\n", t1.to_string(index=False))
    print("\n", overconf_table(s1).to_string(index=False))
    it1.comparison_plot(s1, f"{FIG}/iter1_accuracy_vs_confidence.png")
    plt.close("all")
    base_vs_chat(f"{FIG}/iter1_base_vs_chat.png")

    # Iteration 2
    it2.ITER1_DIR = R1
    runs2 = {n: {c: pd.read_csv(f"{R2}/{it2.slug(n)}_{c}.csv") for c in it2.CONDITIONS} for n in CHAT}
    sv = {n: it2.summarise_verbal(o["verbal"], n) for n, o in runs2.items()}
    sa = {n: it2.summarise_abstain(o["abstain"], n) for n, o in runs2.items()}
    with open("results/summary_iteration2.json", "w") as f:
        json.dump({"verbal": sv, "abstain": sa}, f, indent=2)
    tv, ta = it2.tables(sv, sa)
    tv.to_csv("results/table_iteration2_stated_confidence.csv", index=False)
    ta.to_csv("results/table_iteration2_idk_option.csv", index=False)
    print("\nITERATION 2 - stated confidence\n", tv.to_string(index=False))
    print("\nITERATION 2 - 'I don't know' option\n", ta.to_string(index=False))
    it2.verbal_plot(sv, f"{FIG}/iter2_stated_confidence.png")
    it2.abstain_plot(sa, f"{FIG}/iter2_idk_option.png")
    plt.close("all")

    # Summary across both iterations
    S = signals_summary(f"{FIG}/summary_uncertainty_signals.png")
    print("\nSUMMARY (AUROC)")
    print(pd.DataFrame([{"model": it2.short(n), "lang": l, **{k: round(v, 3) for k, v in S[n][l].items()}}
                        for n in S for l in it2.LANGS]).to_string(index=False))
    print("\nfigures written to", FIG)
