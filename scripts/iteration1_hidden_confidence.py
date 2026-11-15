# %% [markdown]
# # Does the model know when it doesn't know - in Yoruba? (multi-model)
# Iteration 1b: the same English vs Yoruba calibration test on AfriMMLU, run over a
# list of models, including a base model and its chat version.
#
# Run in Google Colab with a T4 GPU:  Runtime -> Change runtime type -> T4 GPU
#
# Gated models (Gemma, Llama) need two things first:
# 1. On each model's Hugging Face page, log in and accept the license
#    (Gemma is instant; Llama can take from minutes to a day).
# 2. Create a "Read" token at https://huggingface.co/settings/tokens and add it in Colab:
#    key icon (Secrets) on the left -> name it HF_TOKEN -> paste -> turn on "Notebook access".
# Models you can't access are skipped with a message; the rest still run.

# %%
# !pip install -q transformers accelerate datasets matplotlib pandas

# %% Log in to Hugging Face (only needed for gated models)
try:
    from google.colab import userdata
    from huggingface_hub import login
    login(token=userdata.get("HF_TOKEN"))
    print("Logged in to Hugging Face")
except Exception as e:
    print("No HF_TOKEN found, gated models will be skipped:", type(e).__name__)

# %%
import ast
import gc
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# "chat": True -> use the model's chat template. False -> plain text prompt (for base models).
# "dtype": float16 by default. Gemma 2 overflows in float16, so it runs in float32 (fits on a T4).
MODELS = [
    {"name": "Qwen/Qwen2.5-3B-Instruct", "chat": True},
    {"name": "Qwen/Qwen2.5-3B", "chat": False},                 # base version of the model above
    {"name": "Qwen/Qwen2.5-1.5B-Instruct", "chat": True},
    {"name": "microsoft/Phi-3.5-mini-instruct", "chat": True},  # 3.8B, not gated
    {"name": "google/gemma-2-2b-it", "chat": True, "dtype": "float32"},  # gated
    {"name": "meta-llama/Llama-3.2-3B-Instruct", "chat": True},          # gated
    # Control: the chat model with the base model's plain prompt. Separates the effect of
    # chat training from the effect of the prompt format.
    {"name": "Qwen/Qwen2.5-3B-Instruct", "chat": False, "label": "Qwen2.5-3B-Instruct (plain prompt)"},
]

LANGS = ["eng", "yor"]
PER_SUBJECT = None   # e.g. 10 for a quick test (10 x 5 subjects = 50 questions per language)
LETTERS = ["A", "B", "C", "D"]
N_BOOT = 2000        # bootstrap resamples for confidence intervals
SEED = 0
OUT_DIR = "results/iteration1"  # one CSV per model; finished models are skipped if you re-run
COLORS = {"eng": "#2a78d6", "yor": "#eb6834"}   # blue = English, orange = Yoruba

# Instruction stays in English for both languages, so only the question language changes.
INSTRUCTION = (
    "Answer the following multiple-choice question. "
    "Reply with only the letter of the correct option (A, B, C, or D)."
)
os.makedirs(OUT_DIR, exist_ok=True)


# %% Data helpers
def parse_choices(raw):
    """Choices are stored as a string like "['p = 4', 'p = 8', ...]"."""
    if isinstance(raw, (list, np.ndarray)):
        return list(raw)
    try:
        out = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        out = re.findall(r"""'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)\"""", raw)
        out = [a or b for a, b in out]
    return [str(c) for c in out]


def load_lang(lang):
    from datasets import load_dataset
    df = load_dataset("masakhane/afrimmlu", lang, split="test").to_pandas()
    df["idx"] = np.arange(len(df))   # question id, shared across languages
    return df


def sample_per_subject(df, k):
    return df if not k else df.groupby("subject", group_keys=False).head(k)


def slug(name):
    return name.replace("/", "__")


def short(name):
    return name.split("/")[-1]


# %% Metrics
def ece(conf, correct, n_bins=10):
    """Expected Calibration Error: weighted average gap between confidence and accuracy."""
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() > 0:
            total += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(total)


def auroc(conf, correct):
    """How well confidence separates right from wrong answers. 0.5 = no better than chance."""
    pos, neg = correct == 1, correct == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = pd.Series(conf).rank().values
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def lang_metrics(g):
    conf, corr = g["confidence"].values, g["correct"].values
    return {
        "n": int(len(g)),
        "accuracy": float(corr.mean()),
        "mean_confidence": float(conf.mean()),
        "overconfidence": float(conf.mean() - corr.mean()),   # >0 means overconfident
        "ECE": ece(conf, corr),
        "AUROC": auroc(conf, corr),
        "brier": float(np.mean((1 - g["p_gold"].values) ** 2)),
        "mean_letter_mass": float(g["letter_mass"].mean()),
    }


def paired_bootstrap(results, a="eng", b="yor", n_boot=N_BOOT, seed=SEED):
    """95% intervals for (b - a) differences, resampling the same questions in both languages."""
    wide = {l: results[results["lang"] == l].set_index("idx").sort_index() for l in (a, b)}
    common = wide[a].index.intersection(wide[b].index)
    A, B = wide[a].loc[common], wide[b].loc[common]
    rng = np.random.default_rng(seed)
    keys = ["accuracy", "mean_confidence", "overconfidence", "ECE", "AUROC"]
    diffs = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, len(common), len(common))
        ma, mb = lang_metrics(A.iloc[pick]), lang_metrics(B.iloc[pick])
        for k in keys:
            diffs[k].append(mb[k] - ma[k])
    ma, mb = lang_metrics(A), lang_metrics(B)
    return {
        k: {"diff": round(mb[k] - ma[k], 3),
            "ci95": [round(float(np.nanpercentile(diffs[k], 2.5)), 3),
                     round(float(np.nanpercentile(diffs[k], 97.5)), 3)]}
        for k in keys
    }


def summarise(results, name=""):
    summary = {lang: {k: round(v, 3) if isinstance(v, float) else v
                      for k, v in lang_metrics(g).items()}
               for lang, g in results.groupby("lang")}
    summary["yor_minus_eng"] = paired_bootstrap(results)
    summary["by_subject"] = {
        subj: {lang: {"acc": round(float(g["correct"].mean()), 3),
                      "conf": round(float(g["confidence"].mean()), 3),
                      "ECE": round(ece(g["confidence"].values, g["correct"].values), 3)}
               for lang, g in sg.groupby("lang")}
        for subj, sg in results.groupby("subject")
    }
    summary["pred_letter_share"] = {
        lang: {LETTERS[k]: round(float(v), 3)
               for k, v in g["pred"].value_counts(normalize=True).sort_index().items()}
        for lang, g in results.groupby("lang")
    }
    # Does confidence on a question in English predict confidence on its translation?
    w = results.pivot(index="idx", columns="lang", values="confidence").dropna()
    summary["conf_corr_eng_yor"] = round(float(np.corrcoef(w["eng"], w["yor"])[0, 1]), 3)
    for lang in LANGS:
        if summary[lang]["mean_letter_mass"] < 0.5:
            print(f"WARNING {name}: {lang} mean letter_mass = {summary[lang]['mean_letter_mass']}. "
                  "The model is not answering with a letter; confidence values are unreliable.")
    return summary


# %% Plots
def reliability_plot(results, summary, title, path):
    fig, axes = plt.subplots(1, len(LANGS), figsize=(5 * len(LANGS), 4.5), sharey=True)
    bins = np.linspace(0, 1, 11)
    for ax, lang in zip(axes, LANGS):
        g = results[results["lang"] == lang]
        centers, accs, counts = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (g["confidence"] > lo) & (g["confidence"] <= hi)
            if m.sum() > 0:
                centers.append((lo + hi) / 2)
                accs.append(g.loc[m, "correct"].mean())
                counts.append(int(m.sum()))
        ax.bar(centers, accs, width=0.09, color=COLORS[lang], label="accuracy")
        for x, y, c in zip(centers, accs, counts):
            ax.text(x, y + 0.02, str(c), ha="center", fontsize=8, color="#555")
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.5, label="perfect calibration")
        s = summary[lang]
        ax.set_title(f"{lang}  acc={s['accuracy']}  ECE={s['ECE']}")
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper left", frameon=False)
    axes[0].set_ylabel("accuracy (numbers = questions per bin)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def comparison_plot(summaries, path="comparison.png"):
    """Left: accuracy vs confidence per model (one line from English to Yoruba).
    Right: AUROC per model in each language."""
    names = list(summaries)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})

    # Left panel: does confidence fall as much as accuracy?
    ax1.plot([0, 1], [0, 1], "--", color="#888", lw=1.5, label="perfect calibration")
    ax1.axvline(0.25, color="#bbb", lw=1, ls=":")
    ax1.text(0.255, 0.42, "chance", color="#888", fontsize=8, rotation=90, va="bottom")
    for n in names:
        e, y = summaries[n]["eng"], summaries[n]["yor"]
        ax1.plot([e["accuracy"], y["accuracy"]], [e["mean_confidence"], y["mean_confidence"]],
                 color="#bbb", lw=2, zorder=1)
        ax1.text(y["accuracy"] - 0.01, y["mean_confidence"], short(n), fontsize=8,
                 ha="right", va="center", color="#333")
    for lang in LANGS:
        ax1.scatter([summaries[n][lang]["accuracy"] for n in names],
                    [summaries[n][lang]["mean_confidence"] for n in names],
                    s=70, color=COLORS[lang], edgecolor="white", linewidth=1.5, zorder=2,
                    label={"eng": "English", "yor": "Yoruba"}[lang])
    lowest = min(summaries[n][l]["mean_confidence"] for n in names for l in LANGS)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(max(0.0, min(0.4, lowest - 0.08)), 1.02)
    ax1.set_xlabel("accuracy")
    ax1.set_ylabel("mean confidence")
    ax1.set_title("Accuracy drops in Yoruba; does confidence?", loc="left")
    ax1.legend(loc="lower right", frameon=False)

    # Right panel: AUROC dot plot
    ypos = np.arange(len(names))[::-1]
    ax2.axvline(0.5, color="#bbb", lw=1, ls=":")
    ax2.text(0.502, ypos.max() + 0.45, "chance", color="#888", fontsize=8)
    for yp, n in zip(ypos, names):
        e, y = summaries[n]["eng"]["AUROC"], summaries[n]["yor"]["AUROC"]
        ax2.plot([e, y], [yp, yp], color="#bbb", lw=2, zorder=1)
    for lang in LANGS:
        ax2.scatter([summaries[n][lang]["AUROC"] for n in names], ypos, s=70, color=COLORS[lang],
                    edgecolor="white", linewidth=1.5, zorder=2,
                    label={"eng": "English", "yor": "Yoruba"}[lang])
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([short(n) for n in names])
    ax2.set_xlim(0.4, 1.0)
    ax2.set_ylim(-0.6, len(names) - 0.2)
    ax2.set_xlabel("AUROC (does confidence separate right from wrong?)")
    ax2.set_title("Confidence as a signal of correctness", loc="left")
    ax2.legend(loc="lower right", frameon=False)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    return fig


def gap_table(summaries):
    rows = []
    for n, s in summaries.items():
        d = s["yor_minus_eng"]
        fmt = lambda k: f"{d[k]['diff']:+.3f} [{d[k]['ci95'][0]:+.3f}, {d[k]['ci95'][1]:+.3f}]"
        rows.append({
            "model": short(n),
            "acc eng": s["eng"]["accuracy"], "acc yor": s["yor"]["accuracy"],
            "conf eng": s["eng"]["mean_confidence"], "conf yor": s["yor"]["mean_confidence"],
            "AUROC eng": s["eng"]["AUROC"], "AUROC yor": s["yor"]["AUROC"],
            "AUROC gap [95% CI]": fmt("AUROC"),
            "overconf gap [95% CI]": fmt("overconfidence"),
            "conf corr eng-yor": s["conf_corr_eng_yor"],
            "letter mass yor": s["yor"]["mean_letter_mass"],
        })
    return pd.DataFrame(rows)


# %% Load data
if __name__ == "__main__":
    data = {lang: sample_per_subject(load_lang(lang), PER_SUBJECT) for lang in LANGS}
    print({lang: len(df) for lang, df in data.items()}, "questions")


# %% Model runner
def find_letter_ids(tok):
    """All single tokens that read as exactly 'A', 'B', 'C' or 'D' (e.g. 'A', ' A', '▁A')."""
    ids = {l: [] for l in LETTERS}
    for token, tid in tok.get_vocab().items():
        clean = token.replace("Ġ", " ").replace("▁", " ").strip()
        if clean in ids:
            ids[clean].append(tid)
    assert all(ids.values()), f"missing letter tokens: {ids}"
    return [sorted(ids[l]) for l in LETTERS]


def run_model(spec, data):
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = spec["name"]
    label = spec.get("label", name)   # a run's name in tables/plots (same model can run twice)
    out_csv = os.path.join(OUT_DIR, slug(label) + ".csv")
    if os.path.exists(out_csv):
        print(f"[{name}] already done, loading {out_csv}")
        return pd.read_csv(out_csv)

    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=getattr(torch, spec.get("dtype", "float16")), device_map="auto")
    except OSError as e:
        print(f"[{name}] SKIPPED: could not load ({str(e).splitlines()[0]})")
        return None
    model.eval()
    letter_ids = find_letter_ids(tok)
    use_chat = spec.get("chat", True) and tok.chat_template is not None
    print(f"[{name}] device={model.device}  chat_template={use_chat}")

    def build_prompt(question, choices):
        options = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, choices))
        user_msg = f"{INSTRUCTION}\n\nQuestion: {question}\n{options}"
        if use_chat:
            # Prompt ends where the assistant's reply starts (no trailing text or space).
            return tok.apply_chat_template([{"role": "user", "content": user_msg}],
                                           tokenize=False, add_generation_prompt=True)
        return user_msg + "\nAnswer:"

    @torch.no_grad()
    def letter_probs(prompt):
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=not use_chat).to(model.device)
        logits = model(**inputs).logits[0, -1].float()
        full = torch.softmax(logits, dim=0)
        per_letter = torch.stack([full[ids].sum() for ids in letter_ids])
        mass = float(per_letter.sum())               # prob. on A-D before renormalising
        return (per_letter / per_letter.sum()).cpu().numpy(), mass

    # Quick check on one question before the full run
    r0 = data["eng"].iloc[0]
    p0, m0 = letter_probs(build_prompt(r0["question"], parse_choices(r0["choices"])))
    print(f"[{name}] check: letter_mass = {m0:.3f} (should be well above 0.5)")

    rows = []
    try:
        for lang, df in data.items():
            for _, r in tqdm(df.iterrows(), total=len(df), desc=f"{short(name)} {lang}"):
                choices = parse_choices(r["choices"])
                if len(choices) != 4:
                    continue
                probs, mass = letter_probs(build_prompt(r["question"], choices))
                pred = int(np.argmax(probs))
                gold = LETTERS.index(str(r["answer"]).strip().upper())
                rows.append({
                    "model": label, "lang": lang, "idx": int(r["idx"]), "subject": r["subject"],
                    "pred": pred, "gold": gold, "correct": int(pred == gold),
                    "confidence": float(probs.max()), "p_gold": float(probs[gold]),
                    "letter_mass": mass,
                })
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    results.to_csv(out_csv, index=False)
    return results


# %% Run all models
if __name__ == "__main__":
    all_results = {}
    for spec in MODELS:
        res = run_model(spec, data)
        if res is not None and len(res):
            all_results[spec.get("label", spec["name"])] = res
    print("finished:", list(all_results))


# %% Summaries, table and plots
if __name__ == "__main__":
    summaries = {}
    os.makedirs("plots", exist_ok=True)
    for name, res in all_results.items():
        summaries[name] = summarise(res, name)
        reliability_plot(res, summaries[name], name, f"plots/{slug(name)}_reliability.png")
    with open("summary_all.json", "w") as f:
        json.dump(summaries, f, indent=2)

    table = gap_table(summaries)
    table.to_csv("comparison_table.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))

    comparison_plot(summaries)
    plt.show()

# %% Download everything as one zip
if __name__ == "__main__":
    os.system("zip -qr afrimmlu_outputs.zip results/iteration1 plots summary_all.json "
              "comparison_table.csv comparison.png")
    try:
        from google.colab import files
        files.download("afrimmlu_outputs.zip")
    except ImportError:
        print("saved afrimmlu_outputs.zip")
