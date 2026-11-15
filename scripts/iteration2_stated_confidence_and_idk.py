# %% [markdown]
# # Iteration 2: Will the model *say* it doesn't know - in Yoruba?
# Iteration 1 showed that a model's hidden confidence (its probability on the chosen
# letter) stops tracking correctness in Yoruba. Here we ask the model directly, in two ways:
#
# **A. Stated confidence.** The model replies "Answer: C / Confidence: 85".
#    Does the number it states drop in Yoruba? Does it track correctness?
# **B. "I don't know" option.** A fifth option "E. I don't know" is added, and the model
#    is told to choose it if it does not know or does not understand the question.
#    Does it choose E more in Yoruba, and on the questions it would get wrong?
#
# Same data as Iteration 1 (AfriMMLU, eng + yor, 500 parallel questions each).
# Run in Google Colab with a T4 GPU. Add your Hugging Face token as the Colab secret HF_TOKEN.

# %%
# !pip install -q transformers accelerate datasets matplotlib pandas

# %% Log in to Hugging Face (needed for Gemma and Llama)
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

# Chat models only: the stated-confidence format needs a model that follows instructions.
# "batch": prompts per forward pass (lowered automatically if the GPU runs out of memory).
MODELS = [
    {"name": "Qwen/Qwen2.5-3B-Instruct"},
    {"name": "Qwen/Qwen2.5-1.5B-Instruct"},
    {"name": "microsoft/Phi-3.5-mini-instruct"},
    {"name": "google/gemma-2-2b-it", "dtype": "float32", "batch": 4},
    {"name": "meta-llama/Llama-3.2-3B-Instruct"},
]
CONDITIONS = ["verbal", "abstain"]   # A = stated confidence, B = "I don't know" option

LANGS = ["eng", "yor"]
PER_SUBJECT = None     # e.g. 10 for a quick test (50 questions per language)
LETTERS = ["A", "B", "C", "D"]
LETTERS5 = LETTERS + ["E"]
IDK_TEXT = "I don't know"
MAX_NEW_TOKENS = 48    # room for "Answer: C\nConfidence: 85" even when the model copies the
                       # Yoruba option text, which takes many tokens (24 cut some replies off)
N_BOOT = 2000
SEED = 0
OUT_DIR = "results/iteration2"   # one CSV per model and condition; finished runs are skipped
ITER1_DIR = "results/iteration1"   # Iteration 1 CSVs, if present, to compare stated vs hidden confidence
COLORS = {"eng": "#2a78d6", "yor": "#eb6834"}
LANG_NAMES = {"eng": "English", "yor": "Yoruba"}

# Instructions stay in English for both languages, so only the question language changes.
VERBAL_INSTRUCTION = (
    "Answer the following multiple-choice question. Then say how confident you are that "
    "your answer is correct, as a number from 0 to 100.\n"
    "Reply in exactly this format and nothing else:\n"
    "Answer: <A, B, C, or D>\n"
    "Confidence: <number from 0 to 100>"
)
ABSTAIN_INSTRUCTION = (
    "Answer the following multiple-choice question. If you do not know the answer, "
    "or do not understand the question, choose E.\n"
    "Reply with only the letter of your choice (A, B, C, D, or E)."
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
    df["idx"] = np.arange(len(df))
    return df


def sample_per_subject(df, k):
    return df if not k else df.groupby("subject", group_keys=False).head(k)


def slug(name):
    return name.replace("/", "__")


def short(name):
    return name.split("/")[-1]


def user_message(condition, question, choices):
    if condition == "verbal":
        options = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, choices))
        return f"{VERBAL_INSTRUCTION}\n\nQuestion: {question}\n{options}"
    options = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS5, list(choices) + [IDK_TEXT]))
    return f"{ABSTAIN_INSTRUCTION}\n\nQuestion: {question}\n{options}"


# %% Parsing the model's written reply (condition A)
ANSWER_RE = re.compile(r"answer\W{0,6}([A-D])\b", re.I)
CONF_RE = re.compile(r"confidence\W{0,6}(\d{1,3}(?:\.\d+)?)", re.I)


def parse_verbal(text):
    """Returns (letter index or None, confidence in [0,1] or None)."""
    m = ANSWER_RE.search(text)
    if m and m.group(1).isupper():
        letter = m.group(1)
    else:  # fallback: first standalone capital A-D in the reply
        m2 = re.search(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", text)
        letter = m2.group(1) if m2 else None
    c = CONF_RE.search(text)
    conf = min(float(c.group(1)), 100.0) / 100 if c else None
    return (LETTERS.index(letter) if letter else None), conf


# %% Metrics
def ece(conf, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() > 0:
            total += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(total)


def auroc(score, label):
    """P(score of a random label-1 item > score of a random label-0 item). 0.5 = chance."""
    score, label = np.asarray(score, float), np.asarray(label)
    pos, neg = label == 1, label == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = pd.Series(score).rank().values
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def verbal_metrics(g):
    ok = g[g["parsed"] == 1]
    conf, corr = ok["confidence"].values, ok["correct"].values
    top = ok["confidence"].round(2).value_counts(normalize=True)
    return {
        "n": int(len(g)),
        "parse_rate": float(g["parsed"].mean()),
        "accuracy": float(corr.mean()) if len(ok) else float("nan"),
        "mean_confidence": float(conf.mean()) if len(ok) else float("nan"),
        "overconfidence": float(conf.mean() - corr.mean()) if len(ok) else float("nan"),
        "ECE": ece(conf, corr) if len(ok) else float("nan"),
        "AUROC": auroc(conf, corr),
        "most_common_conf": float(top.index[0]) if len(top) else float("nan"),
        "share_most_common_conf": float(top.iloc[0]) if len(top) else float("nan"),
    }


def abstain_metrics(g):
    answered = g[g["abstained"] == 0]
    abst = g[g["abstained"] == 1]
    return {
        "n": int(len(g)),
        "abstain_rate": float(g["abstained"].mean()),
        "accuracy_all": float(g["correct4"].mean()),          # best A-D letter, ignoring E
        "accuracy_answered": float(answered["correct4"].mean()) if len(answered) else float("nan"),
        "answered_wrong_rate": float(((g["abstained"] == 0) & (g["correct4"] == 0)).mean()),
        "abstain_precision": float(1 - abst["correct4"].mean()) if len(abst) else float("nan"),
        "mean_p_idk": float(g["p_idk"].mean()),
        "AUROC_idk": auroc(g["p_idk"], 1 - g["correct4"]),  # does P(E) point at wrong answers?
        "mean_letter_mass": float(g["letter_mass"].mean()),
    }


def paired_bootstrap(results, metric_fn, keys, a="eng", b="yor", n_boot=N_BOOT, seed=SEED):
    """95% intervals for (b - a), resampling the same questions in both languages."""
    wide = {l: results[results["lang"] == l].set_index("idx").sort_index() for l in (a, b)}
    common = wide[a].index.intersection(wide[b].index)
    A, B = wide[a].loc[common], wide[b].loc[common]
    rng = np.random.default_rng(seed)
    diffs = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, len(common), len(common))
        ma, mb = metric_fn(A.iloc[pick]), metric_fn(B.iloc[pick])
        for k in keys:
            diffs[k].append(mb[k] - ma[k])
    ma, mb = metric_fn(A), metric_fn(B)
    return {k: {"diff": round(mb[k] - ma[k], 3),
                "ci95": [round(float(np.nanpercentile(diffs[k], 2.5)), 3),
                         round(float(np.nanpercentile(diffs[k], 97.5)), 3)]}
            for k in keys}


def r3(d):
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}


def summarise_verbal(res, name):
    s = {lang: r3(verbal_metrics(g)) for lang, g in res.groupby("lang")}
    s["yor_minus_eng"] = paired_bootstrap(
        res, verbal_metrics, ["parse_rate", "accuracy", "mean_confidence", "overconfidence", "AUROC"])
    # Compare with Iteration 1's hidden confidence on the same questions, if available
    p = os.path.join(ITER1_DIR, slug(name) + ".csv")
    if os.path.exists(p):
        it1 = pd.read_csv(p)[["lang", "idx", "confidence"]].rename(columns={"confidence": "hidden_conf"})
        j = res[res["parsed"] == 1].merge(it1, on=["lang", "idx"])
        for lang, g in j.groupby("lang"):
            rank_corr = g["confidence"].rank().corr(g["hidden_conf"].rank())
            s[lang]["rank_corr_with_hidden_conf"] = round(float(rank_corr), 3)
    for lang in LANGS:
        if s[lang]["parse_rate"] < 0.8:
            print(f"WARNING {name}: only {s[lang]['parse_rate']:.0%} of {lang} replies could be parsed. "
                  "Look at the 'reply' column in the CSV.")
    return s


def summarise_abstain(res, name):
    s = {lang: r3(abstain_metrics(g)) for lang, g in res.groupby("lang")}
    s["yor_minus_eng"] = paired_bootstrap(
        res, abstain_metrics, ["abstain_rate", "accuracy_answered", "answered_wrong_rate", "AUROC_idk"])
    return s


# %% Batched model calls
def run_batched(fn, items, batch_size):
    """Call fn on batches of items; halve the batch size if the GPU runs out of memory."""
    import torch
    from tqdm.auto import tqdm
    out, i = [], 0
    bar = tqdm(total=len(items))
    while i < len(items):
        chunk = items[i:i + batch_size]
        try:
            out.extend(fn(chunk))
            i += len(chunk)
            bar.update(len(chunk))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print("out of memory, batch size ->", batch_size)
    bar.close()
    return out


def find_letter_ids(tok, letters):
    """All single tokens that read as exactly one of the letters (e.g. 'A', ' A', '▁A')."""
    ids = {l: [] for l in letters}
    for token, tid in tok.get_vocab().items():
        clean = token.replace("Ġ", " ").replace("▁", " ").strip()
        if clean in ids:
            ids[clean].append(tid)
    assert all(ids.values()), f"missing letter tokens: {ids}"
    return [sorted(ids[l]) for l in letters]


def run_model(spec, data, conditions=CONDITIONS):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = spec["name"]
    todo = [c for c in conditions
            if not os.path.exists(os.path.join(OUT_DIR, f"{slug(name)}_{c}.csv"))]
    out = {c: pd.read_csv(os.path.join(OUT_DIR, f"{slug(name)}_{c}.csv"))
           for c in conditions if c not in todo}
    if not todo:
        print(f"[{name}] already done, loading")
        return out

    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=getattr(torch, spec.get("dtype", "float16")), device_map="auto")
    except OSError as e:
        print(f"[{name}] SKIPPED: could not load ({str(e).splitlines()[0]})")
        return out
    model.eval()
    tok.padding_side = "left"   # so the last position is the real last token in every row
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    batch_size = spec.get("batch", 16)

    def prompt(condition, question, choices):
        msg = user_message(condition, question, choices)
        return tok.apply_chat_template([{"role": "user", "content": msg}],
                                       tokenize=False, add_generation_prompt=True)

    def encode(prompts):
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        return {k: v.to(model.device) for k, v in enc.items()}

    # Build all items once (both languages), sorted by length to reduce padding
    items = []
    for lang, df in data.items():
        for _, r in df.iterrows():
            ch = parse_choices(r["choices"])
            if len(ch) == 4:
                items.append({"lang": lang, "idx": int(r["idx"]), "subject": r["subject"],
                              "question": r["question"], "choices": ch,
                              "gold": LETTERS.index(str(r["answer"]).strip().upper())})

    try:
        for condition in todo:
            prompts = [prompt(condition, it["question"], it["choices"]) for it in items]
            order = np.argsort([len(p) for p in prompts])[::-1]   # longest first: OOM shows early
            print(f"[{name}] condition {condition}: {len(prompts)} prompts")

            if condition == "verbal":
                @torch.no_grad()
                def fn(idxs):
                    enc = encode([prompts[k] for k in idxs])
                    gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                    new = gen[:, enc["input_ids"].shape[1]:]
                    return tok.batch_decode(new, skip_special_tokens=True)

                replies = run_batched(fn, list(order), batch_size)
                rows = []
                for k, reply in zip(order, replies):
                    it = items[k]
                    pred, conf = parse_verbal(reply)
                    ok = pred is not None and conf is not None
                    rows.append({"model": name, "lang": it["lang"], "idx": it["idx"],
                                 "subject": it["subject"], "gold": it["gold"],
                                 "pred": pred, "confidence": conf, "parsed": int(ok),
                                 "correct": int(ok and pred == it["gold"]), "reply": reply})
            else:
                letter_ids = find_letter_ids(tok, LETTERS5)

                @torch.no_grad()
                def fn(idxs):
                    enc = encode([prompts[k] for k in idxs])
                    pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
                    logits = model(**enc, position_ids=pos).logits[:, -1].float()
                    full = torch.softmax(logits, dim=-1)
                    per = torch.stack([full[:, ids].sum(-1) for ids in letter_ids], dim=-1)
                    return list(zip((per / per.sum(-1, keepdim=True)).cpu().numpy(),
                                    per.sum(-1).cpu().numpy()))

                outs = run_batched(fn, list(order), batch_size)
                rows = []
                for k, (probs, mass) in zip(order, outs):
                    it = items[k]
                    pred4 = int(np.argmax(probs[:4]))
                    rows.append({"model": name, "lang": it["lang"], "idx": it["idx"],
                                 "subject": it["subject"], "gold": it["gold"],
                                 "pred5": int(np.argmax(probs)), "pred4": pred4,
                                 "abstained": int(np.argmax(probs) == 4),
                                 "correct4": int(pred4 == it["gold"]),
                                 "p_idk": float(probs[4]), "letter_mass": float(mass)})
            res = pd.DataFrame(rows).sort_values(["lang", "idx"]).reset_index(drop=True)
            res.to_csv(os.path.join(OUT_DIR, f"{slug(name)}_{condition}.csv"), index=False)
            out[condition] = res
            if condition == "verbal":
                print(res[["lang", "reply"]].head(3).to_string())   # a few raw replies
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return out


# %% Plots
def dumbbell(ax, summaries, key, xlabel, title, xlim=None, ref=None, ref_label=None):
    names = list(summaries)
    ypos = np.arange(len(names))[::-1]
    if ref is not None:
        ax.axvline(ref, color="#bbb", lw=1, ls=":")
        ax.text(ref, ypos.max() + 0.45, f" {ref_label}", color="#888", fontsize=8)
    for yp, n in zip(ypos, names):
        ax.plot([summaries[n]["eng"][key], summaries[n]["yor"][key]], [yp, yp],
                color="#bbb", lw=2, zorder=1)
    for lang in LANGS:
        ax.scatter([summaries[n][lang][key] for n in names], ypos, s=70, color=COLORS[lang],
                   edgecolor="white", linewidth=1.5, zorder=2, label=LANG_NAMES[lang])
    ax.set_yticks(ypos)
    ax.set_yticklabels([short(n) for n in names])
    ax.set_ylim(-0.6, len(names) - 0.2)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.8)


def verbal_plot(summaries, path="verbal.png"):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    dumbbell(axes[0], summaries, "accuracy", "accuracy", "Accuracy", (0, 1), 0.25, "chance")
    dumbbell(axes[1], summaries, "mean_confidence", "mean stated confidence (0-1)",
             "Stated confidence", (0, 1))
    dumbbell(axes[2], summaries, "AUROC", "AUROC of stated confidence",
             "Does stated confidence separate right from wrong?", (0.3, 1), 0.5, "chance")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)
    fig.suptitle("Condition A: the model states its confidence", x=0.01, ha="left", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    return fig


def abstain_plot(summaries, path="abstain.png"):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    dumbbell(axes[0], summaries, "abstain_rate", "share of questions answered 'I don't know'",
             "How often it says 'I don't know'", (0, 1))
    dumbbell(axes[1], summaries, "answered_wrong_rate", "share of all questions answered wrongly",
             "Wrong answers it still gave", (0, 1))
    dumbbell(axes[2], summaries, "AUROC_idk", "AUROC of P('I don't know') for wrong answers",
             "Does 'I don't know' point at its mistakes?", (0.3, 1), 0.5, "chance")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)
    fig.suptitle("Condition B: an 'I don't know' option is offered", x=0.01, ha="left", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    return fig


def tables(verbal_s, abstain_s):
    def ci(d, k):
        return f"{d[k]['diff']:+.3f} [{d[k]['ci95'][0]:+.3f}, {d[k]['ci95'][1]:+.3f}]"
    tv = pd.DataFrame([{
        "model": short(n),
        "parsed eng": s["eng"]["parse_rate"], "parsed yor": s["yor"]["parse_rate"],
        "acc eng": s["eng"]["accuracy"], "acc yor": s["yor"]["accuracy"],
        "stated conf eng": s["eng"]["mean_confidence"], "stated conf yor": s["yor"]["mean_confidence"],
        "stated conf gap [95% CI]": ci(s["yor_minus_eng"], "mean_confidence"),
        "AUROC eng": s["eng"]["AUROC"], "AUROC yor": s["yor"]["AUROC"],
        "most common conf yor (share)": f"{s['yor']['most_common_conf']} ({s['yor']['share_most_common_conf']:.0%})",
        "rank corr w/ hidden conf yor": s["yor"].get("rank_corr_with_hidden_conf", ""),
    } for n, s in verbal_s.items()])
    ta = pd.DataFrame([{
        "model": short(n),
        "IDK rate eng": s["eng"]["abstain_rate"], "IDK rate yor": s["yor"]["abstain_rate"],
        "IDK gap [95% CI]": ci(s["yor_minus_eng"], "abstain_rate"),
        "acc answered eng": s["eng"]["accuracy_answered"], "acc answered yor": s["yor"]["accuracy_answered"],
        "wrong answers eng": s["eng"]["answered_wrong_rate"], "wrong answers yor": s["yor"]["answered_wrong_rate"],
        "AUROC_idk eng": s["eng"]["AUROC_idk"], "AUROC_idk yor": s["yor"]["AUROC_idk"],
        "letter mass yor": s["yor"]["mean_letter_mass"],
    } for n, s in abstain_s.items()])
    return tv, ta


# %% Load data
if __name__ == "__main__":
    data = {lang: sample_per_subject(load_lang(lang), PER_SUBJECT) for lang in LANGS}
    print({lang: len(df) for lang, df in data.items()}, "questions")

# %% Run all models (each model is loaded once and runs both conditions)
if __name__ == "__main__":
    runs = {}
    for spec in MODELS:
        out = run_model(spec, data)
        if out:
            runs[spec["name"]] = out
    print("finished:", list(runs))

# %% Summaries, tables, plots
if __name__ == "__main__":
    verbal_s = {n: summarise_verbal(o["verbal"], n) for n, o in runs.items() if "verbal" in o}
    abstain_s = {n: summarise_abstain(o["abstain"], n) for n, o in runs.items() if "abstain" in o}
    with open("summary_v2.json", "w") as f:
        json.dump({"verbal": verbal_s, "abstain": abstain_s}, f, indent=2)
    tv, ta = tables(verbal_s, abstain_s)
    tv.to_csv("table_verbal.csv", index=False)
    ta.to_csv("table_abstain.csv", index=False)
    pd.set_option("display.width", 250)
    print("CONDITION A - stated confidence\n", tv.to_string(index=False))
    print("\nCONDITION B - 'I don't know' option\n", ta.to_string(index=False))
    verbal_plot(verbal_s)
    abstain_plot(abstain_s)
    plt.show()

# %% Download everything as one zip
if __name__ == "__main__":
    os.system("zip -qr iteration2_outputs.zip results/iteration2 summary_v2.json table_verbal.csv "
              "table_abstain.csv verbal.png abstain.png")
    try:
        from google.colab import files
        files.download("iteration2_outputs.zip")
    except ImportError:
        print("saved iteration2_outputs.zip")
