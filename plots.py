# %%
# imports
from itertools import combinations
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Optional

# increase figure size to (8, 8)
from matplotlib import rcParams

rcParams["figure.figsize"] = (8, 8)
# %%
# Constants
SAVE_DIR = Path("extraction_results")
TEST_ON_TRAIN = False
UQA = "unifiedqa-t5-11b"
GPTJ = "gpt-j-6b"
UQA_GOOD_DS = ["imdb", "amazon-polarity", "ag-news", "dbpedia-14", "copa", "boolq", "story-cloze"]
GPTJ_GOOD_DS = ["imdb", "amazon-polarity", "ag-news", "dbpedia-14"]
RCCS_ITERS = 20

TOGETHER_COL = "blue"
SEPARATE_COL = "orange"
RANDOM_COL = "grey"
LR_COL = "green"
MAX_COL = "red"
MIN_COL = "purple"

# derived constants
if TEST_ON_TRAIN:
    SAVE_DIR = SAVE_DIR / "test_on_train"
TEST_ON_DIR_suffix = "\n(On the train set)" if TEST_ON_TRAIN else ""
# %%
# Utils


def load_probs(model_name: str, train: str, test: str, method: str = "CCS"):
    folder = SAVE_DIR / f"states_{model_name}_{method}" / train
    pattern = f"{test}*_{method}.csv" if test != "all" else f"*_{method}.csv"
    return pd.concat([pd.read_csv(f) for f in folder.glob(pattern)])


def load_stats(model_name: str, train: str, test: str, method: str = "CCS"):
    model_short = {
        UQA: "uqa",
        GPTJ: "gptj",
    }[model_name]
    rccs_infix = f"rccs" if method.startswith("RCCS") else ""
    csvs = SAVE_DIR.glob(f"{model_short}_good{rccs_infix}_*.csv")
    dfs = [pd.read_csv(f) for f in csvs]

    if not dfs:
        raise ValueError(f"No csvs found for {model_name}, {train}, {test}, {method}")

    # Filter by train & method
    dfs = [df[(df["train"] == train) & (df["method"] == method)] for df in dfs]

    if test != "all":
        # Filter by test
        dfs = [df[df["test"] == test] for df in dfs]
        assert all(len(df) == 1 for df in dfs)
        return {
            "accuracy": np.array([df["accuracy"].values[0] for df in dfs]),
            "loss": np.array([df["loss"].values[0] for df in dfs]),
        }

    # Average by dataset
    return {
        "accuracy": np.array([df["accuracy"].mean() for df in dfs]),
        "loss": np.array([df["loss"].mean() for df in dfs]),
    }


def load_rccs_stats(model_name: str, train: str, test: str):
    stats_per_it = [load_stats(model_name, train, test, f"RCCS{it}") for it in range(RCCS_ITERS)]
    return {
        "accuracy": np.array([stats["accuracy"] for stats in stats_per_it]),  # (it, seed)
        "loss": np.array([stats["loss"] for stats in stats_per_it]),  # (it, seed)
    }


def ccs_loss(p0, p1):
    return np.minimum(p0, p1).mean() + np.square(p0 + p1 - 1).mean()

def load_params(model_name: str, train: str, method: str = "CCS"):
    coefs_files = (SAVE_DIR / "params").glob(
        f"coef_{model_name}_normal_{method}_all_{train}_*.npy"
    )
    intercepts_files = (SAVE_DIR / "params").glob(
        f"intercept_{model_name}_normal_{method}_all_{train}_*.npy"
    )
    coefs = [np.load(f) for f in coefs_files]
    intercepts = [np.load(f) for f in intercepts_files]

    # Average by dataset
    return {
        "coefs": np.array([coef for coef in coefs]),  # (seed, it, nhid)
        "intercepts": np.array([intercept for intercept in intercepts]),  # (seed, it, nhid)
    }

# %%
# Plot: distribution of probs on copa
probs = load_probs(UQA, "copa", "copa", "CCS")
p0, p1, labels = probs.values.T
minp = np.minimum(p0, p1)
maxp = np.maximum(p0, p1)
plt.hist(minp, bins=30, alpha=0.5, label="min(p+,p-)", range=(0, 1), color=MIN_COL)
plt.hist(maxp, bins=30, alpha=0.5, label="max(p+,p-)", range=(0, 1), color=MAX_COL)
plt.title(f"Distribution of min(p+,p-) and max(p+,p-)\nCOPA, UQA, CCS{TEST_ON_DIR_suffix}")
plt.legend()
plt.tight_layout()
# %%
# Plot: distribution of probs on all datasets
probs = load_probs(UQA, "all", "all", "CCS")
p0, p1, labels = probs.values.T
minp = np.minimum(p0, p1)
maxp = np.maximum(p0, p1)
plt.hist(minp, bins=30, alpha=0.5, label="min(p+,p-)", range=(0, 1), color=MIN_COL)
plt.hist(maxp, bins=30, alpha=0.5, label="max(p+,p-)", range=(0, 1), color=MAX_COL)
plt.title(f"Distribution of min(p+,p-) and max(p+,p-)\nAll datasets, UQA, CCS{TEST_ON_DIR_suffix}")
plt.legend()
plt.tight_layout()
# %%
# Plot: barchart of losses with CCS
bar_names = UQA_GOOD_DS
separate_losses = [load_stats(UQA, d, d, "CCS")["loss"] for d in bar_names]
together_losses = [load_stats(UQA, "all", d, "CCS")["loss"] for d in bar_names]


def get_mean_and_error(losses):
    means = np.mean(losses, axis=1)
    stds = np.std(losses, axis=1)
    return means, 2 * stds


separate_mean, separate_error = get_mean_and_error(separate_losses)
together_mean, together_error = get_mean_and_error(together_losses)

x_pos = np.arange(len(bar_names))
width = 0.35
plt.bar(x_pos - width / 2, separate_mean, width, label="train separate", yerr=separate_error, color=SEPARATE_COL)
plt.bar(x_pos + width / 2, together_mean, width, label="train together", yerr=together_error, color=TOGETHER_COL)
plt.xticks(x_pos, bar_names, rotation=45)
plt.axhline(0.2, color="black", label="loss for constant guess")
plt.axhline(0, color="lightgray")
plt.title(
    f"Losses of UQA on all datasets\ntrained separately vs. trained together\n(2 std error bars over 10 runs){TEST_ON_DIR_suffix}"
)
plt.legend()
plt.tight_layout()
# %%
# Plot: barchart of accuracies
bar_names = UQA_GOOD_DS
lr_accs = [load_stats(UQA, d, d, "LR")["accuracy"] for d in bar_names]
separate_accs = [load_stats(UQA, d, d, "CCS")["accuracy"] for d in bar_names]
together_accs = [load_stats(UQA, "all", d, "CCS")["accuracy"] for d in bar_names]
random_accs = [load_stats(UQA, d, d, "Random")["accuracy"] for d in bar_names]


def get_mean_and_error(accs):
    means = np.mean(accs, axis=1)
    stds = np.std(accs, axis=1)
    return means, 2 * stds


lr_mean, lr_error = get_mean_and_error(lr_accs)
separate_mean, separate_error = get_mean_and_error(separate_accs)
together_mean, together_error = get_mean_and_error(together_accs)
random_mean, random_error = get_mean_and_error(random_accs)

x_pos = np.arange(len(bar_names))
width = 0.2
kwargs = {
    "capsize": 3,
    "marker": ".",
    "linestyle": "none",
}
plt.errorbar(x_pos - 3 * width / 2, lr_mean, label="Supervised", yerr=lr_error, color=LR_COL, **kwargs)
plt.errorbar(x_pos - width / 2, separate_mean, label="CCS", yerr=separate_error, color=SEPARATE_COL, **kwargs)
plt.errorbar(
    x_pos + width / 2, together_mean, label="CCS, trained together", yerr=together_error, color=TOGETHER_COL, **kwargs
)
plt.errorbar(x_pos + 3 * width / 2, random_mean, label="Random", yerr=random_error, color=RANDOM_COL, **kwargs)
plt.xticks(x_pos, bar_names, rotation=45)
plt.ylim(0.5, 1)
plt.title(
    f"Accuracies of UQA on all datasets\ntrained separately vs. trained together vs random directions\n(2 std error bars over 10 runs){TEST_ON_DIR_suffix}"
)
plt.legend()
plt.tight_layout()
# %%
# Plot: distribution of probs with LR
probs = load_probs(UQA, "all", "all", "LR")
labels, p = probs.values.T
minp = np.minimum(p, 1 - p)
maxp = np.maximum(p, 1 - p)
plt.hist(minp, bins=30, alpha=0.5, label="min(p,1-p)", range=(0, 1), color=MIN_COL)
plt.hist(maxp, bins=30, alpha=0.5, label="max(p,1-p)", range=(0, 1), color=MAX_COL)
plt.title(f"Distribution of min(p+,p-) and max(p+,p-)\nAll datasets, UQA, LR{TEST_ON_DIR_suffix}")
plt.legend()
plt.tight_layout()
# %%
# Plot: distribution of probs with Random
probs = load_probs(UQA, "all", "all", "Random")
p0, p1, labels = probs.values.T
minp = np.minimum(p0, p1)
maxp = np.maximum(p0, p1)
plt.hist(minp, bins=30, alpha=0.5, label="min(p+,p-)", range=(0, 1), color=MIN_COL)
plt.hist(maxp, bins=30, alpha=0.5, label="max(p+,p-)", range=(0, 1), color=MAX_COL)
plt.title(f"Distribution of min(p+,p-) and max(p+,p-)\nAll datasets, UQA, CCS{TEST_ON_DIR_suffix}")
plt.legend()
plt.tight_layout()
# %%
# Plot: RCCS accuracies across iterations
def plot_rccs_accs(ds):
    if TEST_ON_TRAIN:
        return
    
    separate_accs = load_rccs_stats(UQA, ds, ds)["accuracy"]
    together_accs = load_rccs_stats(UQA, "all", ds)["accuracy"]

    rdm_accs = load_stats(UQA, ds, ds, "Random")["accuracy"]
    rdm_mean, rdm_std = np.mean(rdm_accs), np.std(rdm_accs)

    for accs, col, name in (
        [separate_accs, SEPARATE_COL, "separate"], [together_accs, TOGETHER_COL, "together"]
    ):
        if ds == "all" and name == "separate":
            continue
        
        mean_accs = np.mean(accs, axis=1)
        for i, acc in enumerate(accs.T):
            plt.plot(acc, color=col, alpha=0.2)
        plt.plot(mean_accs, color=col, label=f"CCS mean accuracy {name}")
    plt.axhspan(
        rdm_mean - 2 * rdm_std, rdm_mean + 2 * rdm_std, color=RANDOM_COL, alpha=0.2, label="random accuracy (2 std)"
    )
    plt.ylim(0.5, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration of RCCS")
    plt.xticks(np.arange(0, RCCS_ITERS, 4))
    plt.title(f"Accuracies of UQA on {ds}\ntrained together\n(10 runs){TEST_ON_DIR_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()
plot_rccs_accs("all")
# %%
# Plot: RCCS loss across iterations
def plot_rccs_loss(ds):
    if TEST_ON_TRAIN:
        return
    separate_losses = load_rccs_stats(UQA, ds, ds)["loss"]
    together_losses = load_rccs_stats(UQA, "all", ds)["loss"]
    
    for losses, col, name in ([separate_losses, SEPARATE_COL, "separate"], [together_losses, TOGETHER_COL, "together"]):
        if ds == "all" and name == "separate":
            continue
        mean_losses = np.mean(losses, axis=1)
        for i, loss in enumerate(losses.T):
            plt.plot(loss, color=col, alpha=0.2)
        plt.plot(mean_losses, color=col, label=f"CCS mean loss {name}")
        
    plt.axhline(0.2, color="black", linestyle="--", label="loss for constant guess")
    plt.axhline(0, color="lightgray")
    plt.ylabel("Loss")
    plt.xlabel("Iteration of RCCS")
    plt.title(f"Loss of UQA on {ds}\ntrained together\n(10 runs){TEST_ON_DIR_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()
plot_rccs_loss("all")
# %%
# Plot: RCCS accuracies across iterations (but on copa)
plot_rccs_accs("copa")
plot_rccs_accs("imdb")
# %%
# Plot: RCCS loss across iterations (but on copa)
plot_rccs_loss("copa")
plot_rccs_loss("imdb")
# %%
# Plot: barchart of losses on GPT-J
bar_names = GPTJ_GOOD_DS
separate_losses = [load_stats(GPTJ, d, d, "CCS")["loss"] for d in bar_names]
together_losses = [load_stats(GPTJ, "all", d, "CCS")["loss"] for d in bar_names]


def get_mean_and_error(losses):
    means = np.mean(losses, axis=1)
    stds = np.std(losses, axis=1)
    return means, 2 * stds


separate_mean, separate_error = get_mean_and_error(separate_losses)
together_mean, together_error = get_mean_and_error(together_losses)

x_pos = np.arange(len(bar_names))
width = 0.35
plt.bar(x_pos - width / 2, separate_mean, width, label="train separate", yerr=separate_error, color=SEPARATE_COL)
plt.bar(x_pos + width / 2, together_mean, width, label="train together", yerr=together_error, color=TOGETHER_COL)
plt.xticks(x_pos, bar_names, rotation=45)
plt.axhline(0.2, color="black", linestyle="--", label="loss for constant guess")
plt.axhline(0, color="lightgray")
plt.title(
    f"Losses of GPTJ on all datasets\ntrained separately vs. trained together\n(2 std error bars over 10 runs){TEST_ON_DIR_suffix}"
)
plt.legend()
plt.tight_layout()
# %%
# Plot: barchart of accuracies on GPT-J
bar_names = GPTJ_GOOD_DS
lr_accs = [load_stats(GPTJ, d, d, "LR")["accuracy"] for d in bar_names]
separate_accs = [load_stats(GPTJ, d, d, "CCS")["accuracy"] for d in bar_names]
together_accs = [load_stats(GPTJ, "all", d, "CCS")["accuracy"] for d in bar_names]
random_accs = [load_stats(GPTJ, d, d, "Random")["accuracy"] for d in bar_names]


def get_mean_and_error(accs):
    means = np.mean(accs, axis=1)
    stds = np.std(accs, axis=1)
    return means, 2 * stds


lr_mean, lr_error = get_mean_and_error(lr_accs)
separate_mean, separate_error = get_mean_and_error(separate_accs)
together_mean, together_error = get_mean_and_error(together_accs)
random_mean, random_error = get_mean_and_error(random_accs)

x_pos = np.arange(len(bar_names))
width = 0.2
kwargs = {
    "capsize": 3,
    "marker": ".",
    "linestyle": "none",
}
plt.errorbar(x_pos - 3 * width / 2, lr_mean, label="Supervised", yerr=lr_error, color=LR_COL, **kwargs)
plt.errorbar(x_pos - width / 2, separate_mean, label="CCS", yerr=separate_error, color=SEPARATE_COL, **kwargs)
plt.errorbar(
    x_pos + width / 2, together_mean, label="CCS, trained together", yerr=together_error, color=TOGETHER_COL, **kwargs
)
plt.errorbar(x_pos + 3 * width / 2, random_mean, label="Random", yerr=random_error, color=RANDOM_COL, **kwargs)
plt.xticks(x_pos, bar_names, rotation=45)
plt.ylim(0.5, 1)
plt.title(
    f"Accuracies of GPTJ on all datasets\ntrained separately vs. trained together vs random directions\n(2 std error bars over 10 runs){TEST_ON_DIR_suffix}"
)
plt.legend()
plt.tight_layout()
# %%
# Plot: histogram of cosines similarity between directions (all)
params = load_params(UQA, "all", "CCS")["coefs"].squeeze(1)
cosines = np.array([
    abs(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))) for p1, p2 in combinations(params, 2)
])
plt.hist(cosines, bins=20)
plt.xlabel("Cosine similarity")
plt.xlim(0, 1)
plt.title(f"Pairwise cosine similarities between directions on different seeds\nUQA, all datasets{TEST_ON_DIR_suffix}")
# %%
# Plot: histogram of cosines similarity between directions (copa)
params = load_params(UQA, "copa", "CCS")["coefs"].squeeze(1)
cosines = np.array([
    abs(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))) for p1, p2 in combinations(params, 2)
])
plt.hist(cosines, bins=20)
plt.xlabel("Cosine similarity")
plt.xlim(0, 1)
plt.title(f"Pairwise cosine similarities between directions on different seeds\nUQA, copa{TEST_ON_DIR_suffix}")
# %%
# Check cosine similarity of RCCS directions
params = load_params(UQA, "all", "RCCS")["coefs"]
all_cosines = []
for coef_set in params:
    cosines =[
        abs(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))) for p1, p2 in combinations(coef_set, 2)
    ]
    all_cosines += cosines
plt.hist(all_cosines, bins=20)
plt.xlabel("Cosine similarity")
plt.title(f"Pairwise cosine similarities between directions on iterations seeds\nUQA, all datasets{TEST_ON_DIR_suffix}")
# Note: 1e_7, everything is fine

# %%
# Plot: barchart of accuracies with mean difference projection
bar_names = UQA_GOOD_DS
lr_accs = [load_stats(UQA, d, d, "LR-md")["accuracy"] for d in bar_names]
separate_accs = [load_stats(UQA, d, d, "CCS-md")["accuracy"] for d in bar_names]
together_accs = [load_stats(UQA, "all", d, "CCS-md")["accuracy"] for d in bar_names]
random_accs = [load_stats(UQA, d, d, "Random-md")["accuracy"] for d in bar_names]


def get_mean_and_error(accs):
    means = np.mean(accs, axis=1)
    stds = np.std(accs, axis=1)
    return means, 2 * stds


lr_mean, lr_error = get_mean_and_error(lr_accs)
separate_mean, separate_error = get_mean_and_error(separate_accs)
together_mean, together_error = get_mean_and_error(together_accs)
random_mean, random_error = get_mean_and_error(random_accs)

x_pos = np.arange(len(bar_names))
width = 0.2
kwargs = {
    "capsize": 3,
    "marker": ".",
    "linestyle": "none",
}
plt.errorbar(x_pos - 3 * width / 2, lr_mean, label="Supervised", yerr=lr_error, color=LR_COL, **kwargs)
plt.errorbar(x_pos - width / 2, separate_mean, label="CCS", yerr=separate_error, color=SEPARATE_COL, **kwargs)
plt.errorbar(
    x_pos + width / 2, together_mean, label="CCS, trained together", yerr=together_error, color=TOGETHER_COL, **kwargs
)
plt.errorbar(x_pos + 3 * width / 2, random_mean, label="Random", yerr=random_error, color=RANDOM_COL, **kwargs)
plt.xticks(x_pos, bar_names, rotation=45)
plt.ylim(0.5, 1)
plt.title(
    f"Accuracies of UQA on all datasets after mean difference projection\ntrained separately vs. trained together vs random directions\n(2 std error bars over 10 runs){TEST_ON_DIR_suffix}"
)
plt.legend()
plt.tight_layout()
#%%
# Plot: barchart of losses with CCS with mean difference projection

bar_names = UQA_GOOD_DS
separate_losses = [load_stats(UQA, d, d, "CCS-md")["loss"] for d in bar_names]
together_losses = [load_stats(UQA, "all", d, "CCS-md")["loss"] for d in bar_names]


def get_mean_and_error(losses):
    means = np.mean(losses, axis=1)
    stds = np.std(losses, axis=1)
    return means, 2 * stds


separate_mean, separate_error = get_mean_and_error(separate_losses)
together_mean, together_error = get_mean_and_error(together_losses)

x_pos = np.arange(len(bar_names))
width = 0.35
plt.bar(x_pos - width / 2, separate_mean, width, label="train separate", yerr=separate_error, color=SEPARATE_COL)
plt.bar(x_pos + width / 2, together_mean, width, label="train together", yerr=together_error, color=TOGETHER_COL)
plt.xticks(x_pos, bar_names, rotation=45)
plt.axhline(0.2, color="black", label="loss for constant guess")
plt.axhline(0, color="lightgray")
plt.title(
    f"Losses of UQA on all datasets\ntrained separately vs. trained together\n(2 std error bars over 10 runs){TEST_ON_DIR_suffix}"
)
plt.legend()
plt.tight_layout()
# %%
