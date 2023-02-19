# %%
# imports
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
save_dir = Path("extraction_results")
test_on_train = False
UQA = "unifiedqa-t5-11b"
GPTJ = "gpt-j-6b"
Method = Literal["CCS", "LR", "Random"]
uqa_good_datasets = ["imdb", "amazon-polarity", "ag-news", "dbpedia-14", "copa", "boolq", "story-cloze"]
gptj_good_datasets = ["imdb", "amazon-polarity", "ag-news", "dbpedia-14"]

# derived constants
if test_on_train:
    save_dir = save_dir / "test_on_train"
test_on_train_suffix = "\n(On the train set)" if test_on_train else ""
# %%
# Utils


def load_probs(model_name: str, train: str, test: str, method: Method = "CCS"):
    folder = save_dir / f"states_{model_name}_{method}" / train
    pattern = f"{test}*_{method}.csv" if test != "all" else f"*_{method}.csv"
    return pd.concat([pd.read_csv(f) for f in folder.glob(pattern)])


def load_stats(model_name: str, train: str, test: str, method: Method = "CCS"):
    model_short = {
        UQA: "uqa",
        GPTJ: "gptj",
    }[model_name]
    csvs = save_dir.glob(f"{model_short}_good_*.csv")
    dfs = [pd.read_csv(f) for f in csvs]
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


def ccs_loss(p0, p1):
    return np.minimum(p0, p1).mean() + np.square(p0 + p1 - 1).mean()


# %%
# Plot 1.1 distribution of probs on copa
probs = load_probs(UQA, "copa", "copa", "CCS")
p0, p1, labels = probs.values.T
minp = np.minimum(p0, p1)
maxp = np.maximum(p0, p1)
plt.hist(minp, bins=30, alpha=0.5, label="min(p+,p-)", range=(0, 1))
plt.hist(maxp, bins=30, alpha=0.5, label="max(p+,p-)", range=(0, 1))
plt.title(f"Distribution of min(p+,p-) and max(p+,p-)\nCOPA, UQA, CCS{test_on_train_suffix}")
plt.legend()
plt.tight_layout()
# %%
# Plot 1.2 distribution of probs on all datasets
probs = load_probs(UQA, "all", "all", "CCS")
p0, p1, labels = probs.values.T
minp = np.minimum(p0, p1)
maxp = np.maximum(p0, p1)
plt.hist(minp, bins=30, alpha=0.5, label="min(p+,p-)", range=(0, 1))
plt.hist(maxp, bins=30, alpha=0.5, label="max(p+,p-)", range=(0, 1))
plt.title(f"Distribution of min(p+,p-) and max(p+,p-)\nAll datasets, UQA, CCS{test_on_train_suffix}")
plt.legend()
plt.tight_layout()
# %%
# Plot 1.3 barchart of losses with CCS
bar_names = uqa_good_datasets
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
plt.bar(x_pos - width / 2, separate_mean, width, label="train separate", yerr=separate_error)
plt.bar(x_pos + width / 2, together_mean, width, label="train together", yerr=together_error)
plt.xticks(x_pos, bar_names, rotation=45)
plt.axhline(0.2, color="black", label="loss for constant guess")
plt.title(
    f"Losses of UQA on all datasets\ntrained separately vs. trained together\n(2 std error bars over 5 runs){test_on_train_suffix}"
)
plt.legend()
plt.tight_layout()
# %%
# Plot 1.4 barchart of accuracies
bar_names = uqa_good_datasets
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
plt.errorbar(x_pos - 3 * width / 2, lr_mean, label="Supervised", yerr=lr_error, **kwargs)
plt.errorbar(x_pos - width / 2, separate_mean, label="CCS", yerr=separate_error, **kwargs)
plt.errorbar(x_pos + width / 2, together_mean, label="CCS, trained together", yerr=together_error, **kwargs)
plt.errorbar(x_pos + 3 * width / 2, random_mean, label="Random", yerr=random_error, **kwargs)
plt.xticks(x_pos, bar_names, rotation=45)
plt.ylim(0.5, 1)
plt.title(
    f"Accuracies of UQA on all datasets\ntrained separately vs. trained together vs random directions\n(2 std error bars over 5 runs){test_on_train_suffix}"
)
plt.legend()
plt.tight_layout()
# %%
# Plot 1.5 distribution of probs with LR
probs = load_probs(UQA, "all", "all", "LR")
labels, p = probs.values.T
minp = np.minimum(p, 1 - p)
maxp = np.maximum(p, 1 - p)
plt.hist(minp, bins=30, alpha=0.5, label="min(p,1-p)", range=(0, 1))
plt.hist(maxp, bins=30, alpha=0.5, label="max(p,1-p)", range=(0, 1))
plt.title(f"Distribution of min(p+,p-) and max(p+,p-)\nAll datasets, UQA, LR{test_on_train_suffix}")
plt.legend()
plt.tight_layout()
# %%
# Plot 1.6 distribution of probs with Random
probs = load_probs(UQA, "all", "all", "Random")
p0, p1, labels = probs.values.T
minp = np.minimum(p0, p1)
maxp = np.maximum(p0, p1)
plt.hist(minp, bins=30, alpha=0.5, label="min(p+,p-)", range=(0, 1))
plt.hist(maxp, bins=30, alpha=0.5, label="max(p+,p-)", range=(0, 1))
plt.title(f"Distribution of min(p+,p-) and max(p+,p-)\nAll datasets, UQA, CCS{test_on_train_suffix}")
plt.legend()
plt.tight_layout()
# %%
# Plot 2.1 RCCS accuracies across iterations
# %%
# Plot 2.2 RCCS loss across iterations
# %%
# Plot 2.3 RCCS accuracies across iterations (but on copa)
# %%
# Plot 2.4 RCCS loss across iterations (but on copa)
# %%
# Plot 3.1 barchart of losses on GPT-J
bar_names = gptj_good_datasets
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
plt.bar(x_pos - width / 2, separate_mean, width, label="train separate", yerr=separate_error)
plt.bar(x_pos + width / 2, together_mean, width, label="train together", yerr=together_error)
plt.xticks(x_pos, bar_names, rotation=45)
plt.axhline(0.2, color="black", label="loss for constant guess")
plt.title(
    f"Losses of GPTJ on all datasets\ntrained separately vs. trained together\n(2 std error bars over 10 runs){test_on_train_suffix}"
)
plt.legend()
plt.tight_layout()
# %%
# Plot 3.2 barchart of accuracies on GPT-J
bar_names = gptj_good_datasets
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
plt.errorbar(x_pos - 3 * width / 2, lr_mean, label="Supervised", yerr=lr_error, **kwargs)
plt.errorbar(x_pos - width / 2, separate_mean, label="CCS", yerr=separate_error, **kwargs)
plt.errorbar(x_pos + width / 2, together_mean, label="CCS, trained together", yerr=together_error, **kwargs)
plt.errorbar(x_pos + 3 * width / 2, random_mean, label="Random", yerr=random_error, **kwargs)
plt.xticks(x_pos, bar_names, rotation=45)
plt.ylim(0.5, 1)
plt.title(
    f"Accuracies of GPTJ on all datasets\ntrained separately vs. trained together vs random directions\n(2 std error bars over 10 runs){test_on_train_suffix}"
)
plt.legend()
plt.tight_layout()
# %%
# Plot 3.3 scatterplot of accuracies vs. losses on GPT-J
# Not even worth it, GPT-J sucks massively
