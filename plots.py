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
plt.style.use('ggplot')
# %%
# Constants
SAVE_DIR = Path("extraction_results")
TEST_ON_TRAIN = False
UQA = "unifiedqa-t5-11b"
GPTJ = "gpt-j-6B"
UQA_GOOD_DS = ["imdb", "amazon-polarity", "ag-news", "dbpedia-14", "copa", "boolq", "story-cloze"]
GPTJ_GOOD_DS = ["imdb", "amazon-polarity", "ag-news", "dbpedia-14"]
MODEL_GOOD_DS = {UQA: UQA_GOOD_DS, GPTJ: GPTJ_GOOD_DS}
RCCS_ITERS = 20
SHORT_NAMES = {UQA: "UQA", GPTJ: "GPT-J"}

TOGETHER_COL = "blue"
SEPARATE_COL = "orange"
RANDOM_COL = "grey"
LR_COL = "green"
MAX_COL = "red"
MIN_COL = "purple"

# derived constants
if TEST_ON_TRAIN:
    SAVE_DIR = SAVE_DIR / "test_on_train"
TEST_ON_TRAIN_suffix = "\n(On the train set)" if TEST_ON_TRAIN else ""
# %%
# Utils


def load_probs(model_name: str, train: str, test: str, method: str = "CCS", save_dir: Optional[Path] = None):
    save_dir = save_dir or SAVE_DIR
    
    dir = (save_dir / "rccs") if method.startswith("RCCS") else save_dir
    folder = dir / f"states_{model_name}_{method}" / train
    pattern = f"{test}*_{method}.csv" if test != "all" else f"*_{method}.csv"
    return pd.concat([pd.read_csv(f) for f in folder.glob(pattern)])


def load_stats(model_name: str, train: str, test: str, method: str = "CCS", prefix: str = "normal"):
    dir = (SAVE_DIR / "rccs") if method.startswith("RCCS") else SAVE_DIR
    csvs = dir.glob(f"{model_name}_{prefix}_*.csv")
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
            "cons_loss": np.array([df["cons_loss"].values[0] for df in dfs]),
            "sim_loss": np.array([df["sim_loss"].values[0] for df in dfs]),
        }

    # Average by dataset
    return {
        "accuracy": np.array([df["accuracy"].mean() for df in dfs]),
        "loss": np.array([df["loss"].mean() for df in dfs]),
        "cons_loss": np.array([df["cons_loss"].mean() for df in dfs]),
        "sim_loss": np.array([df["sim_loss"].mean() for df in dfs]),
    }


def load_rccs_stats(model_name: str, train: str, test: str):
    stats_per_it = [load_stats(model_name, train, test, f"RCCS{it}") for it in range(RCCS_ITERS)]
    return {
        "accuracy": np.array([stats["accuracy"] for stats in stats_per_it]),  # (it, seed)
        "loss": np.array([stats["loss"] for stats in stats_per_it]),  # (it, seed)
        "cons_loss": np.array([stats["cons_loss"] for stats in stats_per_it]),  # (it, seed)
        "sim_loss": np.array([stats["sim_loss"] for stats in stats_per_it]),  # (it, seed)
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

def boxplots(positions, data, label, color, width=0.5, obj=plt):
    return [(obj.boxplot(data, positions=positions, notch=False, patch_artist=True, showfliers=False,
            boxprops=dict(facecolor="none", color=color),
            capprops=dict(color=color),
            whiskerprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color),
            medianprops=dict(color=color),
            widths=[width]*len(positions),
        )["boxes"][0], label)]

#%%
# CCS is able to find a single direction which correctly classifies statements across datasets
for model in [UQA, GPTJ]:
    bar_names = MODEL_GOOD_DS[model]
    lr_accs = [load_stats(model, d, d, "LR")["accuracy"] for d in bar_names]
    separate_accs = [load_stats(model, d, d, "CCS")["accuracy"] for d in bar_names]
    together_accs = [load_stats(model, "all", d, "CCS")["accuracy"] for d in bar_names]

    x_pos = np.arange(len(bar_names))
    width = 0.2
    legend_info = []
    legend_info += boxplots(x_pos - width, lr_accs, "Logistic Regression (Supervised)", LR_COL, width=width)
    legend_info += boxplots(x_pos, separate_accs, "CCS", SEPARATE_COL, width=width)
    legend_info += boxplots(x_pos + width, together_accs, "CCS, trained together", TOGETHER_COL, width=width)
    plt.legend([l[0] for l in legend_info], [l[1] for l in legend_info])
    plt.xticks(x_pos, bar_names, rotation=45)
    plt.ylim(0.5, 1)
    plt.title(
        f"Accuracies of {SHORT_NAMES[model]} on all datasets\nCCS on each dataset vs. trained together\n(2 std error bars over 10 runs){TEST_ON_TRAIN_suffix}"
    )
    plt.tight_layout()
    plt.show()
#%%
# CCS does so better than random, but not by a huge margin
for model in [UQA, GPTJ]:
    bar_names = MODEL_GOOD_DS[model]
    lr_accs = [load_stats(model, d, d, "LR")["accuracy"] for d in bar_names]
    separate_accs = [load_stats(model, d, d, "CCS")["accuracy"] for d in bar_names]
    together_accs = [load_stats(model, "all", d, "CCS")["accuracy"] for d in bar_names]
    random_accs = [load_stats(model, "all", d, "Random")["accuracy"] for d in bar_names]

    x_pos = np.arange(len(bar_names))
    width = 0.2
    
    legend_info = []
    legend_info += boxplots(x_pos - 3*width/2, lr_accs, "Logistic Regression (Supervised)", LR_COL, width=width)
    legend_info += boxplots(x_pos - width/2, separate_accs, "CCS", SEPARATE_COL, width=width)
    legend_info += boxplots(x_pos + width/2, together_accs, "CCS, trained together", TOGETHER_COL, width=width)
    legend_info += boxplots(x_pos + 3*width/2, random_accs, "Random", RANDOM_COL, width=width)
    plt.legend([l[0] for l in legend_info], [l[1] for l in legend_info])
    plt.xticks(x_pos, bar_names, rotation=45)
    plt.ylim(0.5, 1)
    plt.title(
        f"Accuracies of {SHORT_NAMES[model]} on all datasets\nCCS on each dataset vs. trained together vs. Random\n(over 10 runs){TEST_ON_TRAIN_suffix}"
    )
    plt.tight_layout()
    plt.show()

# %%
# CCS does not find the single direction with high accuracy

def plot_rccs_accs(ds, model):
    if TEST_ON_TRAIN:
        return
    
    separate_accs = load_rccs_stats(model, ds, ds)["accuracy"]
    together_accs = load_rccs_stats(model, "all", ds)["accuracy"]

    rdm_accs = load_stats(model, ds, ds, "Random")["accuracy"]
    rdm_mean, rdm_std = np.mean(rdm_accs), np.std(rdm_accs)

    for accs, col, name in (
        [separate_accs, SEPARATE_COL, ""], [together_accs, TOGETHER_COL, " trained together"]
    ):
        if ds == "all" and col == SEPARATE_COL:
            continue
        
        mean_accs = np.mean(accs, axis=1)
        for i, acc in enumerate(accs.T):
            plt.plot(acc, color=col, alpha=0.2)
        plt.plot(mean_accs, color=col, label=f"CCS{name}, mean accuracy")
    plt.axhspan(
        rdm_mean - 2 * rdm_std, rdm_mean + 2 * rdm_std, color=RANDOM_COL, alpha=0.2, label="random accuracy (2 std)"
    )
    plt.ylim(0.5, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration of RCCS")
    plt.xticks(np.arange(0, RCCS_ITERS, 4))
    ds = "all datasets" if ds == "all" else ds
    plt.title(f"Accuracies of {SHORT_NAMES[model]} on {ds}\n(5 runs){TEST_ON_TRAIN_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()

for ds in ["all", "imdb"]:
    for model in [UQA, GPTJ]:
        plot_rccs_accs(ds, model)

# %%
# Given that there are many “good” directions, does CCS always find roughly the same one?

for dataset in ["all", "imdb"]:
    for model in [UQA, GPTJ]:
        params = load_params(model, dataset, "CCS")["coefs"].squeeze(1)
        cosines = np.array([
            abs(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))) for p1, p2 in combinations(params, 2)
        ])
        plt.hist(cosines, bins=30, range=(0, 1))
        plt.xlabel("Cosine similarity")
        plt.xlim(0, 1)
        plt.title(f"Pairwise cosine similarities between directions on different seeds\n{SHORT_NAMES[model]}, probes trained on {'all datasets (together)' if dataset == 'all' else dataset}{TEST_ON_TRAIN_suffix}")
        plt.show()
# %%
# ablating along the difference of the means direction makes both CCS & Supervised learning fail

for model in [UQA, GPTJ]:
    bar_names = MODEL_GOOD_DS[model]
    lr_accs = [load_stats(model, d, d, "LR-md")["accuracy"] for d in bar_names]
    separate_accs = [load_stats(model, d, d, "CCS-md")["accuracy"] for d in bar_names]
    together_accs = [load_stats(model, "all", d, "CCS-md")["accuracy"] for d in bar_names]
    random_accs = [load_stats(model, "all", d, "Random-md")["accuracy"] for d in bar_names]

    x_pos = np.arange(len(bar_names))
    width = 0.2
    
    legend_info = []
    legend_info += boxplots(x_pos - 3*width/2, lr_accs, "Logistic Regression (Supervised)", LR_COL, width=width)
    legend_info += boxplots(x_pos - width/2, separate_accs, "CCS", SEPARATE_COL, width=width)
    legend_info += boxplots(x_pos + width/2, together_accs, "CCS, trained together", TOGETHER_COL, width=width)
    legend_info += boxplots(x_pos + 3*width/2, random_accs, "Random", RANDOM_COL, width=width)
    plt.legend([l[0] for l in legend_info], [l[1] for l in legend_info])
    plt.xticks(x_pos, bar_names, rotation=45)
    plt.ylim(0.5, 1)
    plt.title(
        f"Accuracies of {SHORT_NAMES[model]} on all datasets\nCCS on each dataset vs. trained together vs. Random\n(over 10 runs){TEST_ON_TRAIN_suffix}"
    )
    plt.tight_layout()
    plt.show()

# %%

def plot_stacked_bar(positions, sim_loss, cons_loss, label, col, width):
    sim_means = np.mean(sim_loss, axis=1)
    cons_means = np.mean(cons_loss, axis=1)
    total_loss_std = np.std(np.array(sim_loss) + np.array(cons_loss), axis=1)
    plt.bar(positions, sim_means, width=width, color=col, label=f"{label}, confidence loss", hatch="O")
    plt.bar(positions, cons_means, width=width, color=col, bottom=sim_means, label=f"{label}, similarity loss", hatch=".",
            yerr=total_loss_std)
    
for model in [UQA, GPTJ]:
    bar_names = MODEL_GOOD_DS[model]
    separate_sim_losses = [load_stats(model, d, d, "CCS")["sim_loss"] for d in bar_names]
    together_sim_losses = [load_stats(model, "all", d, "CCS")["sim_loss"] for d in bar_names]
    separate_cons_losses = [load_stats(model, d, d, "CCS")["cons_loss"] for d in bar_names]
    together_cons_losses = [load_stats(model, "all", d, "CCS")["cons_loss"] for d in bar_names]
    
    
    x_pos = np.arange(len(bar_names))
    width = 0.25
    plot_stacked_bar(x_pos - width / 2, separate_sim_losses, separate_cons_losses, "CCS", SEPARATE_COL, width)
    plot_stacked_bar(x_pos + width / 2, together_sim_losses, together_cons_losses, "CCS, trained together", TOGETHER_COL, width)
    
    plt.xticks(x_pos, bar_names, rotation=45)
    plt.axhline(0.2, color="black", label="loss for constant guess")
    plt.axhline(0, color="lightgray")
    plt.title(
        f"Losses of {SHORT_NAMES[model]} on all datasets\nCCS on each dataset vs. trained together\n(over 10 runs){TEST_ON_TRAIN_suffix}"
    )
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

# %%
# My reproduction of Collin’s plots

for test_on_train in [True, False]:
    save_dir = SAVE_DIR / "test_on_train" if test_on_train else SAVE_DIR
    test_on_train_suffix = "\nTested on the train set" if test_on_train else "\nTested on the test set"
    probs = load_probs(UQA, "copa", "copa", "CCS", save_dir=save_dir)
    p0, p1, labels = probs.values.T
    p = (p1 + (1 - p0)) / 2
    true = p[labels == 1]
    false = p[labels == 0]
    plt.hist(true, bins=30, alpha=0.5, label="True", range=(0, 1))
    plt.hist(false, bins=30, alpha=0.5, label="False", range=(0, 1))
    plt.title(f"$p = (p^+ + 1 - p^-)/2$\nCOPA, UQA, CCS{test_on_train_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
# CCS does not find truth when the model doesn’t try to predict a truth-relevant output

prefixes = ["normal", "normal-dot", "normal-thatsright", "normal-mark"]
prefix_explanations = {
    "normal": "No suffix",
    "normal-dot": "{Q&A}.",
    "normal-thatsright": "{Q&A}That's right!",
    "normal-mark": "{Q&A}Mark for this question:"
}

layer_to_save_dir = {
    "": SAVE_DIR,
    "4th layer before last\n": SAVE_DIR / "layer-5",
}

for layer_name, save_dir in layer_to_save_dir.items():

    for model in [UQA, GPTJ]:
        fig, axs = plt.subplots(len(prefixes), 1, figsize=(8, 16), sharex=True)
        
        for i,prefix in enumerate(["normal", "normal-dot", "normal-thatsright", "normal-mark"]):
            bar_names = MODEL_GOOD_DS[model]
            lr_accs = [load_stats(model, d, d, "LR", prefix=prefix)["accuracy"] for d in bar_names]
            separate_accs = [load_stats(model, d, d, "CCS", prefix=prefix)["accuracy"] for d in bar_names]
            together_accs = [load_stats(model, "all", d, "CCS", prefix=prefix)["accuracy"] for d in bar_names]
            random_accs = [load_stats(model, "all", d, "Random", prefix=prefix)["accuracy"] for d in bar_names]

            x_pos = np.arange(len(bar_names))
            width = 0.2
            
            legend_info = []
            legend_info += boxplots(x_pos - 3*width/2, lr_accs, "Logistic Regression (Supervised)", LR_COL, width=width, obj=axs[i])
            legend_info += boxplots(x_pos - width/2, separate_accs, "CCS", SEPARATE_COL, width=width, obj=axs[i])
            legend_info += boxplots(x_pos + width/2, together_accs, "CCS, trained together", TOGETHER_COL, width=width, obj=axs[i])
            legend_info += boxplots(x_pos + 3*width/2, random_accs, "Random", RANDOM_COL, width=width, obj=axs[i])
            axs[i].legend([l[0] for l in legend_info], [l[1] for l in legend_info])
            axs[i].set_xticks(x_pos)
            axs[i].set_xticklabels(bar_names, rotation=45)
            axs[i].set_ylim(0.5, 1)
            axs[i].set_title(prefix_explanations[prefix])
        plt.suptitle(
            f"Accuracies of {SHORT_NAMES[model]} on all datasets\nCCS on each dataset vs. trained together vs. Random\n{layer_name}(over 10 runs){TEST_ON_TRAIN_suffix}"
        )
        fig.tight_layout()
        fig.show()