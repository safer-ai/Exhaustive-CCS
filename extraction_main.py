from typing import Optional
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import pandas as pd
import numpy as np
import time
from utils_extraction.load_utils import getDic, set_load_dir, get_zeros_acc
from utils_extraction.method_utils import mainResults
from utils_extraction.func_utils import getAvg, adder
import pandas as pd
import random
import json
import argparse

######## JSON Load ########
json_dir = "./registration"

with open("{}.json".format(json_dir), "r") as f:
    global_dict = json.load(f)
registered_dataset_list = global_dict["dataset_list"]
registered_models = global_dict["registered_models"]
registered_prefix = global_dict["registered_prefix"]
models_layer_num = global_dict["models_layer_num"]


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=registered_models)
parser.add_argument("--prefix", nargs="+", default=["normal"], choices=registered_prefix)
parser.add_argument("--datasets", nargs="+", default=registered_dataset_list)
parser.add_argument("--test", type=str, default="testall", choices=["testone", "testall"])
parser.add_argument("--data_num", type=int, default=1000)
parser.add_argument(
    "--method_list",
    nargs="+",
    default=["0-shot", "TPC", "KMeans", "LR", "BSS", "CCS"],
    help=(
        "The name of the method, which should either be in {0-shot, TPC, KMeans, LR, BSS, CCS, Random}\n"
        "or be of the form RCCSi, where i is an integer: to run 10 iteration of RCCS, pass RCCS0, ..., RCCS9 as argument "
        "(it should start by RCCS0). Stats will be saved for each iterations, "
        "and params will be saved for their concatenation under 'RCCS'."
    ),
)
parser.add_argument(
    "--mode", type=str, default="auto", choices=["auto", "minus", "concat"], help="How you combine h^+ and h^-."
)
parser.add_argument("--save_dir", type=str, default="extraction_results", help="where the csv and params are saved")
parser.add_argument("--append", action="store_true", help="Whether to append content in frame rather than rewrite.")
parser.add_argument(
    "--load_dir",
    type=str,
    default="generation_results",
    help="Where the hidden states and zero-shot accuracy are loaded.",
)
parser.add_argument("--location", type=str, default="auto")
parser.add_argument("--layer", type=int, default=-1)
parser.add_argument("--zero", type=str, default="generation_results")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--prompt_save_level", default="all", choices=["single", "all"])
parser.add_argument("--save_states", action="store_true", help="Whether to save the p0, p1, labels.")
parser.add_argument("--test_on_train", action="store_true", help="Whether to test on the train set.")
args = parser.parse_args()

dataset_list = args.datasets
set_load_dir(args.load_dir)
assert args.test != "testone", NotImplementedError(
    "Current extraction program does not support applying method on prompt-specific level."
)

if args.location == "auto":
    args.location = "decoder" if "gpt" in args.model else "encoder"
if args.location == "decoder" and args.layer < 0:
    args.layer += models_layer_num[args.model]

print("-------- args --------")
for key in list(vars(args).keys()):
    print("{}: {}".format(key, vars(args)[key]))
print("-------- args --------")


def methodHasLoss(method):
    return method in ["LR", "BSS", "CCS"] or method.startswith("RCCS")


def saveParams(name, coef, intercept):
    path = os.path.join(args.save_dir, "params")
    np.save(os.path.join(path, "coef_{}.npy".format(name)), coef)
    np.save(os.path.join(path, "intercept_{}.npy".format(name)), intercept)


def saveCsv(csv, prefix, str=""):
    dir = os.path.join(args.save_dir, "{}_{}_{}.csv".format(args.model, prefix, args.seed))
    csv.to_csv(dir, index=False)
    print("{} Saving to {} at {}".format(str, dir, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


if __name__ == "__main__":
    # check the os existence
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    if not os.path.exists(os.path.join(args.save_dir, "params")):
        os.makedirs(os.path.join(args.save_dir, "params"), exist_ok=True)

    # each loop will generate a csv file
    for global_prefix in args.prefix:
        print("---------------- model = {}, prefix = {} ----------------".format(args.model, global_prefix))
        # Set the random seed, in which case the permutation_dict will be the same
        random.seed(args.seed)
        np.random.seed(args.seed)

        # shorten the name
        model = args.model
        data_num = args.data_num

        # Start calculate numbers
        # std is over all prompts within this dataset
        if not args.append:
            csv = pd.DataFrame(
                columns=["model", "prefix", "method", "prompt_level", "train", "test", "accuracy", "std"]
            )
        else:
            dir = os.path.join(args.save_dir, "{}_{}_{}.csv".format(args.model, global_prefix, args.seed))
            csv = pd.read_csv(dir)
            print("Loaded {} at {}".format(dir, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        if "0-shot" in args.method_list:
            # load zero-shot performance
            rawzeros = pd.read_csv(os.path.join(args.load_dir, "{}.csv".format(args.zero)))
            # Get the global zero acc dict (setname, [acc])
            zeros_acc = get_zeros_acc(
                csv_name=args.zero,
                mdl_name=model,
                dataset_list=dataset_list,
                prefix=global_prefix,
            )
            for setname in dataset_list:
                if args.prompt_save_level == "all":
                    csv = adder(
                        csv,
                        model,
                        global_prefix,
                        "0-shot",
                        "",
                        "",
                        setname,
                        np.mean(zeros_acc[setname]),
                        np.std(zeros_acc[setname]),
                        "",
                        "",
                        "",
                    )
                else:  # For each prompt, save one line
                    for idx in range(len(zeros_acc[setname])):
                        csv = adder(
                            csv,
                            model,
                            global_prefix,
                            "0-shot",
                            prompt_level=idx,
                            train="",
                            test=setname,
                            accuracy=zeros_acc[setname][idx],
                            std="",
                            location="",
                            layer="",
                            loss="",
                        )

            saveCsv(csv, global_prefix, "After calculating zeroshot performance.")

        for method in args.method_list:
            if method == "0-shot":
                continue
            print("-------- method = {} --------".format(method))

            method_use_concat = (method in {"CCS", "Random"}) or method.startswith("RCCS")

            mode = args.mode if args.mode != "auto" else ("concat" if method_use_concat else "minus")
            # load the data_dict and permutation_dict
            data_dict, permutation_dict = getDic(
                mdl_name=model,
                dataset_list=dataset_list,
                prefix=global_prefix,
                location=args.location,
                layer=args.layer,
                mode=mode,
            )

            test_dict = {key: range(len(data_dict[key])) for key in dataset_list}

            for train_set in ["all"] + dataset_list:

                train_list = dataset_list if train_set == "all" else [train_set]
                projection_dict = {key: range(len(data_dict[key])) for key in train_list}

                n_components = 1 if method == "TPC" else -1

                # return a dict with the same shape as test_dict
                # for each key test_dict[key] is a unitary list
                save_file_prefix = (
                    f"{args.save_dir}/states_{args.model}_{method}/{train_set}" if args.save_states else None
                )

                method_ = method
                constraints = None
                if method.startswith("RCCS"):
                    method_ = "CCS"
                    params_file_name = "{}_{}_{}_{}_{}_{}".format(
                        model, global_prefix, "RCCS", "all", train_set, args.seed
                    )
                    if method != "RCCS0":
                        constraints = np.load(
                            os.path.join(args.save_dir, "params", "coef_{}.npy".format(params_file_name))
                        )
                        old_biases = np.load(
                            os.path.join(args.save_dir, "params", "intercept_{}.npy".format(params_file_name))
                        )

                res, lss, pmodel, cmodel = mainResults(
                    data_dict=data_dict,
                    permutation_dict=permutation_dict,
                    projection_dict=projection_dict,
                    test_dict=test_dict,
                    n_components=n_components,
                    projection_method="PCA",
                    classification_method=method_,
                    save_file_prefix=save_file_prefix,
                    test_on_train=args.test_on_train,
                    constraints=constraints,
                )

                # save params except for KMeans
                if method in ["TPC", "BSS", "CCS", "Random", "LR"]:
                    if method in ["TPC", "BSS"]:
                        coef, bias = cmodel.coef_ @ pmodel.getDirection(), cmodel.intercept_
                    elif method in ["CCS", "Random"]:
                        coef_and_bias = cmodel.best_theta
                        coef = coef_and_bias[:, :-1]
                        bias = coef_and_bias[:, -1]
                    elif method == "LR":
                        coef, bias = cmodel.coef_, cmodel.intercept_
                    else:
                        assert False
                    saveParams(
                        "{}_{}_{}_{}_{}_{}".format(model, global_prefix, method, "all", train_set, args.seed),
                        coef,
                        bias,
                    )

                if method.startswith("RCCS"):
                    coef_and_bias = cmodel.best_theta
                    coef = coef_and_bias[:, :-1]
                    bias = coef_and_bias[:, -1]
                    if method != "RCCS0":
                        coef = np.concatenate([constraints, coef], axis=0)
                        bias = np.concatenate([old_biases, bias], axis=0)
                    saveParams(params_file_name, coef, bias)

                acc, std, loss = (
                    getAvg(res),
                    np.mean([np.std(lis) for lis in res.values()]),
                    np.mean([np.mean(lis) for lis in lss.values()]),
                )
                print(
                    "method = {:8}, prompt_level = {:8}, train_set = {:20}, avgacc is {:.2f}, std is {:.2f}, loss is {:.4f}".format(
                        method, "all", train_set, 100 * acc, 100 * std, loss
                    )
                )

                for key in dataset_list:
                    if args.prompt_save_level == "all":
                        csv = adder(
                            csv,
                            model,
                            global_prefix,
                            method,
                            "all",
                            train_set,
                            key,
                            accuracy=np.mean(res[key]),
                            std=np.std(res[key]),
                            location=args.location,
                            layer=args.layer,
                            loss=np.mean(lss[key]) if methodHasLoss(method) else "",
                        )
                    else:
                        for idx in range(len(res[key])):
                            csv = adder(
                                csv,
                                model,
                                global_prefix,
                                method,
                                idx,
                                train_set,
                                key,
                                accuracy=res[key][idx],
                                std="",
                                location=args.location,
                                layer=args.layer,
                                loss=lss[key][idx] if methodHasLoss(method) else "",
                            )

        saveCsv(csv, global_prefix, "After finish {}".format(method))
