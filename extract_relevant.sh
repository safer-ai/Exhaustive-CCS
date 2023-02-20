#!/bin/bash

methods="CCS LR Random CCS-md LR-md Random-md"
uqa_good_ds="imdb amazon-polarity ag-news dbpedia-14 copa boolq story-cloze"
gptj_good_ds="imdb amazon-polarity ag-news dbpedia-14"

# Extract UQA states (once)
python extraction_main.py --model unifiedqa-t5-11b --datasets $uqa_good_ds --method_list $methods  --save_states --seed 0
cp extraction_results/unifiedqa-t5-11b_normal_0.csv extraction_results/uqa_good_0.csv

# Extract UQA on datasets where it is good
for seed in {1..9}; do
    python extraction_main.py --model unifiedqa-t5-11b --datasets $uqa_good_ds --method_list $methods --seed $seed
    # copy the csv file so that it doesn't get overwritten
    cp extraction_results/unifiedqa-t5-11b_normal_${seed}.csv extraction_results/uqa_good_${seed}.csv
done

# Extract GPT-J on datasets where it is good
for seed in {0..9}; do
    python extraction_main.py --model gpt-j-6B --datasets $gptj_good_ds --method_list $methods --seed $seed
    # copy the csv file so that it doesn't get overwritten
    cp extraction_results/gpt-j-6B_normal_${seed}.csv extraction_results/gptj_good_${seed}.csv
done


# Same but test on train

# Extract UQA states (once)
python extraction_main.py --model unifiedqa-t5-11b --datasets $uqa_good_ds --method_list $methods --save_states --seed 0 --save_dir extraction_results/train --test_on_train
cp extraction_results/test_on_train/unifiedqa-t5-11b_normal_0.csv extraction_results/test_on_train/uqa_good_0.csv

# Extract UQA on datasets where it is good
for seed in {1..4}; do
    python extraction_main.py --model unifiedqa-t5-11b --datasets $uqa_good_ds --method_list $methods --seed $seed --save_dir extraction_results/train --test_on_train
    # copy the csv file so that it doesn't get overwritten
    cp extraction_results/test_on_train/unifiedqa-t5-11b_normal_${seed}.csv extraction_results/test_on_train/uqa_good_${seed}.csv
done

# Extract GPT-J on datasets where it is good
for seed in {0..9}; do
    python extraction_main.py --model gpt-j-6B --datasets $gptj_good_ds --method_list $methods --seed $seed --save_dir extraction_results/train --test_on_train
    # copy the csv file so that it doesn't get overwritten
    cp extraction_results/test_on_train/gpt-j-6B_normal_${seed}.csv extraction_results/test_on_train/gptj_good_${seed}.csv
done

# RRCS on UQA

RCCS_STRING=$(printf "RCCS%s " $(seq 0 19))

# Extract UQA states (once)
python extraction_main.py --model unifiedqa-t5-11b --datasets $uqa_good_ds --method_list $RCCS_STRING  --save_states --seed 0
cp extraction_results/unifiedqa-t5-11b_normal_0.csv extraction_results/uqa_goodrccs_0.csv

# Extract UQA on datasets where it is good
for seed in {1..9}; do
    python extraction_main.py --model unifiedqa-t5-11b --datasets $uqa_good_ds --method_list $RCCS_STRING --seed $seed
    # copy the csv file so that it doesn't get overwritten
    cp extraction_results/unifiedqa-t5-11b_normal_${seed}.csv extraction_results/uqa_goodrccs_${seed}.csv
done