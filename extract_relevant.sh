#!/bin/bash


# Extract UQA on datasets where it is good
for seed in {1..9}; do
    python extraction_main.py --model unifiedqa-t5-11b --datasets imdb amazon-polarity ag-news dbpedia-14 copa boolq story-cloze --method_list CCS LR Random --seed $seed
    # copy the csv file so that it doesn't get overwritten
    cp extraction_results/unifiedqa-t5-11b_normal_${seed}.csv extraction_results/uqa_good_${seed}.csv
done

# Extract UQA states (once)
python extraction_main.py --model unifiedqa-t5-11b --datasets imdb amazon-polarity ag-news dbpedia-14 copa boolq story-cloze --method_list CCS LR Random --save_states --seed 0
cp extraction_results/unifiedqa-t5-11b_normal_0.csv extraction_results/uqa_good_0.csv

# Extract GPT-J on datasets where it is good
for seed in {0..9}; do
    python extraction_main.py --model gpt-j-6B --datasets imdb amazon-polarity ag-news dbpedia-14 --method_list CCS LR Random --seed $seed
    # copy the csv file so that it doesn't get overwritten
    cp extraction_results/gpt-j-6B_normal_${seed}.csv extraction_results/gptj_good_${seed}.csv
done


# Same but test on train

# Extract UQA on datasets where it is good
for seed in {1..4}; do
    python extraction_main.py --model unifiedqa-t5-11b --datasets imdb amazon-polarity ag-news dbpedia-14 copa boolq story-cloze --method_list CCS LR Random --seed $seed --save_dir extraction_results/train --test_on_train
    # copy the csv file so that it doesn't get overwritten
    cp extraction_results/train/unifiedqa-t5-11b_normal_${seed}.csv extraction_results/train/uqa_good_${seed}.csv
done

# Extract UQA states (once)
python extraction_main.py --model unifiedqa-t5-11b --datasets imdb amazon-polarity ag-news dbpedia-14 copa boolq story-cloze --method_list CCS LR Random --save_states --seed 0 --save_dir extraction_results/train --test_on_train
cp extraction_results/train/unifiedqa-t5-11b_normal_0.csv extraction_results/train/uqa_good_0.csv

# Extract GPT-J on datasets where it is good
for seed in {0..9}; do
    python extraction_main.py --model gpt-j-6B --datasets imdb amazon-polarity ag-news dbpedia-14 --method_list CCS LR Random --seed $seed --save_dir extraction_results/train --test_on_train
    # copy the csv file so that it doesn't get overwritten
    cp extraction_results/train/gpt-j-6B_normal_${seed}.csv extraction_results/train/gptj_good_${seed}.csv
done
