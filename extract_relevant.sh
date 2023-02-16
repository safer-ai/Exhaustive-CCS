#!/bin/bash
# Extract UQA on datasets where it is good
for seed in {0..4}; do
    python extraction_main.py --model unifiedqa-t5-11b --datasets imdb amazon-polarity ag-news dbpedia-14 copa boolq story-cloze --method_list CCS --seed $seed
    # copy the csv file so that it doesn't get overwritten
    cp unifiedqa-t5-11b_normal_${seed}.csv uqa_good_${seed}.csv
done