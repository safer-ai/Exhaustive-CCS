#!/bin/bash

methods="CCS LR Random CCS-md LR-md Random-md"
uqa_good_ds="imdb amazon-polarity ag-news dbpedia-14 copa boolq story-cloze"
gptj_good_ds="imdb amazon-polarity ag-news dbpedia-14"

model_names=(unifiedqa-t5-11b gpt-j-6B)
model_to_ds=(uqa_good_ds gptj_good_ds)
model_to_short=(uqa gptj)
test_on_trains=("" "--test_on_train")
test_on_train_extensions=("" "/test_on_train")

for i_test_on_train in 0 1; do
    for i_model in 0 1; do
        model=${model_names[$i_model]}
        ds=${!model_to_ds[$i_model]}
        short=${model_to_short[$i_model]}
        test_on_train=${test_on_trains[$i_test_on_train]}
        test_on_train_extension=${test_on_train_extensions[$i_test_on_train]}
        for seed in {0..9}; do
            # if seed == 0, save states
            save_states=""
            if [ $seed -eq 0 ]; then
                save_states="--save_states" 
            fi

            python extraction_main.py --model $model --datasets $ds --method_list $methods --seed $seed --save_dir extraction_results$test_on_train_extension $save_states $test_on_train
            # copy the csv file so that it doesn't get overwritten
            cp extraction_results${test_on_train_extension}/${model}_normal_${seed}.csv extraction_results${test_on_train_extension}/${short}_good_${seed}.csv
        done
    done
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