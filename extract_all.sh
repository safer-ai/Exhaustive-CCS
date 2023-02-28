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
        done
    done
done

# Test different prefixes

methods="CCS LR Random"
prefixes=("normal-dot" "normal-thatsright" "normal-mark")
layers=(-1 -5 -9)
save_dir_per_layer=("", "layer-5", "layer-9")
for i_layer in 1 2; do
    layer=${layers[$i_layer]}
    save_dir=${save_dir_per_layer[$i_layer]}
    for prefix in "${prefixes[@]}"; do
        for i_model in 0 1; do
            model=${model_names[$i_model]}
            ds=${!model_to_ds[$i_model]}
            short=${model_to_short[$i_model]}
            for seed in {0..9}; do
                # if seed == 0, save states
                save_states=""
                if [ $seed -eq 0 ]; then
                    save_states="--save_states" 
                fi
                python extraction_main.py --model $model --datasets $ds --method_list $methods --seed $seed $save_states --prefix $prefix --layer $layer --save_dir extraction_results/$save_dir
            done
        done
    done
done

# RRCS

RCCS_STRING=$(printf "RCCS%s " $(seq 0 19))

for i_model in 1 0; do
    model=${model_names[$i_model]}
    ds=${!model_to_ds[$i_model]}
    short=${model_to_short[$i_model]}
    for seed in {0..4}; do
        # if seed == 0, save states
        save_states=""
        if [ $seed -eq 0 ]; then
            save_states="--save_states" 
        fi

        python extraction_main.py --model $model --datasets $ds --method_list $RCCS_STRING --seed $seed --save_dir extraction_results/rccs $save_states
    done
done