#!/bin/bash

# Main experiments
models=(gpt-j-6B unifiedqa-t5-11b roberta-large-mnli t5-11b T0pp deberta-v2-xxlarge-mnli)
for model in "${models[@]}"; do
    python generation_main.py --model $model --datasets all --cal_zeroshot 0 --swipe --states_index -1 -3 -5 -7 -9
done

# Format experiments
models=(gpt-j-6B unifiedqa-t5-11b)
for model in "${models[@]}"; do
    python generation_main.py --model $model --datasets all --cal_zeroshot 0 --swipe  --states_index -1 -3 -5 -7 -9 --prefix normal-dot
    python generation_main.py --model $model --datasets all --cal_zeroshot 0 --swipe  --states_index -1 -3 -5 -7 -9 --prefix normal-thatsright
    python generation_main.py --model $model --datasets all --cal_zeroshot 0 --swipe  --states_index -1 -3 -5 -7 -9 --prefix normal-mark
done