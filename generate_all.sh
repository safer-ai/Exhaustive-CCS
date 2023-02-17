#!/bin/bash

models=(gpt-j-6B unifiedqa-t5-11b roberta-large-mnli t5-11b T0pp deberta-v2-xxlarge-mnli)
for model in "${models[@]}"; do
    python generation_main.py --model $model --datasets all --cal_zeroshot 0 --swipe
done
