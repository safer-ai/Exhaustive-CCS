#!/bin/bash

models=(roberta-large-mnli gpt-j-6B t5-11b T0pp deberta-v2-xxlarge-mnli)
for model in "${models[@]}"; do
    python generation_main.py --model $model --datasets all --cal_zeroshot 0 --swipe
done
