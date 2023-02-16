#!/bin/bash

python generation_main.py --model roberta-large-mnli --datasets all --cal_zeroshot 0 --swipe
python generation_main.py --model unifiedqa-t5-11b --datasets all --cal_zeroshot 0 --swipe
python generation_main.py --model gpt-j-6B --datasets all --cal_zeroshot 0 --swipe
python generation_main.py --model t5-11b --datasets all --cal_zeroshot 0 --swipe
python generation_main.py --model T0pp --datasets all --cal_zeroshot 0 --swipe
python generation_main.py --model deberta-v2-xxlarge-mnli --datasets all --cal_zeroshot 0 --swipe