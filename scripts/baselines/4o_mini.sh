#!/bin/bash

for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --method naive_nodspy \
        --lcb_version release_v2 \
        --result_json_path="results/baselines_4o_mini_${difficulty}.json" \

done
