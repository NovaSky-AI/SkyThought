#!/bin/bash

for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --generator o1-preview \
        --method naive_nodspy \
        --lcb_version release_v2 \
        --result_json_path="results/baselines_o1_preview_${difficulty}.json" \
        
done
