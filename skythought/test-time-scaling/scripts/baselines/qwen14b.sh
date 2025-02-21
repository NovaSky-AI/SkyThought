#!/bin/bash

for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --generator qwen14b \
        --api_name Qwen/Qwen2.5-Coder-14B-Instruct \
        --api_base http://localhost:8000/v1 \
        --method naive_nodspy \
        --lcb_version release_v2 \
        --result_json_path="results/baselines_qwen14b_${difficulty}.json" \

done
