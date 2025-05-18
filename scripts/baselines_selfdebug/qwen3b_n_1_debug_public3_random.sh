#!/bin/bash

MAX_ROUND=3
for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --n=1 \
        --selection=random \
        --api_name openai/Qwen/Qwen2.5-Coder-3B-Instruct \
        --api_base http://localhost:8000/v1 \
        --lcb_version release_v2 \
        --no_dspy_gen \
        --num_round ${MAX_ROUND} \
        --result_json_path="results/final_qwen3b_n_1_debug_public3_select_random_${difficulty}.json"
done
