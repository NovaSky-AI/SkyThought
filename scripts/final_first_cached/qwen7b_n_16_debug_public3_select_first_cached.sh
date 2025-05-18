#!/bin/bash

MAX_ROUND=3
for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=32 \
        --n=16 \
        --selection=first \
        --api_name openai/Qwen/Qwen2.5-Coder-7B-Instruct \
        --api_base http://localhost:8000/v1 \
        --lcb_version release_v2 \
        --num_round ${MAX_ROUND} \
        --result_json_path="results/final_qwen7b_n_16_debug_public3_select_first_cached_${difficulty}.json" \
        --load_cached_preds \
        --cached_preds_path="results/final_qwen7b_n_16_debug_public3_select_oracle_${difficulty}.json"
done
