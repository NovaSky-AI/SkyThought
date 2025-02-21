#!/bin/bash

MAX_ROUND=3
for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=32 \
        --n=16 \
        --selection=generated_tests_no_timeout \
        --api_name Qwen/QwQ-32B-Preview \
        --api_base http://localhost:8000/v1 \
        --test_generator 4o-mini \
        --lcb_version release_v2 \
        --num_round ${MAX_ROUND} \
        --no_dspy_gen \
        --ablation_qwq_vanilla_without_reasoning \
        --ablation_qwq_debug_with_4o_mini \
        --result_json_path="results/final_qwq32b_n_16_debug_public3_select_4omini_cached_${difficulty}.json" \
        --load_cached_preds \
        --cached_preds_path="results/final_qwq32b_n_16_debug_public3_oracle_${difficulty}.json"
done
        
