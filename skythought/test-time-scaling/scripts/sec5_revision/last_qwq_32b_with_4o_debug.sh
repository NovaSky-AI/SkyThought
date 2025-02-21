#!/bin/bash

# Server: vllm serve Qwen/QwQ-32B-Preview --tensor-parallel-size 1
MAX_ROUND=5
for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=8 \
        --n=8 \
        --selection=oracle \
        --lcb_version release_v4 \
        --start_date 2024-08-01 \
        --end_date 2024-12-01 \
        --context last \
        --num_round ${MAX_ROUND} \
        --api_name Qwen/QwQ-32B-Preview \
        --api_base http://localhost:8000/v1 \
        --selection oracle_all_rounds \
        --no_dspy_gen \
        --ablation_qwq_debug_with_4o_mini \
        --result_json_path="results/sec5_revision_last_qwq_32b_${difficulty}_max_round_${MAX_ROUND}_with_4o_debug.json"
done