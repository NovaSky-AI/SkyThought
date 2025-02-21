#!/bin/bash

# Server: vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --tensor-parallel-size 8
MAX_ROUND=5
for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --n=8 \
        --selection=oracle \
        --lcb_version release_v4 \
        --start_date 2024-08-01 \
        --end_date 2024-12-01 \
        --num_round ${MAX_ROUND} \
	--selfdebug_decision refine \
        --api_name openai/Qwen/Qwen2.5-Coder-32B-Instruct \
        --api_base http://localhost:8000/v1 \
        --selection oracle_all_rounds \
        --result_json_path="results/sec5_revision_refine_qwen_32b_${difficulty}_max_round_${MAX_ROUND}.json"
done
