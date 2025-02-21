#!/bin/bash

MAX_ROUND=3
for difficulty in easy medium hard
do
    /root/miniconda3/envs/sstar/bin/python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --n=16 \
        --api_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --api_base http://localhost:8000/v1 \
        --lcb_version release_v2 \
        --num_round ${MAX_ROUND} \
        --selection oracle_all_rounds \
        --no_dspy_gen \
        --ablation_qwq_vanilla_without_reasoning \
        --ablation_qwq_debug_with_4o_mini \
        --result_json_path="results/final_r1qwen32b_n_16_debug_public3_select_oracle_${difficulty}.json"
done
