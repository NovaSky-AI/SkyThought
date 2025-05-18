#!/bin/bash

MAX_ROUND=3
for difficulty in easy medium hard
do
    /root/miniconda3/envs/sstar/bin/python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --n=1 \
        --selection=random \
        --api_name Qwen/QwQ-32B-Preview \
        --api_base http://localhost:8000/v1 \
        --lcb_version release_v2 \
        --num_round ${MAX_ROUND} \
        --no_dspy_gen \
        --ablation_qwq_vanilla_without_reasoning \
        --ablation_qwq_debug_with_4o_mini \
        --result_json_path="results/final_qwq32b_n_1_debug_public3_select_random_${difficulty}.json"
done
