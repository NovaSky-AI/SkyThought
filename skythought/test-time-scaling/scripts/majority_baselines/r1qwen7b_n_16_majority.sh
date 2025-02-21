#!/bin/bash

for difficulty in easy medium hard
do
    /root/miniconda3/envs/sstar/bin/python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=32 \
        --n=16 \
        --test_generator o1-mini \
        --lcb_version release_v2 \
        --num_round 1 \
        --no_dspy_gen \
        --api_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --api_base http://localhost:8000/v1 \
        --selection generated_tests_majority_no_public_tests \
        --result_json_path="results/majority_r1qwen7b_n_16_${difficulty}.json" \

done
