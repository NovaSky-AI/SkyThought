#!/bin/bash

for difficulty in easy medium hard
do
    /root/miniconda3/envs/sstar/bin/python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=128 \
        --n=16 \
        --test_generator 4o-mini \
        --lcb_version release_v2 \
        --num_round 1 \
        --no_dspy_gen \
        --api_name Qwen/Qwen2.5-Coder-32B-Instruct \
        --api_base http://localhost:8000/v1 \
        --selection generated_tests_majority_no_public_tests \
        --result_json_path="results/majority_qwen32b_n_16_${difficulty}.json" \

done
