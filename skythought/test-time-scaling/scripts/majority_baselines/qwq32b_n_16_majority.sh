#!/bin/bash

for difficulty in easy
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=32 \
        --n=16 \
        --test_generator o1-mini \
        --lcb_version release_v2 \
        --num_round 1 \
        --no_dspy_gen \
        --api_name Qwen/QwQ-32B-Preview \
        --api_base http://localhost:8000/v1 \
        --selection generated_tests_majority_no_public_tests \
        --result_json_path="results/majority_qwq32b_n_16_${difficulty}.json" \

done
