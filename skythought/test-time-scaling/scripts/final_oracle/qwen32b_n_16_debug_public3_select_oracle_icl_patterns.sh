#!/bin/bash

MAX_ROUND=3
for difficulty in medium
do
    num_icl=1
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --n=16 \
        --api_name openai/Qwen/Qwen2.5-Coder-32B-Instruct \
        --api_base http://localhost:8000/v1 \
        --selection oracle_all_rounds \
        --lcb_version release_v2 \
        --num_round ${MAX_ROUND} \
        --num_icl_examples ${num_icl} \
        --icl_retriever pattern \
        --result_json_path="results_final/final_qwen32b_n_16_debug_public3_select_oracle_icl_${num_icl}_patterns_${difficulty}.json"
done
