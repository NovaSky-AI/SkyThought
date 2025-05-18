#!/bin/bash

# QwQ-32B
MAX_ROUND=3
for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=32 \
        --n=16 \
        --selection=first \
        --api_name Qwen/QwQ-32B-Preview \
        --api_base http://localhost:8000/v1 \
        --lcb_version release_v2 \
        --num_round ${MAX_ROUND} \
        --no_dspy_gen \
        --ablation_qwq_vanilla_without_reasoning \
        --ablation_qwq_debug_with_4o_mini \
        --result_json_path="results/final_qwq32b_n_16_debug_public3_select_first_cached_${difficulty}.json" \
        --load_cached_preds \
        --cached_preds_path="results/final_qwq32b_n_16_debug_public3_oracle_${difficulty}.json"
done

# R1-Qwen-7B
for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=32 \
        --n=16 \
        --selection=first \
        --api_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --api_base http://localhost:8000/v1 \
        --lcb_version release_v2 \
        --num_round ${MAX_ROUND} \
        --result_json_path="results/final_r1qwen7b_n_16_debug_public3_select_first_cached_${difficulty}.json" \
        --load_cached_preds \
        --cached_preds_path="results/final_r1qwen7b_n_16_debug_public3_select_oracle_${difficulty}.json"
done

# R1-Qwen-14B
for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=32 \
        --n=16 \
        --selection=first \
        --api_name deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --api_base http://localhost:8000/v1 \
        --lcb_version release_v2 \
        --num_round ${MAX_ROUND} \
        --result_json_path="results/final_r1qwen14b_n_16_debug_public3_select_first_cached_${difficulty}.json" \
        --load_cached_preds \
        --cached_preds_path="results/final_r1qwen14b_n_16_debug_public3_select_oracle_${difficulty}.json"
done

# R1-Qwen-32B
for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=32 \
        --n=16 \
        --selection=first \
        --api_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --api_base http://localhost:8000/v1 \
        --lcb_version release_v2 \
        --num_round ${MAX_ROUND} \
        --result_json_path="results/final_r1qwen32b_n_16_debug_public3_select_first_cached_${difficulty}.json" \
        --load_cached_preds \
        --cached_preds_path="results/final_r1qwen32b_n_16_debug_public3_select_oracle_${difficulty}.json"
done
