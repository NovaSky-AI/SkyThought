#!/bin/bash

MAX_ROUND=3
for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=1.0 \
        --num_threads=32 \
        --n=16 \
        --generator o1-mini \
        --selection oracle_all_rounds \
        --lcb_version release_v2 \
        --num_round ${MAX_ROUND} \
        --ablation_qwq_vanilla_without_reasoning \
        --ablation_qwq_debug_with_4o_mini \
        --no_dspy_gen \
        --result_json_path="results/final_o1mini_n_16_debug_public3_select_oracle_${difficulty}.json"
done
