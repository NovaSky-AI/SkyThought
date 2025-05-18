#!/bin/bash

MAX_ROUND=3
for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --n=1 \
        --no_dspy_gen \
	--generator=o1-mini \
        --selection=random \
        --lcb_version release_v2 \
        --ablation_qwq_vanilla_without_reasoning \
        --ablation_qwq_debug_with_4o_mini \
        --num_round ${MAX_ROUND} \
        --result_json_path="results/final_o1_mini_n_1_debug_public3_select_random_${difficulty}.json" \

done
