#!/bin/bash

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
        --selection oracle_all_rounds \
        --result_json_path="results/sec5_revision_refine_4o_mini_${difficulty}_max_round_${MAX_ROUND}.json"
done
