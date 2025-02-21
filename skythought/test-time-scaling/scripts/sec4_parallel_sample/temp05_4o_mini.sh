#!/bin/bash

for difficulty in easy medium hard
do
    for n in 1 2 4 8 16 32 64 128
    do
        python evaluate_multiprocess.py \
            --difficulty=${difficulty} \
            --temperature=0.5 \
            --num_threads=32 \
            --n=${n} \
            --selection=oracle \
            --lcb_version release_v4 \
            --start_date 2024-08-01 \
            --end_date 2024-12-01 \
            --no_refine \
            --num_round 1 \
            --result_json_path="results/sec4_parallel_sample_temp05_4o_mini_${difficulty}_n_${n}.json"
    done
done
