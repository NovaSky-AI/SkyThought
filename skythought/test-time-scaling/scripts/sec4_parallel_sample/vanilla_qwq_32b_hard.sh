#!/bin/bash

# Server: vllm serve Qwen/QwQ-32B-Preview --tensor-parallel-size 8

for difficulty in hard
do
    for n in 1 2 4 8 16 32 64 128
    do
        python evaluate_multiprocess.py \
            --difficulty=${difficulty} \
            --temperature=0.7 \
            --num_threads=8 \
            --n=${n} \
            --selection=oracle \
            --lcb_version release_v4 \
            --start_date 2024-08-01 \
            --end_date 2024-12-01 \
            --no_refine \
            --num_round 1 \
            --api_name Qwen/QwQ-32B-Preview \
            --api_base http://localhost:8000/v1 \
            --no_dspy_gen \
            --result_json_path="results/sec4_parallel_sample_vanilla_qwq_32b_${difficulty}_n_${n}.json"
    done
done
