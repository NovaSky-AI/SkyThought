#!/bin/bash

for difficulty in easy medium hard
do
    /root/miniconda3/envs/sstar/bin/python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --generator qwen32b \
	--api_name Qwen/QwQ-32B-Preview \
        --api_base http://localhost:8000/v1 \
	--no_dspy_gen \
        --method naive_nodspy \
        --lcb_version release_v2 \
        --result_json_path="results/baselines_qwq32b_${difficulty}.json" \

done
