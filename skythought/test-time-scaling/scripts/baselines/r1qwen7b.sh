#!/bin/bash

for difficulty in easy medium hard
do
    /root/miniconda3/envs/sstar/bin/python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --generator r1qwen32b \
	--api_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --api_base http://localhost:8000/v1 \
	--no_dspy_gen \
        --method naive_nodspy \
        --lcb_version release_v2 \
        --result_json_path="results/baselines_r1qwen7b_${difficulty}.json" \

done
