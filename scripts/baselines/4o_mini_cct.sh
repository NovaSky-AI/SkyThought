#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sstar

python codecontest_evaluate_multiprocess.py \
    --temperature=0.7 \
    --num_threads=16 \
    --method naive_nodspy \
    --generator 4o \
    --result_json_path="results/baselines_4o_codecontest.json"


