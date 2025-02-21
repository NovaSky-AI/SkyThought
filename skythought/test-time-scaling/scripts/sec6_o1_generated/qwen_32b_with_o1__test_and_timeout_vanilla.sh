MAX_ROUND=5
for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --n=8 \
        --selection=generated_tests\
        --test_generator o1-mini \
        --lcb_version release_v4 \
        --start_date 2024-08-01 \
        --end_date 2024-12-01 \
        --num_round ${MAX_ROUND} \
        --api_name openai/Qwen/Qwen2.5-Coder-32B-Instruct \
        --api_base http://localhost:8000/v1 \
        --result_json_path="results/sec6_qwen32b_with_o1mini_and_timeout_vanilla_${difficulty}_max_round_${MAX_ROUND}.json" \
        --load_cached_preds \
        --cached_preds_path="results/sec5_revision_vanilla_qwen_32b_${difficulty}_max_round_${MAX_ROUND}.json"
done