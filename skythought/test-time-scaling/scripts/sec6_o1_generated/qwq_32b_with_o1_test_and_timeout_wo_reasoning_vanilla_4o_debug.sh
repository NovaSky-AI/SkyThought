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
        --api_name Qwen/QwQ-32B-Preview \
        --api_base http://localhost:8000/v1 \
        --no_dspy_gen \
        --ablation_qwq_vanilla_without_reasoning \
        --ablation_qwq_debug_with_4o_mini \
        --result_json_path="results/sec6_qwq32b_with_o1mini_and_timeout_wo_reasoning_vanilla_4o_debug_${difficulty}_max_round_${MAX_ROUND}.json" \
        --load_cached_preds \
        --cached_preds_path="results/sec5_revision_vanilla_wo_reasoning_qwq_32b_${difficulty}_max_round_${MAX_ROUND}_with_4o_debug.json"
done