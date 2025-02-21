MAX_ROUND=5
for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --n=8 \
        --selection=first\
        --test_generator 4o-mini \
        --lcb_version release_v4 \
        --start_date 2024-08-01 \
        --end_date 2024-12-01 \
        --num_round ${MAX_ROUND} \
        --load_cached_preds \
        --result_json_path="results/sec6_4o_mini_vanilla_baseline_${difficulty}_max_round_${MAX_ROUND}.json" \
        --cached_preds_path="results/sec5_revision_vanilla_4o_mini_${difficulty}_max_round_${MAX_ROUND}.json"
done