set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English competition datasets
DATA_NAME="aime24 amc23 minerva_math olympiadbench math500"
TOKENIZERS_PARALLELISM=false

for dataset in ${DATA_NAME}; do
    OUTPUT_DIR=${MODEL_NAME_OR_PATH}/${dataset}_eval
    python3 -u math_eval.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_name ${dataset} \
        --output_dir ${OUTPUT_DIR} \
        --split ${SPLIT} \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --seed 0 \
        --temperature 0 \
        --n_sampling 1 \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --overwrite \
        --max_tokens_per_call 32768
done
