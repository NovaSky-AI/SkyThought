# Reproducing results on non-reasoning benchmarks

For the full set of results, see [here](./README.md#results-on-qa-and-instruction-following-benchmarks). 

## Installation instructions

1. For `lm_eval`, install the package by executing the following : 

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
git checkout 703fbff
pip install -e ".[ifeval]"
```

For more details, you can refer to the official instructions [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/703fbffd6fe5e136bbb9d884cb40844e5503ae5d?tab=readme-ov-file#install). We report results with commit https://github.com/EleutherAI/lm-evaluation-harness/commit/703fbffd6fe5e136bbb9d884cb40844e5503ae5d

2. For `fastchat`, follow the instructions [here](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge#install). The current implementation of Fastchat is based on OpenAI version <= 0.28.0. For making use of the latest vllm backend, it is recommended to migrate the `llm_judge` folder to use openai>=1.0.0. You can run `openai migrate` for the fastchat codebase or follow the PR [here](https://github.com/lm-sys/FastChat/pull/2915/files)
3. For `BFCL`, you can follow the official instructions [here](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#basic-installation). We further evaulate on all test categories, which requires [setting up environment variables](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#setting-up-environment-variables), and [obtaining API keys for executable test categories](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#api-keys-for-executable-test-categories). Make sure to use changes from [this PR](https://github.com/ShishirPatil/gorilla/pull/888) for QwQ and Sky-T1 model support.
4. For `Arena-Hard` results, you can follow the instructions [here](https://github.com/lmarena/arena-hard-auto). We use `gpt-4-1106-preview` as the judge.

## Commands for reproducing results

All the benchmarks were run on a 8xH100 machine with the `vllm` backend. If you're running on a different device, make sure to tweak `tensor_parallel_size` and if needed the `batch_size` arguments.  Expect some variance in scores (+/- 1%) for different evaluation settings (ex: `tensor_parallel_size`)

All the commands below are given for `NovaSky-AI/Sky-T1-32B-Preview`. Simply substitute the model name for `Qwen/Qwen-2.5-32B-Instruct`. For `Qwen/QwQ-32B-Preview`, we further make use of two arguments `revision=refs/pr/58,tokenizer_revision=refs/pr/58` to use a corrected revision of QwQ. For more details on this, see https://github.com/NovaSky-AI/SkyThought/pull/26#issuecomment-2606435601. 

### MMLU (0 shot; no CoT)

```bash
lm_eval --model vllm     --model_args pretrained=NovaSky-AI/Sky-T1-32B-Preview,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1,max_model_len=2048     --tasks mmlu --trust_remote_code     --batch_size 8 --apply_chat_template --fewshot_as_multiturn
```

For QwQ, you would do 

```bash
lm_eval --model vllm     --model_args pretrained=Qwen/QwQ-32B-Preview,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1,max_model_len=2048revision=refs/pr/58,tokenizer_revision=refs/pr/58   --tasks mmlu --trust_remote_code     --batch_size 8 --apply_chat_template --fewshot_as_multiturn
```

### MMLU (5 shot; no CoT)

```bash
lm_eval --model vllm     --model_args pretrained=NovaSky-AI/Sky-T1-32B-Preview,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1,max_model_len=2048     --tasks mmlu --trust_remote_code     --batch_size 8 --apply_chat_template --fewshot_as_multiturn --num_fewshot 5
```

### ARC-C (0 shot; no CoT)

```bash
lm_eval --model vllm     --model_args pretrained=NovaSky-AI/Sky-T1-32B-Preview,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1,max_model_len=2048     --tasks arc_challenge --trust_remote_code     --batch_size 8 --apply_chat_template --fewshot_as_multiturn
```

### IFEval

```bash
lm_eval --model vllm     --model_args pretrained=NovaSky-AI/Sky-T1-32B-Preview,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1     --tasks leaderboard_ifeval --trust_remote_code   --batch_size auto --apply_chat_template --fewshot_as_multiturn
```

We use the `prompt_level_strict_acc` metric following Qwen-2.5. 

### MGSM (native CoT)

```bash 
lm_eval --model vllm     --model_args pretrained=NovaSky-AI/Sky-T1-32B-Preview,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1,max_model_len=2048     --tasks mgsm_direct --trust_remote_code     --batch_size 8 --apply_chat_template --fewshot_as_multiturn
```

We report the average value of `flexible-extract` filter. 

### MGSM (8-shot; native CoT)

```bash
lm_eval --model vllm     --model_args pretrained=NovaSky-AI/Sky-T1-32B-Preview,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1,max_model_len=2048     --tasks mgsm_direct --trust_remote_code --batch_size 8 --apply_chat_template --fewshot_as_multiturn --num_fewshot 8
```

### LLM-as-a-Judge

We use the default settings - with `max_tokens` 1024 and the `gpt-4` judge. We observe that some reasoning models like `Qwen/QwQ-32B-Preview` are unable to provide brief responses sometimes and thus get truncated responses at the used `max_tokens`. While this will effect the final rating, given the context length limitations of the commonly used `gpt-4` judge (8K tokens), we stick to the 1024 `max_tokens` budget for consistency. 

1. First, serve the model with vLLM 


```bash
vllm serve NovaSky-AI/Sky-T1-32B-Preview --dtype auto --tensor-parallel-size 8 --gpu-memory-utilization 0.9
```

For `Qwen/QwQ-32B-Preview`,  use 

```bash 
vllm serve Qwen/QwQ-32B-Preview --dtype auto --tensor-parallel-size 8 --gpu-memory-utilization 0.9 --revision refs/pr/58 --tokenizer-revision refs/pr/58
```

2. Next, generate model response 

```bash
python gen_api_answer.py --model NovaSky-AI/Sky-T1-32B-Preview --openai-api-base http://localhost:8000/v1 --parallel 50
```

Note: The generated results will be in `data/model_answer/<repo_id>/<model name>.jsonl` . Move them to the root folder `data/model_answer/`

3. After generating responses for all the models, evaluate with the default settings

```bash
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_judgment.py --model-list Sky-T1-32B-Preview QwQ-32B-Preview Qwen2.5-32B-Instruct --parallel  2
```
4. Get MTBench scores (we use the average score of both turns)

```bash
python show_result.py
```

### BFCL-v3

Our results are reported on `test-category` `all` . Make sure to get the API keys for the executable test categories by following the instructions [here](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#api-keys-for-executable-test-categories)

Run

```bash
bfcl generate --model NovaSky-AI/Sky-T1-32B-Preview --test-category all --backend vllm --num-gpus 8 --gpu-memory-utilization 0.9 
```

For evaluation, you can simply run

```bash
bfcl evaluate --model Qwen/QwQ-32B-Preview,NovaSky-AI/Sky-T1-32B-Preview,Qwen/Qwen2.5-32B-Instruct  --test-category all --api-sanity-check
```
### Arena Hard
For `Arena-Hard`, we use the following script to start a `TGI` service for generating answers 
```bash
hf_pat=
model=NovaSky-AI/Sky-T1-32B-Preview
volume=/mnt/local_storage/data/cache
port=1996

huggingface-cli download $model
sudo docker run --gpus 8 -e HUGGING_FACE_HUB_TOKEN=$hf_pat --shm-size 2000g -p $port:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.2.0 --model-id $model --max-input-length 8192 --max-batch-total-tokens 8193 --max-batch-prefill-tokens 8193 --max-total-tokens 8193 --sharded true
```
For running the `gen_answer.py` script, we use the following `config_api` yaml setting. For `qwq-32b-preview`, we explicitly specify the system prompt as `You are a helpful and harmless assistant. You are Qwen developed by Alibaba.` to avoid the CoT prompt.
```yaml
...
sky-T1-32B-Preview:
    model_name: sky-T1-32B-Preview
    endpoints:
        - api_base: http://localhost:1996/v1
          api_key: empty
    api_type: openai
    parallel: 8
...
```
and finally for `gen_judgment.py`, we use `gpt-4-1106-preview` as the judge.

#### Supplementary results for Arena-Hard

Here are some supplementary results for Arena-Hard, compared with o1-mini which is the best performing model on this benchmark (as of Jan 2025). 

| model | score | rating_q025 | rating_q975 | CI | avg_tokens | date |
|-------|--------|------------|-------------|-------|------------|-------|
| o1-mini-2024-09-12 | 91.98 | 90.88 | 93.12 | (-1.10, +1.14) | 1399.0 | 2025-01-18 |
| sky-T1-32B-Preview | 74.79 | 72.28 | 76.8 | (-2.51, +2.01) | 847.0 | 2025-01-18 |
| qwen2.5-32b-instruct | 66.51 | 64.55 | 68.4 | (-1.96, +1.89) | 611.0 | 2025-01-18 |
| qwq-32b-preview | 52.6 | 50.86 | 54.91 | (-1.74, +2.31) | 1005.0 | 2025-01-23 |

For more details, see: https://github.com/NovaSky-AI/SkyThought/pull/26#issuecomment-2599525551 
