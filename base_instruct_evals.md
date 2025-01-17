# Reproducing results on non-reasoning benchmarks

For the full set of results, see [here](./README.md#results-on-qa-and-instruction-following-benchmarks). 

## Installation instructions

1. For `lm_eval`, please follow the instructions [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/703fbffd6fe5e136bbb9d884cb40844e5503ae5d?tab=readme-ov-file#install) and use commit https://github.com/EleutherAI/lm-evaluation-harness/commit/703fbffd6fe5e136bbb9d884cb40844e5503ae5d 
2. For `fastchat`, follow the instructions [here](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge#install). The current implementation of Fastchat is based on OpenAI version <= 0.28.0. For making use of the latest vllm backend, it is recommended to migrate the `llm_judge` folder to use openai>=1.0.0. You can run `openai migrate` for the fastchat codebase or follow the PR [here](https://github.com/lm-sys/FastChat/pull/2915/files)
3. For `BFCL`, you can follow the official instructions [here](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#basic-installation). We further evaulate on all test categories, which requires [setting up environment variables](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#setting-up-environment-variables), and [obtaining API keys for executable test categories](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#api-keys-for-executable-test-categories). Make sure to use changes from [this PR](https://github.com/ShishirPatil/gorilla/pull/888) for QwQ and Sky-T1 model support. 

## Commands for reproducing results

All the benchmarks were run on a 8xH100 machine with the `vllm` backend on version `0.6.4.post2.dev338+g2c97eca1`. If you're running on a different device, make sure to tweak `tensor_parallel_size` and if needed the `batch_size` arguments. 

All the commands below are given for `NovaSky-AI/Sky-T1-32B-Preview`. Simply substitute the appropriate model name for other models `Qwen/Qwen-2.5-32B-Instruct` and `Qwen/QwQ-32B-Preview`. 

### MMLU (0 shot; no CoT)

```bash
lm_eval --model vllm     --model_args pretrained=NovaSky-AI/Sky-T1-32B-Preview,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1,max_model_len=2048     --tasks mmlu --trust_remote_code     --batch_size 8 --apply_chat_template --fewshot_as_multiturn
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
lm_eval --model vllm     --model_args pretrained=NovaSky-AI/Sky-T1-32B-Preview,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1     --tasks leaderboard_ifeval --trust_remote_code   --batch_size auto --apply_chat_template --fewshot_as_multiturn
```

### MGSM (native CoT)

```bash 
lm_eval --model vllm     --model_args pretrained=NovaSky-AI/Sky-T1-32B-Preview,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1,max_model_len=2048     --tasks mgsm_direct --trust_remote_code     --batch_size 8 --apply_chat_template --fewshot_as_multiturn
```

### LLM-as-a-Judge

We use the default settings - with `max_tokens` 1024 and the `gpt-4` judge. We observe that some reasoning models like `Qwen/QwQ-32B-Preview` are unable to provide brief responses sometimes and thus get truncated responses at the used `max_tokens`. While this will effect the final rating, given the context length limitations of the commonly used `gpt-4` judge (8K tokens), we stick to the 1024 `max_tokens` budget for consistency. 

1. First, serve the model with vLLM 

```bash
vllm serve NovaSky-AI/Sky-T1-32B-Preview --dtype auto --tensor-parallel-size 8 --gpu-memory-utilization 0.9
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


