# Skythought-evals: Data Generation and Evaluation Tools


## Requirements 

Make sure you have installed the `skythought` package as outlined in the [README.md](/README.md#usage).

For running OpenAI model, export the OpenAI key. 
```shell
export OPENAI_API_KEY={openai_api_key}
```

## Usage

We provide three commands in the CLI: 

- `skythought evaluate` : Evaluate a model on a given task.
- `skythought generate`: Generate model outputs for a pre-configured task.
- `skythought score`: Score saved generations for a given task.

For a walkthrough on the basics, please refer to the [example](../../examples/evaluate.ipynb). 

## Generation and Evaluation

### Benchmark Evaluation

Given below are two examples for evaluation.

```shell
skythought evaluate --model NovaSky-AI/Sky-T1-32B-Preview --task aime  --backend vllm --backend-args tensor_parallel_size=8  --sampling-params temperature=0.6,top_p=0.95 --n 8 --result-dir ./

skythought evaluate --model NovaSky-AI/Sky-T1-32B-Preview --task gpqa_diamond --backend vllm --backend-args tensor_parallel_size=8 --sampling-params temperature=0.6,top_p=0.95 --n 8
```

**Note**: The `GPQADiamond` dataset is gated and requires first receiving access at this Huggingface [link](https://huggingface.co/datasets/Idavidrein/gpqa) (which is granted immediately), then logging into your Huggingface account in your terminal session with `huggingface-cli login`. 


The results will be saved in a folder in `result-dir`:

```bash
result-dir/
├── Qwen_QwQ-32B-Preview_aime_myHash
│   ├── results.json # contains the full results for the benchmark
│   └── summary.json # contains summary of the run with configuration and metrics
```

### Scaling evaluation with Ray

You can scale evaluations across multiple model replicas (and across multiple nodes) using [ray](https://docs.ray.io) backend:

```shell
skythought evaluate --model Qwen/QwQ-32B-Preview --task aime --backend ray --backend-args tensor_parallel_size=4,num_replicas=4 --result-dir ./
```

By default, we make use of the configuration in [ray_configs/ray_config.yaml](./ray_configs/ray_config.yaml). You can also customize the following parameters for ray: 

- `tensor_parallel_size`: Tensor parallel size per replica. Defaults to 4.  
- `accelerator_type`: GPU accelerator type. See [the list of available types](https://docs.ray.io/en/latest/ray-core/accelerator-types.html) for more information. Defaults to None, which means any available GPUs in the Ray cluster will be used.  
- `num_replicas`: Number of model replicas to use for inference. Defaults to 2.  
- `batch_size`: Batch size per model replica for inference.  
- `gpu_memory_utilization`: Fraction of GPU memory allocated to the model executor in vLLM. Defaults to 0.9.  
- `dtype`: Data type used for inference. Defaults to "auto".


### Optimized settings for 32B and 7B models

The following are optimized settings on a 8xH100 or a 8xA100 node. We recommend using `ray` backend for best performance. 

For 32B models, we recommend using the default backend configuration for best performance. 

```shell
skythought evaluate --model Qwen/QwQ-32B-Preview --task aime24 --backend ray --result-dir ./
```

For 7B models, we recommend using `tensor_parallel_size=1` and `num_replicas=8` for best performance. For example, the previous command will change to:

```shell
skythought evaluate --model Qwen/Qwen2-7B-Instruct --task math500 --backend ray --backend-args tensor_parallel_size=1,num_replicas=8 --result-dir ./
```

#### Multi-node inference

Note that if you have a ray cluster setup, you can scale the number of replicas as needed with `num_replicas` argument in `backend-args` to make full use of your cluster. Make sure to execute the script on the head node and ensure that `--result-dir` is a valid directory that the head node can write to. 

### Best-of-N Evaluation

You can use the `--n` parameter to specify the number of generations per problem. For `n>1` , we calculate pass

```bash
skythought evaluate --model Qwen/Qwen2-7B-Instruct --task math500 --backend ray --backend-args tensor_parallel_size=1,num_replicas=8 --sampling-params temperature=0.7,max_tokens=4096 --n 64 --result-dir ./
```

### Distill and Reject Sampling
Currently we support distill and reject sampling for NUMINA, APPS, and TACO datasets. For NUMINA, the source can be one from `[amc_aime, math, olympiads]`.

#### Example Usage

```shell
skythought generate --model Qwen/QwQ-32B-Preview --task numina_amc_aime --backend ray --backend-args tensor_parallel_size=8 --sampling-params max_tokens=16384 --result-dir $SKYT_HOME/data
```

Once the generations are saved, you can then apply any postprocessing on the results (saved in a `results.json` file in separate run folder) and then run:

```shell
skythought score --task numina_amc_aime --run-dir <path>
```

### Reproducibility Issues


We've noticed that it can be hard to reproduce results in reasoning benchmarks. Beyond the lack of agreed sampling parameters and metrics in the field at the moment, there can be significant differences in results across different evaluation codebases, and even for the same codebase with a different set of dependencies. In half-precision (bfloat16 or float16), numerical error accumulation will change outputs ever so slightly, which can dramatically alter final performance. There are three factors we've noticed that affect results:

- Long context generations: Errors can accumulate so that the output changes at 1k+ tokens, which compound as you keep generating. Since we typically set max tokens to be 16k or 32k tokens, the final solution will change significantly
- vLLM settings:  With vLLM, we’ve also noticed that at half-precision, different batch sizes can affect downstream evaluation results by a few percentage points. Further, different tensor parallelism settings can also change results in half-precision.
- vLLM version: Different versions of vLLM will use different CUDA-Toolkit or Flash attention versions. Even for the same settings, these differences in the underlying kernels used can change results. 

 We recommend to run evaluation benchmarks at full precision, i.e float32 to avoid this. In full-precision, evaluation results should be robust to changes in batch size, tensor parallel size, version differences, etc.


## Key Concepts

### Tasks

A Task consists of task-specific configuration and implements 
- Dataset loading and preprocessing 
- Creating of input conversation to the model
- Scoring of model responses

The configuration (`TaskConfig`) contains dataset loading related details such as Hugging Face dataset ID, the particular subset for this benchmark (e.g., ”Challenge” subset for ARC), and a task template, which contains task-specific instructions to be used (Eg: `Return your answer in \boxed{}`). Each configuration is stored in a YAML. For example, you can see the YAML in this [aime24.yaml file](./tasks/aime/aime24.yaml)

Internally, a Task implementation is termed a "TaskHandler", you can see one such implementation [here](./tasks/aime/aime_handler.py). 


To add a new task `mytask`: 
- First, see if the task can be simply specified as a configuration (One example is [`aime25`](./tasks/aime/aime25.yaml)). If so, you can add a YAML file in the appropriate folder and re-use an existing handler. (All available handlers are specified [here](./tasks/__init__.py)). 
- If not, you should create a new `TaskHandler` subclass for this task along with a task configuration YAML (`mytask.yaml`). 

### Models

A Model consists of the model ID and templating configuration. This configuration optionally contains the system prompt and an assistant prefill message. Different reasoning models use their own system prompt, and some perform best when the response is prefilled with special tokens. 

We store our pre-configured models as well as a list of system prompt templates [here](./models/model_configs.yaml). 

### Backend

The Backend is concerned with how the LLM instance is created and queried. For flexibility, we support 
- Local inference with vLLM (basic single node) or Ray+vLLM (more scalable single and multi-node inference)
- Remote inference behind an OpenAI-compatible endpoint. 

The Backend also consists of configuration at instantiation (ex; the data type for the model), along with sampling parameters during generation (temperature, max tokens, etc).

During evaluation, the above tie in together and the flow is as follows: 
1. Load dataset and create conversations based on the Task and Model specified by the user
2. Generate model responses from the Backend based on the provided sampling parameters
3. Score model responses based on the Task 
4. Output final results
