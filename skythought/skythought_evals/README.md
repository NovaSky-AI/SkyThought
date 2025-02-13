# Skythought-evals: Data Generation and Evaluation Tools

## Requirements 

Make sure you have installed the `skythought-evals` package as outlined in the [README.md](/README.md#usage).

For running OpenAI model, export the OpenAI key. 
```shell
export OPENAI_API_KEY={openai_api_key}
```

## Generation and Evaluation
The file `inference_and_check.py` provides convenient methods for generating sequences (e.g., for distillation or benchmark evaluation) and checking whether the generated solutions are correct (e.g., for reject sampling or benchmark evaluation).

### Benchmark Evaluation
We provide a wrapper script `eval.py` to conveniently run reasoning benchmarks. This script can be used to launch evaluations for multiple benchmarks, then aggregate and log the accuracy for all benchmarks.  To see the full list of supported args and valid arguments, run `python -m skythought_evals.eval --help`

**Note**: The `GPQADiamond` dataset is gated and requires first receiving access at this Huggingface [link](https://huggingface.co/datasets/Idavidrein/gpqa) (which is granted immediately), then logging into your Huggingface account in your terminal session with `huggingface-cli login`. 

**NOTE**: For reproducing `Sky-T1-32B-Preview` results on `AIME` and `GPQADiamond` dataset, pass in temperatures as `0.7`, and `n=8`. 

```shell
python -m skythought_evals.eval --model NovaSky-AI/Sky-T1-32B-Preview --evals=aime,gpqa_diamond --tp=8 --temperatures 0.7 --n 8
```

#### Example Usage
```shell
python -m skythought_evals.eval --model Qwen/QwQ-32B-Preview --evals=aime,math500,gpqa_diamond --tp=8 --result-dir ./
```

We further recommend streaming all outputs to a log file for reference:

```shell
python -m skythought_evals.eval --model Qwen/QwQ-32B-Preview --evals=aime,math500,gpqa_diamond --tp=8 --result-dir ./ 2>&1 | tee mylogs.txt
```
    
Example result: `{"AIME": <aime_accuracy>, "MATH500": <math500_accuracy>, "GPQADiamond": <gpqa_diamond_accuracy>}` 

### Scaling evaluation with Ray

You can scale evaluations across multiple model replicas (and across multiple nodes) with `inference_and_check` using [ray](https://docs.ray.io):

```shell
python -m skythought_evals.inference_and_check --task math500 --model Qwen/Qwen2-7B-Instruct --max_tokens 4096 --split test --result-dir ./ --temperatures 0.7 --use-ray 
```

By default, we make use of the configuration in [ray_configs/ray_config.yaml](./ray_configs/ray_config.yaml). You can also customize this with `--ray-config /path/to/ray_config.yaml`. 

### Optimized settings for 32B and 7B models

The following are optimized settings on a 8xH100 or a 8xA100 node. 

For 32B models, we recommend using `--use-ray` and the default ray configuration for best performance. 

For 7B models, we recommend adding `--ray-config-tensor-parallel-size 1` and `--ray-config-num-replicas 8` for best performance. FOr example, the previous command will change to:

```shell
python -m skythought_evals.inference_and_check --task math500 --model Qwen/Qwen2-7B-Instruct --max_tokens 4096 --split test --result-dir ./ --temperatures 0.7 --use-ray --ray-config-tensor-parallel-size 1 --ray-config-num-replicas 8
```

#### Multi-node inference

Note that if you have a ray cluster setup, you can scale the number of replicas as needed with `--ray-config-num-replicas` to make full use of your cluster. Make sure to execute the script on the head node and ensure that `--result-dir` is a valid directory that the head node can write to. 

### Best-of-N Evaluation

While we are actively working on a better CLI interface, you can use `-m skythought_evals.inference_and_check` for Best-of-N evaluation. 

```bash
python -m skythought_evals.inference_and_check --task math500 --model Qwen/Qwen2-7B-Instruct --tp 4 --max_tokens 4096 --split test --result-dir ./ --temperatures 0.7 --n 64
```

### Distill and Reject Sampling
Currently we support distill and reject sampling from various self-hosted models for NUMINA, APPS, and TACO datasets. For NUMINA, the source can be one from `[amc_aime, math, olympiads]`.
#### Example Usage

```shell
python -m skythought_evals.inference_and_check --task apps --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split test --difficulty all --result-dir $SKYT_HOME/data

python -m skythought_evals.inference_and_check --task taco --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --difficulty MEDIUM --result-dir $SKYT_HOME/data

python -m skythought_evals.inference_and_check --task taco --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split test --difficulty all --result-dir $SKYT_HOME/data

python -m skythought_evals.inference_and_check --task numina --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --source math --filter-difficulty --result-dir $SKYT_HOME/data --math-difficulty-lower-bound 4 --math-difficulty-upper-bound 9

python -m skythought_evals.inference_and_check --task numina --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --source amc_aime --filter-difficulty --result-dir $SKYT_HOME/data --math-difficulty-lower-bound 1 --math-difficulty-upper-bound 9

python -m skythought_evals.inference_and_check --task numina --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --end 20000--source olympiads --filter-difficulty --result-dir $SKYT_HOME/data --math-difficulty-lower-bound 9 --math-difficulty-upper-bound 9
```

### Reproducibility Issues


We've noticed that it can be hard to reproduce results in reasoning benchmarks. Beyond the lack of agreed sampling parameters and metrics in the field at the moment, there can be significant differences in results across different evaluation codebases, and even for the same codebase with a different set of dependencies. In half-precision (bfloat16 or float16), numerical error accumulation will change outputs ever so slightly, which can dramatically alter final performance. There are three factors we've noticed that affect results:

- Long context generations: Errors can accumulate so that the output changes at 1k+ tokens, which compound as you keep generating. Since we typically set max tokens to be 16k or 32k tokens, the final solution will change significantly
- vLLM settings:  With vLLM, we’ve also noticed that at half-precision, different batch sizes can affect downstream evaluation results by a few percentage points. Further, different tensor parallelism settings can also change results in half-precision.
- vLLM version: Different versions of vLLM will use different CUDA-Toolkit or Flash attention versions. Even for the same settings, these differences in the underlying kernels used can change results. 

 We recommend to run all evaluation benchmarks at full precision, i.e float32 to avoid this.  By default, we run evaluation in `float32`, which can be customized with the `--dtype` flag. In full-precision, evaluation results should be robust to changes in batch size, tensor parallel size, version differences, etc. 