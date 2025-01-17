# Data Generation and Evaluation Tools
This document describes the steps to training data curation and evaluation scripts for Sky-T1. 

## Requirements 
First create the environment as follows.
```shell
conda create -n eval python==3.10
conda activate eval 
pip install -r requirements.txt
```

For running OpenAI model, export the OpenAI key. 
```shell
export OPENAI_API_KEY={openai_api_key}
```

## Training Data Curation
### Step 0 (Optional, only for NUMINA math dataset): Label Math Difficulty from NUMINA
Put one or multiple OPENAI_API_KEY in a file, e.g. keys.txt (one per line). If there is more than one key, the script will use them in a round-robin way to speed up generation. Label Math difficulty using GPT-4o-mini: 
#### Example usage: 
```
python label_math_difficulty.py --source [amc_aime, math, olympiads] --keys keys.txt
```
The expected output is labeled_source_0_-1.json. We also provide instructions to download these files under the labeled_numina_difficulty folder (Download from HuggingFace).

### Step 1: Data Inference
Inference the results from QwQ on several datasets. In preview version, we use data from the following dataset.

```shell
python inference_and_check.py --dataset APPS --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split test --source all --result-dir $SKYT_HOME/data --inference

python inference_and_check.py --dataset TACO --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --source MEDIUM --filter-difficulty --result-dir $SKYT_HOME/data --inference

python inference_and_check.py --dataset TACO --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split test --source all --result-dir $SKYT_HOME/data --inference

python inference_and_check.py --dataset NUMINA --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --source math --filter-difficulty --result-dir $SKYT_HOME/data --inference

python inference_and_check.py --dataset NUMINA --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --source amc_aime --filter-difficulty --result-dir $SKYT_HOME/data --inference

python inference_and_check.py --dataset NUMINA --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --source olympiads --end 20000 --filter-difficulty --result-dir $SKYT_HOME/data --inference
```

### Step 2: Format the response
After obtaining a list file for training data, convert them to a unified format (Note: This uses GPT-4o-mini to rewrite. The output is long and takes ~100 dollars for our preview data).
```shell
python convert_format.py --input_dir $SKYT_HOME/data --keys keys.txt
```

### Step 3: Reject Sampling on the formatted data (Example Usage with previous script)
```shell 
python inference_and_check.py --dataset APPS --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split test --source all --result-dir $SKYT_HOME/data --check
```
Similar for other datasets.

### Convert to ShareGPT format for training
After obtaining multiple converted files, merge them together and convert to the ShareGPT format to perform training. In our preview model, we also add the science and riddle portion from the [STILL-2 model](https://arxiv.org/pdf/2412.09413), where interested readers can download their part of data and simply concatenating to the data obtained above.
```shell
python convert_to_data.py --input_dir $SKYT_HOME/data --output $SKYT_HOME/data/train_data.json
```


## Generation and Evaluation
The file `inference_and_check.py` provides convenient methods for generating sequences (e.g., for distillation or benchmark evaluation) and checking whether the generated solutions are correct (e.g., for reject sampling or benchmark evaluation).

### Distill and Reject Sampling
Currently we support distill and reject sampling from various self-hosted models for NUMINA, APPS, and TACO datasets. For NUMINA, the source can be one from `[amc_aime, math, olympiads]`.
#### Example Usage

```shell
python inference_and_check.py --dataset APPS --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split test --source all --result-dir $SKYT_HOME/data

python inference_and_check.py --dataset TACO --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --source MEDIUM --filter-difficulty --result-dir $SKYT_HOME/data

python inference_and_check.py --dataset TACO --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split test --source all --result-dir $SKYT_HOME/data

python inference_and_check.py --dataset NUMINA --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --source math --filter-difficulty --result-dir $SKYT_HOME/data --math_difficulty_lower_bound 4 --math_difficulty_upper_bound 9

python inference_and_check.py --dataset NUMINA --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --source amc_aime --filter-difficulty --result-dir $SKYT_HOME/data --math_difficulty_lower_bound 1 --math_difficulty_upper_bound 9

python inference_and_check.py --dataset NUMINA --model Qwen/QwQ-32B-Preview --tp 8 --max_tokens 16384 --split train --source olympiads --end 20000 --filter-difficulty --result-dir $SKYT_HOME/data --math_difficulty_lower_bound 9 --math_difficulty_upper_bound 9
```


#### Best-of-N Inference and Check
```bash
python inference_and_check.py --dataset MATH500 --model Qwen/Qwen2-7B-Instruct --tp 4 --max_tokens 4096 --split test --result-dir ./ --inference --temperatures 0.7 --n 64
python inference_and_check.py --dataset MATH500 --model Qwen/Qwen2-7B-Instruct --tp 4 --max_tokens 4096 --split test --result-dir ./ --check --temperatures 0.7 --n 8
```

### Benchmark Evaluations
We provide a wrapper script `eval.py` to conveniently run reasoning benchmarks. We currently support `AIME`, `MATH500`, `GPQADiamond`, and `MMLU`. This script can be used to launch evaluations for multiple benchmarks, then aggregate and log the accuracy for all benchmarks. 

**Note**: The `GPQADiamond` dataset is gated and requires first receiving access at this Huggingface [link](https://huggingface.co/datasets/Idavidrein/gpqa) (which is granted immediately), then logging into your Huggingface account in your terminal session with `huggingface-cli login`. 

**NOTE**: For reproducing `Sky-T1-32B-Preview` results on `AIME` and `GPQADiamond` dataset, pass in temperatures as `0.7`. 

```shell
python eval.py --model NovaSky-AI/Sky-T1-32B-Preview --evals=AIME,GPQADiamond --tp=8 --output_file=results.txt --temperatures 0.7 
```

#### Example Usage
```shell
python eval.py --model Qwen/QwQ-32B-Preview --evals=AIME,MATH500,GPQADiamond --tp=8 --output_file=results.txt
```
    
Example result: `{"AIME": <aime_accuracy>, "MATH500": <math500_accuracy>, "GPQADiamond": <gpqa_diamond_accuracy>}` 
