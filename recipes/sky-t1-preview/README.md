# Sky-T1-32B-Preview 

[Model](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview) | [Dataset](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k) | [Blog](https://novasky-ai.github.io/posts/sky-t1/)

Give below are the instructions to replicate the data preprocessing and training steps for Sky-T1-32B-Preview. 

## Setup

Make sure you have installed the `skythought-evals` package as outlined in the [README.md](/README.md#usage). All the data curation commands are provided from the root directory of the repo.
Set the env variable `SKYT_HOME` as the directory for the final dataset. 

## Training Data Curation

To generate the training data for Sky-T1, we use the QwQ-32B-Preview model. We curate the data mixture to cover diverse domains that require reasoning, and a reject sampling procedure to improve the data quality. We also add the science and riddle portion from the [STILL-2 model](https://arxiv.org/pdf/2412.09413).

The final data contains (1) 5k coding data from APPs and TACO, (2) 10k math data from AIME, MATH, and Olympiads subsets of the NuminaMATH dataset and (3) 1k science and puzzle data from STILL-2.

### Step 0 (Only for NUMINA math dataset): Label Math Difficulty from NUMINA

We provide the labelled NUMINA dataset used for training here: https://huggingface.co/datasets/NovaSky-AI/labeled_numina_difficulty . For replication, read on below.

Put one or multiple OPENAI_API_KEY in a file, e.g. keys.txt (one per line). If there is more than one key, the script will use them in a round-robin way to speed up generation. Label Math difficulty using GPT-4o-mini: 
#### Example usage: 
```
python scripts/label_math_difficulty.py --source [amc_aime, math, olympiads] --keys keys.txt
```
The expected output is labeled_source_0_-1.json. We also provide instructions to download these files under the labeled_numina_difficulty folder (Download from HuggingFace).

### Step 1: Data Inference
Inference the results from QwQ on several datasets. In preview version, we use data from the following dataset.

```shell
skythought generate --task apps --model Qwen/QwQ-32B-Preview --backend vllm --backend-args tp=8 --sampling-params max_tokens=16384 --task-args dataset_split=test --result-dir $SKYT_HOME/data

skythought generate --task taco --model Qwen/QwQ-32B-Preview --backend vllm --backend-args tp=8 --sampling-params max_tokens=16384 --task-args '{"dataset_split": "train", "preprocess_config": {"difficulty": "MEDIUM"}}' --result-dir $SKYT_HOME/data

skythought generate --task numina --model Qwen/QwQ-32B-Preview --backend vllm --backend-args tp=8 --sampling-params max_tokens=16384 --task-args '{"dataset_split": "train", "preprocess_config": {"difficulty": "math"}}' --result-dir $SKYT_HOME/data

skythought generate --task numina --model Qwen/QwQ-32B-Preview --backend vllm --backend-args tp=8 --sampling-params max_tokens=16384 --task-args '{"dataset_split": "train", "preprocess_config": {"difficulty": "amc_aime"}}' --result-dir $SKYT_HOME/data

skythought generate --task numina --model Qwen/QwQ-32B-Preview --backend vllm --backend-args tp=8 --sampling-params max_tokens=16384 --task-args '{"dataset_split": "train", "preprocess_config": {"difficulty": "olympiads"}}' --result-dir $SKYT_HOME/data --start 0 --end 20000
```

This will save the results in individual folders in `result-dir`. The directory structure should be as follows:

```
├── Qwen_QwQ-32B-Preview_numina_myHash
│   ├── results.json
│   └── summary.json
├── Qwen_QwQ-32B-Preview_apps_myHash
│   ├── results.json
│   └── summary.json
...
```

### Step 2: Format the response
After obtaining a list file for training data, convert them to a unified format (Note: This uses GPT-4o-mini to rewrite. The output is long and takes ~100 dollars for our preview data). 
This will overwrite the result files "results.json" in the directory. 

```shell
python scripts/convert_format.py --input_dir $SKYT_HOME/data --keys keys.txt
```

### Step 3: Reject Sampling on the formatted data (Example Usage with previous script)

```shell 
skythought score --task apps --path <path_to_run_folder>
```
Similar for other datasets.

### Convert to ShareGPT format for training
After obtaining multiple converted files, merge them together and convert to the ShareGPT format to perform training. In our preview model, we also add the science and riddle portion from the [STILL-2 model](https://arxiv.org/pdf/2412.09413), where interested readers can download their part of data and simply concatenating to the data obtained above.

```shell
python scripts/convert_to_data.py --input_dir $SKYT_HOME/data --output $SKYT_HOME/data/train_data.json
```

## Training

The model was trained for 3 epochs with a learning rate of 1e-5 and a batch size of 96 using [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory). Our model training was completed in 19 hours on 8 H100 GPUs using DeepSpeed Zero-3 offloading, costing approximately $450 as per Lambda Cloud pricing. 
