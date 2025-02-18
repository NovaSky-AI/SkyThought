# Sky-T1-32B-Flash

[Model](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Flash) | [Dataset](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_preference_data_10k) | [Blog](https://novasky-ai.github.io/posts/reduce-overthinking/)

For a detailed breakdown of the duration curation steps and training methodology, refer to the [blog](https://novasky-ai.github.io/posts/reduce-overthinking/)

## Setup

Make sure you have installed the `skythought-evals` package as outlined in the [README.md](/README.md#usage). All the data curation commands are provided from the root directory of the repo.


## Stage 1: Data Generation

We used `Sky-T1-32B-Preview` to generate responses to the 12K questions in the `PRM800K` dataset. For each question, we used a temperature of 1.0 and generated 8 responses to create a diversity of response lengths. We then formed preference pairs to contrast “verbose” vs. “concise” solutions. Specifically, from the generated responses, we picked the shortest correct response as the positive example and the longest correct response as the negative example. We discarded the rest of the generated responses, and discard any questions that did not produce at least two correct responses. We also incorporated a small number of coding preference pairs simultaneously boosts coding accuracy and further reduces coding generation lengths. 

## Stage 2: Response Rewriting
The file `response_rewrite.py` provides a pipeline for filtering and rewriting responses generated with `inference_and_check.py`. We use `response_rewrite.py` to create preference pairs for preference optimization (e.g., DPO, SimPO), however, the logic can be edited for alternative filtering and rewriting steps. Details of the implemented logic can be found in `response_rewrite.py` or on [this blog post](https://novasky-ai.github.io/posts/reduce-overthinking). 

To use our preference optimization pipeline, first generate and score multiple responses using `inference_and_check.py`. For example:

```shell
skythought evaluate --task math500 --model Qwen/Qwen2-7B-Instruct --backend vllm --backend-args tp=4 --sampling-params max_tokens=4096,temperature=0.7 --n 8 --result-dir ./
```

This will save the results in a directory with the following structure:

```
├── Qwen_Qwen2-7B-Instruct_math500_myHash
│   ├── results.json
│   └── summary.json
```

Then, use `response_rewrite.py` to process the responses into preference pairs. By default, the shortest correct responses will be used as positive examples and the longest correct responses will be used as negative samples. The argument `--SILC` can be used to also include short incorrect responses as negative examples and long correct repsonses as positive samples.

```shell
python scripts/response_rewrite.py --SILC --rewrite-model meta-llama/Meta-Llama-3-8B-Instruct --target-model NovaSky-AI/Sky-T1-32B-Preview --dataset [PATH_TO_GENERATED_RESPONSES] --result-dir ./ --checkpoint --tp 8
```

where `[PATH_TO_GENERATED_RESPONSES]` is the path to the `results.json` file. 
 
The `--checkpoint` argument can optionally be used to save intermediate files of the processed data between steps, in case of failure. 

The resulting `.json` files can be used to train a model with preference optimization algorithms. See the `/train/` directory for more details.

