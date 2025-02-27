"""
This is the recipe for data curation for the Sky T1 Preview model . 
"""

import argparse
import os

import datasets
import ray
from ray.data.llm import (
    HttpRequestProcessorConfig,
    build_llm_processor,
    vLLMEngineProcessorConfig,
)

from skythought.evals.scoring.apps import APPSScorer
from skythought.evals.scoring.math import MathEqualScorer
from skythought.evals.scoring.taco import TACOScorer

from .postprocess import convert_to_sharegpt_format
from .preprocess import (
    APPSPreprocessor,
    NUMINAPreprocessor,
    TACOPreprocessor,
    taco_coerce_types,
)
from .prompts import CONVERT_PROMPT, CONVERT_PROMPT_EXAMPLE

parser = argparse.ArgumentParser()
parser.add_argument("--as-test", action="store_true")
args = parser.parse_args()

SYSTEM_PROMPT = "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."  # noqa: E501
MAX_TOKENS = 16384
# 1. Load datasets
apps_ds = datasets.load_dataset("codeparrot/apps", split="test", trust_remote_code=True)
taco_ds_medium = datasets.load_dataset(
    "BAAI/TACO", split="test", name="MEDIUM", trust_remote_code=True
)
numina_ds = datasets.load_dataset(
    "AI-MO/NuminaMath-CoT", split="train", trust_remote_code=True
)

# convert all to ray dataset
apps_ds = ray.data.from_huggingface(apps_ds)
taco_ds_medium = ray.data.from_huggingface(taco_ds_medium)
taco_ds_medium = taco_ds_medium.map(
    taco_coerce_types, fn_args=(taco_ds_medium.schema(),)
)
numina_ds = ray.data.from_huggingface(numina_ds)


# get subsets from numina based on the source column
numina_ds_amc_aime = numina_ds.filter(lambda x: x["source"] == "amc_aime")
numina_ds_olympiads = numina_ds.filter(lambda x: x["source"] == "olympiads").limit(
    20000
)
numina_ds_math = numina_ds.filter(lambda x: x["source"] == "math")


if args.as_test:
    num_samples = 100
    apps_ds = apps_ds.limit(num_samples)
    taco_ds_medium = taco_ds_medium.limit(num_samples)
    numina_ds_amc_aime = numina_ds_amc_aime.limit(num_samples)
    numina_ds_olympiads = numina_ds_olympiads.limit(num_samples)
    numina_ds_math = numina_ds_math.limit(num_samples)

# 2. Get model responses for each of the datasets
datasets = [
    apps_ds,
    taco_ds_medium,
    numina_ds_amc_aime,
    numina_ds_olympiads,
    numina_ds_math,
]

# these are user-defined simple preprocessing functions to go from entry -> prompt
preprocessors = [
    APPSPreprocessor,
    TACOPreprocessor,
    NUMINAPreprocessor,
    NUMINAPreprocessor,
    NUMINAPreprocessor,
]

dataset_names = ["apps", "taco", "numina_amc_aime", "numina_math", "numina_olympiads"]
scorer_configs = [
    dict(
        cls=APPSScorer, fn_constructor_kwargs=dict(response_column="formatted_response")
    ),
    dict(
        cls=TACOScorer,
        fn_constructor_kwargs=dict(response_column="formatted_response", backend="ray"),
    ),
    dict(
        cls=MathEqualScorer,
        fn_constructor_kwargs=dict(
            response_column="formatted_response", answer_column="solution"
        ),
    ),
    dict(
        cls=MathEqualScorer,
        fn_constructor_kwargs=dict(
            response_column="formatted_response", answer_column="solution"
        ),
    ),
    dict(
        cls=MathEqualScorer,
        fn_constructor_kwargs=dict(
            response_column="formatted_response", answer_column="solution"
        ),
    ),
]

for i, ds in enumerate(datasets):
    if i < 1:
        continue
    # 1. Preprocess and get model prompts
    preprocess_cls = preprocessors[i]
    datasets[i] = ds.map(
        preprocess_cls,
        concurrency=5,
    )

    # 2. Get model responses

    config = vLLMEngineProcessorConfig(
        # model="Qwen/QwQ-32B-Preview",
        model="Qwen/Qwen2-0.5B-Instruct",
        engine_kwargs=dict(
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_num_batched_tokens=16384,
        ),
        concurrency=2,
        batch_size=20,
    )

    processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["user_input"]},
            ],
            sampling_params=dict(
                temperature=0.3,
                max_tokens=MAX_TOKENS,
                detokenize=False,
            ),
        ),
        postprocess=lambda row: dict(
            assistant_response=row["generated_text"],
            **row,  # This will return all the original columns in the dataset.
        ),
    )
    datasets[i] = processor(datasets[i])

    # 3. Reformat the examples into a structured format

    # define a configuration for the reformatter
    config = HttpRequestProcessorConfig(
        url="https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        # number of processors to run in parallel
        # Each handles a batch of requests
        concurrency=1,
        batch_size=64,
    )
    # define the reformatter
    reformatter = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            # define the payload / the exact arguments to the OpenAI chat completions API
            payload=dict(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a solution format convertor.",
                    },
                    {
                        "role": "user",
                        "content": CONVERT_PROMPT.format(
                            example=CONVERT_PROMPT_EXAMPLE,
                            content=f"{row['user_input']}\n{row['assistant_response']}",
                        ),
                    },
                ],
                temperature=0.7,
                max_tokens=MAX_TOKENS,
            ),
        ),
        postprocess=lambda row: dict(
            formatted_response=row["http_response"]["choices"][0]["message"]["content"],
            **row,
        ),
    )
    datasets[i] = reformatter(datasets[i])

    # 4. Rejection Sampling based on scoring
    scorer_cls, fn_constructor_kwargs = (
        scorer_configs[i]["cls"],
        scorer_configs[i]["fn_constructor_kwargs"],
    )
    datasets[i] = datasets[i].map(
        scorer_cls, concurrency=4, fn_constructor_kwargs=fn_constructor_kwargs
    )
    score_column = scorer_cls.SCORE_COLUMN
    datasets[i] = datasets[i].filter(lambda x, sc=score_column: x[sc])

    # 5. Convert to ShareGPT format
    datasets[i] = datasets[i].map(
        convert_to_sharegpt_format,
        fn_kwargs=dict(
            prompt_column="user_input", response_column="formatted_response"
        ),
    )

    # 6. Save datasets
    dir_name = f"data/sky-t1-preview-{dataset_names[i]}"
    datasets[i] = datasets[i].materialize()
    datasets[i].write_json(os.path.abspath(dir_name))


# 7. Union

# final_dataset = datasets[0].union(*datasets[1:])
# dir_name = f"data/sky-t1-preview-full"
# # save in folder as a single JSON file
# final_dataset.repartition(1).write_json(os.path.abspath(dir_name))
