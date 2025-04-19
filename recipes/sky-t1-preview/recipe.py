"""
This is the recipe for data curation for the Sky T1 Preview model . 
"""

# NOTE (sumanthrh): This script still has some rough edges and is a work in progress

import argparse
import os
from pathlib import Path
from aiohttp import ClientSession, ClientTimeout

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
    coerce_types,
)
from .prompts import CONVERT_PROMPT, CONVERT_PROMPT_EXAMPLE
from .resume import resume_from_save_dir
parser = argparse.ArgumentParser()
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--save-dir", type=str, required=True, help="Output directory")
args = parser.parse_args()
args.save_dir = Path(args.save_dir)

SYSTEM_PROMPT = "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."  # noqa: E501
MAX_TOKENS = 16384
ID_COLUMN = "_id"

# We explicitly set the target number of blocks to help tune performance.
# For materialized datasets, the number of blocks determined by ray data can be small
# for a multi-stage pipeline like the one here.
TARGET_NUM_ROWS_PER_BLOCK = 256
SESSION_FACTORY = lambda: ClientSession(timeout=ClientTimeout(sock_connect=500, sock_read=500, total=1000))

BACKEND = "mp"
# Enable more detailed logging of tasks per actor
ray.init(runtime_env={"env_vars": {"RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING": "1"}})

# 1. Load datasets
apps_ds = datasets.load_dataset(
    "codeparrot/apps", split="test", trust_remote_code=True
)  # 10K
taco_ds_medium = datasets.load_dataset(
    "BAAI/TACO", split="train", name="MEDIUM", trust_remote_code=True
)  # 3244
taco_ds_test = datasets.load_dataset(
    "BAAI/TACO", split="test", name="ALL", trust_remote_code=True
)  # 1000
numina_ds = datasets.load_dataset(
    "AI-MO/NuminaMath-CoT", split="train", trust_remote_code=True
)
apps_ds = apps_ds.add_column(name=ID_COLUMN, column=[i for i in range(len(apps_ds))])
taco_ds_medium = taco_ds_medium.add_column(ID_COLUMN, column=[i for i in range(len(taco_ds_medium))])
taco_ds_test = taco_ds_test.add_column(ID_COLUMN, column=[i for i in range(len(taco_ds_test))])
numina_ds = numina_ds.add_column(ID_COLUMN, column=[i for i in range(len(numina_ds))])

if args.as_test:
    num_samples = 10
    apps_ds = apps_ds.select(list(range(num_samples)))
    taco_ds_medium = taco_ds_medium.select(list(range(num_samples)))
    taco_ds_test = taco_ds_test.select(list(range(num_samples)))
    numina_ds = numina_ds.select(list(range(num_samples)))


# convert all to ray dataset
apps_ds = ray.data.from_huggingface(apps_ds)
apps_ds = apps_ds.map(
    coerce_types, fn_args=(apps_ds.schema(),)
)
taco_ds_medium = ray.data.from_huggingface(
    taco_ds_medium,
)
taco_ds_medium = taco_ds_medium.map(
    coerce_types, fn_args=(taco_ds_medium.schema(),)
)
taco_ds_test = ray.data.from_huggingface(
    taco_ds_test,
)
taco_ds_test = taco_ds_test.map(coerce_types, fn_args=(taco_ds_test.schema(),))
numina_ds = ray.data.from_huggingface(
    numina_ds,
)

# get subsets from numina based on the source column
numina_ds_amc_aime = numina_ds.filter(lambda x: x["source"] == "amc_aime")  # 4070
numina_ds_olympiads = numina_ds.filter(lambda x: x["source"] == "olympiads").limit(
    20000
)  # 20k
numina_ds_math = numina_ds.filter(lambda x: x["source"] == "math")  # 7477

# resume from save paths
dataset_names = [
    "apps",
    "taco_train",
    "taco_test",
    "numina_amc_aime",
    "numina_math",
    "numina_olympiads",
]

save_paths = [args.save_dir / f"sky-t1-preview-{dataset_names[i]}" for i in range(len(dataset_names))]
my_datasets = [apps_ds, taco_ds_medium, taco_ds_test, numina_ds_amc_aime, numina_ds_math, numina_ds_olympiads]
my_datasets = resume_from_save_dir(my_datasets, save_paths, "json", ID_COLUMN)

# repartition before processing
for i in range(len(my_datasets)):
    my_datasets[i] = my_datasets[i].repartition(num_blocks=None, target_num_rows_per_block=TARGET_NUM_ROWS_PER_BLOCK)



    # for i in range(len(my_datasets)):
    #     my_datasets[i] = my_datasets[i].limit(num_samples)

# these are user-defined simple preprocessing functions to go from entry -> prompt
preprocessors = [
    APPSPreprocessor,
    TACOPreprocessor,
    TACOPreprocessor,
    NUMINAPreprocessor,
    NUMINAPreprocessor,
    NUMINAPreprocessor,
]


scorer_configs = [
    dict(
        cls=APPSScorer, fn_constructor_kwargs=dict(response_column="formatted_response", backend=BACKEND)
    ),
    dict(
        cls=TACOScorer,
        fn_constructor_kwargs=dict(response_column="formatted_response", backend=BACKEND),
    ),
    dict(
        cls=TACOScorer,
        fn_constructor_kwargs=dict(response_column="formatted_response", backend=BACKEND),
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

for i, ds in enumerate(my_datasets):
    # 1. Preprocess and get model prompts
    if ds.count() == 0:
        print(f"Skipping {dataset_names[i]} because it has no samples")
        continue

    preprocess_cls = preprocessors[i]
    my_datasets[i] = ds.map(
        preprocess_cls,
        concurrency=5,
    )

    # 2. Get model responses

    config = vLLMEngineProcessorConfig(
        # model="Qwen/QwQ-32B-Preview",
        model_source="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        engine_kwargs=dict(
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_num_batched_tokens=4096,
            tensor_parallel_size=2,
        ),
        concurrency=2,
        batch_size=128,
    )

    processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["user_input"]},
            ],
            sampling_params=dict(
                temperature=0,
                max_tokens=MAX_TOKENS,
            ),
        ),
        postprocess=lambda row: dict(
            assistant_response=row["generated_text"],
            **row,  # This will return all the original columns in the dataset.
        ),
    )
    my_datasets[i] = processor(my_datasets[i])

    # 3. Reformat the examples into a structured format

    # define a configuration for the reformatter
    config = HttpRequestProcessorConfig(
        url="https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        # number of processors to run in parallel
        # Each handles a batch of requests
        concurrency=1,
        batch_size=16,
        # Throttle QPS to avoid rate limit errors
        qps=16,
        max_retries=6,
        base_retry_wait_time_in_s=1,
        session_factory=SESSION_FACTORY,
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
    my_datasets[i] = reformatter(my_datasets[i])

    # 4. Rejection Sampling based on scoring
    scorer_cls, fn_constructor_kwargs = (
        scorer_configs[i]["cls"],
        scorer_configs[i]["fn_constructor_kwargs"],
    )
    my_datasets[i] = my_datasets[i].map(
        scorer_cls, concurrency=4, fn_constructor_kwargs=fn_constructor_kwargs
    )
    score_column = scorer_cls.SCORE_COLUMN
    my_datasets[i] = my_datasets[i].filter(lambda x, sc=score_column: x[sc])

    # 5. Convert to ShareGPT format
    my_datasets[i] = my_datasets[i].map(
        convert_to_sharegpt_format,
        fn_kwargs=dict(
            prompt_column="user_input", response_column="formatted_response"
        ),
    )

    # 6. Save datasets
    dir_name = args.save_dir / f"sky-t1-preview-{dataset_names[i]}"
    # use absolute path while saving with ray data
    my_datasets[i].write_json(str(dir_name.expanduser().absolute()))


# 7. Union

# final_dataset = datasets[0].union(*datasets[1:])
# dir_name = f"data/sky-t1-preview-full"
# # save in folder as a single JSON file
# final_dataset.repartition(1).write_json(os.path.abspath(dir_name))
