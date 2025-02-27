import concurrent.futures
import copy
import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ray
from openai import OpenAI
from tqdm import tqdm
from vllm import LLM

from skythought.evals.batch import Pipeline, init_engine_from_config
from skythought.evals.batch.env_config import EnvConfig
from skythought.evals.batch.workload import EvalWorkload
from skythought.evals.common.entities import (
    Backend,
    BackendParameters,
    OpenAISamplingParams,
    RayLLMEngineArgs,
    SamplingParameters,
)
from skythought.evals.models import ModelConfig
from skythought.evals.tasks import (
    ConversationType,
    NUMINATaskHandler,
    TaskHandler,
)
from skythought.evals.util.metrics import pass_at_k
from skythought.evals.util.response import Response, SingleParsedResponse
from skythought.evals.util.results import SummaryResults, save_summary

logger = logging.getLogger(__name__)
module_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAY_CONFIG_RELATIVE_PATH = "ray_configs/ray_config.yaml"

RESULTS_FILENAME = "results.json"
SUMMARY_FILENAME = "summary.json"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_existing_results(result_file) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(result_file):
        return {}
    with open(result_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records)} existing results")
    return records


def save_results(
    result_filepath: os.PathLike, id_to_results: Dict[str, Dict[str, Any]]
):
    with open(result_filepath, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(id_to_results, indent=4, cls=NumpyEncoder, ensure_ascii=False)
        )


def fetch_response_openai(
    client: OpenAI,
    model_config: ModelConfig,
    sampling_params: OpenAISamplingParams,
    prompt,
):
    model_name = model_config.name
    # Ensure model_name has been resolved to a string
    assert model_name
    if model_name.startswith("o1") or model_name.startswith("o3"):
        # O1 doesn't support system prompt
        # NOTE: might want to implement this inside handler instead
        for p in prompt:
            p["role"] = "user"
        response = client.chat.completions.create(
            model=model_config.model_id,
            messages=prompt,
            n=sampling_params.n,
            reasoning_effort=sampling_params.reasoning_effort,
            max_completion_tokens=sampling_params.max_tokens,
        )
    else:
        if sampling_params.reasoning_effort:
            raise ValueError("Reasoning effort is only supported for reasoning models")
        response = client.chat.completions.create(
            model=model_config.model_id,
            messages=prompt,
            n=sampling_params.n,
            temperature=sampling_params.temperature,
            frequency_penalty=sampling_params.frequency_penalty,
            max_completion_tokens=sampling_params.max_tokens,
        )
    return response


def fetch_responses_ray(
    conversations,
    backend_args: RayLLMEngineArgs,
    model_config: ModelConfig,
    sampling_params: SamplingParameters,
):
    config = backend_args.get_ray_llm_config()
    config["model_id"] = model_config.model_id

    engine_cfg = init_engine_from_config(config)
    ds = ray.data.from_items([(idx, conv) for idx, conv in enumerate(conversations)])
    num_replicas = config["env_config"].get("num_replicas", 1)
    if ds.count() < config["env_config"].get("batch_size", 1):
        config["env_config"]["batch_size"] = math.ceil(ds.count() / num_replicas)
    if num_replicas > 1 and num_replicas > ds.num_blocks():
        ds = ds.repartition(num_partitions=num_replicas)
    workload = EvalWorkload(
        dataset=ds,
        sampling_params=sampling_params.to_dict(),
    )
    pipeline = Pipeline(
        engine_cfg,
        env_config=EnvConfig(**config["env_config"]),
    )
    ds = pipeline(workload)
    responses = ds.materialize()
    return responses


def _parse_response_for_idx(
    response: Response,
    sample_idx: int,
) -> Tuple[SingleParsedResponse, Dict[str, int]]:
    content = response.response[sample_idx].strip()
    response_entry = SingleParsedResponse(content=content)

    token_usage_for_response = {
        "completion_tokens": response.num_completion_tokens[sample_idx],
        "prompt_tokens": response.num_input_tokens,
    }
    return response_entry, token_usage_for_response


def inference(
    conversations: List[ConversationType],
    backend: Backend,
    backend_params: BackendParameters,
    model_config: ModelConfig,
    sampling_params: SamplingParameters,
    **kwargs,
):
    if backend == Backend.RAY:
        responses = fetch_responses_ray(
            conversations, backend_params.params, model_config, sampling_params
        )
        responses = [
            Response.from_ray_response(response) for response in responses.iter_rows()
        ]
        # NOTE: This deepcopy is needed to avoid a SIGSEV error related to object cleanup with the ray object store and
        # the later use of ProcessPoolExecutor - see here: https://github.com/NovaSky-AI/SkyThought/pull/63#discussion_r1941899714
        # TODO: revisit the underlying issue and remove the deepcopy if possible
        responses = copy.deepcopy(responses)
        responses = sorted(responses, key=lambda x: x.index)
    elif backend == Backend.OPENAI:
        llm = OpenAI(**backend_params.to_dict())
        assert isinstance(sampling_params.params, OpenAISamplingParams)
        fetch_partial = partial(
            fetch_response_openai,
            llm,
            model_config,
            sampling_params.params,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
            responses = list(e.map(fetch_partial, conversations))

        responses = [Response.from_openai_response(response) for response in responses]
    elif backend == Backend.VLLM:
        batch_size = kwargs.get("batch_size", 1)
        engine_kwargs = copy.deepcopy(backend_params.to_dict())
        engine_kwargs["model"] = model_config.model_id
        llm = LLM(**engine_kwargs)

        response_in_batches = [
            llm.chat(
                messages=conversations[i : i + batch_size],
                sampling_params=sampling_params.params,
                use_tqdm=True,
                add_generation_prompt=model_config.assistant_prefill is None,
                continue_final_message=model_config.assistant_prefill is not None,
            )
            for i in range(0, len(conversations), batch_size)
        ]
        responses = []
        for response_batch in response_in_batches:
            responses.extend(response_batch)
        responses = [Response.from_vllm_response(response) for response in responses]
    else:
        raise ValueError(f"Invalid backend: {backend}")

    return responses


def generate_responses_for_dataset(
    handler: TaskHandler,
    model_config: ModelConfig,
    backend: Backend,
    backend_params: BackendParameters,
    sampling_params: SamplingParameters,
    eval_data: pd.DataFrame,
    id_to_results: Dict[str, Dict[str, Any]],
    **kwargs,
) -> Tuple[Dict[str, Dict[str, Any]], List[int], List[int]]:
    """Generates responses for the given dataset

    Performs the following:
    1. Filter out the items already in `id_to_results` (if resuming).
    2. Build the input conversations.
    3. Perform inference.
    4. Return a dictionary with new results, as well as token usage statistics
    """
    remaining_data = handler.process_remaining_data(eval_data, id_to_results)
    logger.info(f"Generating results for {len(remaining_data)} problems.")

    if not remaining_data:
        logger.info("No remaining data to generate.")
        return id_to_results, [], []

    # Prepare conversations
    conversations = handler.make_conversations(
        remaining_data,
        model_config.system_prompt,
        model_config.user_template,
        model_config.assistant_prefill,
    )

    if not conversations:
        logger.info("No conversations to generate.")
        return id_to_results, [], []

    # Perform inference
    responses = inference(
        conversations, backend, backend_params, model_config, sampling_params, **kwargs
    )

    all_prompt_tokens = []
    all_completion_tokens = []

    unique_ids = [rd["_index"] for rd in remaining_data]
    for idx, response in enumerate(responses):
        # For each problem, we store N responses
        response_entries = []
        token_usages = []
        sum_completion_tokens = 0

        for sample_idx in range(sampling_params.params.n):
            response_entry, token_usage_for_response = _parse_response_for_idx(
                response,
                sample_idx,
            )
            response_entries.append(response_entry.to_dict())
            token_usages.append(token_usage_for_response)
            sum_completion_tokens += token_usage_for_response["completion_tokens"]

        unique_id = unique_ids[idx]

        # Update existing_results
        if unique_id not in id_to_results:
            id_to_results[unique_id] = remaining_data[idx]
            if isinstance(handler, NUMINATaskHandler):
                id_to_results[unique_id]["messages"] = ""
            id_to_results[unique_id]["prompt"] = (
                conversations[idx][1]["content"]
                if len(conversations[idx]) > 1
                else conversations[idx][0]["content"]
            )
            id_to_results[unique_id]["input_conversation"] = conversations[idx]

        id_to_results[unique_id]["responses"] = response_entries
        id_to_results[unique_id]["token_usages"] = token_usages

        all_prompt_tokens.append(response.num_input_tokens)
        all_completion_tokens.append(sum_completion_tokens)

    return id_to_results, all_prompt_tokens, all_completion_tokens


def score_responses(
    handler: TaskHandler,
    id_to_results: Dict[str, Dict[str, Any]],
    *,
    max_workers: int = 32,
) -> Tuple[float, Dict[str, List[int]], int]:
    """Computes correctness for model responses for the given task

    The 'id_to_results' dictionary is assumed to be a mapping between problem ID -> { responses: [...], ... },
    This is updated in-place.

    Returns:
       - overall accuracy
       - `id_to_scores` dictionary mapping IDs -> list of scores (used for metrics like pass@k)
       - total number of responses processed
    """
    if not id_to_results:
        return 0.0, {}, 0

    total_correct = 0
    total_finish = 0
    id_to_scores = {}

    # Figure out how many generations per problem
    N = len(next(iter(id_to_results.values()))["responses"])

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {}
        for unique_id, record in id_to_results.items():
            for i in range(N):
                content = record["responses"][i]["content"]
                future = executor.submit(handler.update_results, record, content)
                # track which problem and which response index
                future_to_info[future] = (unique_id, i)

        for future in tqdm(as_completed(future_to_info), total=len(future_to_info)):
            unique_id, i = future_to_info[future]
            new_response_entry = future.result()

            # Update correctness and reason in the original results dict
            id_to_results[unique_id]["responses"][i]["correctness"] = (
                new_response_entry["correctness"]
            )
            id_to_results[unique_id]["responses"][i]["reason"] = new_response_entry[
                "reason"
            ]

            # Track scores separately for metrics like pass@k
            # TODO (sumanthrh): this can be improved
            if unique_id not in id_to_scores:
                id_to_scores[unique_id] = [0 for _ in range(N)]
            id_to_scores[unique_id][i] = int(new_response_entry["correctness"])

            total_correct += new_response_entry["correctness"]
            total_finish += 1

    accuracy = round(total_correct / total_finish, 4) if total_finish else 0
    return accuracy, id_to_scores, total_finish


def score_responses_for_indices(
    handler: TaskHandler,
    id_to_results: Dict[str, Dict[str, Any]],
    *,
    indices: List[str],
) -> List[int]:
    """Computes correctness for model responses for the given task for the unique index `idx`.

    The 'id_to_results' dictionary is assumed to be a mapping between problem ID -> { responses: [...], ... },
    This is updated in-place.

    Returns:
       - list of scores
    """
    if not id_to_results:
        return []
    logger.info(f"Computing scores for {len(indices)} samples")
    for idx in indices:
        # Figure out how many generations per problem
        N = len(next(iter(id_to_results.values()))["responses"])
        record = id_to_results[idx]
        scores = []
        for i in range(N):
            content = record["responses"][i]["content"]
            response_entry = handler.update_results(record, content)

            # Update correctness and reason in the original results dict
            id_to_results[idx]["responses"][i]["correctness"] = response_entry[
                "correctness"
            ]
            id_to_results[idx]["responses"][i]["reason"] = response_entry["reason"]
            scores.append(response_entry["correctness"])
    return scores


def generate_and_score(
    handler: TaskHandler,
    model_config: ModelConfig,
    backend: Backend,
    backend_params: BackendParameters,
    sampling_params: SamplingParameters,
    output_dir: Path,
    start: int,
    end: int,
    run_config_dict: dict,
    **kwargs,
):
    result_file = output_dir / RESULTS_FILENAME
    summary_file = output_dir / SUMMARY_FILENAME

    eval_data = handler.load_and_filter_dataset(start, end)
    id_to_results = {}

    id_to_results, prompt_tokens, completion_tokens = generate_responses_for_dataset(
        handler=handler,
        model_config=model_config,
        backend=backend,
        backend_params=backend_params,
        sampling_params=sampling_params,
        eval_data=eval_data,
        id_to_results=id_to_results,
        **kwargs,
    )

    accuracy, id_to_scores, total_finish = score_responses(
        handler, id_to_results, max_workers=32
    )
    logger.info(f"Accuracy: {accuracy}")

    num_responses_total = len(id_to_results) * sampling_params.params.n

    pass_at_k_metrics = None
    if sampling_params.params.n > 1:
        pass_at_k_metrics = pass_at_k(sampling_params.params.n, id_to_scores)

    total_completion_tokens = int(sum(completion_tokens))
    total_prompt_tokens = int(sum(prompt_tokens))
    summary_data = SummaryResults(
        configuration=run_config_dict,
        total_completion_tokens=total_completion_tokens,
        total_prompt_tokens=total_prompt_tokens,
        avg_completion_tokens=(
            round(total_completion_tokens / num_responses_total, 3)
            if total_completion_tokens
            else 0
        ),
        avg_prompt_tokens=(
            round(total_prompt_tokens / num_responses_total, 3)
            if total_prompt_tokens
            else 0
        ),
        accuracy=accuracy,
        pass_at_k=pass_at_k_metrics,
    )

    save_summary(summary_file, summary_data)
    save_results(result_file, id_to_results)
    logger.info(f"Saved results to {result_file}")
    logger.info(f"Summary saved to {summary_file}")


def generate_and_save(
    handler: TaskHandler,
    model_config: ModelConfig,
    backend: Backend,
    backend_params: BackendParameters,
    sampling_params: SamplingParameters,
    output_dir: Path,
    start: int,
    end: int,
    run_config_dict: dict,
    resume_from: Optional[os.PathLike] = None,
    **kwargs,
):
    if resume_from is not None:
        resume_from = Path(resume_from)
        result_file = resume_from / RESULTS_FILENAME
        summary_file = resume_from / SUMMARY_FILENAME
        id_to_results = load_existing_results(result_file)
    else:
        id_to_results = {}
        result_file = output_dir / RESULTS_FILENAME
        summary_file = output_dir / SUMMARY_FILENAME

    eval_data = handler.load_and_filter_dataset(start, end)

    # Step 3: generate responses (no scoring here)
    id_to_results, prompt_tokens, completion_tokens = generate_responses_for_dataset(
        handler=handler,
        model_config=model_config,
        backend=backend,
        backend_params=backend_params,
        sampling_params=sampling_params,
        eval_data=eval_data,
        id_to_results=id_to_results,
        **kwargs,
    )

    total_completion_tokens = int(sum(completion_tokens))
    total_prompt_tokens = int(sum(prompt_tokens))
    num_responses_total = len(eval_data) * sampling_params.params.n
    summary_data = SummaryResults(
        configuration=run_config_dict,
        total_completion_tokens=total_completion_tokens,
        total_prompt_tokens=total_prompt_tokens,
        avg_completion_tokens=(
            round(total_completion_tokens / num_responses_total, 3)
            if total_completion_tokens
            else 0
        ),
        avg_prompt_tokens=(
            round(total_prompt_tokens / num_responses_total, 3)
            if total_prompt_tokens
            else 0
        ),
    )

    save_summary(summary_file, summary_data)
    save_results(result_file, id_to_results)

    logger.info(f"Saved results to {result_file}")
    logger.info(f"Saved summary to {summary_file}")


def score_results(
    handler: TaskHandler,
    run_dir: Path,
    run_summary: SummaryResults,
    indices: Optional[List[str]] = None,
) -> None:
    # load existing results
    result_file = run_dir / RESULTS_FILENAME
    summary_file = run_dir / SUMMARY_FILENAME
    id_to_results = load_existing_results(result_file)
    logger.info(f"Loaded {len(id_to_results)} existing results for scoring.")

    if not indices:
        accuracy, id_to_scores, total_finish = score_responses(handler, id_to_results)
    else:
        N = len(next(iter(id_to_results.values()))["responses"])
        score_responses_for_indices(handler, id_to_results, indices=indices)
        id_to_scores = {
            index: [
                id_to_results[index]["responses"][i]["correctness"] for i in range(N)
            ]
            for index in id_to_results
        }
        accuracy = round(
            sum(map(sum, id_to_scores.values())) / (len(id_to_scores) * N), 4
        )

    sample_count = 0
    if id_to_results:
        sample_count = len(next(iter(id_to_results.values()))["responses"])
    pass_at_k_metrics = None
    if sample_count > 1:
        pass_at_k_metrics = pass_at_k(sample_count, id_to_scores)

    run_summary.accuracy = accuracy
    run_summary.pass_at_k = pass_at_k_metrics

    logger.info(f"Accuracy: {accuracy}")
    save_summary(summary_file, run_summary)

    save_results(result_file, id_to_results)
    logger.info(f"Scored results saved to {result_file}")
