import concurrent.futures
import copy
import json
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ray
from openai import OpenAI
from skythought_evals.batch import Pipeline, init_engine_from_config
from skythought_evals.batch.env_config import EnvConfig
from skythought_evals.batch.workload import EvalWorkload
from skythought_evals.common.entities import (
    Backend,
    BackendParameters,
    OpenAISamplingParams,
    RayLLMEngineArgs,
    SamplingParameters,
)
from skythought_evals.models import ModelConfig
from skythought_evals.tasks import (
    ConversationType,
    NUMINATaskHandler,
    TaskHandler,
)
from skythought_evals.util.metrics import pass_at_k
from skythought_evals.util.response import Response, SingleParsedResponse
from skythought_evals.util.results import SummaryResults, save_summary
from tqdm import tqdm
from vllm import LLM

logger = logging.getLogger(__name__)
module_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAY_CONFIG_RELATIVE_PATH = "ray_configs/ray_config.yaml"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_existing_results(result_file):
    if not os.path.exists(result_file):
        return {}
    with open(result_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    return records


def fetch_response_openai(
    client: OpenAI,
    model_config: ModelConfig,
    sampling_params: OpenAISamplingParams,
    prompt,
):
    model_name = model_config.model_id.replace("openai/", "")
    if "o1" in model_name:
        # O1 doesn't support system prompt
        # NOTE: might want to implement this inside handler instead
        for p in prompt:
            p["role"] = "user"

        response = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            n=sampling_params.n,
            temperature=sampling_params.temperature,
            max_tokens=sampling_params.max_tokens,
            reasoning_effort=sampling_params.reasoning_effort,
            frequency_penalty=sampling_params.frequency_penalty,
            max_completion_tokens=sampling_params.max_tokens,
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            n=sampling_params.n,
            temperature=sampling_params.temperature,
            max_tokens=sampling_params.max_tokens,
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
        llm = OpenAI(**backend_params)
        fetch_partial = partial(
            fetch_response_openai,
            llm,
            model_config,
            sampling_params,
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
    # flush logs to stdout
    sys.stdout.flush()

    return responses


def generate_responses_for_dataset(
    handler: TaskHandler,
    model_config: ModelConfig,
    backend: Backend,
    backend_params: BackendParameters,
    sampling_params: SamplingParameters,
    eval_data: pd.DataFrame,
    existing_results: Dict[int, Dict[str, Any]],
    **kwargs,
) -> Tuple[Dict[int, Dict[str, Any]], List[int], List[int]]:
    """Generates responses for the given dataset

    Performs the following:
    1. Filter out the items already in `existing_results` (if resuming).
    2. Build the input conversations.
    3. Perform inference.
    4. Return a dictionary with new results, as well as token usage statistics
    """
    remaining_data = handler.process_remaining_data(eval_data, existing_results)
    logger.info(f"Generating results for {len(remaining_data)} problems.")

    if not remaining_data:
        logger.info("No remaining data to generate.")
        return existing_results, [], []

    # Prepare conversations
    conversations = handler.make_conversations(
        remaining_data,
        model_config.system_prompt,
        model_config.user_template,
        model_config.assistant_prefill,
    )

    if not conversations:
        logger.info("No conversations to generate.")
        return existing_results, [], []

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
        if unique_id not in existing_results:
            existing_results[unique_id] = remaining_data[idx]
            if isinstance(handler, NUMINATaskHandler):
                existing_results[unique_id]["messages"] = ""
            existing_results[unique_id]["prompt"] = (
                conversations[idx][1]["content"]
                if len(conversations[idx]) > 1
                else conversations[idx][0]["content"]
            )
            existing_results[unique_id]["input_conversation"] = conversations[idx]

        existing_results[unique_id]["responses"] = response_entries
        existing_results[unique_id]["token_usages"] = token_usages

        all_prompt_tokens.append(response.num_input_tokens)
        all_completion_tokens.append(sum_completion_tokens)

    return existing_results, all_prompt_tokens, all_completion_tokens


def score_responses(
    handler: TaskHandler, results: Dict[int, Dict[str, Any]], max_workers: int = 32
) -> Tuple[float, Dict[int, List[int]], int]:
    """Computes correctness for model responses for the given task

    The 'results' dictionary is assumed to be problem IDs -> { responses: [...], ... },

    Returns:
       - overall accuracy
       - `id_to_scores` dictionary mapping IDs -> list of scores (used for metrics like pass@k)
       - total number of responses processed
    """
    if not results:
        return 0.0, {}, 0

    total_correct = 0
    total_finish = 0
    id_to_scores = {}

    # Figure out how many generations per problem
    N = len(next(iter(results.values()))["responses"])

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {}
        for unique_id, record in results.items():
            for i in range(N):
                content = record["responses"][i]["content"]
                future = executor.submit(handler.update_results, record, content)
                # track which problem and which response index
                future_to_info[future] = (unique_id, i)

        for future in tqdm(as_completed(future_to_info), total=len(future_to_info)):
            unique_id, i = future_to_info[future]
            new_response_entry = future.result()

            # Update correctness and reason in the original results dict
            results[unique_id]["responses"][i]["correctness"] = new_response_entry[
                "correctness"
            ]
            results[unique_id]["responses"][i]["reason"] = new_response_entry["reason"]

            # Track scores separately for metrics like pass@k
            # TODO (sumanthrh): this can be improved
            if unique_id not in id_to_scores:
                id_to_scores[unique_id] = [0 for _ in range(N)]
            id_to_scores[unique_id][i] = new_response_entry["correctness"]

            total_correct += new_response_entry["correctness"]
            total_finish += 1

    accuracy = round(total_correct / total_finish, 4) if total_finish else 0
    return accuracy, id_to_scores, total_finish


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
    result_file = output_dir / "results.json"
    summary_file = output_dir / "summary.json"

    eval_data = handler.load_and_filter_dataset(start, end)
    results = {}

    results, prompt_tokens, completion_tokens = generate_responses_for_dataset(
        handler=handler,
        model_config=model_config,
        backend=backend,
        backend_params=backend_params,
        sampling_params=sampling_params,
        eval_data=eval_data,
        existing_results=results,
        **kwargs,
    )

    accuracy, id_to_scores, total_finish = score_responses(
        handler, results, max_workers=32
    )
    logger.info(f"Accuracy: {accuracy}")

    num_responses_total = len(results) * sampling_params.params.n

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

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)

    save_summary(summary_file, summary_data)
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
    breakpoint()
    if resume_from is not None:
        resume_from = Path(resume_from)
        result_file = resume_from / "results.json"
        summary_file = resume_from / "summary.json"
        results = load_existing_results(result_file)
    else:
        results = {}
        result_file = output_dir / "results.json"
        summary_file = output_dir / "summary.json"

    eval_data = handler.load_and_filter_dataset(start, end)

    # Step 3: generate responses (no scoring here)
    results, prompt_tokens, completion_tokens = generate_responses_for_dataset(
        handler=handler,
        model_config=model_config,
        backend=backend,
        backend_params=backend_params,
        sampling_params=sampling_params,
        eval_data=eval_data,
        existing_results=results,
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
    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    logger.info(f"Saved results to {result_file}")
    logger.info(f"Saved summary to {summary_file}")


def score_results(
    handler: TaskHandler, run_dir: Path, run_summary: SummaryResults
) -> None:
    # load existing results
    result_file = run_dir / "results.json"
    summary_file = run_dir / "summary.json"
    results = load_existing_results(result_file)
    logger.info(f"Loaded {len(results)} existing results for scoring.")

    accuracy, id_to_scores, total_finish = score_responses(handler, results)

    logger.info(f"Accuracy: {accuracy}")

    sample_count = 0
    if results:
        sample_count = len(next(iter(results.values()))["responses"])
    pass_at_k_metrics = None
    if sample_count > 1:
        pass_at_k_metrics = pass_at_k(sample_count, id_to_scores)

    run_summary.accuracy = accuracy
    run_summary.pass_at_k = pass_at_k_metrics
    save_summary(summary_file, run_summary)

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    logger.info(f"Re-scored results saved to {result_file}")
