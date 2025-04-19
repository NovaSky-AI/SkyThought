import copy
import json
import multiprocessing
from multiprocessing import Manager
from typing import Any, Dict, List, Literal

import numpy as np
import ray
from ray.exceptions import GetTimeoutError

from skythought.evals.scoring.base import Scorer
from skythought.evals.util.common import has_code

from .apps_util import run_test as apps_run_test


class APPSScorer(Scorer):
    """Scorer for the APPS dataset

    For the APPS dataset format, see https://huggingface.co/datasets/codeparrot/apps

    Args:
        response_column: The column name for the response (str).
        solutions_column: The column name with solutions (str).
        input_output_column: The column name with the test inputs and outputs (str).
        keyword_args_column: The column name for the keyword arguments to the instruction builder (str).
        key_column: The column name for the unique identifier (str).
        backend: The backend to use for scoring. Supports "ray" or "mp" (str).
    """

    SCORE_COLUMN = "apps_score"
    # timeout per sample
    TIMEOUT = 10

    def __init__(
        self,
        response_column="response",
        solutions_column="solutions",
        input_output_column="input_output",
        backend: Literal["mp", "ray"] = "ray",
    ) -> None:
        super().__init__()
        self.response_column = response_column
        self.solutions_column = solutions_column
        self.input_output_column = input_output_column
        self.backend = backend
        if self.backend not in ["mp", "ray"]:
            raise ValueError(f"Invalid backend for `APPSScorer`: {self.backend}")

    def score(self, row: Dict[str, Any]):

        code_filter_result = has_code(row[self.response_column])
        if len(code_filter_result) == 0:
            return {self.SCORE_COLUMN: False}
        else:
            last_code = code_filter_result[-1]
            problem_to_check = copy.deepcopy(row)
            problem_to_check[self.input_output_column] = json.loads(
                row[self.input_output_column]
            )
            try:
                problem_to_check[self.solutions_column] = json.loads(
                    row[self.solutions_column]
                )
            except Exception:
                problem_to_check[self.solutions_column] = ""

        if self.backend == "ray":
            score = _run_test_ray(
                problem_to_check[self.input_output_column],
                last_code,
                self.TIMEOUT,
                False,
            )
        else:
            score = _run_test_mp(
                problem_to_check[self.input_output_column],
                last_code,
                self.TIMEOUT,
                False,
            )
        return {self.SCORE_COLUMN: score}


# NOTE (sumanthrh): We make sure that scoring for code generation is run on a separate process for isolation
# We need to run scoring for each data sample in a separate process. Since ray doesn't play well with
# multiprocessing, we launch scoring as a standalone ray task. Further, to make sure that resource requests
# don't blow up for batched processing- for example, in a ray data pipeline, we reduce `num_cpus` to 0.01 from the default
# value of 1. That way, scoring for different samples can timeshare on the same set of cpus.
@ray.remote(num_cpus=0.001)
def _temp_run_ray(input_outputs, generation, debug) -> List[bool]:
    try:
        result: List[bool] = apps_run_test(input_outputs, test=generation, debug=debug)
        return result
    except Exception:
        pass
    return []


def _run_test_ray(input_outputs, generation, timeout, debug):
    try:
        result = ray.get(
            _temp_run_ray.remote(input_outputs, generation, debug),
            timeout=timeout + 1,
        )
    except GetTimeoutError:
        result = []
    return bool(result and np.all(result))

def _temp_run(input_outputs, generation, debug, result):
    try:
        result.append(
            apps_run_test(input_outputs=input_outputs, test=generation, debug=debug)
        )
    except Exception:
        pass

def _run_test_mp(input_outputs, generation, timeout, debug):
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    result = manager.list()
    p = ctx.Process(
        target=_temp_run, args=(input_outputs, generation, False, result)
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    return bool(result and np.all(result[0]))
