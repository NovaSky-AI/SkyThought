import multiprocessing
from multiprocessing import Manager
from typing import Any, Dict, Literal

import ray

from skythought.evals.tasks.livecodebench.livecodebench_util import restore_original_references, save_original_references
from skythought.evals.util.common import has_code

from ..base import Scorer
from .taco_util import run_test as taco_run_test


class TACOScorer(Scorer):
    SCORE_COLUMN = "taco_score"

    def __init__(
        self,
        response_column="response",
        input_output_column="input_output",
        backend: Literal["ray", "mp"] = "ray",
    ) -> None:
        super().__init__()
        self.response_column = response_column
        self.input_output_column = input_output_column
        self.backend = backend
        if backend not in ["ray", "mp"]:
            raise ValueError(f"Unsupported backend for launching tests: {backend}")

    def score(self, row: Dict[str, Any]):
        # Initialize the response structure
        response = row[self.response_column]
        input_outputs = row[self.input_output_column]

        code_filter_result = has_code(response)
        if len(code_filter_result) == 0:
            return {self.SCORE_COLUMN: False}
        else:
            last_code = code_filter_result[-1]
            if self.backend == "mp":
                curr_res, _ = _taco_run_tests_mp(input_outputs, generation=last_code)
            else:
                curr_res, _ = _taco_run_tests_ray(input_outputs, generation=last_code)

            if curr_res:
                return {self.SCORE_COLUMN: True}
            else:
                return {self.SCORE_COLUMN: False}


def _temp_run(input_outputs, generation, debug, result):
    try:
        result.append(taco_run_test(input_outputs, test=generation, debug=debug))
    except Exception as e:
        print(f"Error in _temp_run: {e}")

def _taco_run_tests_mp(input_outputs, generation):

    # run the test in a separate process for safety
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    result = manager.list()
    p = ctx.Process(
        target=_temp_run, args=(input_outputs, generation, False, result)
    )
    p.start()
    p.join()
    if p.is_alive():
        p.kill()
    # get the first element in ListProxy - this is the result
    result = result[0]
    return bool(result and all(result)), result


# NOTE (sumanthrh): We make sure that scoring for code generation is run on a separate process for isolation
# We need to run scoring for each data sample in a separate process. Since ray doesn't play well with
# multiprocessing, we launch scoring as a standalone ray task. Further, to make sure that resource requests
# don't blow up for batched processing- for example, in a ray data pipeline, we reduce `num_cpus` to 0.001 from the default
# value of 1. That way, scoring for different samples can timeshare on the same set of cpus.
@ray.remote(num_cpus=0.001)
def _temp_run_ray(input_outputs, generation, debug):
    result = []
    try:
        result = taco_run_test(input_outputs, test=generation, debug=debug)
    except Exception as e:
        print(f"Error in _temp_run: {e}")
    return result


def _taco_run_tests_ray(input_outputs, generation):
    # run the test in a separate process for safety
    obj_ref = _temp_run_ray.remote(input_outputs, generation, False)
    result = ray.get(obj_ref)
    return bool(result and all(result)), result
