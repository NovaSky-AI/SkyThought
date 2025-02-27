import asyncio
import copy
from typing import Any, AsyncIterator, Dict, List, Literal, Tuple

from skythought.evals.util.common import has_code

from ..base import BatchScorer, Scorer
from .livecodebench_util import (
    _ray_wrapper,
    has_test_type,
    post_process_code,
    unsafe_lcb_runTests_mp,
    unsafe_lcb_runTests_ray,
)


class LiveCodeBenchScorer(Scorer):
    """Scorer for LiveCodeBench

    For the LiveCodeBench dataset format, see https://huggingface.co/datasets/livecodebench/code_generation_lite

    Args:
        question_content_column: The column name for the question (str).
        private_test_cases_column: The column name for the private test cases (str).
        public_test_cases_column: The column name for the public test cases (str).
        starter_code_column: The column name for the starter code (str).
        difficulty_column: The column name for the difficulty level (str).
        question_id_column: The column name for the question id (str).
        response_column: The column name for the response (str).
        backend: The backend to use for scoring. Supports "ray" or "mp" (str).
    """

    TIMEOUT = 6
    SCORE_COLUMN = "livecodebench_score"

    def __init__(
        self,
        question_content_column: str = "question_content",
        private_test_cases_column: str = "private_test_cases",
        public_test_cases_column: str = "public_test_cases",
        starter_code_column: str = "starter_code",
        difficulty_column: str = "difficulty",
        question_id_column: str = "question_id",
        response_column: str = "response",
        backend: Literal["ray", "mp"] = "ray",
    ):

        self.question_content_column = question_content_column
        self.private_test_cases_column = private_test_cases_column
        self.public_test_cases_column = public_test_cases_column
        self.starter_code_column = starter_code_column
        self.difficulty_column = difficulty_column
        self.question_id_column = question_id_column
        self.response_column = response_column
        self.backend = backend

    def score(self, row: dict) -> Dict[str, Any]:
        row = self.map_to_example(row)

        code_filter_result = has_code(row[self.response_column])
        last_code = None
        if len(code_filter_result) == 0:
            return {self.SCORE_COLUMN: False}
        else:
            last_code = code_filter_result[-1]
            problem_to_check = copy.deepcopy(row)

        if self.backend == "ray":
            result_list = unsafe_lcb_runTests_ray(
                problem_to_check,
                post_process_code(last_code),
                self.TIMEOUT,
                runtime_debug=False,
                is_extracted=not row["is_stdin"],
            )
        else:
            result_list = unsafe_lcb_runTests_mp(
                problem_to_check,
                post_process_code(last_code),
                self.TIMEOUT,
                runtime_debug=False,
                is_extracted=not row["is_stdin"],
            )
        details = [r[0] for r in result_list]
        all_passed = all(details)

        result = ""
        if result_list and all_passed:
            result = "passed"

        return {self.SCORE_COLUMN: result == "passed", "all_results": result_list}

    @property
    def expected_keys(self) -> List[str]:
        return [
            self.question_content_column,
            self.private_test_cases_column,
            self.public_test_cases_column,
            self.difficulty_column,
            self.question_id_column,
            self.starter_code_column,
            self.response_column,
        ]

    def map_to_example(self, row):
        return {
            "prompt": row[self.question_content_column],
            "test": row[self.private_test_cases_column],
            "entry_point": row[self.starter_code_column],
            "canonical_solution": "",  # seems like live code bench lite does not have this field
            "task_id": row[self.question_id_column],
            "is_stdin": has_test_type(row[self.public_test_cases_column], "stdin"),
            "public_test_cases": row[self.public_test_cases_column],
            "difficulty": row[self.difficulty_column],
            self.response_column: row[self.response_column],
        }


class LiveCodeBenchBatchScorer(BatchScorer):
    """Batch scorer for LiveCodeBench

    For the LiveCodeBench dataset format, see https://huggingface.co/datasets/livecodebench/code_generation_lite

    Args:
        question_content_column: The column name for the question (str).
        private_test_cases_column: The column name for the private test cases (str).
        public_test_cases_column: The column name for the public test cases (str).
        starter_code_column: The column name for the starter code (str).
        difficulty_column: The column name for the difficulty level (str).
        question_id_column: The column name for the question id (str).
        response_column: The column name for the response (str).
    """

    TIMEOUT = 6
    SCORE_COLUMN = "livecodebench_score"

    def __init__(
        self,
        question_content_column: str = "question_content",
        private_test_cases_column: str = "private_test_cases",
        public_test_cases_column: str = "public_test_cases",
        starter_code_column: str = "starter_code",
        difficulty_column: str = "difficulty",
        question_id_column: str = "question_id",
        response_column: str = "response",
    ):
        self.question_content_column = question_content_column
        self.private_test_cases_column = private_test_cases_column
        self.public_test_cases_column = public_test_cases_column
        self.starter_code_column = starter_code_column
        self.difficulty_column = difficulty_column
        self.question_id_column = question_id_column
        self.response_column = response_column

    async def score(self, rows: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:

        inputs = []
        ids = []
        for row in rows:
            row = self.map_to_example(row)
            code_filter_result = has_code(row[self.response_column])
            last_code = None
            if len(code_filter_result) == 0:
                yield {
                    self.INTERNAL_IDX_KEY: row[self.INTERNAL_IDX_KEY],
                    self.SCORE_COLUMN: False,
                    "reason": "No code found",
                }
            else:
                last_code = code_filter_result[-1]
                problem_to_check = copy.deepcopy(row)

            inputs.append(
                {
                    "problem": problem_to_check,
                    "completion": post_process_code(last_code),
                    "timeout": self.TIMEOUT,
                    "runtime_debug": False,
                    "is_extracted": row["is_stdin"],
                }
            )
            ids.append(row[self.INTERNAL_IDX_KEY])

        async for output in _unsafe_lcb_runTests_ray_batch(ids, inputs):
            idx, result_list = output
            details = [r[0] for r in result_list]
            all_passed = all(details)

            result = ""
            if result_list and all_passed:
                result = "passed"

            yield {
                self.INTERNAL_IDX_KEY: idx,
                self.SCORE_COLUMN: result == "passed",
                "reason": [r[1] for r in result_list],
            }

    def map_to_example(self, row):
        return {
            "prompt": row[self.question_content_column],
            "test": row[self.private_test_cases_column],
            "entry_point": row[self.starter_code_column],
            "canonical_solution": "",  # seems like live code bench lite does not have this field
            "task_id": row[self.question_id_column],
            "is_stdin": has_test_type(row[self.public_test_cases_column], "stdin"),
            "public_test_cases": row[self.public_test_cases_column],
            "difficulty": row[self.difficulty_column],
            self.response_column: row[self.response_column],
            self.INTERNAL_IDX_KEY: row[self.INTERNAL_IDX_KEY],
        }


async def _unsafe_lcb_runTests_ray_batch(
    ids, inputs
) -> AsyncIterator[Tuple[int, List[Tuple[bool, str, str, float]]]]:
    refs = []
    for idx, _input in zip(ids, inputs):
        problem = _input["problem"]
        completion = _input["completion"]
        timeout = _input["timeout"]
        runtime_debug = _input["runtime_debug"]
        is_extracted = _input["is_extracted"]
        test_cases = problem["test"]

        result_ref = _ray_wrapper.remote(
            test_cases, completion, timeout, runtime_debug, is_extracted, idx
        )
        refs.append(result_ref)

    futs = [asyncio.wrap_future(ref.future()) for ref in refs]
    for fut in asyncio.as_completed(futs):
        idx, result = await fut
        _input = inputs[ids.index(idx)]
        ## This is supposed to be the case where not all test passed in the given timeout
        for _i in range(len(_input["problem"]["test"]) - len(result)):
            result.append((False, "Time out!.", "Error: Time out!", float("inf")))
        yield idx, result
