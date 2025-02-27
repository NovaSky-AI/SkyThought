import copy
import json
from typing import Any, Dict, Literal

import numpy as np
import ray
from ray.exceptions import GetTimeoutError

from skythought.evals.scoring.base import Scorer
from skythought.evals.tasks.apps.apps_util import run_test as apps_run_test
from skythought.evals.util.common import has_code


class APPSScorer(Scorer):
    SCORE_COLUMN = "apps_score"

    def __init__(
        self,
        response_column="response",
        answer_column="solutions",
        input_column="input_output",
        backend: Literal["mp", "ray"] = "ray",
    ) -> None:
        super().__init__()
        self.response_column = response_column
        self.answer_column = answer_column
        self.input_column = input_column
        self.backend = backend

    def score(self, row: Dict[str, Any]):
        TIMEOUT = 10
        code_filter_result = has_code(row[self.response_column])
        if len(code_filter_result) == 0:
            return {self.SCORE_COLUMN: False}
        else:
            last_code = code_filter_result[-1]
            problem_to_check = copy.deepcopy(row)
            problem_to_check[self.input_column] = json.loads(row[self.input_column])
            try:
                problem_to_check[self.answer_column] = json.loads(
                    row[self.answer_column]
                )
            except Exception:
                problem_to_check[self.answer_column] = ""

        @ray.remote
        def _temp_run(problem, generation, debug):
            try:
                result = apps_run_test(problem=problem, test=generation, debug=debug)
                return result
            except Exception:
                pass

        try:
            result = ray.get(
                _temp_run.remote(problem_to_check, last_code, False),
                timeout=TIMEOUT + 1,
            )
        except GetTimeoutError:
            result = []

        score = bool(result and np.all(result[0]))
        return {self.SCORE_COLUMN: score}
