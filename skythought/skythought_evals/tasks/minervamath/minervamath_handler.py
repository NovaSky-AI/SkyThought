from skythought_evals.util.math_parsing_util import (
    extract_answer,
    math_equal,
    strip_answer_string,
)

from ..math.math_handler import MathTaskHandler


class MinervaMathTaskHandler(MathTaskHandler):

    def check_correctness(self, problem, generation):
        answer = extract_answer(problem[self.task_config.answer_key])
        answer = strip_answer_string(answer)

        pred = extract_answer(generation)
        pred = strip_answer_string(pred)
        return math_equal(pred, answer)
