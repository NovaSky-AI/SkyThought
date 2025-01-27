from skythought_evals.util.math_parsing_util import (
    extract_answer,
    math_equal,
    strip_answer_string,
)

from ..math.math_handler import MathTaskHandler


class OlympiadBenchMathTaskHandler(MathTaskHandler):
    def check_correctness(self, problem, generation):
        # all problems have final answer in a list
        answer = strip_answer_string(problem[self.task_config.answer_key][0])
        pred = extract_answer(generation)
        pred = strip_answer_string(pred)
        return math_equal(pred, answer)
