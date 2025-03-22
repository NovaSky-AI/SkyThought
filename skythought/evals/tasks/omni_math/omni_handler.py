from skythought.evals.util.math_parsing_util import (
    extract_answer,
    math_equal,
    strip_answer_string,
)

from ..math.math_handler import MathTaskHandler


class OMNIMathTaskHandler(MathTaskHandler):
    def generate_prompt(self, problem):
        return self.task_config.templating_parameters["template"].format(**problem)

    def check_correctness(self, problem, generation):
        # no preprocessing needed
        answer = problem[self.task_config.answer_key]
        pred = extract_answer(generation)
        pred = strip_answer_string(pred)
        return math_equal(pred, answer)
