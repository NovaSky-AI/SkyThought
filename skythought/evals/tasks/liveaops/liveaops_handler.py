from skythought.evals.util.math_parsing_util import (
    extract_answer,
    math_equal,
    strip_answer_string,
)

from ..math.math_handler import MathTaskHandler


class LiveAOPSTaskHandler(MathTaskHandler):
    def generate_prompt(self, problem):
        return self.task_config.templating_parameters["template"].format(**problem)

    def check_correctness(self, problem, generation):
        # no preprocessing needed
        answer = problem[self.task_config.answer_key]
        pred = extract_answer(generation)
        pred = strip_answer_string(pred)
        return math_equal(pred, answer)

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None
    ):
        assert difficulty is None, "LiveAOPS does not support `difficulty` argument"
        dataset = self.load_dataset(subset=subset, split=split).to_pandas()
        return dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]
