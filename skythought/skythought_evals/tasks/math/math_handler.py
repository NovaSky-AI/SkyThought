from skythought_evals.util.math_parsing_util import (
    extract_answer,
    math_equal,
    strip_answer_string,
)

from ..base import TaskHandler


class MathTaskHandler(TaskHandler):
    def generate_prompt(self, problem):
        return self.task_config.templating_parameters["template"].format(**problem)

    def check_correctness(self, problem, generation):
        answer = strip_answer_string(problem[self.task_config.answer_key])
        pred = extract_answer(generation)
        pred = strip_answer_string(pred)
        return math_equal(pred, answer)

    def update_results(self, problem, response):
        # Initialize the response structure
        response_entry = {
            "content": response,
            "correctness": None,
            "reason": None,
        }
        curr_res = self.check_correctness(problem, generation=response)
        if curr_res:
            response_entry["correctness"] = True
            response_entry["reason"] = ""
        else:
            response_entry["correctness"] = False
            response_entry["reason"] = "Solution is incorrect."

        return response_entry

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None
    ):
        dataset = self.load_dataset(subset=subset, split=split).to_pandas()
        return dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]
