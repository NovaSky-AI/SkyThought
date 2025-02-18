import json
import multiprocessing
from multiprocessing import Manager

import numpy as np
from skythought_evals.util.common import has_code

from ..base import TaskHandler
from .taco_util import run_test as taco_run_test


class TACOTaskHandler(TaskHandler):

    def generate_prompt(self, problem):
        prompt = problem["question"]
        starter_code = (
            None if len(problem["starter_code"]) == 0 else problem["starter_code"]
        )
        try:
            input_outpout = json.loads(problem["input_output"])
            fn_name = (
                None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
            )
        except ValueError:
            fn_name = None

        _input = self.task_config.templating_parameters["initial_template"].format(
            prompt=prompt
        )

        if starter_code:
            _input = self.task_config.templating_parameters[
                "starter_code_template"
            ].format(input=_input, starter_code=starter_code)
        else:
            _input = self.task_config.templating_parameters["initial_template"].format(
                prompt=prompt
            )
        if (not fn_name) and (not starter_code):
            _input = self.task_config.templating_parameters["stdin_template"].format(
                input=_input
            )
        else:
            _input = self.task_config.templating_parameters["call_template"].format(
                input=_input
            )

        return _input

    def check_correctness(self, problem, generation):
        TIME_OUT = 300

        def _temp_run(problem, generation, debug, result):
            try:
                result.append(taco_run_test(problem, test=generation, debug=debug))
            except Exception as e:
                print(f"Error in _temp_run: {e}")

        manager = Manager()
        result = manager.list()
        p = multiprocessing.Process(
            target=_temp_run, args=(problem, generation, False, result)
        )
        p.start()
        p.join(timeout=TIME_OUT + 1)
        if p.is_alive():
            p.kill()
        return bool(result and np.all(result[0]))

    def update_results(self, problem, response):
        # Initialize the response structure
        response_entry = {
            "content": response,
            "correctness": None,
            "reason": None,
        }
        code_filter_result = has_code(response)
        if len(code_filter_result) == 0:
            response_entry["correctness"] = False
            response_entry["reason"] = "Does not contain code component."
        else:
            last_code = code_filter_result[-1]
            curr_res = self.check_correctness(problem, generation=last_code)
            if curr_res:
                response_entry["correctness"] = True
                response_entry["reason"] = ""
            else:
                response_entry["correctness"] = False
                response_entry["reason"] = "Code is incorrect."

        return response_entry

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None
    ):
        dataset = self.load_dataset(subset=subset, split=split).to_pandas()
        if difficulty or "difficulty" in self.task_config.preprocess_config:
            difficulty = (
                difficulty
                if difficulty
                else self.task_config.preprocess_config["difficulty"]
            )
            dataset = dataset.filter(
                lambda example: example["difficulty"] == difficulty
            )

        return dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]
