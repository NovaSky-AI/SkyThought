import copy
import json
import multiprocessing
from multiprocessing import Manager
from typing import Any, Dict, List, Optional

import numpy as np
from skythought_evals.util.common import has_code

from ..apps.apps_util import run_test as apps_run_test
from ..base import TaskHandler


class APPSTaskHandler(TaskHandler):

    def generate_prompt(self, test_case, prompt, starter_code=None):
        if not test_case.get("fn_name"):
            _input = self.task_config.templating_parameters[
                "with_fn_name_template"
            ].format(prompt=prompt)
        else:
            _input = self.task_config.templating_parameters[
                "without_fn_name_template"
            ].format(prompt=prompt)

        if starter_code is not None:
            _input = self.task_config.templating_parameters[
                "with_starter_code_template"
            ].format(input=_input, starter_code=starter_code)
        return _input

    def check_correctness(self, problem, generation):
        TIMEOUT = 10

        def _temp_run(problem, generation, debug, result):
            try:
                result.append(
                    apps_run_test(problem=problem, test=generation, debug=debug)
                )
            except Exception:
                pass

        manager = Manager()
        result = manager.list()
        p = multiprocessing.Process(
            target=_temp_run, args=(problem, generation, False, result)
        )
        p.start()
        p.join(timeout=TIMEOUT + 1)
        if p.is_alive():
            p.kill()
        return bool(result and np.all(result[0]))

    def update_results(self, problem, response):
        if not isinstance(response, str):
            response = response.outputs[0].text.strip()
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
            problem_to_check = copy.deepcopy(problem)
            problem_to_check["input_output"] = json.loads(problem["input_output"])
            try:
                problem_to_check["solutions"] = json.loads(problem["solutions"])
            except Exception:
                problem_to_check["solutions"] = ""
                print("Empty solution from the dataset")
            curr_res = self.check_correctness(problem_to_check, generation=last_code)
            if curr_res:
                response_entry["correctness"] = True
                response_entry["reason"] = ""
            else:
                response_entry["correctness"] = False
                response_entry["reason"] = "Code is incorrect."

        return response_entry

    def make_conversations(
        self,
        data: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        conversations = []
        for problem in data:
            test_case = json.loads(problem["input_output"])
            starter_code = problem["starter_code"]
            prompt_text = self.generate_prompt(
                test_case, problem["question"], starter_code
            )
            conversations.append(
                self.make_conversation_from_contents(
                    [prompt_text],
                    system_prompt=system_prompt,
                    user_template=user_template,
                )
            )
        return conversations

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None, args=None
    ):
        train_data = self.load_dataset(subset=subset, split=split)
        if difficulty or "difficulty" in self.task_config.preprocess_config:
            difficulty = (
                self.task_config.preprocess_config["difficulty"]
                if not difficulty
                else difficulty
            )
            train_data = train_data.filter(lambda x: x["difficulty"] == difficulty)

        train_data = train_data.to_pandas()

        return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]

    def process_remaining_data(self, train_data, results):
        return [
            row.to_dict()
            for _, row in train_data.iterrows()
            if str(row[self.question_key]) not in results
        ]
