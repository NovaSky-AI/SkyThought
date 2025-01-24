import copy
import json
import multiprocessing
from multiprocessing import Manager

import numpy as np
from datasets import load_dataset

from tasks.apps.apps_util import run_test as apps_run_test
from util.common import has_code

from ..common import TaskHandler


class APPSTaskHandler(TaskHandler):
    @staticmethod
    def get_question_key():
        return "question"

    def generate_prompt(self, test_case, prompt, starter_code=None):
        _input = ""
        data = test_case
        if not data.get("fn_name"):
            _input += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."  # "\nUse Standard Input format"#\n"
        else:
            _input += "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."  # "\nUse Call-Based format"#\n"
        data = prompt
        _input += data
        if starter_code is not None:
            data = starter_code
            data = "\n" + data  # + "\n"
            _input += data
        else:
            # _input += "\n\n"
            pass

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

    def make_conversations(self, data, system_prompt, model=None):
        conversations = []
        for problem in data:
            test_case = json.loads(problem["input_output"])
            starter_code = problem["starter_code"]
            prompt_text = self.generate_prompt(
                test_case, problem["question"], starter_code
            )
            conversations.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ]
            )
        return conversations

    def load_and_filter_dataset(
        self, start, end, split="train", source=None, filter_difficulty=False, args=None
    ):
        dataset = load_dataset("codeparrot/apps", trust_remote_code=True)
        train_data = dataset[split].to_pandas()
        if not filter_difficulty:
            return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]
        return (
            train_data.query("difficulty == @source").iloc[start:end]
            if end > 0
            else train_data.query("difficulty == @source").iloc[start:]
        )

    def process_remaining_data(self, train_data, results):
        return [
            row.to_dict()
            for _, row in train_data.iterrows()
            if str(row["question"]) not in results
        ]
