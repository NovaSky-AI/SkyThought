import json
import multiprocessing
from multiprocessing import Manager

import numpy as np
from datasets import load_dataset

from tasks.taco.taco_util import run_test as taco_run_test
from util.common import has_code

from ..common import TaskHandler


class TACOTaskHandler(TaskHandler):
    @staticmethod
    def get_question_key():
        return "question"

    @staticmethod
    def generate_prompt(prompt, starter_code=None, fn_name=None):
        _input = "\nQUESTION:\n"
        _input += prompt
        if starter_code:
            _input += starter_code
        if (not fn_name) and (not starter_code):
            call_format = "\nUse Standard Input format"
            _input += call_format
        else:
            call_format = "\nUse Call-Based format"
            _input += call_format
        _input += "\nANSWER:\n"

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
            curr_res = self.check_correctness(problem, generation=last_code)
            if curr_res:
                response_entry["correctness"] = True
                response_entry["reason"] = ""
            else:
                response_entry["correctness"] = False
                response_entry["reason"] = "Code is incorrect."

        return response_entry

    def make_conversations(self, data, system_prompt, model=None):
        conversations = []
        for idx, problem in enumerate(data):
            starter_code = (
                None if len(problem["starter_code"]) == 0 else problem["starter_code"]
            )
            try:
                input_outpout = json.loads(problem["input_output"])
                fn_name = (
                    None
                    if not input_outpout.get("fn_name")
                    else input_outpout["fn_name"]
                )
            except ValueError:
                fn_name = None
            prompt_text = self.generate_prompt(
                problem["question"], starter_code, fn_name
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
        dataset = load_dataset("BAAI/TACO", "ALL", trust_remote_code=True)
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
