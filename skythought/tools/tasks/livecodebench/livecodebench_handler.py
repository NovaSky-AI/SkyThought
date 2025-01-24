import copy
from typing import Dict

from datasets import load_dataset

from tasks.common import TaskHandler
from tasks.livecodebench.livecodebench_util import (
    map_to_example,
    post_process_code,
    translate_private_test_cases,
    unsafe_lcb_runTests,
)
from util.common import has_code


class LiveCodeBenchTaskHandler(TaskHandler):
    def generate_prompt(self, problem):
        # print(problem)
        prompt = problem["prompt"]
        if problem["is_stdin"]:
            return (
                "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."
                + prompt
            )
        else:
            return (
                "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
                + prompt
            )

    @staticmethod
    def get_question_key():
        return "task_id"

    def check_correctness(
        self,
        problem: Dict,
        completion: str,
        timeout: float,
        runtime_debug=False,
        is_extracted=False,
    ) -> Dict:
        """
        Evaluates the functional correctness of a completion by running the test
        suite provided in the problem.

        :param completion_id: an optional completion ID so we can match
            the results later even if execution finishes asynchronously.
        """
        result_list = unsafe_lcb_runTests(
            problem, completion, timeout, runtime_debug, is_extracted
        )
        details = [r[0] for r in result_list]
        all_passed = all(details)

        result = ""
        if result_list and all_passed:
            result = "passed"

        return result == "passed"

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
        # print(response)
        if len(code_filter_result) == 0:
            response_entry["correctness"] = False
            response_entry["reason"] = "Does not contain code component."
        else:
            last_code = code_filter_result[-1]
            problem_to_check = copy.deepcopy(problem)

            curr_res = self.check_correctness(
                problem=problem_to_check,
                completion=post_process_code(last_code),
                timeout=6,
                is_extracted=not problem_to_check["is_stdin"],
            )
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
            prompt_text = self.generate_prompt(problem)
            conversations.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ]
            )
        return conversations

    def load_and_filter_dataset(
        self, start, end, split="test", source=None, filter_difficulty=False, args=None
    ):
        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            version_tag="release_v2",
            split=split,
            trust_remote_code=True,
        )
        if filter_difficulty:
            dataset = dataset.filter(lambda example: example["difficulty"] == source)
        dataset = dataset.map(
            lambda example: {
                "private_test_cases": translate_private_test_cases(
                    example["private_test_cases"]
                )
            }
        )
        # Apply the mapping function
        dataset = dataset.map(
            map_to_example, remove_columns=dataset.column_names
        ).to_pandas()
        return dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]

    def process_remaining_data(self, train_data, results):
        return [
            row.to_dict()
            for _, row in train_data.iterrows()
            if str(row["task_id"]) not in results
        ]
