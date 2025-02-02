import copy
from typing import Any, Dict, List, Optional

from datasets import Dataset as HFDataset
from skythought_evals.util.common import has_code

from ..base import TaskHandler
from .livecodebench_util import (
    map_to_example,
    post_process_code,
    translate_private_test_cases,
    unsafe_lcb_runTests,
)


class LiveCodeBenchTaskHandler(TaskHandler):

    def generate_prompt(self, problem):
        if problem["is_stdin"]:
            return self.task_config.templating_parameters["stdin_template"].format(
                **problem
            )
        else:
            return self.task_config.templating_parameters["non_stdin_template"].format(
                **problem
            )

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

    def make_conversations(
        self,
        data: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        conversations = []
        for problem in data:
            prompt_text = self.generate_prompt(problem)
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
        dataset: HFDataset = self.load_dataset(subset=subset, split=split)
        # Filter by CLI or config
        if difficulty or "difficulty" in self.task_config.preprocess_config:
            difficulty = (
                difficulty
                if difficulty
                else self.task_config.preprocess_config["difficulty"]
            )
            dataset = dataset.filter(
                lambda example: example["difficulty"] == difficulty
            )
        # We use a lower writer_batch_size to avoid pyarrow issues. JSON entries with LiveCodeBench are large.
        # See: https://github.com/NovaSky-AI/SkyThought/pull/45 for details.
        dataset = dataset.map(
            lambda example: {
                "private_test_cases": translate_private_test_cases(
                    example["private_test_cases"]
                )
            },
            writer_batch_size=100,
        )
        # Apply the mapping function
        dataset = dataset.map(
            map_to_example, remove_columns=dataset.column_names, writer_batch_size=100
        ).to_pandas()
        return dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]

    def process_remaining_data(self, train_data, results):
        return [
            row.to_dict()
            for _, row in train_data.iterrows()
            if str(row[self.question_key]) not in results
        ]
