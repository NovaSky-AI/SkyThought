from typing import Any, Dict, List, Optional

from datasets import load_dataset
from skythought_evals.util.common import TimeoutException, timeout
from skythought_evals.util.math_parsing_util import (
    extract_answer,
    math_equal,
    strip_answer_string,
)

from ..base import TaskHandler


class NUMINATaskHandler(TaskHandler):

    def generate_prompt(self, prompt):
        return self.task_config.templating_parameters["template"].format(prompt=prompt)

    @timeout(5)  # Add timeout of 5 seconds
    def check_correctness(self, problem, generation):
        solution = extract_answer(problem[self.task_config.answer_key])
        solution = strip_answer_string(solution)
        pred = extract_answer(generation)
        pred = strip_answer_string(pred)
        return math_equal(pred, solution)

    def update_results(self, problem, response):
        if not isinstance(response, str):
            response = response.outputs[0].text.strip()
        # Initialize the response structure
        response_entry = {
            "content": response,
            "correctness": None,
            "reason": None,
        }

        try:
            curr_res = self.check_correctness(problem, generation=response)
            if curr_res:
                response_entry["correctness"] = True
                response_entry["reason"] = ""
            else:
                response_entry["correctness"] = False
                response_entry["reason"] = "Solution is incorrect."
        except TimeoutException as e:
            response_entry["correctness"] = False
            response_entry["reason"] = str(e)

        return response_entry

    @staticmethod
    def get_difficulty_dict(source, start, end):
        diff_dict = {}
        dataset = load_dataset(
            "NovaSky-AI/labeled_numina_difficulty_859K",
            trust_remote_code=True,
            split="train",
        )
        for example in dataset:
            # print(example)
            diff_dict[example["problem"]] = example["gpt_difficulty_parsed"]
        return diff_dict

    def make_conversations(
        self,
        data: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        conversations = []
        for problem in data:
            prompt_text = self.generate_prompt(problem["problem"])
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
        dataset = self.load_dataset(subset=subset, split=split).to_pandas()

        if args.source:
            dataset = dataset[dataset["source"] == args.source]
        dataset = dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]
        dataset = dataset[dataset["solution"].str.contains("boxed", na=False)]

        if (
            args.filter_difficulty
            or "filter_difficulty" in self.task_config.preprocess_config
        ):
            lower_bound = (
                args.math_difficulty_lower_bound
                if args.filter_difficulty
                else self.task_config.preprocess_config["math_difficulty_lower_bound"]
            )
            upper_bound = (
                args.math_difficulty_upper_bound
                if args.filter_difficulty
                else self.task_config.preprocess_config["math_difficulty_upper_bound"]
            )
            diff_dict = self.get_difficulty_dict(args.source, start, end)
            dataset = dataset[
                dataset["problem"]
                .map(diff_dict)
                .apply(lambda x: x >= lower_bound and x <= upper_bound)
            ]

        return dataset

    def process_remaining_data(self, train_data, results):
        return [
            row.to_dict()
            for _, row in train_data.iterrows()
            if str(row["problem"]) not in results
        ]
