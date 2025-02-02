from typing import Any, Dict, List, Optional

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
        if not isinstance(response, str):
            response = response.outputs[0].text.strip()
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

    def process_remaining_data(self, train_data, results):
        return [
            row.to_dict()
            for _, row in train_data.iterrows()
            if str(row[self.question_key]) not in results
        ]

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None, args=None
    ):
        dataset = self.load_dataset(subset=subset, split=split).to_pandas()
        return dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]
