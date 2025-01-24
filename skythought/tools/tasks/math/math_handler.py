from datasets import load_dataset

from tasks.common import TaskHandler
from util.math_parsing_util import extract_answer, math_equal, strip_answer_string


class MathTaskHandler(TaskHandler):
    def generate_prompt(self, prompt):
        return self.task_config.templating_parameters["instruction"] + prompt

    def check_correctness(self, problem, generation):
        answer = strip_answer_string(problem["answer"])
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

    def make_conversations(self, data, system_prompt, model=None):
        conversations = []
        for problem in data:
            prompt_text = self.generate_prompt(problem[self.task_config.question_key])
            conversations.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ]
            )
        return conversations

    def process_remaining_data(self, train_data, results):
        return [
            row.to_dict()
            for _, row in train_data.iterrows()
            if str(row[self.task_config.question_key]) not in results
        ]

    def load_and_filter_dataset(
        self, start, end, split="test", source=None, filter_difficulty=False, args=None
    ):
        dataset = load_dataset(self.dataset)
        train_data = dataset[split].to_pandas()
        return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]
