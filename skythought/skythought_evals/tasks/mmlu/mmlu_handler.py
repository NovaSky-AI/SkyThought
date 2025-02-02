from typing import Any, Dict, List, Optional

from skythought_evals.util.math_parsing_util import (
    get_multiple_choice_answer,
    mmlu_pro_extract_answer,
)

from ..base import TaskConfig, TaskHandler


class MMLUTaskHandler(TaskHandler):
    def generate_prompt(self, prompt):
        return self.task_config.templating_parameters["template"].format(prompt=prompt)

    def check_correctness(self, problem, generation):
        pred = get_multiple_choice_answer(generation)
        abcd = "ABCD"
        answer = abcd[problem[self.task_config.answer_key]]
        return answer == pred

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

    def get_multiple_choice_answers(self, problem):
        options = problem["choices"]
        options_str = ""
        for _, (label, option) in enumerate(zip("ABCD", options)):
            options_str += f"({label}) {str(option).strip()} "
        options_str = options_str[:-1]  # remove the last space
        return f"Answer Choices: {options_str}"

    def make_conversations(
        self,
        data: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        conversations = []
        for problem in data:
            multiple_choice_string = self.get_multiple_choice_answers(problem)
            prompt_text = self.generate_prompt(
                problem["question"] + "\n" + multiple_choice_string
            )
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
            if str(row["question"]) not in results
        ]

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None, args=None
    ):
        dataset = self.load_dataset(subset=subset, split=split).to_pandas()
        return dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]


class MMLUProTaskHandler(MMLUTaskHandler):
    def __init__(self, task_config: TaskConfig):
        super().__init__(task_config)
        self.choices = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
        ]

    def generate_prompt(self, prompt):
        return self.task_config.templating_parameters["template"].format(prompt=prompt)

    def check_correctness(self, problem, generation):
        pred = mmlu_pro_extract_answer(generation)
        answer = self.choices[problem["answer_index"]]
        return answer == pred

    def get_multiple_choice_answers(self, problem):
        options = problem["options"]
        for i, (label, option) in enumerate(zip(self.choices[: len(options)], options)):
            options[i] = f"({label}) {str(option).strip()}"
        options = " ".join(options)
        return f"Answer Choices: {options}"

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None, args=None
    ):
        dataset = self.load_dataset(subset=subset, split=split).to_pandas()
        return dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]
