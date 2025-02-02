import random
from typing import Any, Dict, List, Optional

from skythought_evals.util.math_parsing_util import get_multiple_choice_answer

from ..base import TaskHandler


class GPQADiamondTaskHandler(TaskHandler):

    def generate_prompt(self, problem):
        return self.task_config.templating_parameters["template"].format(**problem)

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

    def check_correctness(self, problem, generation):
        pred = get_multiple_choice_answer(generation)
        answer = problem[self.task_config.answer_key]
        return answer == pred

    def get_multiple_choice_answers(self, data):
        answers = [
            data["Correct Answer"],
            data["Incorrect Answer 1"],
            data["Incorrect Answer 2"],
            data["Incorrect Answer 3"],
        ]
        random.shuffle(answers)

        # Map options to letters
        options = ["A", "B", "C", "D"]
        options_to_answers = {
            letter: answer for letter, answer in zip(options, answers)
        }

        # Format the options into the string
        multiple_choice_string = ", ".join(
            f"{letter}) {options_to_answers[letter]}" for letter in options
        )

        # Save the letter corresponding to the correct answer
        correct_answer_letter = next(
            letter
            for letter, answer in options_to_answers.items()
            if answer == data["Correct Answer"]
        )

        return multiple_choice_string, correct_answer_letter

    def make_conversations(
        self,
        data: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        conversations = []
        for problem in data:
            (
                multiple_choice_string,
                correct_answer_letter,
            ) = self.get_multiple_choice_answers(problem)
            problem["Answer"] = correct_answer_letter
            problem["prompt"] = problem["Question"] + "\n" + multiple_choice_string
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
        train_data = self.load_dataset(subset=subset, split=split).to_pandas()
        return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]

    def process_remaining_data(self, train_data, results):
        return [
            row.to_dict()
            for _, row in train_data.iterrows()
            if str(row["Question"]) not in results
        ]
