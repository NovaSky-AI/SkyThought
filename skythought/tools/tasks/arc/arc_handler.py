import re
from typing import Any, Dict

from datasets import load_dataset

from tasks.common import TaskHandler
from util.math_parsing_util import extract_answer


class ARCChallengeTaskHandler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = "allenai/ai2_arc"
        self.ans_re = re.compile(r"[Tt]he best answer is ([A-D])[\.\,]*", re.IGNORECASE)
        self.letter_re = re.compile(r"([A-D])[\.\,]*")
        self.canonical_options = ["A", "B", "C", "D"]
        self.invalid_ans = "[invalid]"

    @staticmethod
    def get_question_key():
        return "question"

    def generate_prompt(self, problem):
        question = problem["question"]
        choices = problem["choices"]
        choices_text = "\n".join(
            [
                f"{label}.{choice}"
                for label, choice in zip(["A", "B", "C", "D"], choices["text"])
            ]
        )
        full_prompt = (
            'Given the following question and four candidate answers (A, B, C and D), choose the best answer. Your response should end with "The best answer is [the_answer_letter]" where [the_answer_letter] is one of the four letter choice (A, B, C, or D).\n'
            + f"{question}\n{choices_text}"
        )
        return full_prompt

    def check_correctness(self, problem: Dict[str, Any], generation: str) -> bool:
        gt_answer = problem["answerKey"]
        if gt_answer not in self.canonical_options:
            gt_answer = self.canonical_options[int(problem["answerKey"]) - 1]
        model_answer = self.get_answer(generation)
        return model_answer == gt_answer

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
            prompt_text = self.generate_prompt(problem)
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
        dataset = load_dataset(self.dataset, "ARC-Challenge")
        train_data = dataset[split].to_pandas()
        return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]

    def process_remaining_data(self, train_data, results):
        return [
            row.to_dict()
            for _, row in train_data.iterrows()
            if str(row["question"]) not in results
        ]

    def get_answer(self, completion):
        # First, we try to extract similar to MATH answers
        answer = extract_answer(completion)
        match = None
        if answer:
            # match for the letter answer needed.
            match = self.letter_re.search(answer)
            if match:
                return match.group(1).strip()

        if not answer or not match:
            # try basic-regex based search
            patterns_to_remove = [
                ",",  # Remove commas
                r"\$",  # Remove dollar signs
                r"\.$" r"\\",  # Remove trailing period  # Remove stray backslashes
                r"\*",  # Remove asterisks
            ]
            answer = completion
            for pattern in patterns_to_remove:
                answer = re.sub(pattern, "", answer)
            matches = self.ans_re.findall(answer)
            if not matches:
                return self.invalid_ans
            return matches[-1].strip()
