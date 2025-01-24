from datasets import load_dataset

from util.math_parsing_util import get_multiple_choice_answer, mmlu_pro_extract_answer

from ..common import TaskHandler


class MMLUTaskHandler(TaskHandler):
    def __init__(self):
        self.dataset = "cais/mmlu"

    def generate_prompt(self, prompt):
        return "Return your final response within \\boxed{{}}. " + prompt

    @staticmethod
    def get_question_key():
        return "question"

    def check_correctness(self, problem, generation):
        pred = get_multiple_choice_answer(generation)
        abcd = "ABCD"
        answer = abcd[problem["answer"]]
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
        for i, (label, option) in enumerate(zip("ABCD", options)):
            options[i] = f"({label}) {str(option).strip()}"
        options = " ".join(options)
        return f"Answer Choices: {options}"

    def make_conversations(self, data, system_prompt, model=None):
        conversations = []
        for problem in data:
            multiple_choice_string = self.get_multiple_choice_answers(problem)
            prompt_text = self.generate_prompt(
                problem["question"] + "\n" + multiple_choice_string
            )
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
            if str(row["question"]) not in results
        ]

    def load_and_filter_dataset(
        self, start, end, split="test", source=None, filter_difficulty=False, args=None
    ):
        dataset = load_dataset(self.dataset, "all")
        train_data = dataset[split].to_pandas()
        return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]


class MMLUProTaskHandler(MMLUTaskHandler):
    def __init__(self):
        super().__init__()
        self.dataset = "TIGER-Lab/MMLU-Pro"
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

    @staticmethod
    def generate_prompt(prompt):
        return "Return your final response within \\boxed{{}}. " + prompt

    @staticmethod
    def get_question_key():
        return "question"

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
        self, start, end, split="test", source=None, filter_difficulty=False, args=None
    ):
        dataset = load_dataset(self.dataset, "default")
        train_data = dataset[split].to_pandas()
        return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]
