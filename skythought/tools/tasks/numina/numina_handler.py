from datasets import load_dataset

from tasks.common import TaskHandler
from util.common import TimeoutException, timeout
from util.math_parsing_util import extract_answer, math_equal, strip_answer_string


class NUMINATaskHandler(TaskHandler):
    @staticmethod
    def get_question_key():
        return "problem"

    def generate_prompt(self, prompt):
        return "Return your final response within \\boxed{{}}. " + prompt

    @timeout(5)  # Add timeout of 5 seconds
    def check_correctness(self, problem, generation):
        solution = extract_answer(problem["solution"])
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

    def make_conversations(self, data, system_prompt, model=None):
        conversations = []
        for problem in data:
            prompt_text = self.generate_prompt(problem["problem"])
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
        dataset = load_dataset("AI-MO/NuminaMath-CoT")
        train_data = dataset[split].to_pandas()
        train_data = (
            train_data.query("source == @source").iloc[start:end]
            if end > 0
            else train_data.query("source == @source").iloc[start:]
        )
        train_data = train_data[train_data["solution"].str.contains("boxed", na=False)]
        if filter_difficulty:
            diff_dict = self.get_difficulty_dict(source, start, end)
            train_data = train_data[
                train_data["problem"]
                .map(diff_dict)
                .apply(
                    lambda x: x >= args.math_difficulty_lower_bound
                    and x <= args.math_difficulty_upper_bound
                )
            ]
        return train_data

    def process_remaining_data(self, train_data, results):
        return [
            row.to_dict()
            for _, row in train_data.iterrows()
            if str(row["problem"]) not in results
        ]
