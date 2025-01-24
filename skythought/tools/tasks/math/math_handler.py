from datasets import load_dataset
from typing import Dict, Any
from multiprocessing import Manager
from tasks.apps.apps_util import run_test as apps_run_test
from tasks.taco.taco_util import run_test as taco_run_test
from util.math_parsing_util import strip_answer_string, get_multiple_choice_answer, extract_answer, math_equal, mmlu_pro_extract_answer
from tasks.livecodebench.livecodebench_util import unsafe_lcb_runTests, map_to_example, has_test_type, post_process_code, translate_private_test_cases
from util.common import TimeoutException, timeout
from util.model_utils import SYSTEM_PROMPT, MODEL_TO_NAME
from tasks.common import TaskHandler


class MathTaskHandler(TaskHandler):
    @staticmethod
    def generate_prompt(prompt):
        return "Return your final response within \\boxed{{}}. " + prompt
    
    def check_correctness(self, problem, generation):
        answer = strip_answer_string(problem["answer"])
        pred = extract_answer(generation)
        # print(problem)
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
            prompt_text = self.generate_prompt(problem["problem"])
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ])
        return conversations
    
    def process_remaining_data(self, train_data, results):
        return [row.to_dict() for _, row in train_data.iterrows() if str(row["problem"]) not in results]

    def load_and_filter_dataset(self, start, end, split="test", source=None, filter_difficulty=False, args=None):
        dataset = load_dataset(self.dataset)
        train_data = dataset[split].to_pandas()
        return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]


class MATH500TaskHandler(MathTaskHandler):
    def __init__(self):
        self.dataset = "qq8933/MATH500"
    
    @staticmethod
    def get_question_key():
        return "problem"
