import re 
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


class GSM8KTaskHandler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = "openai/gsm8k"
        self.ans_re = re.compile(r"((-?[$0-9.,]{2,})|(-?[0-9]+))")
        self.gt_re =  re.compile(r"#### (\-?[0-9\.\,]+)")
        self.invalid_ans = "[invalid]"

    @staticmethod
    def get_question_key():
        return "question"

    @staticmethod
    def generate_prompt(problem):
        question = problem["question"] 
        full_prompt = f"Given the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
        return full_prompt
    
    def check_correctness(self, problem: Dict[str, Any], generation: str) -> bool: 
        gt_answer = self.extract_gt_answer(problem["answer"])
        model_answer = extract_answer(generation)
        model_answer = self.sanitize_answer(model_answer)
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
        curr_res= self.check_correctness(problem, generation=response)
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
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ])
        return conversations

    def load_and_filter_dataset(self, start, end, split="train", source=None, filter_difficulty=False, args=None):
        dataset = load_dataset(self.dataset, "main")
        train_data = dataset[split].to_pandas()
        return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]

    def process_remaining_data(self, train_data, results):
        return [row.to_dict() for _, row in train_data.iterrows() if str(row["question"]) not in results]
    
    def extract_gt_answer(self, completion):
        match = self.gt_re.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return self.invalid_ans

    def sanitize_answer(self, answer):
        patterns_to_remove = [
            ',',           # Remove commas
            r'\$',         # Remove dollar signs
            r'\.$'         # Remove trailing period
            r"\*",           # Remove asterisks
        ]
        for pattern in patterns_to_remove:
            answer = re.sub(pattern, '', answer)
        
        matches = self.ans_re.findall(answer)
        if matches:
            # get the last match (i.e final response) and the first / outer capturing group
            match_str = matches[-1][0].strip()
            return match_str
        else:
            return self.invalid_ans