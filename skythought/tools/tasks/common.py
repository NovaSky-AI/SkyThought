import copy
import json
import multiprocessing
import os
import random
import re
import numpy as np
from datasets import load_dataset
from typing import Dict, Any
from multiprocessing import Manager
from tasks.apps.apps_util import run_test as apps_run_test
from tasks.taco.taco_util import run_test as taco_run_test
from ..util.math_parsing_util import strip_answer_string, get_multiple_choice_answer, extract_answer, math_equal, mmlu_pro_extract_answer
from tasks.livecodebench.livecodebench_util import unsafe_lcb_runTests, map_to_example, has_test_type, post_process_code, translate_private_test_cases
from ..util.common import TimeoutException, timeout
from util.model_utils import *

def has_code(response):
    pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    # Use re.DOTALL to match multiline content inside backticks
    matches = re.findall(pattern, response, re.DOTALL)
    # print(matches)
    return matches

class TaskHandler:
    @staticmethod
    def get_question_key():
        raise NotImplementedError("Subclasses should implement this method.")

    def check_correctness(self, problem, generation):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def update_results(self, problem, response):
        raise NotImplementedError("Subclasses should implement this method.")

    def make_conversations(self, data, system_prompt, model=None):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_existing_results(self, result_file):
        if not os.path.exists(result_file):
            return {}
        with open(result_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
        return records

    def load_and_filter_dataset(self, start, end, split="train", source=None, filter_difficulty=False, args=None):
        raise NotImplementedError("Subclasses should implement this method.")

    def process_remaining_data(self, train_data, results):
        raise NotImplementedError("Subclasses should implement this method.")

    
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

    @staticmethod
    def generate_prompt(problem):
        question = problem["question"] 
        choices = problem["choices"]
        choices_text = '\n'.join([f"{label}.{choice}" for label, choice in zip(["A", "B", "C", "D"], choices["text"])])
        full_prompt = "Given the following question and four candidate answers (A, B, C and D), choose the best answer. Your response should end with \"The best answer is [the_answer_letter]\" where [the_answer_letter] is one of the four letter choice (A, B, C, or D).\n" + f"{question}\n{choices_text}"
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
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ])
        return conversations

    def load_and_filter_dataset(self, start, end, split="train", source=None, filter_difficulty=False, args=None):
        dataset = load_dataset(self.dataset, "ARC-Challenge")
        train_data = dataset[split].to_pandas()
        return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]

    def process_remaining_data(self, train_data, results):
        return [row.to_dict() for _, row in train_data.iterrows() if str(row["question"]) not in results]

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
                ',',           # Remove commas
                r'\$',         # Remove dollar signs
                r'\.$'         # Remove trailing period
                r"\\",         # Remove stray backslashes
                r"\*",           # Remove asterisks
            ]
            answer = completion
            for pattern in patterns_to_remove:
                answer = re.sub(pattern, '', answer)
            matches = self.ans_re.findall(answer)
            if not matches: 
                return self.invalid_ans
            return matches[-1].strip()


TASK_HANDLERS = {
    "NUMINA": NUMINATaskHandler,
    "APPS": APPSTaskHandler,
    "TACO": TACOTaskHandler,
    "MATH500": MATH500TaskHandler,
    "AIME": AIMETaskHandler,
    "GPQADiamond": GPQADiamondTaskHandler,
    "MMLU": MMLUTaskHandler,
    "MMLUPro": MMLUProTaskHandler,
    "LiveCodeBench": LiveCodeBenchTaskHandler,
    "GSM8K": GSM8KTaskHandler,
    "ARC-C": ARCChallengeTaskHandler,
    "AMC23": AMC23TaskHandler,
}
