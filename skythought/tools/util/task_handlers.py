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
from .apps.testing_util import run_test as apps_run_test
from .taco.testing_util import run_test as taco_run_test
from .math.testing_util import strip_answer_string, get_multiple_choice_answer, extract_answer, math_equal, mmlu_pro_extract_answer
from .livecodebench.testing_util import unsafe_lcb_runTests, map_to_example, has_test_type, post_process_code, translate_private_test_cases
from .common import TimeoutException, timeout
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

class MATH500TaskHandler(MathTaskHandler):
    def __init__(self):
        self.dataset = "qq8933/MATH500"
    
    @staticmethod
    def get_question_key():
        return "problem"

class AIMETaskHandler(MathTaskHandler):
    def __init__(self):
        self.dataset = "AI-MO/aimo-validation-aime"
    
    @staticmethod
    def generate_prompt(prompt, model):
        if MODEL_TO_NAME[model] == "Sky-T1-32B-Preview":
            return prompt + "\nReturn your final response within \\boxed{{}}"
        else:
            return "Return your final response within \\boxed{{}}. " + prompt
    
    @staticmethod
    def get_question_key():
        return "problem"
    
    def make_conversations(self, data, system_prompt, model=None):
        conversations = []
        for problem in data:
            prompt_text = self.generate_prompt(problem["problem"], model)
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ])
        return conversations
    
    def load_and_filter_dataset(self, start, end, split="train", source=None, filter_difficulty=False, args=None):
        dataset = load_dataset(self.dataset)
        train_data = dataset[split].to_pandas()
        filtered_data = train_data[train_data['url'].str.contains("2024", na=False)]
        return filtered_data.iloc[start:end] if end > 0 else filtered_data.iloc[start:]
    
class GPQADiamondTaskHandler(TaskHandler):
    def __init__(self):
        self.dataset = "Idavidrein/gpqa"

    @staticmethod
    def generate_prompt(prompt):
        return "Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response. " + prompt

    @staticmethod
    def get_question_key():
        return "Question"

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
        answer = problem["Answer"]
        return answer == pred
    
    def get_multiple_choice_answers(self, data):
        answers = [
            data["Correct Answer"],
            data["Incorrect Answer 1"],
            data["Incorrect Answer 2"],
            data["Incorrect Answer 3"]
        ]
        random.shuffle(answers)

        # Map options to letters
        options = ["A", "B", "C", "D"]
        options_to_answers = {letter: answer for letter, answer in zip(options, answers)}

        # Format the options into the string
        multiple_choice_string = ", ".join(f"{letter}) {options_to_answers[letter]}" for letter in options)

        # Save the letter corresponding to the correct answer
        correct_answer_letter = next(letter for letter, answer in options_to_answers.items() if answer == data["Correct Answer"])

        return multiple_choice_string, correct_answer_letter
    
    def make_conversations(self, data, system_prompt, model=None):
        conversations = []
        for problem in data:
            multiple_choice_string, correct_answer_letter = self.get_multiple_choice_answers(problem)
            problem["Answer"] = correct_answer_letter
            prompt_text = self.generate_prompt(problem["Question"] + "\n" + multiple_choice_string)
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ])
        return conversations

    def load_and_filter_dataset(self, start, end, split="train", source=None, filter_difficulty=False, args=None):
        dataset = load_dataset(self.dataset, "gpqa_diamond")
        train_data = dataset[split].to_pandas()
        return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]

    def process_remaining_data(self, train_data, results):
        return [row.to_dict() for _, row in train_data.iterrows() if str(row["Question"]) not in results]

class MMLUTaskHandler(TaskHandler):
    def __init__(self):
        self.dataset = "cais/mmlu"

    @staticmethod
    def generate_prompt(prompt):
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
            prompt_text = self.generate_prompt(problem["question"] + "\n" + multiple_choice_string)
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ])
        return conversations

    def process_remaining_data(self, train_data, results):
        return [row.to_dict() for _, row in train_data.iterrows() if str(row["question"]) not in results]

    def load_and_filter_dataset(self, start, end, split="test", source=None, filter_difficulty=False, args=None):
        dataset = load_dataset(self.dataset, "all")
        train_data = dataset[split].to_pandas()
        return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]

class MMLUProTaskHandler(MMLUTaskHandler):
    def __init__(self):
        super().__init__()
        self.dataset = "TIGER-Lab/MMLU-Pro"
        self.choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

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
        for i, (label, option) in enumerate(zip(self.choices[:len(options)], options)):
            options[i] = f"({label}) {str(option).strip()}"
        options = " ".join(options)
        return f"Answer Choices: {options}"

    def load_and_filter_dataset(self, start, end, split="test", source=None, filter_difficulty=False):
        dataset = load_dataset(self.dataset, "default")
        train_data = dataset[split].to_pandas()
        return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]
    
class NUMINATaskHandler(TaskHandler):
    @staticmethod
    def get_question_key():
        return "problem"

    @staticmethod
    def generate_prompt(prompt):
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
        dataset = load_dataset("NovaSky-AI/labeled_numina_difficulty_859K", trust_remote_code=True, split="train")
        for example in dataset:
            # print(example)
            diff_dict[example["problem"]] = example["gpt_difficulty_parsed"]
        return diff_dict

    def make_conversations(self, data, system_prompt, model=None):
        conversations = []
        for problem in data:
            prompt_text = self.generate_prompt(problem["problem"])
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ])
        return conversations

    def load_and_filter_dataset(self, start, end, split="train", source=None, filter_difficulty=False, args=None):
        dataset = load_dataset("AI-MO/NuminaMath-CoT")
        train_data = dataset[split].to_pandas()
        train_data = train_data.query('source == @source').iloc[start:end] if end > 0 else train_data.query('source == @source').iloc[start:]
        train_data = train_data[train_data["solution"].str.contains("boxed", na=False)]
        if filter_difficulty:
            diff_dict = self.get_difficulty_dict(source, start, end)
            train_data = train_data[train_data["problem"].map(diff_dict).apply(lambda x: x >= args.math_difficulty_lower_bound and x <= args.math_difficulty_upper_bound)]
        return train_data

    def process_remaining_data(self, train_data, results):
        return [row.to_dict() for _, row in train_data.iterrows() if str(row["problem"]) not in results]

class APPSTaskHandler(TaskHandler):
    @staticmethod
    def get_question_key():
        return "question"

    @staticmethod
    def generate_prompt(test_case, prompt, starter_code=None):
        _input = ""
        data = test_case
        if not data.get("fn_name"):
            _input += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."# "\nUse Standard Input format"#\n"
        else:
            _input += "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution." #"\nUse Call-Based format"#\n"
        data = prompt
        _input += data
        if starter_code != None:
            data = starter_code
            data = "\n" + data #+ "\n"
            _input += data
        else:
            #_input += "\n\n"
            pass
        
        return _input
    
    def check_correctness(self, problem, generation):
        TIMEOUT = 10
        def _temp_run(problem, generation, debug, result):
            try:
                result.append(apps_run_test(problem=problem, test=generation, debug=debug))
            except Exception as e:
                pass

        manager = Manager()
        result = manager.list()
        p = multiprocessing.Process(target=_temp_run, args=(problem, generation, False, result))
        p.start()
        p.join(timeout=TIMEOUT + 1)
        if p.is_alive():
            p.kill()
        return bool(result and np.all(result[0]))
    
    def update_results(self, problem, response):
        if not isinstance(response, str):
            response = response.outputs[0].text.strip()
        # Initialize the response structure
        response_entry = {
            "content": response,
            "correctness": None,
            "reason": None,
        }
        code_filter_result = has_code(response)
        if len(code_filter_result) == 0:
            response_entry["correctness"] = False
            response_entry["reason"] = "Does not contain code component."
        else:	
            last_code = code_filter_result[-1]
            problem_to_check = copy.deepcopy(problem)
            problem_to_check["input_output"] = json.loads(problem["input_output"])
            try:
                problem_to_check["solutions"] = json.loads(problem["solutions"]) 
            except:
                problem_to_check["solutions"] = ""
                print(f"Empty solution from the dataset")
            curr_res = self.check_correctness(problem_to_check, generation=last_code)
            if curr_res:
                response_entry["correctness"] = True
                response_entry["reason"] = ""
            else:
                response_entry["correctness"] = False
                response_entry["reason"] = "Code is incorrect."
        
        return response_entry

    def make_conversations(self, data, system_prompt, model=None):
        conversations = []
        for problem in data:
            test_case = json.loads(problem["input_output"])
            starter_code = problem["starter_code"]
            prompt_text = self.generate_prompt(test_case, problem["question"], starter_code)
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ])
        return conversations

    def load_and_filter_dataset(self, start, end, split="train", source=None, filter_difficulty=False, args=None):
        dataset = load_dataset("codeparrot/apps", trust_remote_code=True)
        train_data = dataset[split].to_pandas()
        if not filter_difficulty:
            return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]
        return train_data.query('difficulty == @source').iloc[start:end] if end > 0 else train_data.query('difficulty == @source').iloc[start:]

    def process_remaining_data(self, train_data, results):
        return [row.to_dict() for _, row in train_data.iterrows() if str(row["question"]) not in results]

class TACOTaskHandler(TaskHandler):
    @staticmethod
    def get_question_key():
        return "question"

    @staticmethod
    def generate_prompt(prompt, starter_code=None, fn_name=None):
        _input = "\nQUESTION:\n"
        _input += prompt
        if starter_code:
            _input += starter_code
        if (not fn_name) and (not starter_code):
            call_format = "\nUse Standard Input format"
            _input += call_format
        else:
            call_format = "\nUse Call-Based format"
            _input += call_format
        _input += "\nANSWER:\n"
        
        return _input
    
    def check_correctness(self, problem, generation):
        TIME_OUT = 300
        def _temp_run(problem, generation, debug, result):
            try:
                result.append(taco_run_test(problem, test=generation, debug=debug))
            except Exception as e:
                print(f"Error in _temp_run: {e}")

        manager = Manager()
        result = manager.list()
        p = multiprocessing.Process(target=_temp_run, args=(problem, generation, False, result))
        p.start()
        p.join(timeout=TIME_OUT + 1)
        if p.is_alive():
            p.kill()
        return bool(result and np.all(result[0]))
    
    def update_results(self, problem, response):
        if not isinstance(response, str):
            response = response.outputs[0].text.strip()
        # Initialize the response structure
        response_entry = {
            "content": response,
            "correctness": None,
            "reason": None,
        }
        code_filter_result = has_code(response)
        if len(code_filter_result) == 0:
            response_entry["correctness"] = False
            response_entry["reason"] = "Does not contain code component."
        else:	
            last_code = code_filter_result[-1]
            curr_res = self.check_correctness(problem, generation=last_code)
            if curr_res:
                response_entry["correctness"] = True
                response_entry["reason"] = ""
            else:
                response_entry["correctness"] = False
                response_entry["reason"] = "Code is incorrect."
        
        return response_entry

    def make_conversations(self, data, system_prompt, model=None):
        conversations = []
        for idx, problem in enumerate(data):
            starter_code = None if len(problem["starter_code"]) == 0 else problem["starter_code"]
            try:
                input_outpout = json.loads(problem["input_output"])
                fn_name = (
                    None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
                )
            except ValueError:
                fn_name = None
            prompt_text = self.generate_prompt(problem["question"], starter_code, fn_name)
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ])
        return conversations

    def load_and_filter_dataset(self, start, end, split="train", source=None, filter_difficulty=False, args=None):
        dataset = load_dataset("BAAI/TACO", "ALL", trust_remote_code=True)
        train_data = dataset[split].to_pandas()
        if not filter_difficulty:
            return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]
        return train_data.query('difficulty == @source').iloc[start:end] if end > 0 else train_data.query('difficulty == @source').iloc[start:]

    def process_remaining_data(self, train_data, results):
        return [row.to_dict() for _, row in train_data.iterrows() if str(row["question"]) not in results]

class LiveCodeBenchTaskHandler(TaskHandler):
    @staticmethod
    def generate_prompt(problem):
        # print(problem)
        prompt = problem["prompt"]
        if problem["is_stdin"]:
            return "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition." + prompt
        else:
            return "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution." + prompt
    
    @staticmethod
    def get_question_key():
        return "task_id"
    
    def check_correctness(
        self,
        problem: Dict,
        completion: str,
        timeout: float,
        runtime_debug=False,
        is_extracted=False,
    ) -> Dict:
        """
        Evaluates the functional correctness of a completion by running the test
        suite provided in the problem.

        :param completion_id: an optional completion ID so we can match
            the results later even if execution finishes asynchronously.
        """
        result_list = unsafe_lcb_runTests(problem, completion, timeout, runtime_debug, is_extracted)
        details = [r[0] for r in result_list]
        all_passed = all(details)
        
        result = ""
        if result_list and all_passed:
            result = "passed"

        return result == "passed"
    
    def update_results(self, problem, response):
        if not isinstance(response, str):
            response = response.outputs[0].text.strip()
        # Initialize the response structure
        response_entry = {
            "content": response,
            "correctness": None,
            "reason": None,
        }
        code_filter_result = has_code(response)
        # print(response)
        if len(code_filter_result) == 0:
            response_entry["correctness"] = False
            response_entry["reason"] = "Does not contain code component."
        else:	
            last_code = code_filter_result[-1]
            problem_to_check = copy.deepcopy(problem)

            curr_res = self.check_correctness(problem=problem_to_check, completion=post_process_code(last_code), timeout=6, is_extracted=not problem_to_check["is_stdin"])
            if curr_res:
                response_entry["correctness"] = True
                response_entry["reason"] = ""
            else:
                response_entry["correctness"] = False
                response_entry["reason"] = "Code is incorrect."
        
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

    def load_and_filter_dataset(self, start, end, split="test", source=None, filter_difficulty=False, args=None):
        dataset = load_dataset("livecodebench/code_generation_lite", version_tag="release_v2", split=split, trust_remote_code=True)
        if filter_difficulty:
            dataset = dataset.filter(lambda example: example['difficulty'] == source)
        dataset = dataset.map(
            lambda example: {
                "private_test_cases": translate_private_test_cases(example["private_test_cases"])
            }
        )
        # Apply the mapping function
        dataset = dataset.map(map_to_example, remove_columns=dataset.column_names).to_pandas()
        return dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]

    def process_remaining_data(self, train_data, results):
        return [row.to_dict() for _, row in train_data.iterrows() if str(row["task_id"]) not in results]

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

    def make_conversations(self, data, system_prompt):
        conversations = []
        for problem in data:
            prompt_text = self.generate_prompt(problem)
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ])
        return conversations

    def load_and_filter_dataset(self, start, end, split="train", source=None, filter_difficulty=False):
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

    def make_conversations(self, data, system_prompt):
        conversations = []
        for problem in data:
            prompt_text = self.generate_prompt(problem)
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ])
        return conversations

    def load_and_filter_dataset(self, start, end, split="train", source=None, filter_difficulty=False):
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
}
