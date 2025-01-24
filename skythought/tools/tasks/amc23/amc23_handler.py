from datasets import load_dataset
from typing import Dict, Any
from multiprocessing import Manager
from tasks.apps.apps_util import run_test as apps_run_test
from tasks.taco.taco_util import run_test as taco_run_test
from util.math_parsing_util import strip_answer_string, get_multiple_choice_answer, extract_answer, math_equal, mmlu_pro_extract_answer
from tasks.livecodebench.livecodebench_util import unsafe_lcb_runTests, map_to_example, has_test_type, post_process_code, translate_private_test_cases
from util.common import TimeoutException, timeout
from util.model_utils import SYSTEM_PROMPT, MODEL_TO_NAME
from tasks.common import MathTaskHandler

class AMC23TaskHandler(MathTaskHandler):
    def __init__(self):
        self.dataset = "AI-MO/aimo-validation-amc"

    @staticmethod
    def get_question_key():
        return "problem"
    
    def load_and_filter_dataset(self, start, end, split="train", source=None, filter_difficulty=False, args=None):
        dataset = load_dataset(self.dataset)
        train_data = dataset[split].to_pandas()
        filtered_data = train_data[train_data['url'].str.contains("2023", na=False)]
        return filtered_data.iloc[start:end] if end > 0 else filtered_data.iloc[start:]