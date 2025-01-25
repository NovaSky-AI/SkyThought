from tasks.common import TaskHandler
from util.math_parsing_util import extract_answer, math_equal, strip_answer_string
from tasks.math.math_handler import MathTaskHandler


class OlympiadBenchMathTaskHandler(MathTaskHandler): 
    def __init__(self):
        self.dataset = "Hothan/OlympiadBench"
        self.source = "OE_TO_maths_en_COMP"

    @staticmethod
    def get_question_key():
        return "question"
    
    def load_and_filter_dataset(self, start, end, split="train", source=None, filter_difficulty=False, args=None):
        dataset = self.load_dataset(source=source, split=split).to_pandas()
        return dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]
    
    def check_correctness(self, problem, generation):
        # all problems have final answer in a list 
        answer = strip_answer_string(problem["final_answer"][0])
        pred = extract_answer(generation)
        pred = strip_answer_string(pred)
        return math_equal(pred, answer)