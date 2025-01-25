from tasks.common import TaskHandler
from util.math_parsing_util import extract_answer, math_equal, strip_answer_string
from tasks.math.math_handler import MathTaskHandler

class MinervaMathTaskHandler(MathTaskHandler):
    def __init__(self):
        self.dataset = "svc-huggingface/minerva-math"
    
    @staticmethod
    def get_question_key():
        return "problem"

    def check_correctness(self, problem, generation):
        answer = extract_answer(problem["solution"])
        answer = strip_answer_string(answer)

        pred = extract_answer(generation)
        pred = strip_answer_string(pred)
        return math_equal(pred, answer)