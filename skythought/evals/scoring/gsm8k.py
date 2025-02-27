import re
from typing import Any, Dict, List

from skythought.evals.util.math_parsing_util import extract_answer, math_equal

from .base import Scorer


class GSM8KScorer(Scorer):
    """Scorer for GSM8K based on the `math_equal` function from Qwen Math

    Args:
        response_column: The column name for the model generated response.
        answer_column: The column name for the ground truth answer.
    """

    SCORE_COLUMN = "gsm8k_score"
    INVALID_ANS = "[invalid]"
    GT_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    ANS_RE = re.compile(r"((-?[$0-9.,]{2,})|(-?[0-9]+))")

    def __init__(self, response_column: str, answer_column: str):

        self.response_column = response_column
        self.answer_column = answer_column

    def score(self, row: dict) -> Dict[str, Any]:
        try:
            pred = self.extract_pred_from_response(row[self.response_key])
            ref = self.extract_gt_answer(row[self.answer_key])
        except Exception:
            return False
        return {
            self.SCORE_COLUMN: math_equal(pred, ref),
        }

    def extract_gt_answer(self, completion):
        match = self.GT_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return self.INVALID_ANS

    def extract_pred_from_response(self, response):
        answer = extract_answer(response)
        answer = self.sanitize_answer(response)
        return answer

    def sanitize_answer(self, answer):
        patterns_to_remove = [
            ",",  # Remove commas
            r"\$",  # Remove dollar signs
            r"\.$" r"\*",  # Remove trailing period  # Remove asterisks
        ]
        for pattern in patterns_to_remove:
            answer = re.sub(pattern, "", answer)

        matches = self.ANS_RE.findall(answer)
        if matches:
            # get the last match (i.e final response) and the first / outer capturing group
            match_str = matches[-1][0].strip()
            return match_str
        else:
            return self.INVALID_ANS

    @property
    def expected_keys(self) -> List[str]:
        return [self.response_column, self.answer_column]
