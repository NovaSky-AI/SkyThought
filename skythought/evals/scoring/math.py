from typing import Any, Dict, List

from ..util.math_parsing_util import extract_answer, math_equal
from .base import Scorer

try:
    from math_verify import parse as mv_parse
    from math_verify import verify as mv_verify
except ImportError:
    mv_parse = None
    mv_verify = None


class MathEqualScorer(Scorer):
    """Scorer for math based on the `math_equal` function from Qwen Math

    Args:
        response_column: The column name for the model generated response.
        answer_column: The column name for the ground truth answer.
    """

    SCORE_COLUMN = "math_equal_score"

    def __init__(self, response_column: str, answer_column: str):
        self.response_column = response_column
        self.answer_column = answer_column

    def score(self, row: dict) -> Dict[str, Any]:
        try:
            pred = extract_answer(row[self.response_column])
            ref = extract_answer(row[self.answer_column])
        except Exception:
            return False
        return {self.SCORE_COLUMN: math_equal(pred, ref)}

    @property
    def expected_keys(self) -> List[str]:
        return [self.response_column, self.answer_column]


class MathVerifyScorer(Scorer):
    """Scorer for math based on the `math_verify` function from HuggingFace

    Args:
        response_column: The column name for the model generated response.
        answer_column: The column name for the ground truth answer.
    """

    SCORE_COLUMN = "math_verify_score"

    def __init__(self, response_column: str, answer_column: str):
        self.response_column = response_column
        self.answer_column = answer_column
        if mv_parse is None or mv_verify is None:
            raise ImportError(
                "`math_verify` is not installed. Please install it with `pip install math_verify`."
            )

    def score(self, row: dict) -> Dict[str, Any]:
        try:
            pred = mv_parse(row[self.response_key])
            ref = mv_parse(row[self.answer_key])
        except Exception:
            return False
        return {self.SCORE_COLUMN: mv_verify(pred, ref)}

    @property
    def expected_keys(self) -> List[str]:
        return [self.response_column, self.answer_column]
