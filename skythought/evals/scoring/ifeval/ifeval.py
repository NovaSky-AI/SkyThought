from typing import Any, Dict, List

from skythought.evals.scoring.ifeval.instructions_main import (
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)

from ..base import Scorer


def process_results(doc, response):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


class IfEvalScorer(Scorer):
    """Scorer for the IF-Eval task

    For the IFEval dataset format, see https://huggingface.co/datasets/google/IFEval

    Args:
        instruction_ids_column: The column name for the list of instruction ids (str).
        prompt_column: The column name for the prompt (str).
        keyword_args_column: The column name for the keyword arguments to the instruction builder (str).
        key_column: The column name for the unique identifier (str).
        response_column: The column name for the response (str).
    """

    SCORE_COLUMN = "ifeval_score"

    def __init__(
        self,
        instruction_ids_column: str = "instruction_id_list",
        prompt_column: str = "prompt",
        keyword_args_column: str = "kwargs",
        key_column: str = "key",
        response_column: str = "response",
    ):
        self.instruction_ids_column = instruction_ids_column
        self.response_column = response_column

    def score(self, row: dict) -> Dict[str, Any]:
        return {self.SCORE_COLUMN: process_results(row, row[self.response_column])}

    @property
    def expected_keys(self) -> List[str]:
        return [
            self.instruction_ids_column,
            self.prompt_column,
            self.keyword_args_column,
            self.key_column,
            self.response_column,
        ]
