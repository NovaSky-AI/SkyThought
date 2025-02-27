from typing import Any, AsyncIterator, Dict, List

import pytest

from skythought.evals.scoring.base import BatchScorer


class TestBatchScorer:
    class BasicBatchScorer(BatchScorer):
        def __init__(self, response_column: str, answer_column: str):
            self.response_column = response_column
            self.answer_column = answer_column

        async def score(
            self, batch: List[Dict[str, Any]]
        ) -> AsyncIterator[Dict[str, Any]]:
            # emit out of order
            for row in batch[::-1]:
                yield {
                    self.INTERNAL_IDX_KEY: row[self.INTERNAL_IDX_KEY],
                    self.SCORE_COLUMN: row[self.response_column]
                    == row[self.answer_column],
                }

    @pytest.mark.asyncio
    async def test_basic_scoring(self):
        scorer = self.BasicBatchScorer("response", "answer")
        batch = {
            "question": ["A", "B", "C", "D"],
            "response": ["match", "not match", "match", "match"],
            "answer": ["match", "match", "match", "match"],
        }
        async for result in scorer(batch):
            assert scorer.INTERNAL_IDX_KEY in result.keys()
            # get index from the result dict
            idx = result[scorer.INTERNAL_IDX_KEY]
            # lookup expected answer from the original batch
            expected_answer = batch["response"][idx] == batch["answer"][idx]
            # check if the output matches expected answer
            assert result[scorer.SCORE_COLUMN] == expected_answer
            # ensure other keys in the batch are kept as-is
            assert batch["question"][idx] == result["question"]
