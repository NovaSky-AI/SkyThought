from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List


class Scorer(ABC):
    """Abstract base class for scorers."""

    SCORE_COLUMN = "score"

    @abstractmethod
    def score(self, row: dict) -> Dict[str, Any]:
        """Scores a single row of data

        Args:
            row: A dictionary containing the data to score.

        Returns:
            A dictionary containing the score and any other relevant information.
        """
        pass

    def __call__(self, row: dict):
        return {**row, **self.score(row)}


class BatchScorer(ABC):
    """
    Abstract base class for batch scorers.
    """

    SCORE_COLUMN = "score"

    INTERNAL_IDX_KEY = "__internal_idx__"

    @abstractmethod
    async def score(self, rows: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        """Scores a batch of data

        Args:
            batch: A dictionary containing the data to score.

        Returns:
            An async iterator of dictionaries containing the score and any other relevant information.
        """
        pass

    async def __call__(self, batch: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Scores a batch of data

        Yields results for each row in the batch as they finish.

        Args:
            batch: A dictionary containing the data to score.

        Returns:
            An async iterator of dictionaries containing the score and any other relevant information.
        """
        key = next(iter(batch.keys()))
        value = batch[key]
        num_rows = len(value)
        if hasattr(value, "tolist"):
            batch = {k: v.tolist() for k, v in batch.items()}
        else:
            batch = {k: list(v) for k, v in batch.items()}
        batch[self.INTERNAL_IDX_KEY] = list(range(num_rows))
        rows = [{k: batch[k][i] for k in batch.keys()} for i in range(num_rows)]
        async for result in self.score(rows):
            if self.INTERNAL_IDX_KEY not in result:
                raise ValueError(
                    f"`score` function must yield dictionaries with the key {self.INTERNAL_IDX_KEY}"
                )
            idx = result[self.INTERNAL_IDX_KEY]
            row = rows[idx]
            yield {**row, **result}
