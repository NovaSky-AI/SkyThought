import logging
from collections import defaultdict
from typing import Any, Dict

import numpy as np


def _pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def pass_at_k(N: int, temp_to_scores: Dict[str, Dict[str, Any]]):
    # pass at k per temperature
    # scores = list(correct[temp].values())
    pass_values = {}  # temp -> value
    for temp in temp_to_scores:
        scores = temp_to_scores[temp]  # dict mapping idx -> list of scores
        final_passk_scores = {}
        k_to_passk_scores = defaultdict(list)  # k -> list of scores
        for _, sample_scores in scores.items():
            k = N
            while k > 0:
                # calculate pass @ k
                num_correct = np.sum(sample_scores)
                pass_k = _pass_at_k(N, num_correct, k)
                k_to_passk_scores[k].append(pass_k)
                k = k // 2

        for k in k_to_passk_scores:
            final_passk_scores[f"{k=}"] = round(np.mean(k_to_passk_scores[k]) * 100, 3)

        # print("Final pass @ k:")
        for k, s in final_passk_scores.items():
            logging.info(f"temp: {temp}, k: {k}, pass @ k: {s}")
        pass_values[f"{temp=}"] = final_passk_scores
        # temp_correct = sum([any(x) for x in scores])
    return pass_values
