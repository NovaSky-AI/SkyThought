import logging
import math
from collections import defaultdict
from typing import Dict, List

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


def pass_at_k(N: int, id_to_scores: Dict[str, List[int]]):
    final_passk_scores = {}
    k_to_passk_scores = defaultdict(list)  # k -> list of scores
    for _, sample_scores in id_to_scores.items():
        # Start at N
        k = N
        is_power_of_2 = N == 2 ** (int(math.log2(N)))
        while k > 0:
            # calculate pass @ k
            num_correct = np.sum(sample_scores)
            pass_k = _pass_at_k(N, num_correct, k)
            k_to_passk_scores[k].append(pass_k)
            # corner case: when N is not a power of 2
            if not is_power_of_2 and k == N:
                k = 2 ** (int(math.log2(N)))
            else:
                # otherwise, just divide by 2
                k = k // 2

    for k in k_to_passk_scores:
        final_passk_scores[f"{k=}"] = round(np.mean(k_to_passk_scores[k]) * 100, 3)

    # print("Final pass @ k:")
    for k, s in final_passk_scores.items():
        logging.info(f"k: {k}, pass @ k: {s}")
    return final_passk_scores
