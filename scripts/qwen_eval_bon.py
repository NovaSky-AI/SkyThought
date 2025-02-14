import argparse
import json
import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

set_seed(args.seed)


with open(args.path, "r") as f:
    data = [json.loads(line) for line in f.readlines()]


final_bon_scores = {}

num_scores = len(data[0]["score"])
N = num_scores
k = num_scores

actual_accuracy = sum([sum(sample["score"]) for sample in data]) / (
    len(data) * num_scores
)
print(f"Actual accuracy: {actual_accuracy}")

while k > 0:
    new_scores = []
    for sample in data:
        # calculate pass @ k
        num_correct = np.sum(sample["score"])
        # pass_k = 1 - (math.comb(N - num_correct, k) / math.comb(N, k))
        pass_k = pass_at_k(N, num_correct, k)
        new_scores.append(pass_k)
    final_bon_scores[k] = round(float(np.mean(new_scores)) * 100, 3)
    k = k // 2

print(f"Final pass @ k for {args.path}:")
print(final_bon_scores)
