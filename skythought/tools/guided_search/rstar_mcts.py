"""
rStar Implementation
"""

import copy
import json
import math

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type
from distributed.utils import print_rank_0, print_with_rank
from envs.base_env import CoTEnv
import pdb
from tqdm import tqdm
import heapq
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict

from .tree import Node, LanguageNode, SearchTree
from envs.rstar.rstar_utils import *


class RstarSearchTree(SearchTree):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.parent2children: Dict[RstarLanguageNode, List[RstarLanguageNode]] = dict()
        self.explored_nodes = set()
        self.N: Dict[RstarLanguageNode, int] = defaultdict(
            lambda: 0
        )  # total visit count for each node
        self.Q: Dict[MCTS_Node, float] = defaultdict(
            lambda: 0.0
        )  # total reward of each node
        self.weight_scheduler = "const"
        self.mcts_exploration_weight = 2.0
        self.max_depth_allowed = 5
        self.show_tree_expansion = True

    @override
    def _select_child(
        self, node: RstarLanguageNode, simulate_env: Type[CoTEnv], rollout_id: int
    ) -> Tuple[RstarLanguageNode, bool]:

        # for select, if there is unexplored children, select it randomly. if all children nodes
        # have been explored, select UCB, a leaf node means it has no children, return True
        # when there is unexplored node

        if node not in self.parent2children.keys():
            return node, True

        # if there are children unexplored
        unexplored = [
            n for n in self.parent2children[node] if n not in self.explored_nodes
        ]
        if unexplored:
            next_node = random.choice(unexplored)
            return next_node, True

        # if all have been explord, from parent2children dict, select one node with highest UCB score

        # Get the list of children for the current node
        children = self.parent2children[node]

        # Compute UCT values for each child node
        uct_values = {
            n: self._compute_uct(parent_node=node, node=n, rollout_id=rollout_id)
            for n in children
        }
        # print(f"@@@ uct = {uct_values}, node type = {[i.node_type for i in children]}")
        # Find the child with the maximum UCT value
        next_node = max(uct_values, key=uct_values.get)

        return next_node, False

    def _compute_uct(
        self, parent_node: RstarLanguageNode, node: RstarLanguageNode, rollout_id: int
    ):
        "Upper confidence bound for trees"
        if parent_node is None:  # invalid UCT: the node is the root
            return 666
        else:
            if self.N[node] == 0:  # invalid UCT: the node has not been explored yet
                return 999
            else:
                weight = self._get_weight(rollout_id)
                return self.Q[node] / self.N[node] + weight * math.sqrt(
                    math.log(self.N[parent_node]) / self.N[node]
                )

    def _get_weight(self, rollout_id: int):
        # start with exploration weight, end with 0.1 * exploration weight
        if self.weight_scheduler == "exp":
            return self.mcts_exploration_weight * (0.1 ** (rollout_id / self.num_path))
        elif self.weight_scheduler == "lin":
            return self.mcts_exploration_weight * (
                1 - 0.9 * (rollout_id / self.num_path)
            )
        elif self.weight_scheduler == "const":
            return self.mcts_exploration_weight

    def rstar_mcts(
        self,
        simulate_env: Type[CoTEnv],
        num_path: int,
        reward_model_fn: Optional[Callable] = None,
        select_by_prior: bool = False,
    ) -> List[Dict]:
        simulate_env.reset()
        # api_call_completion_tokens += info["api_completion_token"]
        if self.root is None:
            root = RstarLanguageNode(
                id=0,
                parent=None,
                depth=0,
                node_type=Node_Type.USER_QUESTION,
                disable_a5=False,
                user_question=simulate_env.math_problem["question"],
                expected_answer=simulate_env.math_problem["answer"],
                max_depth_allowed=self.max_depth_allowed,
                disable_a1=False,
            )
            self.root = root

        traj_list = []

        self.num_path = num_path
        model_solutions = []
        model_all_solutions = []
        model_rollout_nodes = []

        for i_path in tqdm(range(num_path), desc=f"Running {num_path} MCTS paths"):
            node = self.root
            env_copy = simulate_env.copy()
            done = False
            node_path = []  # for boostrapping
            while not done:
                # this is the whole process of navigating from root to a terminate node, along the way,
                # there are explored nodes with children where we do UCB, and there are unexplored nodes where we
                # expand its children through legal_action_update,
                # for select, if there is unexplored children, select it randomly. if all children nodes
                # have been explored, select UCB
                next_node, is_leaf = self._select_child(
                    node, env_copy, i_path
                )  # find a leaf node or
                # simulate the remaining
                node_path.append(next_node)

                done = env_copy.is_terminal(next_node)  # checking terminal condition
                # update legal action (expand) when the current code is not a leaf (no children)
                # no env.step only use for checking termination when is not leaf, and update legal action
                # when the current node is leaf
                # print(f"Path {i_path}: depth = {next_node.depth}, done = {done}, is leaf = {is_leaf}")

                if (
                    not done and is_leaf
                ):  # expand when encounter a leaf node one step further
                    next_node_children = env_copy.try_update_legal_action(
                        node=next_node
                    )
                    for c in next_node_children:
                        c.set_rollout_id(i_path)
                    self.parent2children[next_node] = next_node_children

                if self.show_tree_expansion:
                    self.draw_tree()

                if done:
                    self.explored_nodes.add(next_node)

                node = next_node

            else:
                # boostrapping
                reward = next_node.calculate_reward()

                for node in reversed(node_path):
                    self.Q[node] += reward
                    self.N[node] += 1
                    self.explored_nodes.add(node)

            model_rollout_nodes.append(next_node)
            _, best_solution, _, chosen_node, all_solution_nodes, all_solutions = (
                stochastic_find_best_solution(
                    self.root, env_copy.evaluator, enable_potential_score=False
                )
            )

            # model_solutions.append(best_solution)
            # model_all_solutions.append(all_solutions)
            assert best_solution is not None

            traj_data = {
                "path_idx": i_path,
                "text": best_solution,
                "value": reward,
                "api_completion_tokens": env_copy.total_api_call_completion,
                "tree_completion_tokens": env_copy.total_tree_completion,
            }

            traj_list.append(traj_data)

        return traj_list

    def draw_tree(self):

        def display_tree(node):
            print("|" + "-" * (node.depth * 4) + str(node))
            for child in node.children:
                display_tree(child)

        print(f"\n---------Expanded Tree---------")
        display_tree(self.root)