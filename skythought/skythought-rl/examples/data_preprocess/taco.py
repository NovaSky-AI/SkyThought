# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets
import json

from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/shared/sycao/rlef/data/taco_simple')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    num_few_shot = 5
    data_source = 'BAAI/TACO'

    dataset = datasets.load_dataset(data_source, "ALL")

    train_dataset = dataset['train'].filter(lambda x: x['difficulty'] == 'EASY')
    test_dataset = dataset['test'].filter(lambda x: x['difficulty'] == 'EASY')

    instruction_following = "Let's think step by step and output the final answer in python code block."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):

            # question_raw = example['question']

            # question = question_raw + ' ' + instruction_following
            prompt = "\nQUESTION:\n"
            prompt += example["question"]
            starter_code = None if len(example["starter_code"]) == 0 else example["starter_code"]
            try:
                input_outpout = json.loads(example["input_output"])
                fn_name = (
                    None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
                )
            except ValueError:
                fn_name = None
            if starter_code:
                prompt += starter_code
            if (not fn_name) and (not starter_code):
                call_format = "\nUse Standard Input format"
                prompt += call_format
            else:
                call_format = "\nUse Call-Based format"
                prompt += call_format
            prompt += "\nANSWER:\n"

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "task_id": idx,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
