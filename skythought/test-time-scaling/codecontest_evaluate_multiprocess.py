from datasets import load_dataset
import sys
from functools import partial
import importlib.util
import dspy
import numpy as np
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
from multiprocessing import Lock, Manager
import sys

from live_code_bench_program import (
#    NaiveCodeGeneratorScaling,
#    generate_tests_for_whole_dataset,
    generate_tests_repeat,
    post_process_tests,
    post_process_code,
    filter_extracted_tests,
    extract_tests,
    has_test_type,
    NaiveCodeGeneratorNoDSPy,
#    CodeGeneratorWithIteratedRanker,
    CodeGeneratorWithSelfDebug,
#    NaiveCodeGenerator,
#    CodeGeneratorWithRanker,
#    NaiveCodeGeneratorScalingPattern,
    generate_tests_for_one_example,
    NaiveCodeGenerator,
    generate_timeout_tests_repeat
)
import random
from datetime import datetime
from live_code_bench_execute import check_correctness, check_correctness_oracle, unsafe_lcb_run_timeout_tests

import json
import base64
import zlib
import pickle
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from util import name_map

############################################## HyperParameters ##########################################

MODEL = "gpt-4o-mini"
TIMEOUT_CONSTANT=6
# make DIFFICULTY is_stdin and METHOD as command line arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_threads", type=int, default=1, help="Number of threads to use")
parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
parser.add_argument('--n', type=int, default=16, help='Number of code generation samples.')
parser.add_argument('--context', type=str, default="all", help="'all' means use all history for self-debug. 'last' means only use latest code for self-debug.")
parser.add_argument('--selection', type=str, default="first", help="How to select the final solution.")
parser.add_argument('--num_round', type=int, default=3, help="How many rounds for generation for self-debug. 1 means only using the initial solution (with a self-refine stage).")
parser.add_argument('--selfdebug_decision', type=str, default="exit", help="What to do when self debug has passed all public tests. Exit means directly save the sample, 'refine' means continue to improve its own answer based on self-generated tests.")
parser.add_argument('--judge', type=str, default="4o-mini", help="The model used to judge.")
parser.add_argument('--generator', type=str, default="4o-mini", help="The model used to generate solution.")
parser.add_argument('--result_json_path', type=str, required=True, help="Json file to store the current result.")
parser.add_argument('--ablation_judge_api_name', type=str, default=None, help="The model used to judge.")
parser.add_argument('--ablation_judge_api_base', type=str, default="4o-mini", help="The model used to judge.")
parser.add_argument('--method', type=str, default="selfdebug", help="The method used to generate the code.")
parser.add_argument('--test_generator', type=str, default="4o-mini", help="The model used to judge.")
parser.add_argument('--num_test_suites', type=int, default=1, help="The number of test suites to generate..")
parser.add_argument('--api_name', type=str, default=None, help="Whether to use local served model.")
parser.add_argument('--api_base', type=str, default=None, help="API Base for local served model.")
parser.add_argument('--no_refine', action="store_true", help="Whether to use reflection after one round of generation.")
parser.add_argument('--no_dspy_gen', action="store_true", help="Whether to use dspy for generating response.")
parser.add_argument('--num_icl_examples', type=int, default=0, help="The number of ICL examples to use.")
parser.add_argument('--enable_llm_reflection_with_tool', action="store_true", help="Whether to use reflection after one round of generation.")
parser.add_argument('--enable_vanilla_reflection', action="store_true", help="Whether to use vanilla reflection.")
parser.add_argument('--ablation_qwq_vanilla_without_reasoning', action="store_true", help="Whether to include reasoning when doing vanilla self debug.")
parser.add_argument('--ablation_qwq_debug_with_4o_mini', action="store_true", help="Whether to use 4o-mini for debug.")
parser.add_argument('--load_cached_preds', action="store_true", help="Whether to load the cached predictions for selection.")
parser.add_argument('--cached_preds_path', type=str, default=None, help="Path to the cached predictions for selection.")
parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

args = parser.parse_args()
print(args)
if os.path.exists(args.result_json_path):
    with open(args.result_json_path, "r") as f:
        lines = f.readlines()
    
    if len(lines) > 0 and "final_accuracy" in lines[-1]:
        print(f"This run has been finished previously. Skipped to the next experiment.")
        sys.exit(0)

############################################## setting up #################################################
# turbo = dspy.OpenAI(model=MODEL, max_tokens=4096)
use_dspy_cache = True
if args.api_name is not None:
    #  2 hourfor a single request for self-served model
    lm = dspy.LM(args.api_name, api_base=args.api_base, api_key="None", timeout=7200)
    args.generator = args.api_name
else:
    if "o1" in args.generator:
        lm = dspy.LM(name_map[args.generator], max_tokens=65536, temperature=1.0, cache=False)
    else:
        lm = dspy.LM(name_map[args.generator], cache=use_dspy_cache)
judge_lm = dspy.LM(name_map[args.judge], cache=use_dspy_cache)
if args.ablation_judge_api_name is not None:
    ablation_judge_lm = dspy.LM(args.ablation_judge_api_name, api_base=args.ablation_judge_api_base, api_key="None")
else:
    ablation_judge_lm = dspy.LM('openai/gpt-4o-mini', cache=use_dspy_cache)
if "o1" in args.test_generator:
    test_generator_lm = dspy.LM(name_map[args.test_generator], max_tokens=65536, temperature=1.0)
    print("Using o1")
else:
    test_generator_lm = dspy.LM(name_map[args.test_generator], cache=use_dspy_cache)
    print("Using 4omini")
dspy.settings.configure(lm=lm)#turbo)
dspy.configure(trace=[])

lcb_codegen = load_dataset(
            "deepmind/code_contests", split="test", trust_remote_code=True
            )

import json

def modify_entry(entry, idx):
    entry["question_id"] = idx

    # Ensure lists exist
    entry["private_tests"]["input"] = list(entry["private_tests"].get("input", []))
    entry["private_tests"]["output"] = list(entry["private_tests"].get("output", []))
    entry["generated_tests"]["input"] = list(entry["generated_tests"].get("input", []))
    entry["generated_tests"]["output"] = list(entry["generated_tests"].get("output", []))

    # Extend private tests with generated tests
    entry["private_tests"]["input"].extend(entry["generated_tests"]["input"])
    entry["private_tests"]["output"].extend(entry["generated_tests"]["output"])

    # Convert to desired format
    private_test_cases = []
    for inp, out in zip(entry["private_tests"]["input"], entry["private_tests"]["output"]):
        private_test_cases.append({
            "input": inp.strip(),  # Ensure clean formatting
            "output": out.strip(),
            "testtype": "stdin"  # Set test type as "stdin"
        })

    # Apply the new format
    entry["private_test_cases"] = json.dumps(private_test_cases)
    entry["public_test_cases"] = json.dumps(entry.pop("public_tests"))  # Store as JSON string

    return entry


# add question_id to each entry
# selected_idx = [10]
# lcb_codegen = lcb_codegen.select(selected_idx).map(modify_entry, with_indices=True)
lcb_codegen = lcb_codegen.map(modify_entry, with_indices=True)

## load the cached predictions for selection
def load_json_cached_preds(filename):
    results = {}
    with open(filename, 'r') as f:
        content = f.read()

    # Split content by newlines and process each line
    lines = content.split('\n')
    for line in lines:
        if not line.strip():
            continue
            
        try:
            obj = json.loads(line)
            if 'task_id' in obj:
                results[obj['task_id']] = obj
        except json.JSONDecodeError:
            print(f"Error decoding JSON line: {line[:100]}...")  # Print first 100 chars of problematic line
            continue

    return results
if args.load_cached_preds:
    cached_preds_dict = load_json_cached_preds(args.cached_preds_path)
else:
    cached_preds_dict = None
############################################ Generate dspy examples #######################################

def map_to_dspy_example(row):
    return {
        "prompt": row["description"],
        "test": json.loads(row["private_test_cases"]),
        "entry_point": "",
        "canonical_solution": "",  # seems like live code bench lite does not have this field
        "task_id": row["question_id"],
        "is_stdin": True,
        "public_test_cases": row["public_test_cases"],
    }



random.seed(41)
print(f"Before: {len(lcb_codegen)}")
filtered_lcb_codegen_list = lcb_codegen

extracted_tests = {}
for entry in filtered_lcb_codegen_list:
    extracted_tests[entry["question_id"]] = json.loads(entry["public_test_cases"])

# update_dataset_in_place(filtered_lcb_codegen_list)  ## decode the private test cases

## extract the private test cases for calculating pass at 20
extracted_private_tests = {}
for entry in filtered_lcb_codegen_list:
    extracted_private_tests[entry["question_id"]] =  json.loads(entry["private_test_cases"])


live_code_bench_dataset = [
    dspy.Example(**map_to_dspy_example(row)).with_inputs(
        "prompt", "test", "entry_point", "canonical_solution", "task_id", "is_stdin", "public_test_cases"
    )
    for row in filtered_lcb_codegen_list
]

################################### Define evaluation functions ###############################################
## Currently only support method 

def get_accuracy(dataset, num_process_evaluate, method = "selfdebug", timeout = 6):
    """Take in a dataset or subset of dataset, evaluate accuracy using multiprocessing"""
    total_passed = 0

    # Create a lock for safe concurrent file access
    manager = Manager()
    lock = manager.Lock()

    with tqdm(total=len(dataset) , desc="Progress") as pbar:
        with ProcessPoolExecutor(
            max_workers = num_process_evaluate
        ) as executor:
            futures = {
                executor.submit(generate_and_evaluate, (example, timeout, method, args.result_json_path, lock)): i
                for i, example in enumerate(dataset)
            }

            results = {}
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                results[index] = result
                # Update running statistics
                total_passed += int(result["passed"])
                current_accuracy = (total_passed / (len(results))) * 100
                
                # Update progress bar with accuracy
                pbar.set_postfix({
                    'Accuracy': f'{current_accuracy:.2f}%',
                    'Passed': f'{total_passed}/{len(results)}'
                })
                
                pbar.update(1)

    assert len(results) == len(
        dataset
    ), f"results = {len(results)} inputs = {len(dataset)}"

    final_accuracy = total_passed / len(dataset)

    return final_accuracy

def get_accuracy_all_rounds(dataset, num_process_evaluate, method="selfdebug", timeout=6):
    """Evaluate accuracy across multiple rounds using multiprocessing."""
    manager = Manager()
    lock = manager.Lock()

    # Initialize total_passed as a list of zeros for each round
    total_passed = [0] * args.num_round
    num_results_per_round = [0] * args.num_round  # Track the number of results per round

    with tqdm(total=len(dataset), desc="Progress") as pbar:
        with ProcessPoolExecutor(max_workers=num_process_evaluate) as executor:
            futures = {
                executor.submit(generate_and_evaluate, (example, timeout, method, args.result_json_path, lock)): i
                for i, example in enumerate(dataset)
            }

            results = {}
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                results[index] = result

                # Update total_passed for each round
                # print(result["passed"])
                for r in range(args.num_round):
                    total_passed[r] += int(result["passed"][r])
                    num_results_per_round[r] += 1

                # Update progress bar with accuracy for the last round
                current_accuracy = (total_passed[-1] / num_results_per_round[-1]) * 100
                pbar.set_postfix({
                    'Last Round Accuracy': f'{current_accuracy:.2f}%',
                    'Passed': f'{sum(total_passed)}/{len(results) * args.num_round}'
                })
                pbar.update(1)

    # Calculate final accuracy for all rounds
    final_accuracy = [
        (total_passed[r] / num_results_per_round[r]) * 100 if num_results_per_round[r] > 0 else 0
        for r in range(args.num_round)
    ]

    return final_accuracy


def generate_and_evaluate(arguments): ##TODO Alex, take in a method here to support all methods
    """
    Takes in a single dspy example, generate code and evaluate it. 

    Append the result to a shared list.
    """

    example = arguments[0]
    timeout = arguments[1]
    method = arguments[2]
    result_json_path = arguments[3]
    lock = arguments[4]
    
    # print(example)
    ## First generate the precomputed test 
    tests = None
    if "generated_tests" in args.selection or args.selfdebug_decision == "refine":
        if "o1" in args.test_generator:
            generation_fun = partial(generate_tests_repeat, temperature_base=1.0)
            retry = 1
        else:
            generation_fun = partial(generate_tests_repeat, temperature_base=0.7)
            retry = 3
        with dspy.context(lm=test_generator_lm):
            if "o1" in args.test_generator:
                tests = generate_tests_for_one_example(example,generation_fun=generation_fun, num_test_suites=args.num_test_suites, retry=retry, o1=True)
            else:
                tests = generate_tests_for_one_example(example,generation_fun=generation_fun, num_test_suites=args.num_test_suites, retry=retry)
        
        # print(tests) 
        # assert False
    ## Initialize the code generator
    if method == "selfdebug":
        ## initialize debug lm to be 40mini : TODO(Alex): delete this if not work, or add a new argument for this if this works
        debug_lm = dspy.LM('openai/gpt-4o-mini', cache=use_dspy_cache)
        test_program = CodeGeneratorWithSelfDebug(extracted_tests, num_round=args.num_round, n=args.n, temperature=args.temperature, 
                                                  lm=lm, selection=args.selection, context=args.context, judge_lm=judge_lm, pre_computed_tests=tests, 
                                                  selfdebug_decision=args.selfdebug_decision, ablation_judge_lm=ablation_judge_lm, debug_lm=debug_lm, num_icl_examples=args.num_icl_examples, args=args, enable_llm_reflection_with_tool=args.enable_llm_reflection_with_tool, enable_vanilla_reflection=args.enable_vanilla_reflection, ablation_qwq_vanilla_without_reasoning=args.ablation_qwq_vanilla_without_reasoning, ablation_qwq_debug_with_4o_mini=args.ablation_qwq_debug_with_4o_mini, cached_preds_dict=cached_preds_dict)
    elif "naive" in method:
        # method == "naive":
        if method == "naive":
            test_program = NaiveCodeGenerator(cot=False)
        elif method == "naive_cot":
            test_program = NaiveCodeGenerator(cot=True)
        elif method == "naive_nodspy":
            test_program = NaiveCodeGeneratorNoDSPy(args)
        

    # TODO: @DL support oracle
    # if args.selection == "debug_all":
    #    eval_metric=live_code_bench_evaluate_batch
    # else:
    #    eval_metric=live_code_bench_evaluate

    is_stdin = example.is_stdin
    task_id = example.task_id
    if not is_stdin:
        is_extracted=True
    else:
        is_extracted=False
    # print(f"Task ID: {task_id} is_stdin: {is_stdin}")
    
    # print("=" * 10 + "Generating prediction" + "=" * 10)
    if method == "selfdebug":
        (pred, reflections), _ = test_program(example, None, task_id, None, None, is_stdin)
        # try:
        #     (pred, reflections), _ = test_program(example, None, task_id, None, None, is_stdin)  #TODO , Prune the arguments that are obsolete, Also, added timeout tests so now take in whole example 
        # except Exception as e:
        #     print(f"Error in test program {e}")
        #     reflections = None
        #     if args.selection == "oracle":
        #         pred = dspy.Prediction(codes=[dspy.Prediction(code="")] * args.n, prompts=[example.prompt] * args.n)
        #     else:
        #         pred = dspy.Prediction(code=dspy.Prediction(code=""))
    # elif method == "naive":
    elif "naive" in method:
        pred = test_program(example["prompt"], is_stdin)
    # print("=" * 10 + "Finished generating prediction" + "=" * 10)
    
    ## Evaluate the prediction
    
    if args.selection == "oracle":  ## the oracle method to see pass@n
        result_dict = check_correctness_oracle(example, pred, timeout, is_extracted = is_extracted, debug_mode = True, fast_check=True)
        completion_and_pass = result_dict["completion_and_pass"]
        codes = result_dict["codes"]
        result_json = {
            'task_id': example.task_id,
            'codes': codes,
            'passed': completion_and_pass
        }
        
        with lock:
            with open(result_json_path, 'a') as f:
                f.write(json.dumps(result_json) + '\n')
    elif args.selection == "oracle_all_rounds":  ## the oracle method to see pass@n for all rounds
        result_json = {
            'task_id': example.task_id,
            'codes': [[] for _ in range(args.num_round)],
            'passed': [0] * args.num_round #[None for _ in range(args.num_round)]
        }

        for r in range(args.num_round):
            # print(r, pred)
            result_dict = check_correctness_oracle(example, pred[r], timeout, is_extracted = is_extracted, debug_mode = True, fast_check=True)
            completion_and_pass = result_dict["completion_and_pass"]
            codes = result_dict["codes"]

            result_json["codes"][r] = codes
            result_json["passed"][r] = result_dict["passed"]
            # print(completion_and_pass)
        with lock:
            with open(result_json_path, 'a') as f:
                f.write(json.dumps(result_json) + '\n')
        return result_json
    else: 
        result_dict = check_correctness(example, post_process_code(pred.code), timeout, is_extracted = is_extracted, fast_check=True)
        code = result_dict["code"]
        result_json = {
            'task_id': example.task_id,
            'code': code,
            'passed': result_dict["passed"]
        }
        with lock:
            with open(result_json_path, 'a') as f:
                f.write(json.dumps(result_json) + '\n')
    # print(result_dict["maybe_error_messages"])

   #  print(result_dict)
    return result_dict

########################################### Main ##########################################################

# Custom function to convert namespace to a serializable dictionary
def namespace_to_serializable_dict(namespace):
    def default_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to ISO 8601 string
        raise TypeError(f"Type {type(obj)} not serializable")
    
    # Convert namespace to a dictionary
    namespace_dict = vars(namespace)
    
    # Serialize to JSON (if required)
    return json.dumps(namespace_dict, default=default_serializer)

if __name__ == "__main__":
    # Ensure the shared file exists and is empty
    open(args.result_json_path, 'w').close()
    with open(args.result_json_path, "a") as f:
        f.write(json.dumps(namespace_to_serializable_dict(args)) + '\n')
    if args.seed is not None: ## for reproducibility of randomly selecting answers
        random.seed(args.seed)
    if args.selection != "oracle_all_rounds":
        accuracy = get_accuracy(live_code_bench_dataset, num_process_evaluate=args.num_threads,method=args.method, timeout=TIMEOUT_CONSTANT)
        print(f"Accuracy: {accuracy*100:.2f}%")
        with open(args.result_json_path, "a") as f:
            f.write(json.dumps({"final_accuracy": f"{accuracy*100:.2f}%"}) + '\n')
    else:
        accuracy = get_accuracy_all_rounds(live_code_bench_dataset, num_process_evaluate=args.num_threads,method=args.method, timeout=TIMEOUT_CONSTANT)
        print(f"Accuracy: {accuracy}")
        with open(args.result_json_path, "a") as f:
            f.write(json.dumps({"final_accuracy list": accuracy}) + '\n')
