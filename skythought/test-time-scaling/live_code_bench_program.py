import math
import random
import json
from datasets import load_dataset
import dspy.evaluate
import numpy as np
import tqdm
from live_code_bench_execute import (
    check_correctness,
    unsafe_lcb_runTests,
    time_limit,
    post_process_timeout_tests_func,
    post_process_timeout_tests_stdin,
    unsafe_lcb_run_timeout_tests,
)
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
import concurrent.futures
import openai
import re
import threading
import copy
from itertools import chain
import time
from rich import print as rprint
from collections import defaultdict
import multiprocessing
from multiprocessing import Process, Manager
from openai import OpenAI
from util import name_map
TIMEOUT_CONSTANT = 6
# os.environ["DSP_NOTEBOOK_CACHEDIR"] = "./human_eval_dspy_cache"

# Settings for prompt without demo
# NUM_SAMPLES = 50
# TEMPARATURE_BASE = 0.0
# TEMPARATURE_STEP = 0.01
# NUM_TESTS = 5

# Settings for prompt with demo
# lock = threading.Lock()

NUM_SAMPLES = 20
# TEMPARATURE_BASE = 0.7
TEMPARATURE_STEP = 0.0001
NUM_TESTS = 5


import dspy
from dspy.evaluate import Evaluate
from util import post_process_code, name_map, ICL_EXAMPLES

class CodeProblem(dspy.Signature):
    prompt = dspy.InputField(format=str)
    code = dspy.OutputField()

def post_process_tests(tests, prompt=None):
    fun_name = (
        prompt[prompt.find("def") + 3 : prompt.find("(")].strip() if prompt else "None"
    )
    result = []
    for l in tests.split("\n"):
        if l.strip().startswith("assert"):
            assert_test = l.strip()
            eqns = assert_test[7:].split("==")
            # assert_test = f"{assert_test}"
            if len(eqns) != 2:
                result.append(assert_test)
                continue
            actual, expected = eqns
            assert_message = f"Expected {expected.strip()}"
            actual_result = "result = " + actual.strip() + "\n"
            result.append(f"{actual_result}\n{assert_test}, {repr(assert_message)}")
    result = [f"candidate = {fun_name}\n{test}" for test in result]
    return result


def has_test_type(tests, type):  ## helper to select specific type of problems
    """
    Check if any test in the test list has 'testtype' set to 'type'.
    """
    test_list = json.loads(tests)
    for test in test_list:
        if test.get("testtype") == type:
            return True
    return False

def post_process_tests_inputs(raw_text, is_stdin):
    # raw_text = raw_text.strip().strip("```json").strip("```").strip() # raw_text.strip()
    # print(raw_text)
    if is_stdin: 
        blocks = raw_text.split("Input:")

        formatted_tests = []

        for block in blocks:
            if not block.strip():
                continue

            input_output = block.split("Output:")

            if len(input_output) == 2:
                input_value = input_output[0].strip()
                output_value = input_output[1].strip()

                formatted_tests.append(
                    {
                        "input": input_value + "\n",
                        "output": output_value + "\n",
                        "testtype": "stdin",
                    }
                )
        return formatted_tests 
    else:
        # Step 1: Clean the input string by removing surrounding markdown syntax and extra spaces
        cleaned_string = raw_text.strip().strip("```json").strip("```").strip()

        # Step 2: Check if it's a JSON array
        if cleaned_string.startswith("[") and cleaned_string.endswith("]"):
            test_cases = json.loads(cleaned_string)
            for test_case in test_cases:
                test_case["testtype"] = "functional"
            return test_cases

        # Step 3: Handle cases where multiple JSON objects are concatenated without commas
        else:
            # Use regex to separate JSON objects by detecting patterns starting with {"input": and ending with the last }
            json_pattern = re.compile(r'(\{"input":.*?"output":.*?\})', re.DOTALL)
            matches = json_pattern.findall(cleaned_string)

            if matches:
                # Combine matches into a valid JSON array by inserting commas between objects
                json_array_string = "[" + ",".join(matches) + "]"
                try:
                    test_cases = json.loads(json_array_string)
                    for test_case in test_cases:
                        test_case[
                            "testtype"
                        ] = "functional"  # Add 'testtype' for each test case
                    return test_cases
                except json.JSONDecodeError as e:
                    print(f"Error parsing concatenated JSON: {e}")

            # If no matches are found, fall back to line-by-line parsing
            cleaned_lines = cleaned_string.split("\n")
            if test_cases == None:
                test_cases = []
            for line in cleaned_lines:
                try:
                    test_case = json.loads(line)
                    test_case["testtype"] = "functional"
                    test_cases.append(test_case)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")
                    with open("DEBUG NOT JSON RETURN TEST.txt", "a") as log_file:
                        log_file.write(f"{line}\n")
                    continue

            return test_cases

def human_eval_evaluate(
    example: dspy.Example, pred: dspy.Prediction, target: str = None
):
    result = check_correctness(example.toDict(), post_process_code(pred.code), 2)
    return result["passed"]


class GenerateTests(dspy.Signature):
    prompt = dspy.InputField(format=str)
    tests = dspy.OutputField(
        desc="Executable tests using assert, you can use the one in prompts. \
                             Do not put tests in another local function, directly write them."
    )


class GenerateTests_std(dspy.Signature):
    prompt = dspy.InputField(format=str)
    tests = dspy.OutputField(
        desc="Input and output pair as a list that can be used to test a solution to the prompt. Assume input will be used via stdin and output is caputred stdout .You can use the one in prompts. \
                            Do not give anything else than the input output pair, in the format of Input: input Output: output."
    )

class GenerateTests_timeout_std(dspy.Signature):
    prompt = dspy.InputField(format=str)
    tests = dspy.OutputField(
        desc="Generate python code expression to be evaluated to a single long input to test whether an AI-generated solution will timeout to the coding problem. \
        Assume the input will be used via stdin, do not attempt to write out the complete long input, instead, give an input expression directly, and the input expression should be evaluated directly to a string representation of the stdin input. Do not name the expressions. \
        The length of the input should not exceed the upper limit sepcified in prompt. Do not give anything else (names, comments) than the input expression."
    )

class GenerateTests_std_inputs(dspy.Signature):
    prompt = dspy.InputField(format=str)
    tests = dspy.OutputField(
            desc="Generate a complete set of potential input and output pair as a list to test an AI-generated solution to the coding problem.  Assume input will be used via stdin and output is caputred stdout. Include (1) Edge cases, such as empty string or arrays, (3) Complex and diffucult inputs, but make sure do not include very long inputs. (3) Other ones that can maximize the chance of catching a bug. Do not give anything else than the input output pair, in the format of Input: input Output: output."
    )

class GenerateTests_func_inputs(dspy.Signature):
    prompt = dspy.InputField(format=str)
    tests_inputs = dspy.OutputField(
            desc='Generate a complete set of potential inputs to test an AI-generated solution to the coding problem. Cover: (i) Edge cases, such as empty string or arrays, (ii) Complex and diffucult inputs, but do not include very long inputs. (iii) Other ones that can maximize the chance of catching a bug. Provide the input in JSON format as follows: \
        {"input": <example_input>} Ensure the input match the types and structure expected for the problem. Do not predict output. \
        Do not include any additional text or explanations, just the JSON object.'
    )

class GenerateTests_LLM_reflect_with_tool_std(dspy.Signature):
    prompt = dspy.InputField(format=str)
    completionA = dspy.InputField(format=str)
    completionB = dspy.InputField(format=str)
    tests = dspy.OutputField(
            desc="Generate a complete set of potential input and output pair to test the two given AI-generated solutions to the coding problem.  Assume input will be used via stdin and output is caputred stdout. Include (1) Cases where one implementation may fail but the other succeeds. (2) Edge cases, such as empty string or arrays, (3) Complex and diffucult inputs, but make sure do not include very long inputs. (4) Other ones that can maximize the chance of catching a bug. Do not give anything else than the input output pair, in the format of Input: input Output: output."
    )

class GenerateTests_LLM_reflect_with_tool_func(dspy.Signature):
    prompt = dspy.InputField(format=str)
    completionA = dspy.InputField(format=str)
    completionB = dspy.InputField(format=str)
    tests = dspy.OutputField(
            desc='Generate a complete set of potential inputs to test the two given AI-generated solutions to the coding problem. Cover: (i) Cases where one implementation may fail but the other succeeds.(ii) Edge cases, such as empty string or arrays, (iii) Complex and diffucult inputs, but do not include very long inputs. (iv) Other ones that can maximize the chance of catching a bug. Provide the input and output in JSON format as follows: \
        {"input": <example_input>, "output": <expected_output>} Ensure the input and output match the types and structure expected for the problem. \
        Do not include any additional text or explanations, just the JSON object.'
    )

class GenerateTests_timeout_func(dspy.Signature):
    prompt = dspy.InputField(format=str)
    tests = dspy.OutputField(
        desc='Generate python code expression to be evaluated to a single long input to test whether an AI-generated solution will timeout to the coding problem. \
        The input expression should be evaluated to the same type as the input provided in the prompt. \
        Do not attempt to write out the expression, but simply give a short expression to be evluated to the long actual input. Please do not give the expression a name. The length of the input should not exceed the upper limit sepcified in prompt.Do not give anything else (names, comments) than the single input expression.'
    )

class GenerateTests_func_inputs(dspy.Signature):
    prompt = dspy.InputField(format=str)
    tests = dspy.OutputField(
        desc='Generate a complete set of potential inputs to test an AI-generated solution to the coding problem. Cover: (i) Edge cases, such as empty string or arrays, (ii) Complex and diffucult inputs, but do not include very long inputs. (iii) Other ones that can maximize the chance of catching a bug.  Provide the input and output in JSON format as follows: \
        {"input": <example_input>, "output": <expected_output>} Ensure the input and output match the types and structure expected for the problem. \
        Do not include any additional text or explanations, just the JSON object.'
    )
class GenerateTests_func(dspy.Signature):
    prompt = dspy.InputField(format=str)
    tests = dspy.OutputField(
        desc='Generate input and output pairs for testing a coding problem. Provide the input and output in JSON format as follows: \
        {"input": <example_input>, "output": <expected_output>} Ensure the input and output match the types and structure expected for the problem. \
        Do not include any additional text or explanations, just the JSON object.'
    )

class ExtractTests(dspy.Signature):
    """
    Extract tests/examples from the prompt, and convert them to executable asserts. DO NOT INVENT YOUR OWN TESTS!
    """

    prompt = dspy.InputField(format=str)
    tests = dspy.OutputField(
        desc="Executable tests using assert, you should directly extract the test from the prompt. \
                             Do not invent your own tests!"
    )


class GenerateLCBcodestdin(dspy.Signature):
    """ """

    prompt = dspy.InputField(format=str)
    code = dspy.OutputField(
        desc="Executable Python function generated from the given prompt.  \
                      the function should take stdin as input and print the output. Simply call the function after the definition. DO NOT give me anything else! "
    )

class GenerateLCBcodestdinConditional(dspy.Signature):
    """ """

    prompt = dspy.InputField(format=str)
    past_solution_list = dspy.InputField(format=str)
    code = dspy.OutputField(
        desc="Executable Python function generated from the given prompt.  \
                      the function should take stdin as input and print the output. Simply call the function after the definition. DO NOT give me anything else! Also, here is a list of past solutions you have written, give me something that is different from any of the solutions in the list."
    )

class GenerateLCBcodefunctional(dspy.Signature):
    prompt = dspy.InputField(format=str)
    code = dspy.OutputField(
        desc="Executable Python function generated from the given prompt.  \
                     DO NOT include anything other than function body! Give me only the function itself! "
    )
    
class GenerateLCBcodestdinICL(dspy.Signature):
    """ """

    prompt = dspy.InputField(format=str)
    code = dspy.OutputField(
        desc="Executable Python function generated from the given prompt.  \
                      the function should take stdin as input and print the output. Simply call the function after the definition. DO NOT give me anything else! \
                          Also, here are some example prompt-code pairs to use as reference."
    )
    
class GenerateLCBcodefunctionalICL(dspy.Signature):
    prompt = dspy.InputField(format=str)
    code = dspy.OutputField(
        desc="Executable Python function generated from the given prompt.  \
                     DO NOT include anything other than function body! Give me only the function itself \
                         Also, here are some example prompt-code pairs to use as reference."
    )

class SelfDebug(dspy.Signature):
    prompt = dspy.InputField(format=str)
    code = dspy.OutputField(
        desc="Here is the past history of your code and the test case feedback. Please reason why your code fail in the last round, and correct the code. Do not write non-code content in the code field.", max_length=2048
    )

class LLMSelect(dspy.Signature):
    prompt = dspy.InputField(format=str)
    samples_with_idx = dspy.InputField(format=str)
    best_sample_idx = dspy.OutputField(
        desc="Given the prompt, an AI have generated several candidates. Please compare these candidates carefully, and return the index of the best sample."
    )

class LLMSelectWithTestResult(dspy.Signature):
    prompt = dspy.InputField(format=str)
    test_and_samples_with_idx = dspy.InputField(format=str)
    best_idx = dspy.OutputField(
        desc="Given the prompt, an AI have generated several code candidates. These code samples are passed through an example test input, and result in several output results. There is only one output that is correct. Please think carefully according to the prompt, and return the index of the best output. The test could also be broken in the sense that all samples result in compile error like missing positional arguments. In this case, simply return -1 to me."
    )

class LLMSelectPairwise(dspy.Signature):
    prompt = dspy.InputField(format=str)
    samples_with_idx = dspy.InputField(format=str)
    best_sample_idx = dspy.OutputField(
        desc="Given the prompt, an AI have generated two candidates. Please compare these two candidates carefully, and return the index of the best sample (0 for the first sample, 1 for the second sample)."
    )

class LLMSelectPairwiseBT(dspy.Signature):
    prompt = dspy.InputField(format=str)
    samples_with_idx = dspy.InputField(format=str)
    best_sample_idx = dspy.OutputField(
        desc="Given the prompt, an AI have generated two candidates. Please compare these two candidates carefully, and return the index of the best sample (0 for the first sample, 1 for the second sample, 2 if they are equally good/bad)."
    )

class LLMSelectPairwiseTestAwareBT(dspy.Signature):
    prompt = dspy.InputField(format=str)
    samples_with_idx = dspy.InputField(format=str)
    best_sample_idx = dspy.OutputField(
            desc="Given the prompt, an AI have generated two candidates. You will be provided with the code of each candidate, and the execution result of the same given input. Utilize the provided samples and execution output to compare these two candidates carefully, and return the index of the best sample (0 for the first sample, 1 for the second sample, 2 if they are equally good/bad). Hint: These samples can be tricky! First analyze the problem yourself, then analyze the solutions."
    )

class LLMSelfReflect(dspy.Signature):
    prompt = dspy.InputField(format=str)
    completion = dspy.InputField(format=str)
    feedback = dspy.OutputField(
        desc="You are given this prompt for code generation and this code completion to the prompt. Please tell if the code completion is correct. Only return 'Correct' or 'Incorrect', DO NOT include anything else!"
    )

class LLMSelfReflect_with_tool(dspy.Signature):
    prompt = dspy.InputField(format=str)
    completion = dspy.InputField(format=str)
    execution_results = dspy.InputField(format=str)
    feedback = dspy.OutputField(
        desc="You are given this prompt for code generation and this code completion to the prompt, as well as the execution results of the code completion on test cases. Please tell if the code completion is correct. Only return 'Correct' or 'Incorrect', DO NOT include anything else!"
    )

def generate_tests(prompt):
    test_gen = dspy.ChainOfThought(GenerateTests)
    tests = test_gen(prompt=prompt)
    return tests.tests


def extract_tests(prompt):
    test_gen = dspy.ChainOfThought(ExtractTests)
    tests = test_gen(prompt=prompt)
    return tests.tests


def filter_extracted_tests(prompt, tests):
    return tests
    results = []
    for test in tests:
        assert_line = [
            l.strip()[7:] for l in test.split("\n") if l.strip().startswith("assert")
        ][0]
        eqns = assert_line.split("==")
        # actual = eqns[0].strip()
        # # actual is like do_algebra(['+'], [1, 2]), we want to extract ['+'], [1, 2]
        # actual_operands = actual[actual.find("(") + 1: -1]
        # print(actual)
        # print(actual_operands)
        # actual_operands = eval(actual_operands)
        # if not isinstance(actual_operands, tuple):
        #     actual_operands = (actual_operands,)
        try:
            expected = eqns[1][: eqns[1].find("Expected") - 3].strip()
            print(expected)

            # for actual_operand in actual_operands:
            #     if str(actual_operand) not in prompt:
            #         continue
            if str(expected) not in prompt:
                continue
        except Exception as e:
            continue
        results.append(test)
    return results

def generate_tests_repeat(prompt, is_stdin, r, temperature_base):
    if is_stdin:
        # test_gen = dspy.ChainOfThought(GenerateTests_std)
        test_gen = dspy.ChainOfThought(GenerateTests_std_inputs)
    else:
        # test_gen = dspy.ChainOfThought(GenerateTests_func)
        test_gen = dspy.ChainOfThought(GenerateTests_func_inputs)
    tests = []
    raw_tests = test_gen(prompt=prompt, config=dict(temperature=temperature_base + (0.01 * r)))

    if is_stdin:
        tests = post_process_tests_inputs(raw_tests.tests, is_stdin)
    else:
        tests = post_process_tests_inputs(raw_tests.tests, is_stdin)
    result = tests
    return result

def llm_generate_tests_reflect(prompt, completions, is_stdin, r, temperature_base, judge_lm):
    if is_stdin:
        test_gen = dspy.ChainOfThought(GenerateTests_LLM_reflect_with_tool_std)
    else:
        test_gen = dspy.ChainOfThought(GenerateTests_LLM_reflect_with_tool_func)
    tests = []
    completionA = completions[0]
    completionB = completions[1]
    
    with dspy.context(lm=judge_lm):
        # print(f"Using judge lm: {judge_lm}")
        raw_tests = test_gen(prompt=prompt, completionA=completionA, completionB=completionB, config=dict(temperature=temperature_base + (0.01 * r)))
    
    try:
        tests = post_process_tests_inputs(raw_tests.tests, is_stdin)
    except Exception as e:
        print(f"Error in post_process_tests_inputs: {e}")   
    result = tests
    return result

def generate_timeout_tests_repeat(prompt, is_stdin, r):
    if is_stdin:
        test_gen = dspy.ChainOfThought(GenerateTests_timeout_std)
    else:
        test_gen = dspy.ChainOfThought(GenerateTests_timeout_func)
    raw_tests = test_gen(prompt=prompt, config=dict(temperature=0.7 + (0.01 * r)))
    # Create a process-safe way to handle the test generation
    def test_generation_worker(raw_tests, r, result_queue, is_stdin):
        try:
            if is_stdin:
                result = post_process_timeout_tests_stdin(raw_tests.tests, r)
            else:
                result = post_process_timeout_tests_func(raw_tests.tests, r)
            result_list.append(result)
        except Exception as e:
            result_list.append(None)
            print(f"Error in worker process: {e}")
    manager = multiprocessing.Manager()
    result_list = manager.list()
    p = multiprocessing.Process(target=test_generation_worker, args=(raw_tests, r, result_list, is_stdin))
    p.start()
    
    p.join(timeout=2)
    
    if p.is_alive():
        p.kill()
        return None
    
    if not result_list:
        return None
    tests = result_list[0]
    return tests


def filter_test(tests, canonical_solution, prompt):
    return list(
        filter(
            lambda test: check_test([test], canonical_solution, 0, prompt, raw=True)[0],
            tests,
        )
    )


def map_test(tests, canonical_solution, prompt):
    return list(
        map(
            lambda test: check_test([test], canonical_solution, 0, prompt, raw=True)[0],
            tests,
        )
    )


def check_test(
    tests,
    pred,
    task_id,
    prompt,
    entry_point="dummy",
    raw=False,
    verbose=False,
    runtime_debug=False,
    is_extracted=False,
):  ## added runtime debug to see specific test case
    code = post_process_code(pred.code) if not raw else pred

    if len(tests) == 0:
        return True, "No tests found", "No tests found"
    for test in tests:
        result = check_correctness(
            {
                "prompt": prompt,
                "entry_point": entry_point,
                "test": [test],
                "task_id": task_id,
            },
            code,
            TIMEOUT_CONSTANT,
            eval_fun=unsafe_lcb_runTests,
            verbose=verbose,
            runtime_debug=runtime_debug,
            is_extracted=is_extracted,
        )
        if not result["passed"]:
            break
    # print(result["passed"], test, result["result"])
    return result["passed"], test, result["result"], result["maybe_error_messages"], result["maybe_output_values"]

def get_execution_feedback(
    tests,
    pred_code,
    task_id,
    prompt,
    entry_point="dummy",
    raw=False,
    verbose=False,
    runtime_debug=False,
    is_extracted=False,
):  ## added runtime debug to see specific test case
    # code = post_process_code(pred.code) if not raw else pred
    assert not raw
    code = post_process_code(pred_code)

    assert len(tests) == 1, "Only support a single test feedback now."
    test = tests[0]
    result = check_correctness(
        {
            "prompt": prompt,
            "entry_point": entry_point,
            "test": [test],
            "task_id": task_id,
        },
        code,
        TIMEOUT_CONSTANT, # @Dacheng: Change this
        eval_fun=unsafe_lcb_runTests,
        verbose=verbose,
        runtime_debug=runtime_debug,
        is_extracted=is_extracted,
    )
    return result

def reduce_preds(preds):
    preds = [pred.code for pred in preds]
    preds = list(set(preds))
    preds = [dspy.Prediction(code=pred) for pred in preds]
    return preds


class CodeGeneratorWithRanker(dspy.Module):
    total_num_tests = 0
    total_filtered_tests = 0

    def __init__(
        self,
        pre_computed_tests: dict,
        extracted_tests,
        temperature: float,
        n=1,
        collect_stats=False,
        visualize=False,
        log_file=None,
        cache_execution=None
    ):
        self.stdin_prog = dspy.ChainOfThought(GenerateLCBcodestdin)
        self.functional_prog = dspy.ChainOfThought(GenerateLCBcodefunctional)
        self.log_file = log_file
        self.visualize = visualize
        self.collect_stats = collect_stats
        self.pre_computed_tests = pre_computed_tests
        self.extracted_tests = extracted_tests
        self.num_generations = n
        import datetime

        now = datetime.datetime.now()
        self.init_time = now.strftime("%Y%m%d%H%M%S")
        self.temperature = temperature

        self.cache_execution = cache_execution

    def forward(
        self, prompt, canonical_solution, task_id, test, entry_point, is_stdin, **kargs
    ):
        # print(list(self.pre_computed_tests.values())[0], len(list(self.pre_computed_tests.values())[0]))
        if task_id in self.cache_execution:
             zipped_stats = self.cache_execution[task_id]
        #    print(len(self.cache_execution[task_id]))
        #assert False
        else:
            self.prog = self.stdin_prog if is_stdin else self.functional_prog
            tests = self.pre_computed_tests[task_id]

            if len(tests) == 0:
                dspy.logger.warning(f"No tests found for prompt: {prompt.splitlines()[0]}")
                return self.prog(prompt=prompt)

            preds = [
                self.prog(
                    prompt=prompt,
                    config=dict(temperature=self.temperature + (TEMPARATURE_STEP * i)),
                )
                for i in range(self.num_generations)
            ]

            preds = reduce_preds(preds)

            tests_as_strings = [json.dumps(test, sort_keys=True) for test in tests]
            tests_counter = Counter(tests_as_strings)
            tests = [
                {"test": json.loads(test_str), "count": count}
                for test_str, count in tests_counter.items()
            ]

            preds_pass = [
                list(
                    map(
                        lambda test: test["count"]
                        if check_test(
                            [test["test"]], pred, 0, prompt, "dummy", runtime_debug=True
                        )[0]
                        else 0,
                        tests,
                    )
                )
                for pred in preds
            ]
            zipped_stats = sorted(zip([p.code for p in preds], map(sum, preds_pass)), key=lambda x: x[1])
            sorted_arr =  {task_id: zipped_stats}
            if self.log_file:
                # with lock:
                self.log_file.write(json.dumps(sorted_arr) + "\n")

        # print(zipped_stats)
        preds_pass_rank = max(zipped_stats, key=lambda x:x[1])[ 
            0
        ]#.code

        pred = preds_pass_rank
        
        return dspy.Prediction(code=pred)

class CodeGeneratorWithIteratedRanker(dspy.Module):
    total_num_tests = 0
    total_filtered_tests = 0

    def __init__(
        self,
        pre_computed_tests: dict,
        extracted_tests: dict,
        temperature: float,
        n=1,
        collect_stats=False,
        visualize=False,
        log_file=None,
        cache_execution=None
    ):
        self.stdin_prog = dspy.ChainOfThought(GenerateLCBcodestdin)
        self.functional_prog = dspy.ChainOfThought(GenerateLCBcodefunctional)
        self.visualize = visualize
        self.log_file = log_file
        self.collect_stats = collect_stats
        self.pre_computed_tests = pre_computed_tests
        self.extracted_tests = extracted_tests
        self.num_generations = n
        import datetime

        now = datetime.datetime.now()
        self.init_time = now.strftime("%Y%m%d%H%M%S")
        self.temperature = temperature
        self.cache_execution = cache_execution

    def forward(
        self, prompt, canonical_solution, task_id, test, entry_point, is_stdin, **kargs
    ):
        if task_id in self.cache_execution:
            zipped_stats = self.cache_execution[task_id]
        else:
            self.prog = self.stdin_prog if is_stdin else self.functional_prog

            tests = self.pre_computed_tests[task_id]
            extracted_tests = self.extracted_tests[task_id]

            if len(tests) == 0:
                dspy.logger.warning(f"No tests found for prompt: {prompt.splitlines()[0]}")
                return self.prog(prompt=prompt)

            preds = [
                self.prog(
                    prompt=prompt,
                    config=dict(temperature=self.temperature + (TEMPARATURE_STEP * i)),
                )
                for i in range(self.num_generations)
            ]
            preds = reduce_preds(preds)

            # tests = Counter(tests)
            tests_as_strings = [json.dumps(test, sort_keys=True) for test in tests]
            tests_counter = Counter(tests_as_strings)
            tests = [
                {"test": json.loads(test_str), "count": count}
                for test_str, count in tests_counter.items()
            ]

            preds_pass = [
                list(
                    map(
                        lambda test: test["count"]
                        if check_test(
                            [test["test"]], pred, 0, prompt, "dummy", runtime_debug=True
                        )[0]
                        else 0,
                        tests,
                    )
                )
                for pred in preds
            ]

            extracted_tests_pass = [
                list(
                    map(
                        lambda test: 1
                        if check_test([test], pred, 0, prompt, "dummy", is_extracted=True)[
                            0
                        ]
                        else 0,
                        extracted_tests,
                    )
                )
                for pred in preds
            ]

            max_extracted_tests_pass = max(
                zip(preds, map(sum, extracted_tests_pass)), key=lambda x: x[1]
            )[1]
            # Good first candidates
            good_pred_indices = [
                i
                for i, x in enumerate(map(sum, extracted_tests_pass))
                if x == max_extracted_tests_pass
            ]
            # Bad first candidates
            bad_pred_indices = [
                i
                for i, x in enumerate(map(sum, extracted_tests_pass))
                if x != max_extracted_tests_pass
            ]

        preds_pass_rank = max(zipped_stats, key=lambda x:x[1])[
            0
        ]# .code

        # preds_pass_rank = #max(zip(preds, map(sum, new_preds_pass)), key=lambda x: x[1])[

        pred = preds_pass_rank

        """
        pred = (
            visualize_matrix(
                preds,
                new_preds_pass,
                canonical_solution,
                prompt,
                test,
                entry_point,
                new_tests,
                task_id,
                extracted_tests,
                prefix=self.init_time,
            )
            if self.visualize
            else preds_pass_rank
        )

        visualize_lcb(
            new_preds_pass,
            extracted_tests_pass,
            task_id,
            output_dir="./functional_naive_easy_iterated",
        )
        """
        return dspy.Prediction(code=pred)


class CodeGeneratorWithSelfDebug(dspy.Module):
    total_num_tests = 0
    total_filtered_tests = 0

    def __init__(
        self,
        extracted_tests: dict,
        temperature: float,
        num_round: int,
        n=1,
        lm=None,
        selection=None,
        pre_computed_tests=None,
        context=None,
        judge_lm=None,
        debug_lm=None,
        selfdebug_decision=None,
        ablation_judge_lm=None,
        num_icl_examples=0,
        args=None,
        enable_llm_reflection_with_tool = False,
        enable_vanilla_reflection = False,
        ablation_qwq_vanilla_without_reasoning = False,
        ablation_qwq_debug_with_4o_mini = False,
        cached_preds_dict = None
    ):  
        self.enable_llm_reflection_with_tool = enable_llm_reflection_with_tool
        self.enable_vanilla_reflection = enable_vanilla_reflection
        self.icl_examples = ''
        if num_icl_examples == 0:
            self.stdin_prog = dspy.ChainOfThought(GenerateLCBcodestdin)
            self.functional_prog = dspy.ChainOfThought(GenerateLCBcodefunctional)
        else:
            self.stdin_prog = dspy.ChainOfThought(GenerateLCBcodestdinICL)
            self.functional_prog = dspy.ChainOfThought(GenerateLCBcodefunctionalICL)
            for i in range(num_icl_examples):
                self.icl_examples += f"Prompt: {ICL_EXAMPLES[i]['prompt']}\nCode: {ICL_EXAMPLES[i]['codes'][0]}\n\n"
            # print(f"ICL Examples:\n{self.icl_examples}")
            
        self.stdin_prog_prompt = "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition ."
        self.functional_prog_prompt = "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."

        self.self_debug_prog = dspy.ChainOfThought(SelfDebug)
        # self.self_debug_timeout_prior_prog = dspy.ChainOfThought(SelfDebugTimeoutPrior)
        self.lm = lm
        self.judge_lm = judge_lm
        self.ablation_judge_lm = ablation_judge_lm
        self.debug_lm = debug_lm
        self.selfdebug_decision = selfdebug_decision
        self.reflect_prog = dspy.ChainOfThought(LLMSelfReflect)
        self.llm_with_tool_reflect_prog = dspy.ChainOfThought(LLMSelfReflect_with_tool)

        # Context management
        self.context = context
        
        # Selection Policy
        self.pre_computed_tests = pre_computed_tests
        self.llm_select_prog = dspy.ChainOfThought(LLMSelect)
        self.llm_select_pairwise_prog = dspy.ChainOfThought(LLMSelectPairwise)
        self.llm_select_pairwise_bt_prog = dspy.ChainOfThought(LLMSelectPairwiseBT)
        self.llm_select_pairwise_test_aware_bt_prog = dspy.ChainOfThought(LLMSelectPairwiseTestAwareBT)
        self.llm_select_with_test_result_prog = dspy.ChainOfThought(LLMSelectWithTestResult)

        self.extracted_tests = extracted_tests
        self.num_generations = n
        self.selection = selection

        self.num_round = num_round
        import datetime
        now = datetime.datetime.now()
        self.init_time = now.strftime("%Y%m%d%H%M%S")
        self.temperature = temperature

        self.args = args
        self.ablation_qwq_vanilla_without_reasoning = ablation_qwq_vanilla_without_reasoning
        self.ablation_qwq_debug_with_4o_mini = ablation_qwq_debug_with_4o_mini
        self.cached_preds_dict = cached_preds_dict

    # An updated verion of forward, getting rid of previous conditional etc logic
    # Only supported some features. Also assume growing tree.
    def forward(
        self, example, canonical_solution, task_id, test, entry_point, is_stdin, **kargs
    ):
        prompt = example["prompt"]

        ## TODO: (Alex) Here make sure this is the right place to read cache
        if self.args.load_cached_preds:
            ## load the cached prediction completions and get the public accuracy to replicate zipped_history
            codes = self.cached_preds_dict[task_id]['codes']
            zipped_history = [[] for _ in range(self.num_generations)]  ## This is duplicated from below, but I don't know whether the icl examples are being affected so temporarily keep this here, can move the below line earlier and delete this line
            extracted_tests = self.extracted_tests[task_id] ## this line too
            for n in range(self.num_generations):
                for r in range(self.num_round):
                    pred = dspy.Prediction(code=codes[r][n])
                    feedback_string, public_test_acc, public_test_details, public_test_timeout_details, public_test_time_elapsed = self.get_public_tests_feedback(prompt, pred, extracted_tests)
                    zipped_history[n].append((pred.code, public_test_acc, public_test_details, public_test_timeout_details, public_test_time_elapsed))
            print("HITTED CACHE")
            return self.selection_function(zipped_history, task_id, prompt, is_stdin, example), None
        if self.icl_examples != '':
            prompt = f"{self.icl_examples}Prompt: {prompt}\nCode: "
        
        zipped_history = [[] for _ in range(self.num_generations)]

        self.prog = self.stdin_prog if is_stdin else self.functional_prog
        extracted_tests = self.extracted_tests[task_id]

        for n in range(self.num_generations):
            try:
                prompt_with_trace = copy.deepcopy(prompt)

                # Iterate on generated tests, based on https://arxiv.org/pdf/2401.08500 
                is_in_refinement_stage = False
                generated_test_anchors = []
                if self.selfdebug_decision != "exit":
                    generated_tests_as_strings = [json.dumps(test, sort_keys=True) for test in self.pre_computed_tests]
                    generated_tests_counter = Counter(generated_tests_as_strings)
                    generated_tests = [
                        {"test": json.loads(generated_test_str), "count": count}
                        for generated_test_str, count in generated_tests_counter.items()
                    ]
                    remaining_generated_tests = generated_tests
                    effective_round = 0
                    prompt_with_trace_anchor = None
                    pred_anchor = None

                for r in range(self.num_round):
                    if r == 0: 
                        cur_temp = self.temperature + TEMPARATURE_STEP * (n * self.num_round + r)
                        if "o1" in self.lm.model:
                            cur_temp = 1.0
                        if not self.args.no_dspy_gen:
                            cur_prog = self.prog
                            pred = cur_prog(
                                    prompt=prompt_with_trace,
                                    config=dict(temperature=cur_temp),
                            )
                        else:
                            question_prompt = self.stdin_prog_prompt if is_stdin else self.functional_prog_prompt
                            prompt = question_prompt + prompt
                            if self.args.api_name is not None:
                                client = OpenAI(
                                    api_key="EMPTY",
                                    base_url=self.args.api_base,
                                )
                                
                                assert "qwen" in self.args.api_name.lower(), "Only Qwen system prompt is supported. Please modify the below system prompt for other models."
                                try:
                                    chat_response = client.chat.completions.create(
                                        model=self.args.api_name,
                                        messages=[
                                            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                                            {"role": "user", "content": prompt},
                                        ]
                                    )
                                    response = chat_response.choices[0].message.content
                                    # pred = prog(prompt=prompt) 
                                except Exception as e:
                                    print(e)
                                    response = ""
                            else:
                                client = OpenAI(
                                    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
                                )
                                # print(f"Using {self.args.generator} to generate code")
                                # TODO: DL, please clean up these massy naming
                                generator = self.args.generator
                                if generator == "4o-mini":
                                    generator = "gpt-4o-mini"
                                try:
                                    chat_response = client.chat.completions.create(
                                        model=generator,
                                        messages=[
                                            {"role": "user", "content": prompt},
                                        ]
                                    )
                                    response = chat_response.choices[0].message.content
                                    # pred = prog(prompt=prompt) 
                                except Exception as e:
                                    print(e)
                                    response = ""

                            pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"

                            # Use re.DOTALL to match multiline content inside backticks
                            matches = re.findall(pattern, response, re.DOTALL)

                            if matches:
                                response = matches[-1]
                                # print(response)
                            else:
                                response = ""
                            # assert False	
                            pred = dspy.Prediction(code=response, reasoning="") 
                            # print(pred)

                        # zipped_history[0].append((pred.code, None, None, None, None))
                    else:
                        cur_temp = self.temperature + TEMPARATURE_STEP * (n * self.num_round + r)
                        if "o1" in self.lm.model:
                            cur_temp = 1.0
                        if self.ablation_qwq_debug_with_4o_mini:
                            with dspy.context(lm=self.debug_lm):
                                print("debug lm: ", self.debug_lm.model)
                                pred = self.self_debug_prog(
                                        prompt=prompt_with_trace,
                                        config=dict(temperature=cur_temp),
                                        )
                        else:
                            pred = self.self_debug_prog(
                                        prompt=prompt_with_trace,
                                        config=dict(temperature=cur_temp),
                                        )

                    # Gather test feedback
                    feedback_string, public_test_acc, public_test_details, public_test_timeout_details, public_test_time_elapsed = self.get_public_tests_feedback(prompt, pred, extracted_tests)
                    # print(public_test_acc, len(extracted_tests))
                    if not is_in_refinement_stage:
                        # Update context
                        if self.context == "all": 
                            if not self.ablation_qwq_vanilla_without_reasoning:
                                prompt_with_trace += f"\n [Round {r} Reasoning]: {pred.reasoning} \n"
                            prompt_with_trace += f"\n [Round {r} Generated code]: {pred.code} \n"
                            prompt_with_trace += (f"[Round {r} Test Feedback]: {feedback_string}")
                        elif self.context == "last":
                            prompt_with_trace = prompt + f"\n [Round 0 Generated code]: {pred.code} \n"
                            prompt_with_trace += (f"[Round 0 Test Feedback]: {feedback_string}")
                        else: 
                            assert False
                        
                        zipped_history[n].append((pred.code, public_test_acc, public_test_details, public_test_timeout_details, public_test_time_elapsed))
                        if public_test_acc == 1.0: #all_passed:
                            if self.selfdebug_decision == "exit":
                                # print(f"Task {task_id} (sample {n}) Finish in {r} round of self-debug")
                                break
                            else:
                                print(f"Task {task_id} (sample {n}) passed all public tests in {r} round of self-debug")
                                assert self.selfdebug_decision == "refine"
                                is_in_refinement_stage = True
                                effective_round = r
                                
                                # anchor is a checkpoint for rollback, prompt_with_trace contains a (maybe wrong) feeback from the next generated tests 
                                generated_test_anchors, remaining_generated_tests, prompt_with_trace_anchor, prompt_with_trace = self.maybe_iterate_single_generated_test(prompt, pred, generated_test_anchors, remaining_generated_tests, prompt_with_trace)
                                # print(f"Task {task_id} (sample {n}) starts to iterate on generated tests in round {r}, initially passed: {len(generated_test_anchors)}, not passed: {len(remaining_generated_tests)}.")
                                if len(remaining_generated_tests) == 0:
                                    # print(f"Task {task_id} (sample {n}) finishes iteration on generated tests in round {r}")
                                    break
                    else: # Already debug in generated tests
                        assert self.selfdebug_decision != "exit"
                        # Check whether the newly updated code passed all the anchor
                        anchor_break, anchor_feedback_string = self.get_anchor_break_and_feedback(prompt, pred, extracted_tests, public_test_acc, feedback_string, generated_test_anchors)
                        # if anchor break, roll back to the previous code

                        cur_iterate_generated_test = remaining_generated_tests.pop(0)
                        # If fix the last test, add it into anchor
                        result = check_test([cur_iterate_generated_test["test"]], post_process_code(pred.code), 0, prompt, "dummy", runtime_debug=True, raw=True, is_extracted=False)
                        # print(anchor_break, result[0])
                        if anchor_break or not result[0]:
                            # print(f"Task {task_id} (sample {n}) anchor breaks or fail to fix the last code in round {r}, roll back to the previous code.")
                            prompt_with_trace = prompt_with_trace_anchor
                            # pop the last generated test
                            # restore the last prediction
                            pred = dspy.Prediction(code=zipped_history[n][-1][0])
                        else:
                            # print(f"Task {task_id} (sample {n}) fixes a generatd test!")
                            # Otherwise, adopt the change
                            effective_round += 1
                            generated_test_anchors.append(cur_iterate_generated_test)
                            
                            if self.context == "all": 
                                prompt_with_trace += f"\n [Round {effective_round} Reasoning]: {pred.reasoning} \n"
                                prompt_with_trace += f"\n [Round {effective_round} Generated code]: {pred.code} \n"
                                prompt_with_trace += (f"[Round {effective_round} Test Feedback]: {anchor_feedback_string}")
                            elif self.context == "last":
                                prompt_with_trace = prompt + f"\n [Round 0 Generated code]: {pred.code} \n"
                                prompt_with_trace += (f"[Round 0 Test Feedback]: {anchor_feedback_string}")
                                                    
                            zipped_history[n].append((pred.code, public_test_acc, public_test_details, public_test_timeout_details, public_test_time_elapsed))

                            # anchor is a checkpoint for rollback, prompt_with_trace contains a (maybe wrong) feeback from the next generated tests 
                        # print(f"history length: {len(zipped_history[n])}")
                        generated_test_anchors, remaining_generated_tests, prompt_with_trace_anchor, prompt_with_trace = self.maybe_iterate_single_generated_test(prompt, pred, generated_test_anchors, remaining_generated_tests, prompt_with_trace)
                        # print(f"Task {task_id} (sample {n}) iterating on generated tests in round {r}, passed: {len(generated_test_anchors)}, not passed: {len(remaining_generated_tests)}.")
                        if len(remaining_generated_tests) == 0:
                            # print(f"Task {task_id} (sample {n}) finishes iteration on generated tests in round {r}")
                            break
                    if r == self.num_round - 1:
                        # print(f"Task {task_id} (sample {n}) finishes with public acc: {public_test_acc}")
                        pass
            except Exception as e:
                    print(f"Exception in selfdebug: {e}, n: {n}, task_id: {task_id}")
                    zipped_history[n].append(("", 0, "", "", 0)) ## if any exception occur (like context window limit exceeded, fallback to simply empty completion)  
        # print(zipped_history[-1][-1])
        # print(f"=" * 10 + "Finished generating selfdebug prediction" + "=" * 10)
        return self.selection_function(zipped_history, task_id, prompt, is_stdin, example), None ## (Alex) example is newly added, could get rid of the redundancy for prompt and is_stdin

    def get_anchor_break_and_feedback(self, prompt, pred, extracted_tests, public_test_acc, public_test_feedback_string, generated_test_anchors):
        anchor_break = False
        anchor_feedback = public_test_feedback_string
        if len(extracted_tests) != 0 and public_test_acc != 1.0:
            anchor_break = True
        for generated_test_id, generated_test in enumerate(generated_test_anchors):
            result = check_test([generated_test["test"]], post_process_code(pred.code), 0, prompt, "dummy", runtime_debug=True, raw=True, is_extracted=False)
            cur_test_id = len(extracted_tests) + generated_test_id
            # Add additioinal test achnor information in
            passed = result[0]
            cur_test_id = len(self.extracted_tests) + generated_test_id
            # Add additioinal test achnor information in
            # print(f"*********** in anchor {cur_test_id} ")
            anchor_feedback += (f"[Test {cur_test_id} - ]" + result[3][0] + "\n")
            if not passed:
                anchor_break = True
        return anchor_break, anchor_feedback

    # (1) Update anchor, (2) Save prompt anchor, (3) Generate the feedback of the next test.
    def maybe_iterate_single_generated_test(self, prompt, pred, generated_test_anchors, remaining_generated_tests, prompt_with_trace):
        # always make sure to store anchor first for potential future rollback the entire call of this function
        new_remaining_generated_tests = []
        for generated_test_id, generated_test in enumerate(remaining_generated_tests):
            # print(generated_test)
            result = check_test([generated_test["test"]], post_process_code(pred.code), 0, prompt, "dummy", runtime_debug=True, raw=True, is_extracted=False)
            passed = result[0]
            error_message = result[3]
            # print(passed)
            # print(error_message)

            # Filtered out generated tests that are obviously wrong (those with positional error)
            if "required positional argument" in error_message:
                print(f"Filtered out low-quality generated test with error {error_message}")
                continue
            if passed:
                cur_test_id = len(self.extracted_tests) + len(generated_test_anchors)
                # Add additioinal test achnor information in
                prompt_with_trace += (f"[Test {cur_test_id} - ]" + error_message[0] + "\n")
                generated_test_anchors.append(generated_test)
                # print(f"ahiuhduiw********* {len(generated_test_anchors)}")
            else:
                new_remaining_generated_tests.append(generated_test)
        prompt_with_trace_anchor = copy.deepcopy(prompt_with_trace)
        # if there are remaining generated tests that we can iterate on
        if len(new_remaining_generated_tests) > 0:
            cur_iterate_generated_test = new_remaining_generated_tests[0]# .pop(0)
            result = check_test([cur_iterate_generated_test["test"]], pred.code, 0, prompt, "dummy", is_extracted=False, runtime_debug=True, raw=True)
            cur_test_id = len(self.extracted_tests) + len(generated_test_anchors)
            # print(f"*********** in iterate {cur_test_id} ")
            prompt_with_trace += (f"[Test {cur_test_id} - ]" + result[3][0] + "\n")
        return generated_test_anchors, new_remaining_generated_tests, prompt_with_trace_anchor, prompt_with_trace

    def get_public_tests_feedback(self, prompt, pred, extracted_tests):
        # Gather public tests feedback
        feedback_string = ""
        public_test_passed = []
        public_test_details = []
        public_test_timeout_details = []
        public_test_time_elapsed = []
        for extracted_test_id, extracted_test in enumerate(extracted_tests):
            result = get_execution_feedback([extracted_test], pred.code, 0, prompt, "dummy", is_extracted=True, runtime_debug=True)
            public_test_passed.append(int(result["passed"]))
            assert len(result["time_elapsed"]) <= 1, "More than one result."
            public_test_time_elapsed.append(sum(result["time_elapsed"]))
            feedback_string += (f"[Test {extracted_test_id} - ]" + result["maybe_error_messages"][0] + "\n")
            assert len(result["details"]) <= 1, "More than one test result."
            assert len(result["timeout_details"]) <= 1, "More than one test result."
            public_test_details.append(int(result["details"][0]))
            public_test_timeout_details.append(int(result["timeout_details"][0]))

        public_test_time_elapsed = sum(public_test_time_elapsed)
        if len(public_test_passed) == 0:
            public_test_acc = 0
        else:
            public_test_acc = sum(public_test_passed) / len(public_test_passed)
        
        return feedback_string, public_test_acc, public_test_details, public_test_timeout_details, public_test_time_elapsed

    def selection_function(self, zipped_history, task_id, prompt, is_stdin, example):
        if self.selection == "first":
            # select from the best of n
            best_acc = -1
            best_sample = None
            for n in range(self.num_generations):
                sample, acc, _, _, _ = zipped_history[n][-1]
                if acc > best_acc:
                    best_acc = acc
                    best_sample = sample
            # self.lm.inspect_history()
        elif self.selection == "fast":
            # select from the best of n
            best_acc = -1
            best_sample = None
            correct_fastest_time = float("inf")
            correct_fastest_sample = None
            for n in range(self.num_generations):
                sample, acc, _, _, time = zipped_history[n][-1]
                if acc > best_acc:
                    best_acc = acc
                    best_sample = sample
                if acc == 1.0:
                    if time < correct_fastest_time:
                        correct_fastest_time = time
                        correct_fastest_sample = sample
        elif self.selection == "oracle":
            if self.enable_llm_reflection_with_tool:
                return self.llm_reflection_with_tool(zipped_history, example)
            all_samples = []
            reflections = []
            for n in range(self.num_generations):
                sample, acc, _, _, _ = zipped_history[n][-1]
                sample = dspy.Prediction(code=sample)
                all_samples.append(sample)
                # print(f"=" * 10 + "Generating oracle reflection" + "=" * 10)
                if self.enable_vanilla_reflection:
                    with dspy.context(lm=self.ablation_judge_lm):
                        # print(f"=" * 10 + "Running reflect_prog" + "=" * 10)
                        reflection = self.reflect_prog(
                            prompt=prompt,
                            completion = sample,
                            config=dict(temperature=self.temperature + TEMPARATURE_STEP*n),
                        )
                    reflections.append(reflection.feedback)
            # print("=" * 10 + "Finished generating oracle prediction" + "=" * 10)
            return dspy.Prediction(codes=all_samples, prompts=[prompt] * self.num_generations), reflections
#             return dspy.Prediction(codes=all_samples, prompts=[prompt] * self.num_generations)
        elif self.selection == "oracle_all_rounds":
            ret = []
            for r in range(self.num_round):
                all_samples = [] # [[] for _ in range(self.num_round)]
                for n in range(self.num_generations):
                    sample, acc, _, _, _ = zipped_history[n][min(r, len(zipped_history[n]) - 1)]
                    sample = dspy.Prediction(code=sample)
                    all_samples.append(sample)
                ret.append(dspy.Prediction(codes=all_samples, prompts=[prompt] * self.num_generations))
                
            return ret, None
        elif self.selection == "timeout_tests":
            best_acc = -1
            best_samples = []
            for n in range(self.num_generations):
                sample, acc, _, _, _ = zipped_history[n][-1]
                if acc > best_acc:
                    best_acc = acc
                    best_samples = [sample]
                elif acc == best_acc:
                    best_samples.append(sample)
            ## Use timeout tests to do tie breaking. TODO: should we save time by not tie breaking if the acc on public tests is not 1.0?
            if len(best_samples) > 1:
                timeout_input_list = generate_tests_for_one_example(example, generation_fun=generate_timeout_tests_repeat,num_timeout_tests=3)
                best_rate = -1
                for sample in best_samples:
                    result = unsafe_lcb_run_timeout_tests(timeout_input_list, sample, TIMEOUT_CONSTANT, is_stdin)
                    if len(result) == 0:
                        pass_rate = 0
                    else:
                        pass_rate = sum([int(result[i][0]) for i in range(len(result))]) / len(result)
                    if pass_rate > best_rate:
                        best_rate = pass_rate
                        best_sample = sample
            else:
                assert len(best_samples) == 1
                best_sample = best_samples[0]
        elif self.selection == "generated_tests_majority_no_public_tests":
            feedbacks = []
            samples = []
            generated_tests = self.pre_computed_tests
            if len(generated_tests) > 0:
                generated_tests_as_strings = [json.dumps(test, sort_keys=True) for test in generated_tests]
                generated_tests_counter = Counter(generated_tests_as_strings)
                generated_tests = [
                        {"test": json.loads(generated_test_str), "count": count}
                        for generated_test_str, count in generated_tests_counter.items()
                ]
                for n in range(self.num_generations):
                    sample, acc, _, _, _ = zipped_history[n][-1]
                    samples.append(sample)
                    feedbacks.append([])
                    for generated_test in generated_tests:
                        feedbacks[n].append(check_test([generated_test["test"]], post_process_code(sample), 0, prompt, "dummy", runtime_debug=True, raw=True, is_extracted=False)[3])
                # print(feedbacks)
                best_sample_idx= self.find_maximum_agreement(feedbacks)
                best_sample = samples[best_sample_idx]
            else:
                # Select the first sample if there is no pre-computed tests (rare case)
                print(f"Warning: No pre-computed tests, selecting the first sample.")
                best_sample = zipped_history[0][-1][0]
        elif self.selection == "random":
            # select randomly
            random_idx = random.randint(0, self.num_generations - 1)
            best_sample, _, _, _, _ = zipped_history[random_idx][-1]
        else: # elif "private_tests" in self.selection_policy: # self.selection_policy == "private_tests" or self.selection_policy == "private_tests_debug_all":
            # select from the best of n
            # print("Selecting USING ELSE")
            best_acc = -1
            best_sample = None
            public_correct_samples = []
            for n in range(self.num_generations):
                sample, acc, _, _, time = zipped_history[n][-1]
                if acc > best_acc:
                    best_acc = acc
                    best_sample = sample
                if acc == 1.0:
                    public_correct_samples.append(sample) 
            
            if len(public_correct_samples) > 0:
                if "generated_tests" in self.selection:
                    # Force on passing timeout tests
                    generated_tests = self.pre_computed_tests# [task_id]
                    print(f"{task_id} has {len(public_correct_samples)} / {self.num_generations} samples that pass all the public tests. Number of generated tests: {len(generated_tests)}")
                    # print(generated_tests)
                    if len(generated_tests) > 0:
                        generated_tests_as_strings = [json.dumps(test, sort_keys=True) for test in generated_tests]
                        generated_tests_counter = Counter(generated_tests_as_strings)
                        generated_tests = [
                             {"test": json.loads(generated_test_str), "count": count}
                             for generated_test_str, count in generated_tests_counter.items()
                        ]
                        # print(private_tests)
                        if self.selection == "generated_tests":
                            with dspy.context(lm=self.debug_lm): ## TODO (Alex): here I simply used debug_lm because debug_lm is fixed to be 4omini, we can make this an argument
                                timeout_input_list = generate_tests_for_one_example(example, generation_fun=generate_timeout_tests_repeat,num_timeout_tests=3)
                            best_rate = -1
                            public_correct_samples_pass_rate = []
                            for sample in public_correct_samples:
                                result = unsafe_lcb_run_timeout_tests(timeout_input_list, sample, TIMEOUT_CONSTANT, is_stdin)
                                if len(result) == 0:
                                    pass_rate = 0
                                else:
                                    pass_rate = sum([int(result[i][0]) for i in range(len(result))]) / len(result)
                                public_correct_samples_pass_rate.append(pass_rate)
                            print(f"Timeout pass rate: {public_correct_samples_pass_rate}")
                            max_pass_rate = max(public_correct_samples_pass_rate)

                            new_public_correct_samples = []
                            for sample_id, sample in enumerate(public_correct_samples):
                                if public_correct_samples_pass_rate[sample_id] == max_pass_rate:
                                    new_public_correct_samples.append(sample)

                            print(f"Before forcing timeout check, {task_id} has {len(public_correct_samples)} samples.")
                            public_correct_samples = new_public_correct_samples
                            print(f"After forcing timeout check, {task_id} has {len(public_correct_samples)} samples.")
                            # print(generated_tests)
                            preds_pass = [
                                list(
                                    map(
                                        lambda test: 1# test["count"] # @DL: This is weird, should just reduce the same tests.
                                        if check_test(
                                            [test["test"]], post_process_code(public_correct_sample), 0, prompt, "dummy", runtime_debug=True, raw=True, is_extracted=False
                                        )[0]
                                        else 0,
                                        generated_tests,
                                    )
                                )
                                for public_correct_sample in public_correct_samples
                            ]
                            zipped_stats = sorted(zip(public_correct_samples, map(sum, preds_pass)), key=lambda x: x[1])
                            preds_pass_rank = max(zipped_stats, key=lambda x:x[1])[0]
                            print([sum(p) for p in preds_pass])
                            best_sample = preds_pass_rank
                        elif self.selection == "generated_tests_majority":
                            feedbacks = [[] for _ in range(len(public_correct_samples))]
                            for public_correct_sample_idx, public_correct_sample in enumerate(public_correct_samples): 
                                for generated_test in generated_tests:
                            #        print(private_test)
                            #        print(private_test["test"], len(private_test["test"]))
                                    # if public_correct_sample_idx == 0: print(private_test, len(private_tests))
                                    feedbacks[public_correct_sample_idx].append(check_test([generated_test["test"]], post_process_code(public_correct_sample), 0, prompt, "dummy", runtime_debug=True, raw=True, is_extracted=False)[3])
                            # print(feedbacks)
                            best_sample_idx= self.find_maximum_agreement(feedbacks)
                            best_sample = public_correct_samples[best_sample_idx]
                        elif self.selection == "generated_tests_aware_llm_judge":
                            feedbacks = [[] for _ in range(len(public_correct_samples))]
                            for public_correct_sample_idx, public_correct_sample in enumerate(public_correct_samples): 
                                for generated_test in generated_tests:
                                    feedbacks[public_correct_sample_idx].append(check_test([generated_test["test"]], post_process_code(public_correct_sample), 0, prompt, "dummy", runtime_debug=True, raw=True, is_extracted=False)[4]) # 4 is the output values

                            assert len(generated_tests) == len(feedbacks[0])
                            concensus_groups_record = self.find_concensus_groups(feedbacks)
                            selected_sample_idx = self.find_maximum_score_test_aware_bt(prompt, generated_tests, public_correct_samples, concensus_groups_record)
                            best_sample = public_correct_samples[selected_sample_idx]
                            # print(best_sample)
                        elif self.selection == "generated_tests_majority_llm":
                            feedbacks = [[] for _ in range(len(public_correct_samples))]
                            for public_correct_sample_idx, public_correct_sample in enumerate(public_correct_samples): 
                                for generated_test in generated_tests:
                                    feedbacks[public_correct_sample_idx].append(check_test([generated_test["test"]], post_process_code(public_correct_sample), 0, prompt, "dummy", runtime_debug=True, raw=True, is_extracted=False)[4]) # 4 is the output values
                            broken, best_sample_idx, concensus_groups_record = self.find_maximum_score_llm(prompt, generated_tests, feedbacks)
                            best_sample = public_correct_samples[best_sample_idx]
                        elif self.selection == "generated_tests_tool_assisted":
                            print("Using generated_tests_tool_assisted")
                            feedbacks = [[] for _ in range(len(public_correct_samples))]
                            for public_correct_sample_idx, public_correct_sample in enumerate(public_correct_samples): 
                                for generated_test in generated_tests:
                                    feedbacks[public_correct_sample_idx].append(check_test([generated_test["test"]], post_process_code(public_correct_sample), 0, prompt, "dummy", runtime_debug=True, raw=True, is_extracted=False)[4]) # 4 is the output values
                            assert len(generated_tests) == len(feedbacks[0])
                            concensus_groups_record = self.find_concensus_groups(feedbacks)
                            
                           
                            processed_groups = []
                            for group in concensus_groups_record:
                                if group is not None:
                                    new_group = {}
                                    for key, value in group.items():
                                        if not key.startswith("Error:"):
                                            new_group[key] = value
                                    if new_group:  # Only append if there are non-error entries
                                        processed_groups.append(new_group)
                                else:
                                    processed_groups.append(None)
                                    
                            concensus_groups_record = processed_groups
                        
                            selected_sample_idx = self.find_maximum_score_tool_assisted_bt(prompt, public_correct_samples, concensus_groups_record, example)
                            best_sample = public_correct_samples[selected_sample_idx]
                            # print(best_sample)

                elif self.selection == "llm_judge_pairwise":
                    selected_sample_idx = self.find_maximum_score(prompt, public_correct_samples)
                    best_sample = public_correct_samples[selected_sample_idx]
                elif self.selection == "llm_judge_pairwise_bt":
                    selected_sample_idx = self.find_maximum_score_bt(prompt, public_correct_samples)
                    best_sample = public_correct_samples[selected_sample_idx]
                

        return dspy.Prediction(code=best_sample), None
    
    def find_maximum_score(self, prompt, responses):
        score = np.zeros(len(responses))
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                samples_with_index = ""
                samples_with_index += f"\n[Sample 0]: {responses[i]} \n"
                samples_with_index += f"\n[Sample 1]: {responses[j]} \n"
                with dspy.context(lm=self.judge_lm):
                    pred = self.llm_select_pairwise_prog(prompt=prompt, samples_with_idx=samples_with_index)
                try:
                    selected_sample_idx = int(pred.best_sample_idx)
                    # print(f"LLM return idx: {selected_sample_idx}")
                    assert selected_sample_idx in [0,1]
                    if selected_sample_idx == 0:
                        score[i] += 1
                    else:
                        score[j] += 1
                except Exception as e:
                    print(e)
        max_score_index = np.argmax(score)
        print(score, max_score_index)
        return max_score_index

    def find_maximum_score_bt(self, prompt, responses):
        score = np.zeros(len(responses))
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                choice = []
                for k in range(2):
                    samples_with_index = ""
                    if k == 0:
                        samples_with_index += f"\n[Sample 0]: {responses[i]} \n"
                        samples_with_index += f"\n[Sample 1]: {responses[j]} \n"
                    else:
                        samples_with_index += f"\n[Sample 0]: {responses[j]} \n"
                        samples_with_index += f"\n[Sample 1]: {responses[i]} \n"
                    with dspy.context(lm=self.judge_lm):
                        pred = self.llm_select_pairwise_bt_prog(prompt=prompt, samples_with_idx=samples_with_index)
                    try:
                        selected_sample_idx = int(pred.best_sample_idx)
                        assert selected_sample_idx in [0,1,2]
                        # print(f"LLM return idx (Round {k}): {selected_sample_idx}")
                        choice.append(selected_sample_idx)
                    except Exception as e:
                        print(e)
                if len(choice) < 2: 
                    continue

                if (choice[0] == 0 and choice[1] == 1): # i wins
                    score[i] += 1
                elif (choice[0] == 1 and choice[1] == 0): # j wins
                    score[j] += 1
                elif (choice[0] == 2 and choice [1] == 2): # tie
                    score[i] += 0.5
                    score[j] += 0.5
        max_score_index = np.argmax(score)
        print(score, max_score_index)
        return max_score_index
    
    def test_aware_single_match(self, prompt, test_input, output_a, output_b, sample_a, sample_b):
        choice = []
        for k in range(2):
            samples_with_idx = f"\n[Test input]: {test_input} \n"
            if k == 0:
                samples_with_idx += f"\n[Sample 0]: ### Code ### \n {sample_a} \n ### Execution output ### {output_a} \n"
                samples_with_idx += f"\n[Sample 1]: ### Code ### \n {sample_b} \n ### Execution output ### {output_b} \n"
            else:
                samples_with_idx += f"\n[Sample 0]: ### Code ### \n {sample_b} \n ### Execution output ### {output_b} \n"
                samples_with_idx += f"\n[Sample 1]: ### Code ### \n {sample_a} \n ### Execution output ### {output_a} \n"
            # print(samples_with_idx)
            with dspy.context(lm=self.judge_lm):
                pred = self.llm_select_pairwise_test_aware_bt_prog(prompt=prompt, samples_with_idx=samples_with_idx)
            try:
                selected_sample_idx = int(pred.best_sample_idx)
                assert selected_sample_idx in [0,1,2]
                # print(f"LLM return idx (Round {k}): {selected_sample_idx}")
                choice.append(selected_sample_idx)
            except Exception as e:
                print(e)
        if len(choice) < 2: 
            return 0, 0

        if (choice[0] == 0 and choice[1] == 1): # i wins
            return 1, 0
        elif (choice[0] == 1 and choice[1] == 0): # j wins
            return 0, 1
        elif (choice[0] == 2 and choice [1] == 2): # tie
            return 0.5, 0.5
        else: # inconsistent
            return 0, 0

    def find_maximum_agreement(self, feedbacks):
        agreement_score = np.zeros(len(feedbacks))
        for i in range(len(feedbacks)):
            for j in range(i + 1, len(feedbacks)):
                for k in range(len(feedbacks[i])):
                    if feedbacks[i][k] == feedbacks[j][k]:
                        agreement_score[i] += 1
                        agreement_score[j] += 1
        max_agreement_index = np.argmax(agreement_score)
        print(agreement_score, max_agreement_index)
        return max_agreement_index

    def find_concensus_groups(self, feedbacks):
        score = np.zeros(len(feedbacks))
        concensus_groups_record = [None] * len(feedbacks[0])
        for i in range(len(feedbacks[0])):
            # For each response, we form a consensus group and let LLM to decide which work the best
            consensus_groups = defaultdict(list)
            for j in range(len(feedbacks)):
                assert len(feedbacks[j][i]) == 1
                consensus_groups[str(feedbacks[j][i][0])].append(j)
          #  print(consensus_groups)
            concensus_groups_record[i] = consensus_groups
        return concensus_groups_record

    def find_maximum_score_test_aware_bt(self, prompt, tests, samples, concensus_groups_record):
        score = np.zeros(len(samples))
        # print(concensus_groups_record)
        for group_idx, group in enumerate(concensus_groups_record):
            keys = list(group.keys())
            cur_cluster_score = np.zeros(len(keys))
            if len(keys) == 1:
                continue
            # draw sample a and b
            for cluster_a_idx in range(len(keys)):
                for cluster_b_idx in range(cluster_a_idx + 1, len(keys)):
                    print(cluster_a_idx, cluster_b_idx) 
                    cluster_a_element = group[keys[cluster_a_idx]]
                    cluster_b_element = group[keys[cluster_b_idx]]
                    for sample_a_idx in cluster_a_element:
                        for sample_b_idx in cluster_b_element:
                            sample_a_score, sample_b_score = self.test_aware_single_match(prompt, tests[group_idx]['test']['input'], keys[cluster_a_idx], keys[cluster_b_idx], samples[sample_a_idx], samples[sample_b_idx])
                            cur_cluster_score[cluster_a_idx] += sample_a_score
                            cur_cluster_score[cluster_b_idx] += sample_b_score
              #              print(f"Match between cluster {cluster_a_idx} and cluster {cluster_b_idx}, sample {sample_a_idx} and sample {sample_b_idx}: {sample_a_score} {sample_b_score}")
              #              print(f"After update: cluster scores: {cur_cluster_score}")

            max_value = np.max(cur_cluster_score)
            count_max = np.sum(cur_cluster_score == max_value)
            if count_max > 1: # The clusters have more than one maximum scores, skip.
                pass
                print(f"Test {group_idx} result: Clusters ties with group element: {group} and score {cur_cluster_score}, current sample score: {score}")
            else:
                best_cluster_index = np.argmax(cur_cluster_score)
                for element_idx in group[keys[best_cluster_index]]:
                    score[element_idx] += 1
                print(f"Test {group_idx} result: Cluster {best_cluster_index} wins with group element: {group}, current sample score: {score}")

        if np.max(score) == 0: # the previous procedure breaks, we fall back to pairwise comparison on code.
            max_sample_index = self.find_maximum_score_bt(prompt, samples)
        else:
            max_sample_index = np.argmax(score)
        return max_sample_index
    
    def find_maximum_score_tool_assisted_bt(self, prompt, samples, concensus_groups_record, example):
        score = np.zeros(len(samples))
        # print(concensus_groups_record)
        for group_idx, group in enumerate(concensus_groups_record):
            keys = list(group.keys())
            cur_cluster_score = np.zeros(len(keys))
            if len(keys) == 1:
                continue
            # draw sample a and b
            for cluster_a_idx in range(len(keys)):
                for cluster_b_idx in range(cluster_a_idx + 1, len(keys)):
                    print(cluster_a_idx, cluster_b_idx) 
                    cluster_a_element = group[keys[cluster_a_idx]]
                    cluster_b_element = group[keys[cluster_b_idx]]
                    for sample_a_idx in cluster_a_element:
                        for sample_b_idx in cluster_b_element:
                            sample_a_score, sample_b_score = self.tool_assisted_single_match(prompt, samples[sample_a_idx], samples[sample_b_idx], example)
                            cur_cluster_score[cluster_a_idx] += sample_a_score
                            cur_cluster_score[cluster_b_idx] += sample_b_score
              #              print(f"Match between cluster {cluster_a_idx} and cluster {cluster_b_idx}, sample {sample_a_idx} and sample {sample_b_idx}: {sample_a_score} {sample_b_score}")
              #              print(f"After update: cluster scores: {cur_cluster_score}")

            max_value = np.max(cur_cluster_score)
            count_max = np.sum(cur_cluster_score == max_value)
            if count_max > 1: # The clusters have more than one maximum scores, skip.
                pass
                print(f"Test {group_idx} result: Clusters ties with group element: {group} and score {cur_cluster_score}, current sample score: {score}")
            else:
                best_cluster_index = np.argmax(cur_cluster_score)
                for element_idx in group[keys[best_cluster_index]]:
                    score[element_idx] += 1
                print(f"Test {group_idx} result: Cluster {best_cluster_index} wins with group element: {group}, current sample score: {score}")

        if np.max(score) == 0: # the previous procedure breaks, we fall back to pairwise comparison on code.
            max_sample_index = self.find_maximum_score_bt(prompt, samples)
        else:
            max_sample_index = np.argmax(score)
        return max_sample_index

    def find_maximum_score_llm(self, prompt, tests, feedbacks):
        score = np.zeros(len(feedbacks))
        assert len(tests) == len(feedbacks[0])
        # print(f"There are {len(tests)} tests and {len(feedbacks)} samples.")
        # print(feedbacks)
        num_bad_tests = 0
        concensus_groups_record = [None] * len(feedbacks[0])
        # print(tests)
        # print(feedbacks)
        for i in range(len(feedbacks[0])):
            # For each response, we form a consensus group and let LLM to decide which work the best
            consensus_groups = defaultdict(list)
            for j in range(len(feedbacks)):
                assert len(feedbacks[j][i]) == 1
                # print(f)
                # print(feedbacks[j][i])
                consensus_groups[str(feedbacks[j][i][0])].append(j)
            print(consensus_groups)
            concensus_groups_record[i] = consensus_groups
            test_and_samples_with_idx = f"\n [Test input]: {tests[i]['test']['input']} \n"
            keys = list(consensus_groups.keys())
            if len(keys) == 1: # This test cannot distinguish inputs, and thus should be skipped
                continue
            else:
                for c_idx, c in enumerate(keys):
                    test_and_samples_with_idx += f"\n[Test result {c_idx}]: {c} \n"
                 
                pred = self.llm_select_with_test_result_prog(prompt=prompt, test_and_samples_with_idx=test_and_samples_with_idx)
         #        print(prompt)
                print(test_and_samples_with_idx)
                # print(pred)

                try:
                    selected_sample_idx = int(pred.best_idx)
                    # print(f"LLM return idx: {selected_sample_idx}")
                    if selected_sample_idx == -1:
                        num_bad_tests += 1
                        print(f"Bad test. Skipped")
                        continue
                    assert selected_sample_idx in list(range(len(consensus_groups)))
                    best_c = keys[selected_sample_idx]
                    for best_sample_idx in consensus_groups[best_c]:
                        score[best_sample_idx] += 1
                except Exception as e:
                    print(e)
        # if np.max(score) == 0: # No decision has been made, fall back to bt
        max_score_index = np.argmax(score)
        print(score, max_score_index)
        return num_bad_tests == len(tests), max_score_index, concensus_groups_record
    
    def llm_reflection_with_tool(self,zipped_history,example):
        reflections = []
        all_samples = []
        for n in range(self.num_generations):
            sample, _, _, _, _ = zipped_history[n][-1]
            all_samples.append(dspy.Prediction(code=sample))
            llm_generated_tests = generate_tests_for_one_example(example,generation_fun=llm_generate_tests_reflect, completions= sample, num_test_suites=1)
            execution_results = ""
            for test in llm_generated_tests:
                exec_feedback = get_execution_feedback([test], sample, 0, example.prompt)
                #TODO: print out exec_feedback to see what is its structure
                execution_results += str(exec_feedback["maybe_error_messages"][0]) + "\n"
            reflection = self.llm_with_tool_reflect_prog( ##TODO: Here should we provide the previous chain of thought of generating the tests so LLM has context?
                    prompt=example.prompt,
                    completion = sample,
                    execution_results = execution_results,
                    config=dict(temperature=self.temperature + TEMPARATURE_STEP*n),
            )
            reflections.append(reflection.feedback)
        return dspy.Prediction(codes=all_samples, prompts=[example.prompt] * self.num_generations), reflections
    

    def tool_assisted_single_match(self, prompt, sample_a, sample_b, example):
        choice = []
        llm_generated_tests = generate_tests_for_one_example(example,generation_fun=llm_generate_tests_reflect, completions= [sample_a, sample_b], num_test_suites=1, judge_lm=self.judge_lm)
        for k in range(2):
            samples_with_idx = ""
            if k == 0:
                # First code and all its test results
                samples_with_idx += f"\n[Sample 0]: ### Code ### \n {sample_a}\n"
                samples_with_idx += "### Execution outputs ###\n"
                for test in llm_generated_tests:
                    samples_with_idx += f"\n[Test input]: {test['input']}"
                    exec_feedback_a = get_execution_feedback([test], sample_a, 0, example.prompt)
                    samples_with_idx += f"\n[Output]: {exec_feedback_a['maybe_error_messages'][0]}\n"

                # Second code and all its test results  
                samples_with_idx += f"\n[Sample 1]: ### Code ### \n {sample_b}\n"
                samples_with_idx += "### Execution outputs ###\n"
                for test in llm_generated_tests:
                    samples_with_idx += f"\n[Test input]: {test['input']}"
                    exec_feedback_b = get_execution_feedback([test], sample_b, 0, example.prompt)
                    samples_with_idx += f"\n[Output]: {exec_feedback_b['maybe_error_messages'][0]}\n"
            else:
                # First code and all its test results
                samples_with_idx += f"\n[Sample 0]: ### Code ### \n {sample_b}\n"
                samples_with_idx += "### Execution outputs ###\n"
                for test in llm_generated_tests:
                    samples_with_idx += f"\n[Test input]: {test['input']}"
                    exec_feedback_b = get_execution_feedback([test], sample_b, 0, example.prompt)
                    samples_with_idx += f"\n[Output]: {exec_feedback_b['maybe_error_messages'][0]}\n"

                # Second code and all its test results
                samples_with_idx += f"\n[Sample 1]: ### Code ### \n {sample_a}\n"
                samples_with_idx += "### Execution outputs ###\n" 
                for test in llm_generated_tests:
                    samples_with_idx += f"\n[Test input]: {test['input']}"
                    exec_feedback_a = get_execution_feedback([test], sample_a, 0, example.prompt)
                    samples_with_idx += f"\n[Output]: {exec_feedback_a['maybe_error_messages'][0]}\n"
            
            with dspy.context(lm=self.judge_lm):
                pred = self.llm_select_pairwise_test_aware_bt_prog(prompt=prompt, samples_with_idx=samples_with_idx)
            try:
                selected_sample_idx = int(pred.best_sample_idx)
                assert selected_sample_idx in [0,1,2]
                # print(f"LLM return idx (Round {k}): {selected_sample_idx}")
                choice.append(selected_sample_idx)
            except Exception as e:
                print(e)
        if len(choice) < 2: 
            return 0, 0

        if (choice[0] == 0 and choice[1] == 1): # i wins
            return 1, 0
        elif (choice[0] == 1 and choice[1] == 0): # j wins
            return 0, 1
        elif (choice[0] == 2 and choice [1] == 2): # tie
            return 0.5, 0.5
        else: # inconsistent
            return 0, 0
def visualize_matrix(
    preds,
    preds_pass,
    canonical_solution,
    prompt,
    golden_test,
    entry_point,
    tests,
    task_id,
    extracted_tests=None,
    prefix=None,
):
    tests, tests_weights = list(tests.keys()), list(tests.values())
    correct_test = map_test(tests, canonical_solution, prompt)
    correct_extracted_test = map_test(extracted_tests, canonical_solution, prompt)

    preds_pass_extracted_test = [
        list(
            map(
                lambda test: 1
                if check_test([test], pred, 0, prompt, "dummy")[0]
                else 0,
                extracted_tests,
            )
        )
        for pred in preds
    ]

    preds_pass_extracted_test_array = np.array(preds_pass_extracted_test)

    processed_golden_tests = post_process_tests_human_eval(golden_test)
    correct_code = [
        check_test_temp(processed_golden_tests, pred, 0, prompt, entry_point)
        for pred in preds
    ]
    task_id_num = task_id.split("/")[-1]

    # Convert preds_pass to numpy array for easier handling
    preds_array = np.array(preds_pass)
    num_tests_passed = np.sum(preds_array, axis=1)

    # Sorting indices for codes and tests
    # code_indices = np.argsort(correct_code)[::-1]  # Reverse to put True (good) first
    code_indices = np.argsort(num_tests_passed)[::-1]
    test_indices = np.argsort(correct_test)[::-1]  # Reverse to put True (good) first
    extract_tests_indices = np.argsort(correct_extracted_test)[::-1]

    # sorted_code and tests
    preds_code = [preds[i].code for i in code_indices]
    sorted_tests = np.array(tests)[test_indices]
    sorted_tests_weights = np.array(list(tests_weights))[test_indices]
    sorted_extracted_tests = np.array(extracted_tests)[extract_tests_indices]

    if correct_code[code_indices[0]]:
        passed = "pass"
    else:
        passed = "fail"

    log_name = f"data_{prefix}/{passed}/{task_id_num}.log"
    os.makedirs(os.path.dirname(log_name), exist_ok=True)
    with open(log_name, "w") as f:
        for i, pred in enumerate(preds_code):
            f.write(f"code_{i}: " + pred + "\n\n")
        for i, test in enumerate(sorted_tests):
            f.write(f"test_{i}: " + test + "\n\n")

    # Apply sorting
    sorted_preds = preds_array[code_indices, :]
    sorted_preds = sorted_preds[:, test_indices]
    sorted_codes_good = np.array(correct_code)[code_indices]
    sorted_tests_good = np.array(correct_test)[test_indices]
    sorted_extracted_tests_good = np.array(correct_extracted_test)[
        extract_tests_indices
    ]

    sorted_tests_good = np.concatenate(
        (sorted_extracted_tests_good, sorted_tests_good), axis=0
    )

    if extracted_tests:
        # prepend preds_pass_extracted_test_array to sorted_preds
        sorted_preds = np.concatenate(
            (preds_pass_extracted_test_array[code_indices, :], sorted_preds), axis=1
        )

    max_passed_tests = np.max(num_tests_passed)
    highest_candidates = num_tests_passed[code_indices] == max_passed_tests

    sorted_codes_with_stars = [f"{code}" for code in range(len(preds_code))]
    sorted_codes_with_stars = [
        f"{code}*" if is_highest else f"{code}"
        for code, is_highest in zip(sorted_codes_with_stars, highest_candidates)
    ]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a colormap where True is green and False is red
    cmap = mcolors.ListedColormap(["red", "green"])
    bounds = [0, 0.5, 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot the matrix
    cax = ax.matshow(sorted_preds, cmap=cmap, norm=norm)

    # Set axis labels
    ax.set_xlabel("Tests")
    ax.set_ylabel("Codes")

    # Set ticks
    # Set ticks
    ax.set_xticks(np.arange(len(sorted_tests_good)))
    ax.set_yticks(np.arange(len(sorted_codes_good)))

    # print(len(sorted_tests_good))

    sorted_tests_with_weights_labels = [
        f"{i}\n({weight})"
        for i, weight in zip(
            list(range(len(sorted_extracted_tests))) + list(range(len(sorted_tests))),
            [1 for _ in range(len(sorted_extracted_tests))]
            + list(sorted_tests_weights),
        )
    ]

    # Set tick labels
    ax.set_xticklabels(sorted_tests_with_weights_labels, fontsize=8)
    ax.set_yticklabels(sorted_codes_with_stars)

    # Set tick labels with color differentiation for good/bad candidates
    # Set the color of the tick labels
    for tick_label, is_good in zip(ax.get_xticklabels(), sorted_tests_good):
        tick_label.set_color("green" if is_good else "red")

    for tick_label, is_good in zip(ax.get_yticklabels(), sorted_codes_good):
        tick_label.set_color("green" if is_good else "red")

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, len(sorted_tests_good), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sorted_codes_good), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)

    good_code_idx = len(sorted_extracted_tests)
    ax.axvline(good_code_idx - 0.5, color="black", linewidth=3)

    plt.tight_layout()
    # Add a colorbar
    # plt.colorbar(cax, ticks=[0, 1])
    post_fix = "_no_sample" if not any(correct_code) else ""

    plt.savefig(f"data_{prefix}/{passed}/task_{task_id_num}{post_fix}.png")

    return preds_code[0]


def visualize_lcb(preds_pass, extracted_pass, taskid, output_dir="."):
    # Convert preds_pass and extracted_pass to numpy arrays
    preds_array = np.array(preds_pass)
    extracted_array = np.array(extracted_pass)

    # Determine the number of tests passed by each prediction
    num_tests_passed = np.sum(preds_array, axis=1)

    # Sort predictions by the number of tests passed in descending order
    code_indices = np.argsort(num_tests_passed)[::-1]

    # Combine preds_pass and extracted_pass test results for plotting
    combined_preds = np.concatenate((extracted_array, preds_array), axis=1)
    combined_preds = combined_preds.astype(int)

    # Sort the combined results according to sorted prediction indices
    sorted_preds = combined_preds[code_indices, :]

    # Create a figure and axis for the matrix plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a colormap where True (pass) is green and False (fail) is red
    cmap = mcolors.ListedColormap(["red", "green"])
    bounds = [0, 0.5, 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot the matrix with the sorted data
    cax = ax.matshow(sorted_preds, cmap=cmap, norm=norm)

    # Set axis labels
    ax.set_xlabel("Tests")
    ax.set_ylabel("Predictions")

    # Set ticks
    num_extracted = extracted_array.shape[1]
    num_preds = preds_array.shape[1]
    total_tests = num_extracted + num_preds
    ax.set_xticks(np.arange(total_tests))
    ax.set_yticks(np.arange(len(preds_pass)))

    # Set tick labels with distinct labels for extracted and other tests
    sorted_test_labels = ["E{}".format(i) for i in range(num_extracted)] + [
        "P{}".format(i) for i in range(num_preds)
    ]

    # Set tick labels
    ax.set_xticklabels(sorted_test_labels, fontsize=8)
    ax.set_yticklabels(["Pred_{}".format(i) for i in range(len(preds_pass))])

    # Color the tick labels to distinguish extracted (green) and other tests (black)
    for i, tick_label in enumerate(ax.get_xticklabels()):
        tick_label.set_color("green" if i < num_extracted else "black")

    # Draw grid lines for better visualization
    ax.set_xticks(np.arange(-0.5, total_tests, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(preds_pass), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)

    # Add a vertical line to separate extracted tests from other tests
    ax.axvline(num_extracted - 0.5, color="blue", linewidth=2)

    # Add a colorbar to indicate pass/fail
    plt.colorbar(cax, ticks=[0, 1])

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot to the specified directory and filename
    save_path = os.path.join(output_dir, f"{taskid}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved to {save_path}")

def process_example(example, retry, generation_fun):
    """Helper function to process each example from the dataset."""
    for _ in range(retry):
        try:
            is_stdin = has_test_type(example.public_test_cases, "stdin")
            tests = generation_fun(example.prompt, is_stdin)
            break
        except Exception as e:
            print(e)
            continue

    # Return the task_id and tests for aggregation later
    return example.task_id, tests

def generate_tests_for_whole_dataset(
    dataset,
    # generation_fun,
    post_process_test=lambda prompt, tests: tests,
    retry=3,
    verbose=False,
):
    results = {}
    all_tests_count = 0
    correct_tests_count = 0
    zero_tests_count = 0
    zero_correct_tests_count = 0
    not_all_correct = 0

    # Using ProcessPoolExecutor for parallel processing
    # num_process = 32
    #with concurrent.futures.ProcessPoolExecutor(num_process) as executor:
        # Wrap dataset with tqdm for progress bar
    #    futures = [executor.submit(process_example, example, retry, generation_fun) for example in dataset]
    #    for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(dataset)):
    #        task_id, tests = future.result()
    #        results[task_id] = tests

    #        all_tests_count += len(tests)
    #        if len(tests) == 0:
    #            zero_tests_count += 1

    for example in tqdm.tqdm(dataset):
        for r in range(retry):
            try:
                is_stdin = has_test_type(example.public_test_cases, "stdin")
                tests = generate_tests_repeat(example.prompt, is_stdin, r)
                break
            except Exception as e:
                print(e)
                continue
#
        # print(tests)
        results[example.task_id] = tests

        all_tests_count += len(tests)

        if len(tests) == 0:
            zero_tests_count += 1

    if verbose:
        print(f"Total number of prompts with not all correct tests: {not_all_correct}")
    return (
        results,
        all_tests_count,
        correct_tests_count,
        zero_tests_count,
        zero_correct_tests_count,
    )

def generate_tests_for_one_example(example,  ##TODO :Alex, make sure what ever takes the output of this function to be able to handle the new output format
    generation_fun,
    completions = None,
    judge_lm=None,
    post_process_test=lambda prompt, tests: tests,
    retry=3,
    verbose=False, num_timeout_tests=None, num_test_suites=1, o1=False):
    results = {}
    tests = []
    if num_timeout_tests is not None:
        assert num_test_suites == 1, "Multiple generation is not supported for timeout tests."
        tests = []
        r_counter = 0
        for _ in range(num_timeout_tests):
            for r in range(retry):
                try:
                    is_stdin = has_test_type(example.public_test_cases, "stdin")
                    test = generation_fun(example.prompt, is_stdin, r_counter)
                    if test is not None:
                        tests.append(test)
                    r_counter += 1
                    break
                except Exception as e:
                    print(e)
                    r_counter += 1
                    continue
    else:
        r_counter = 0
        for t in range(num_test_suites):
            for _ in range(retry):
                r_counter += 1
                if o1:
                    r_counter = 0 ## o1 is not using r_counter
                try:
                    is_stdin = has_test_type(example.public_test_cases, "stdin")
                    if completions is not None:
                        cur_tests = generation_fun(example.prompt, completions, is_stdin, r_counter, 0.7, judge_lm) ## Hard Coded temperature base
                    else:
                        cur_tests = generation_fun(example.prompt, is_stdin, r_counter)
                    tests.append(cur_tests)
                    break
                except Exception as e:
                    print(e)
                    continue
        tests = list(chain(*tests))
        # print(tests)

    # assert len(tests) > 0, f"No tests were generated for task {example.task_id}"

    if verbose:
        print(f"Number of tests generated for task {example.task_id}: {len(tests)}")
    return tests

class NaiveCodeGenerator(dspy.Module):
    def __init__(self, cot):
        if cot:
            self.stdin_prog = dspy.ChainOfThought(GenerateLCBcodestdin)
            self.functional_prog = dspy.ChainOfThought(GenerateLCBcodefunctional)
        else:
            self.stdin_prog = dspy.Predict(GenerateLCBcodestdin)
            self.functional_prog = dspy.Predict(GenerateLCBcodefunctional)

    def forward(self, prompt, is_stdin, **kargs):
        prog = self.stdin_prog if is_stdin else self.functional_prog
        pred = prog(prompt=prompt) 
        # print(pred)
        return pred

class NaiveCodeGeneratorNoDSPy(dspy.Module):
    def __init__(self, args):
        self.stdin_prog_prompt = "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition ."
        self.functional_prog_prompt = "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
        self.args = args

    def forward(self, prompt, is_stdin, **kargs):
        question_prompt = self.stdin_prog_prompt if is_stdin else self.functional_prog_prompt
        prompt = question_prompt + prompt
        print(prompt)
        response = ""
        
        if self.args.generator in name_map.keys():
            mapped_name = name_map[self.args.generator]
        else:
            mapped_name = self.args.generator
        
        print(mapped_name, self.args.generator)
        if "openai" in mapped_name:
            client = OpenAI()
            for i in range(3):
                try:
                    print(f"mapped_name: {mapped_name}")
                    if "o3-mini" in mapped_name:
                        chat_response = client.chat.completions.create(
                            messages=[{
                                    "role": "user",
                                    "content": prompt,
                            }],
                            model=mapped_name.replace("openai/", ""),
                            reasoning_effort="high"
                        )
                    else:
                        chat_response = client.chat.completions.create(
                            messages=[{
                                    "role": "user",
                                    "content": prompt,
                            }],
                            model=mapped_name.replace("openai/", ""),
                        )

                    response = chat_response.choices[0].message.content
                    # print(response)
                    break
                except Exception as e:
                    print(e)
                    time.sleep(5)
                
        else:
            client = OpenAI(
                api_key="EMPTY",
                base_url=self.args.api_base,
            )
            assert "qwen" in self.args.api_name.lower(), "Only Qwen system prompt is supported. Please modify the below system prompt for other models."
            try:
                chat_response = client.chat.completions.create(
                    model=self.args.api_name,
                    messages=[
                        # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
                        {"role": "user", "content": prompt},
                    ]
                )
                response = chat_response.choices[0].message.content
                # pred = prog(prompt=prompt) 
            except Exception as e:
                print(e)
                response = ""

        # print(response)
        # Regular expression to match text enclosed by triple backticks
        print("response before", response)
        # o3 will not use ```
        if "```" in response:
            pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"

            # Use re.DOTALL to match multiline content inside backticks
            matches = re.findall(pattern, response, re.DOTALL)

            # Print the extracted code blocks
            # for idx, block in enumerate(matches):
            #    print(f"Code Block {idx}:\n{block}")

            if matches:
                response = matches[-1]
                # print(response)
            else:
                response = ""
        # assert False	
        
        print("response after", response)

        pred = dspy.Prediction(code=response)       
        return pred

