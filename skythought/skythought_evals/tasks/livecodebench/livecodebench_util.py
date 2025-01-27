import base64
import builtins
import contextlib
import copy
import faulthandler
import io
import json
import multiprocessing
import os
import pickle
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zlib
from io import StringIO
from typing import Optional


def post_process_code(code):
    code = code.split("</code>")[0]
    code = code.replace("```python", "")
    code = code.split("```")[0]
    code = code.replace("<code>", "")
    # print(f"postprocessed code: {code}")
    return code


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
        # TODO: .strip() with multiple characters is misleading, this will look for the individual characters in any order.
        # we should switch to .replace and test
        # TODO: Do not ignore B005
        cleaned_string = (
            raw_text.strip().strip("```json").strip("```").strip()  # noqa: B005
        )

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
                        test_case["testtype"] = (
                            "functional"  # Add 'testtype' for each test case
                        )
                    return test_cases
                except json.JSONDecodeError as e:
                    print(f"Error parsing concatenated JSON: {e}")

            # If no matches are found, fall back to line-by-line parsing
            cleaned_lines = cleaned_string.split("\n")
            if test_cases is None:
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


def prepare_test_input_output_functional(test_case, is_extracted):
    if not is_extracted:
        # Extract input and expected output from JSON directly
        test_input = test_case["input"]
        test_output = test_case["output"]
        return test_input, test_output
    else:
        # TODO THIS IS VERY HACKY, CAN TAKE OFF A LOT OF THINGS THAT ARE NOT NEEDED
        # Extract input and expected output
        input_str = test_case["input"]
        expected_output = test_case["output"].strip()

        # Prepare the input
        inputs = []
        if "=" in input_str:
            if "," in input_str:
                # Multiple key-value pairs case
                input_parts = [part.strip() for part in input_str.split(",")]
                for part in input_parts:
                    key, value = part.split("=")
                    key = key.strip()
                    value = value.strip()
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            value = value.strip('"')
                    inputs.append(value)
            else:
                # Single key-value pair case
                key, value = input_str.split("=")
                key = key.strip()
                value = value.strip()
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        value = value.strip('"')
                inputs.append(value)
        else:
            # Handle standard input formats with newlines
            input_parts = input_str.split("\n")
            for part in input_parts:
                part = part.strip()
                if part:
                    if part.startswith('"') and part.endswith('"'):
                        # If the part is wrapped in quotes, treat it as a string
                        inputs.append(part.strip('"'))
                    elif part.startswith("[") and part.endswith("]"):
                        # If part looks like a list, parse it using json.loads
                        inputs.append(json.loads(part))
                    else:
                        # Otherwise, attempt to convert it to a number (int or float)
                        try:
                            inputs.append(int(part))
                        except ValueError:
                            try:
                                inputs.append(float(part))
                            except ValueError:
                                # If it's neither a number nor a list, treat it as a string
                                inputs.append(part)

        # Try to parse the expected output into a proper data type
        try:
            expected_output = json.loads(expected_output)
        except json.JSONDecodeError:
            # If it's not JSON, keep it as a string
            expected_output = expected_output.strip()
        return inputs, expected_output


def prepare_test_input_output_std(test_case):
    test_input = test_case["input"]
    test_output = test_case["output"].strip()
    if test_output.endswith("-"):
        test_output = test_output[
            : test_output.rfind("-")
        ].rstrip()  # Remove '-' if present and trailing
    return test_input, test_output


def run_test_func(completion, is_extracted, test_input, test_output):
    # print(f"inside: {completion}")
    if not is_extracted:
        # Define the namespace in which to execute the completion code
        namespace = {}

        # Execute the generated code in the namespace

        exec(completion, namespace)

        # Retrieve the function name from the completion (assuming the function name is consistent)
        func_name = completion.split("(")[0].split()[-1]

        # Ensure inputs and expected_output are correctly parsed from JSON
        if isinstance(test_input, dict):
            # If the input is a dictionary, pass it as keyword arguments
            args = []
            kwargs = test_input  # The dictionary is passed as keyword arguments
        else:
            # If the input is not a dictionary, pass it as a single argument
            args = [test_input]  # Wrap inputs in a list, keeping it as one argument
            kwargs = {}

        # Redirect stdout to capture any printed output
        output = io.StringIO()
        sys.stdout = output

        # Call the function and get the result
        try:
            # Execute the function with the prepared inputs
            result_output = namespace[func_name](*args, **kwargs)

            # Compare the result with expected output
            if result_output != test_output:
                return False, result_output

        except Exception as e:
            # Capture and return the exception if one occurs
            return False, f"Error: {str(e)}"

        finally:
            # Reset stdout
            sys.stdout = sys.__stdout__

        return True, result_output
    else:
        namespace = {}

        # Execute the generated code in the namespace

        exec(completion, namespace)

        # Retrieve the function name from the completion (assuming the function name is consistent)
        func_name = completion.split("(")[0].split()[-1]

        # Redirect stdout
        output = io.StringIO()
        sys.stdout = output

        # Call the function and get the result
        try:
            # Execute the function with the prepared inputs
            result_output = namespace[func_name](*test_input)

            if result_output != test_output:
                return False, result_output
        except Exception as e:
            return False, str(e)

        finally:
            # Reset stdout
            sys.stdout = sys.__stdout__

        return True, result_output


def run_test_std(completion, test_input, test_output):
    sys.stdin = StringIO(test_input)

    output = StringIO()
    sys.stdout = output

    if '__name__ == "__main__"' in completion:
        # Simulate that the code is being run as the main script
        completion = '__name__ = "__main__"\n' + completion

    namespace = {}
    exec(completion, namespace)

    output_value = output.getvalue().strip()
    return output_value == test_output, output_value


def unsafe_lcb_runTests(problem, completion, timeout, runtime_debug, is_extracted):
    test_cases = problem["test"]
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(
        target=run_tests_for_one_example,
        args=(test_cases, completion, result, runtime_debug, is_extracted),
    )
    p.start()
    p.join(
        timeout=(timeout + 1) * len(test_cases) + 5
    )  # TODO Alex: Check whether number of task cases is correct
    if p.is_alive():
        p.kill()

    # if len(result) < len(test_cases): ## This is supposed to be the case where not all test passed in the given timeout
    for _i in range(len(test_cases) - len(result)):
        result.append((False, "Time out!.", "Error: Time out!", float("inf")))
    return result


def run_tests_for_one_example(
    test_cases, completion, result_list, runtime_debug, is_extracted
):
    time_elapsed = float("inf")
    test_type = test_cases[0]["testtype"]
    reliability_guard()
    for _i, test_case in enumerate(test_cases):
        output_error = ""
        output_value = ""
        try:
            time_start = time.time()
            if test_type == "functional":
                test_input, test_output = prepare_test_input_output_functional(
                    test_case, is_extracted
                )
                passed, output_value = run_test_func(
                    completion,
                    is_extracted,
                    copy.deepcopy(test_input),
                    copy.deepcopy(test_output),
                )
            else:
                test_input, test_output = prepare_test_input_output_std(test_case)
                passed, output_value = run_test_std(
                    completion, copy.deepcopy(test_input), copy.deepcopy(test_output)
                )
            # print(test_input, test_output, output_value)
            if not passed:
                output_error = f"For test input: {test_input}. Expected output is: {test_output}, but got: {output_value}."
            time_elapsed = time.time() - time_start

        except Exception as e:
            print(f"is_extracted: {is_extracted}")
            print(f"Caught a generic exception: {e}")
            passed = False
            output_error = f"For test input: {test_input}. Expected output is: {test_output}, but got error: {e}."
            output_value = f"Error: {e}."
        if output_error == "":
            output_error = f"For test input: {test_input}. Expected output is: {test_output}, your solution correctly passes this test with output {output_value}."  # noqa: E501

        result_list.append((passed, output_error, output_value, time_elapsed))
        if not passed:
            return


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io(redirect_input=True):
    """
    A context manager that redirects stdout and stderr to a write-only stream,
    and optionally redirects stdin based on the redirect_input flag.
    """
    stream = WriteOnlyStringIO()  # Create a stream to capture stdout and stderr

    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        if redirect_input:
            with contextlib.redirect_stdin(StringIO()):  # Redirect stdin if enabled
                yield stream
        else:
            yield stream  # Do not redirect stdin


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    # __builtins__["help"] = None   # this line is commented out as it results into error

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def save_original_references():
    global originals
    originals = {
        "builtins": {
            "exit": builtins.exit,
            "quit": builtins.quit,
        },
        "os": {
            "kill": os.kill,
            "system": os.system,
            "putenv": os.putenv,
            "remove": os.remove,
            "removedirs": os.removedirs,
            "rmdir": os.rmdir,
            "fchdir": os.fchdir,
            "setuid": os.setuid,
            "fork": os.fork,
            "forkpty": os.forkpty,
            "killpg": os.killpg,
            "rename": os.rename,
            "renames": os.renames,
            "truncate": os.truncate,
            "replace": os.replace,
            "unlink": os.unlink,
            "fchmod": os.fchmod,
            "fchown": os.fchown,
            "chmod": os.chmod,
            "chown": os.chown,
            "chroot": os.chroot,
            "getcwd": os.getcwd,
            "chdir": os.chdir,
        },
        "shutil": {
            "rmtree": shutil.rmtree,
            "move": shutil.move,
            "chown": shutil.chown,
        },
        "subprocess": {"Popen": subprocess.Popen},
        "sys_modules": {
            "ipdb": sys.modules.get("ipdb"),
            "joblib": sys.modules.get("joblib"),
            "resource": sys.modules.get("resource"),
            "psutil": sys.modules.get("psutil"),
            "tkinter": sys.modules.get("tkinter"),
        },
    }


def restore_original_references():
    global originals
    # Restore 'builtins' functions
    builtins.exit = originals["builtins"]["exit"]
    builtins.quit = originals["builtins"]["quit"]

    # Restore 'os' functions
    for func_name, original_func in originals["os"].items():
        setattr(os, func_name, original_func)

    # Restore 'shutil' functions
    for func_name, original_func in originals["shutil"].items():
        setattr(shutil, func_name, original_func)

    # Restore 'subprocess' functions
    subprocess.Popen = originals["subprocess"]["Popen"]

    # Restore sys modules
    for module_name, original_module in originals["sys_modules"].items():
        if original_module is not None:
            sys.modules[module_name] = original_module


def has_test_type(tests, type):  ## helper to select specific type of problems
    """
    Check if any test in the test list has 'testtype' set to 'type'.
    """
    test_list = json.loads(tests)
    for test in test_list:
        if test.get("testtype") == type:
            return True
    return False


def translate_private_test_cases(encoded_data):
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    return json.loads(original_data)


"""
def update_dataset_in_place(
    dataset,
):  ## helper functions to translate the private test cases
    for entry in dataset:
        try:
            # Translate the private test cases
            decoded_private_test_cases = translate_private_test_cases(
                entry["private_test_cases"]
            )
            # Update the entry in place
            entry["private_test_cases"] = decoded_private_test_cases
            # print(decoded_private_test_cases)
        except Exception as e:
            print(e)
        # break
"""


def map_to_example(row):
    return {
        "prompt": row["question_content"],
        "test": row["private_test_cases"],
        "entry_point": row["starter_code"],
        "canonical_solution": "",  # seems like live code bench lite does not have this field
        "task_id": row["question_id"],
        "is_stdin": has_test_type(row["public_test_cases"], "stdin"),
        "public_test_cases": row["public_test_cases"],
        "difficulty": row["difficulty"],
    }
