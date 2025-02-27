import json

import pyarrow as pa
from ray.data import Schema


class APPSPreprocessor:
    WITH_FN_NAME_TEMPLATE = "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition. {prompt}"  # noqa: E501

    WITHOUT_FN_NAME_TEMPLATE = "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution. {prompt}"  # noqa: E501

    WITH_STARTER_CODE_TEMPLATE = "{input}\n{starter_code}"

    def __call__(self, row):
        test_case = json.loads(row["input_output"])
        starter_code = row["starter_code"]
        prompt = row["question"]
        if not test_case.get("fn_name"):
            _input = self.WITH_FN_NAME_TEMPLATE.format(prompt=prompt)
        else:
            _input = self.WITHOUT_FN_NAME_TEMPLATE.format(prompt=prompt)

        if starter_code is not None:
            _input = self.WITH_STARTER_CODE_TEMPLATE.format(
                input=_input, starter_code=starter_code
            )

        return {**row, "user_input": _input}


class TACOPreprocessor:
    INITIAL_TEMPLATE = "\nQUESTION:\n{prompt}"
    STARTER_CODE_TEMPLATE = "{input}\n{starter_code}"
    STDIN_TEMPLATE = "{input}\nUse Standard Input format\nANSWER:\n"
    CALL_TEMPLATE = "{input}\nUse Call-Based format\nANSWER:\n"

    def __call__(self, problem):

        prompt = problem["question"]
        starter_code = (
            None if len(problem["starter_code"]) == 0 else problem["starter_code"]
        )
        try:
            input_outpout = json.loads(problem["input_output"])
            fn_name = (
                None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
            )
        except ValueError:
            fn_name = None

        _input = self.INITIAL_TEMPLATE.format(prompt=prompt)

        if starter_code:
            _input = self.STARTER_CODE_TEMPLATE.format(
                input=_input, starter_code=starter_code
            )
        else:
            _input = self.INITIAL_TEMPLATE.format(prompt=prompt)
        if (not fn_name) and (not starter_code):
            _input = self.STDIN_TEMPLATE.format(input=_input)
        else:
            _input = self.CALL_TEMPLATE.format(input=_input)

        return {**problem, "user_input": _input}


class NUMINAPreprocessor:
    TEMPLATE = "Return your final response within \\boxed{{}}. {prompt}"

    def __call__(self, row):
        prompt = row["problem"]
        _input = self.TEMPLATE.format(prompt=prompt)
        return {**row, "user_input": _input}


def taco_coerce_types(row, schema: Schema):
    for key, schema_type in zip(schema.names, schema.types):
        value = pa.array([row[key]])
        if value.type != schema_type:
            if schema_type == pa.string():
                try:
                    row[key] = str(row[key])
                except Exception:
                    row[key] = ""
            elif schema_type == pa.null():
                row[key] = None
    return row
