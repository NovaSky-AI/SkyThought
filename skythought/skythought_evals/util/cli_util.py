from ast import literal_eval
from typing import Any

import msgpack
import typer
import xxhash
from click.core import ParameterSource


def _parse_multi_args(vals: str) -> dict:
    """Parse a multi-value argument into a dictionary.
    The argument can either be a comma separated list of key=value pairs, or a dictionary.
    """
    try:
        # try to parse as a dictionary first
        my_dict = literal_eval(vals)
        assert isinstance(my_dict, dict)
        return my_dict
    except Exception:
        # try to parse as a comma separated list of key=value pairs
        vals = vals.replace(" ", "")
        if not len(vals):
            return {}
        ret = {}
        for val in vals.split(","):
            k, v = val.split("=")
            try:
                ret[k] = literal_eval(v)
            except ValueError:
                # if literal eval fails, propagate as a string
                ret[k] = v
        return ret


def parse_multi_args(vals: str) -> dict:
    try:
        return _parse_multi_args(vals)
    except Exception as err:
        raise ValueError(
            f"Expected comma separated list of parameters arg1=val1,args2=val2 or a dictionary, got invalid argument {vals}. "
        ) from err


def get_user_provided_params(ctx: typer.Context) -> dict:
    # click_ctx: click.Context = ctx.get_click_context()
    # Initialize a dictionary to hold user-provided parameters
    user_provided_params = {}

    # Iterate over all parameters of the command
    for param in ctx.command.params:
        # Get the value of the parameter
        value = ctx.params.get(param.name)

        # Determine the source of the parameter's value
        source = ctx.get_parameter_source(param)

        # Check if the parameter was set via the command line
        if source == ParameterSource.COMMANDLINE:
            user_provided_params[param.name] = value
    return user_provided_params


def to_tuple(d) -> tuple:
    if isinstance(d, dict):
        return tuple(map(to_tuple, d.items()))
    elif isinstance(d, (set, list, tuple)):
        return tuple(map(to_tuple, d))
    else:
        return d


def get_deterministic_hash(d: Any, num_digits: int = 6) -> str:
    """Get deterministic hash"""
    tuple_form = to_tuple(d)
    serialized = msgpack.packb(tuple_form, use_bin_type=True)
    return xxhash.xxh32(serialized).hexdigest()[:num_digits]
