from ast import literal_eval
from typing import Any, List

import msgpack
import xxhash


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
            except (ValueError, SyntaxError):
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


def comma_separated_to_list(vals: str) -> List[str]:
    vals = vals.replace(" ", "")
    return vals.split(",")


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
