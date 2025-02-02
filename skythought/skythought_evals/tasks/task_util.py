import glob
import os
from typing import Dict

# NOTE: (sumanthrh): this dictionary is initialized once during module import per process and stored.
TASK_HANDLER_MAP = {}


def register_handler(handler_name):
    if handler_name in TASK_HANDLER_MAP:
        raise ValueError(f"Handler {handler_name} already registered")

    def wraps(handler_cls):
        TASK_HANDLER_MAP[handler_name] = handler_cls
        return handler_cls

    return wraps


def get_tasks(task_root_dir: str) -> Dict[str, str]:
    """Returns a dictionary of task names and their corresponding yaml file paths"""
    # list all yamls in subdirectories
    name_to_yaml = {}
    for yaml_file in glob.glob(
        os.path.join(task_root_dir, "**", "*.yaml"), recursive=True
    ):
        # arc.yaml -> arc
        name = os.path.basename(yaml_file).split(".")[0]

        name_to_yaml[name] = yaml_file

    return name_to_yaml
