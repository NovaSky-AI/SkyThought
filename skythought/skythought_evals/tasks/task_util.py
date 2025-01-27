import glob
import os
from typing import Dict


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
