# setup module skyevals in tools directory
from pathlib import Path

import setuptools


def get_requirements():
    req_path = Path("skythought/skythought_evals/requirements.txt")
    with open(req_path, "r") as f:
        return f.read().splitlines()


setuptools.setup(
    name="skythought_evals",
    version="0.0.1",
    package_dir={"": "skythought"},
    packages=setuptools.find_packages(
        where="skythought",
        include=["skythought_evals*"],  # Only pick up skythought_evals, skip 'train'
    ),
    install_requires=get_requirements(),
    python_requires=">=3.9,<3.12",  # pyext doesn't work with python 3.12
)
