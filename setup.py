# setup module skyevals in tools directory
import setuptools

setuptools.setup(
    name="skythought_evals",
    version="0.0.1",
    package_dir={
        "skythought_evals": "skythought/skythought_evals"
    },  # map skythought_evals to skythought/skythought_evals
    packages=["skythought_evals"]
    + [
        f"skythought_evals.{pkg}"
        for pkg in setuptools.find_packages(where="skythought/skythought_evals")
    ],
)
