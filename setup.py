# setup module skyevals in tools directory
import setuptools

setuptools.setup(
    name="skyevals",
    version="0.0.1",
    package_dir={"skyevals": "skythought/tools"},  # map skyevals to skythought/tools
    packages=["skyevals"]
    + [f"skyevals.{pkg}" for pkg in setuptools.find_packages(where="skythought/tools")],
)
