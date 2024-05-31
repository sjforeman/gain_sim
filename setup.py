import sys

from setuptools import setup, find_packages, Extension

import numpy as np

# Enable OpenMP support if available
if sys.platform == "darwin":
    compile_args = []
    link_args = []
else:
    compile_args = ["-fopenmp"]
    link_args = ["-fopenmp"]

setup(
    name="gain_sim",
    version=24.6,
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    author="Simon Foreman",
    description="Code for simple gain-variation simulations",
)
