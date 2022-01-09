"""Setup.py for standard project."""
from setuptools import setup


common = [
    "mip",
    "numpy",
    "pyyaml",
    "sympy",
]

dev = [
    "black",
    "flake8",
    "pydocstyle",
    "pylint",
    "pytest",
    "pytest-cov",
    "yapf",
]

setup(
    name="ayto",
    version="0.1",
    packages=["ayto"],
    python_requires=">=3.8",
    description="Are You The One Solver",
    url="https://github.com/nbgit10/ayto",
    platforms="any",
    install_requires=common,
    extras_require={
        "dev": dev,
    },
)
