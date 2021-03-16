"""Setup.py for standard project."""
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line]

extra_requirements = {}

with open(path.join(here, "requirements_dev.txt"), encoding="utf-8") as f:
    extra_requirements["dev"] = [line.strip() for line in f if line]

setup(
    name="ayto",
    version="0.1",
    packages=["ayto"],
    python_requires=">=3.8",
    description="Are You The One Solver",
    url="https://github.com/nbgit10/ayto",
    platforms="any",
    install_requires=requirements,
    extras_require=extra_requirements,
)
