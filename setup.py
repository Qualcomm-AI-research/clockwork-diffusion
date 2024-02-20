# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from pathlib import Path

from setuptools import setup, find_packages

ROOT = Path(__file__).parent.absolute()

# Package meta-data.
NAME = "clockwork"
DESCRIPTION = "Python package implementing Clockwork Diffusion UNet wrapper for the Hugging Face diffusers library."
URL = "https://github.com/Qualcomm-AI-research/clockwork-diffusion"
EMAIL = "{ahabibia, ghodrati, noor, gsautie, rgarrepa, fporikli, jpeterse}@qti.qualcomm.com"
AUTHOR = "Amirhossein Habibian and Amir Ghodrati and Noor Fathima and Guillaume Sautiere and Risheek Garrepalli and Fatih Porikli and Jens Petersen"
REQUIRES_PYTHON = ">=3.9"
VERSION = None
REQUIREMENTS = [
    "accelerate",
    "diffusers",
    "torch",
    "transformers",
]


# version
if VERSION is None:
    # load _version.py module as a dictionary
    module = {}
    with (ROOT / NAME / "_version.py").open() as f:
        exec(f.read(), module)
    VERSION = module["__version__"]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    install_requires=REQUIREMENTS,
    packages=find_packages(exclude=("tests")),
)
