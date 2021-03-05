# coding=utf-8
# Copyright 2018 Nafise Sadat Moosavi (ns.moosavi at gmail dot com)

"""Setup script for CoVal.
This script allows pip-installing CoVal as a Python module.
"""

import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()


setuptools.setup(
    name="CoVal",  # Replace with your own username
    version="0.0.1",
    author="Nafise Sadat Moosavi, Michael Strube",
    description="CoVal: A coreference evaluation tool for the CoNLL and ARRAU datasets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ns-moosavi/coval",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
    ],
    license="LICENSE",
    python_requires=">=3",
    install_requires=["numpy", "scipy>=0.17.0"])
