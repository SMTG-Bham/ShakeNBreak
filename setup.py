# -*- coding: utf-8 -*-
"""
This is a setup.py script to install ShakeNBreak
"""

import os
from setuptools import setup, find_packages


# https://stackoverflow.com/questions/27664504/how-to-add-package-data-recursively-in-python-setup-py
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


data_files = package_files("data/")

setup(
    name="shakenbreak",
    version="0.2.4",
    description="Package to generate and analyse distorted defect structures, in order to "
    "identify ground-state and metastable defect configurations.",
    author="Irea Mosquera, Seán Kavanagh",
    author_email="irea.lois.20@ucl.ac.uk",
    packages=find_packages(),
    license="MIT",
    install_requires=[
        "numpy",
        "pymatgen",
        "matplotlib",
        "ase",
        "pandas",
        "seaborn",
        "hiphive",
        "monty",
        "click",
    ],
    extras_require={
        "tests": [
            "pytest",
            "pytest-mpl==0.15.1", # New version 0.16.0 has a bug
        ]
    },
    package_data={"shakenbreak": ["shakenbreak/*"] + data_files},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "snb = shakenbreak.cli:snb",
            "snb-generate = shakenbreak.cli:generate",
            "snb-generate_all = shakenbreak.cli:generate_all",
            "snb-parse = shakenbreak.cli:parse",
            "snb-analyse = shakenbreak.cli:analyse",
            "snb-plot = shakenbreak.cli:plot",
            "snb-regenerate = shakenbreak.cli:regenerate",
            "shakenbreak = shakenbreak.cli:snb",
            "shakenbreak-generate = shakenbreak.cli:generate",
            "shakenbreak-generate_all = shakenbreak.cli:generate_all",
            "shakenbreak-parse = shakenbreak.cli:parse",
            "shakenbreak-analyse = shakenbreak.cli:analyse",
            "shakenbreak-plot = shakenbreak.cli:plot",
            "shakenbreak-regenerate = shakenbreak.cli:regenerate",
        ],
    },
    scripts=["shakenbreak/bash_scripts/*"],
)
