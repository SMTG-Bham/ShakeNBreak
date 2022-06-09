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
    version="0.2.3",
    description="Package to generate and analyse distorted defect structures, in order to "
    "identify ground-state and metastable defect configurations.",
    author="Irea Mosquera, Seán Kavanagh",
    author_email="irea.lois.20@ucl.ac.uk",
    packages=find_packages(),
    license="MIT",
    install_requires=[
        "doped>=0.0.5",
        "numpy",
        "pymatgen",
        "matplotlib",
        "ase",
        "pandas",
        "seaborn",
        "hiphive",
        "monty",
    ],
    extras_require={"tests": ["pytest", "pytest-mpl"]},
    package_data={"shakenbreak": ["shakenbreak/*"] + data_files},
    include_package_data=True,
)
