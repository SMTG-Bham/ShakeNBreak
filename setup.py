# -*- coding: utf-8 -*-
"""
This is a setup.py script to install BDM
"""

from setuptools import setup, find_packages

setup(name='BDM',
      version='0.0',
      description='Collection of python fucntions to generate and analyse distorted defect structures.',
      author='Irea Mosquera',
      author_email='irea.lois.20@ucl.ac.uk',
      py_modules=['BDM', 'plot_BDM', 'analyse_defects', 'champion_defects_rerun'],
      packages=find_packages(),
      license="MIT",
      install_requires=[
        "doped>=0.0.5",
        "numpy",
        "pymatgen",
        "matplotlib",
        "ase",
        "pandas",
    ],
      )
