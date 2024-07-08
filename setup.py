#!/usr/bin/env python
import os

from setuptools import find_packages
from setuptools import setup

readme = open("README.rst").read()

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

required_packages = open('requirements.txt').read().splitlines()

setup(
	name='paltas',
	version='0.2.0',
	description='Strong lens substructure package.',
	long_description=readme,
	author='Sebastian Wagner-Carena',
	author_email='sebaswagner@outlook.com',
	url='https://github.com/swagnercarena/paltas',
	packages=find_packages(PACKAGE_PATH),
	package_dir={'paltas': 'paltas'},
	include_package_data=True,
	install_requires=required_packages,
	license='MIT',
	zip_safe=False
)
