#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "plotly",
    "pandas",
    "numpy",
    "scipy",
    "kaleido",
    "tqdm"
]

setup_requirements = requirements.copy()

test_requirements = [ ]

setup(
    author="Julian Lehrer",
    author_email='jmlehrer@ucsc.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A library for computing statistical depth for univariate and multivariate functional data, and pointcloud data. Additionally, methods for homogeneity testing and visualization are provided.",
    install_requires=[
        "plotly",
        "pandas",
        "numpy",
        "scipy",
        "tqdm"
    ],
    license="GPL license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='statdepth',
    name='statdepth',
    packages=find_packages(exclude=['tests', '*tutorial/*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/braingeneers/statdepth',
    version='1.0.0',
    zip_safe=False,
)
