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
    "numba"
]

setup_requirements = [
    "plotly",
    "pandas",
    "numpy",
    "scipy",
    "numba"
]

test_requirements = [ ]

setup(
    author="Julian Lehrer",
    author_email='jmlehrer@ucsc.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A library for computing statistical band depth for multivariate time series data",
    install_requires=[
    "plotly",
    "pandas",
    "numpy",
    "scipy",
    "numba"],
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='statdepth',
    name='statdepth',
    packages=find_packages(exclude=['tests']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/braingeneers/functional_depth_methods',
    version='0.7.1',
    zip_safe=False,
)
