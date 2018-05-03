#!/usr/bin/env python

requirements = [
    'torch>=0.4.0',
    'numpy',
    'scikit-learn',
]

test_requirements = [
    'pytest',
]

from setuptools import setup, find_packages

setup(
    name='torchsample',
    version='0.1.3',
    description='Personal Training, Augmentation, and Sampling for Pytorch',
    author='NC Cullen',
    author_email='nickmarch31@yahoo.com',
    install_requires=requirements,
    packages=find_packages(exclude=['examples', 'tests']),
    license='MIT license',
    classifiers=[
      'Development Status :: 2 - Pre-Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: MIT License',
      'Natural Language :: English',
      'Programming Language :: Python :: 2',
      'Programming Language :: Python :: 2.7',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.4',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements
)