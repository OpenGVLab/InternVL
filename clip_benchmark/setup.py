#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()


def load_requirements(f):
    return [l.strip() for l in open(f).readlines()]


requirements = load_requirements('requirements.txt')

test_requirements = requirements + ['pytest', 'pytest-runner']

setup(
    author='Mehdi Cherti',
    author_email='mehdicherti@gmail.com',
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
    description='CLIP-like models benchmarks on various datasets',
    entry_points={
        'console_scripts': [
            'clip_benchmark=clip_benchmark.cli:main',
            'clip_benchmark_export_wds=clip_benchmark.webdataset_builder:main',
        ],
    },
    install_requires=requirements,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='clip_benchmark',
    name='clip_benchmark',
    packages=find_packages(include=['clip_benchmark', 'clip_benchmark.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mehdidc/clip_benchmark',
    version='1.4.0',
    zip_safe=False,
    extra_require={
        'vtab': ['task_adaptation==0.1', 'timm>=0.5.4'],
        'tfds': ['tfds-nightly', 'timm>=0.5.4'],
        'coco': ['pycocotools>=2.0.4'],
    }
)
