import os

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="neo_force_scheme",
    version="0.0.1",
    author="Leonardo Christino",
    author_email="christinoleo@dal.ca",
    description="NeoForceSceme, an extention of the original ForceScheme with performance improvements",
    license="MIT",
    keywords="gpu numba forcescheme projection dimenstionality reduction",
    url="https://github.com/visml/neo_force_scheme",
    packages=['neo_force_scheme'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: GPU :: NVIDIA CUDA",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
    ],
)
