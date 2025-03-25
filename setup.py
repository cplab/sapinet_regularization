from setuptools import setup, find_packages
from src import __version__

with open("README.md") as f:
    long_description = f.read()

setup(
    name="sapinet_regularization",
    version=__version__,
    description="Heterogeneous quantization regularizes spiking neural network activity",
    long_description=long_description,
    url="https://github.com/cplab/sapinet_regularization",
    author="Roy Moyal, Kyrus R. Mama, Matthew Einhorn, Ayon Borthakur, Thomas A. Cleland",
    author_email="rm875@cornell.edu, krm74@cornell.edu, me263@cornell.edu, ayon.borthakur@iitg.ac.in, tac29@cornell.edu",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    packages=find_packages(),
    install_requires=[
        "sapicore==0.4.0",
        "numpy<2",
        "scipy",
        "pandas",
        "torch",
        "scikit-learn",
        "networkx",
        "matplotlib",
        "seaborn",
        "pyvista",
        "psutil",
        "PyYAML",
        "dill",
        "natsort",
        "h5py",
        "jenkspy",
        "tree-config",
        "alive_progress",
        "importlib-metadata<4.3",
    ],
    extras_require={
        "dev": ["pytest", "coverage", "flake8", "sphinx<7.0.0", "sphinx-rtd-theme", "m2r2"],
    },
)
