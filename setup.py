from setuptools import setup, find_packages

setup(
    name="mpi4dl",
    version="1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
