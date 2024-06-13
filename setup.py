"""Placeholder"""

from setuptools import setup, find_packages

setup(
    name="iot",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click",
        "matplotlib",
        "pandas",
        "rich",
        "scikit-image",
        "seaborn",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "iot=iot.cli:cli",
        ],
    },
)
