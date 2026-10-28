"""Packaging for wiebe-curriculum-rl."""
from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent

setup(
    name="wiebe-curriculum-rl",
    version="1.0.0",
    author="Mehrdad Raeesi, Mohammad Hassan Shojaeefard",
    author_email="mehrdadraeesi@gmail.com",
    description=("Physics-informed curriculum reinforcement learning for real-time "
                 "calibration of double Wiebe combustion models"),
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/mehrdadraeesi1-svg/wiebe-curriculum-rl",
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24,<3", "scipy>=1.10", "numba>=0.57", "pandas>=2.0", "scikit-learn>=1.3",
        "torch>=2.0", "gymnasium>=0.29", "stable-baselines3>=2.0,<3", "matplotlib>=3.7",
        "pyyaml>=6.0", "tqdm>=4.65",
    ],
    extras_require={
        "xgboost": ["xgboost>=2.0"],
        "dev": ["pytest>=7.4", "pytest-cov>=4.1", "xgboost>=2.0"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
