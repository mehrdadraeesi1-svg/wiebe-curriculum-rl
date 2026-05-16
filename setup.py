"""Setup script for RL Wiebe Framework."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name='rl-wiebe-framework',
    version='0.1.0',
    author='Mehrdad Raeesi',
    author_email='mehrdadraeesi@gmail.com',
    description='Reinforcement Learning framework for Wiebe coefficient prediction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/rl-wiebe-framework',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'pandas>=2.0.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'torch>=2.0.0',
        'gymnasium>=0.28.0',
        'stable-baselines3>=2.0.0',
        'numba>=0.57.0',
        'scikit-learn>=1.3.0',
        'tensorboard>=2.13.0',
        'tqdm>=4.65.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.1.0',
            'mypy>=1.5.0',
        ],
        'docs': [
            'sphinx>=7.1.0',
            'sphinx-rtd-theme>=1.3.0',
        ],
        'wandb': [
            'wandb>=0.15.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'rl-wiebe-train=scripts.train_rl_agent:main',
            'rl-wiebe-eval=scripts.evaluate_agent:main',
        ],
    },
)
