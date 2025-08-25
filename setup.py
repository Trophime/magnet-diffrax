#!/usr/bin/env python3
"""
Setup script for magnet_diffrax package
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Magnetic coupling simulation package for RL circuits with adaptive PID control"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'jax>=0.4.0',
        'jaxlib>=0.4.0',
        'diffrax>=0.4.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
    ]

setup(
    name="magnet_diffrax",
    version="0.1.0",
    author="Magnet Diffrax Team",
    author_email="team@magnetdiffrax.com",
    description="Magnetic coupling simulation for RL circuits with adaptive PID control using JAX and Diffrax",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/magnetdiffrax/magnet_diffrax",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-mock>=3.10.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'isort>=5.10.0',
            'mypy>=0.950',
        ],
        'docs': [
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
            'sphinxcontrib-napoleon>=0.7',
        ]
    },
    entry_points={
        'console_scripts': [
            'magnet-diffrax=magnet_diffrax.main:main',
            'magnet-diffrax-coupled=magnet_diffrax.coupled_main:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
