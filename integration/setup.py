#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="ai-ecosystem-cli",
    version="1.0.0",
    description="CLI tool for AI Ecosystem Integration",
    author="AI Ecosystem",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.7",
        "aiohttp>=3.9.1",
        "python-dateutil>=2.8.2",
    ],
    entry_points={
        "console_scripts": [
            "ai-cli=cli:cli",
        ],
    },
    python_requires=">=3.8",
)