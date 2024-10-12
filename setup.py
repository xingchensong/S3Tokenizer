from pathlib import Path
from setuptools import find_packages, setup

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as file:
        lines = (line.strip() for line in file)
        return [line for line in lines if line and not line.startswith('#')]

setup(
    name="s3tokenizer",
    version="0.0.8",
    description="Reverse Engineering of Supervised Semantic Speech Tokenizer (S3Tokenizer) proposed in CosyVoice",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    author="xingchensong",
    url="https://github.com/xingchensong/S3Tokenizer",
    license="Apache2.0",
    packages=find_packages(),
    install_requires=parse_requirements(Path(__file__).with_name("requirements.txt")),
    entry_points={
        "console_scripts": ["s3tokenizer=s3tokenizer.cli:main"],
    },
    include_package_data=True,
    extras_require={"dev": ["pytest", "scipy", "black", "flake8", "isort"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
)
