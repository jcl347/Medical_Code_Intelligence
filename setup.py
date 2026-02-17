from setuptools import setup, find_packages

setup(
    name="medical-code-intelligence",
    version="0.1.0",
    description="State-of-the-art Medical/Biomedical Named Entity Recognition",
    author="Medical Code Intelligence",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "accelerate>=0.25.0",
        "seqeval>=1.2.2",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
        "tracking": ["wandb>=0.16.0"],
    },
)
