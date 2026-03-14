from setuptools import setup, find_packages

setup(
    name="flash_attn_with_score_bp",
    version="0.1.0",
    description="Flash Attention with Scores and Backward Pass Support",
    author="Your Name",
    url="https://github.com/yourusername/Flash_Attn_with_Score_bp_support",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.0.0",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
