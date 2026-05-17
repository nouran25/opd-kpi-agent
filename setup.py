"""Setup script for OPD KPI Agent"""

from setuptools import setup, find_packages

setup(
    name="opd-kpi-agent",
    version="1.0.0",
    description="AI-powered OPD KPI Analytics Agent",
    author="nouran25",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "openpyxl>=3.1.0",
        "langchain>=1.0.0",
        "langchain-groq>=1.0.0",
        "langchain-community>=0.4.0",
        "chromadb>=0.4.0",
        "gradio>=4.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
    ],
)
