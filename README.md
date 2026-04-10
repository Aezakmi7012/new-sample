# 📦 Retrieval Pipeline Setup

# READ THIS FIRST 
Read env example and give out necessary config

# Install uv (package manager)
pip install uv

# Create virtual environment
uv venv

# Activate virtual environment for bash/linux
source .venv/bin/activate

# Install project in editable mode
uv pip install -e .

# Install pre-commit hooks
uv pip install pre-commit
pre-commit install
pre-commit run

# Run the application
python -m retrieval_pipeline.main

# For Docling (codespaces didnt have it so...) PS: Ignore if you dont face this erroe
sudo apt-get update
sudo apt-get install -y libgl1