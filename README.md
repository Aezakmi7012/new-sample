# 📦 Retrieval Pipeline Setup

# Install uv (package manager)
pip install uv

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Create README
echo "# Retrieval Pipeline" > README.md

# Install project in editable mode
uv pip install -e .

# Install pre-commit hooks
uv pip install pre-commit

# Navigate to source directory
cd src

# Run the application
python -m retrieval_pipeline.main

# For Docling
sudo apt-get update
sudo apt-get install -y libgl1