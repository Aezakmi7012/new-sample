# Retrieval Pipeline
pip install uv
uv venv
source .venv/bin/activate
echo "# Retrieval Pipeline" > README.md
uv pip install -e .
uv pip install pre-commit

cd src
python -m retrieval_pipeline.main