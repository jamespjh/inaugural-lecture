uv venv
source .venv/bin/activate
brew/dnf install ffmpeg
uv pip install -e .
uv pip install -e '.[dev]'
py.test
flake8
