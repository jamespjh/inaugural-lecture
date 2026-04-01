python -m venv venv
source venv/bin/activate
brew/dnf install ffmpeg
pip install -e .
pip install -e '.[dev]'
py.test
flake8
