python -m venv venv
source venv/bin/activate
brew/dnf install ffmpeg
pip install -e .
pip install -r requirements.txt
py.test
flake8
