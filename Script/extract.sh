#!/usr/bin/env bash

python -m virtualenv ./venv --always-copy

source ./venv/bin/activate

pip install -r ./requirements.txt

python ./main.py


