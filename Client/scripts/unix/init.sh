#!/bin/bash

# Python virtual environment
_SRC_PATH="../../"
_ENVIRONMENT_PATH="venv"
python3.7 -m venv "$_SRC_PATH"/"$_ENVIRONMENT_PATH"
. "$_SRC_PATH"/"$_ENVIRONMENT_PATH"/bin/activate

pip install wheel

cd "$_SRC_PATH"/player/
pip install -r requirements.txt
pip install . ## Install the Tablut player library

deactivate
