#!/bin/bash

# Python virtual environment
_ENVIRONMENT_PATH="venv"
python3.7 -m venv $_ENVIRONMENT_PATH
. "$_ENVIRONMENT_PATH"/bin/activate

pip install wheel

cd player/
pip install -r requirements.txt
pip install . ## Install the Tablut player library
