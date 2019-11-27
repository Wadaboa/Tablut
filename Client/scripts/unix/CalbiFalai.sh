#!/bin/bash

_SRC_PATH="../../"
_ENVIRONMENT_PATH="venv"
source "$_SRC_PATH"/"$_ENVIRONMENT_PATH"/bin/activate

tablut_player $1 -t $2 -s $3

deactivate
