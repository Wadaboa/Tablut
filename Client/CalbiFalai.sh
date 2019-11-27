#!/bin/bash

_ENVIRONMENT_PATH="venv"
source "$_ENVIRONMENT_PATH"/bin/activate

tablut_player $1 -t $2 -s $3

deactivate
