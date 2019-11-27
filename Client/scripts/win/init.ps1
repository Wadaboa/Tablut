$_SRC_PATH="..\..\"
$_ENVIRONMENT_PATH="venv"
py -3.7 -m venv $_SRC_PATH$_ENVIRONMENT_PATH --clear
$_SRC_PATH$_ENVIRONMENT_PATH\Scripts\activate

cd $_SRC_PATH\player\
pip install wheel
pip install -r requirements.txt
pip install .

deactivate
