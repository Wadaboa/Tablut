Set-Location $PSScriptRoot
$_SRC_PATH = "..\..\"
Set-Location $_SRC_PATH
$_ENVIRONMENT_PATH = "venv"
py -3.7 -m venv $_ENVIRONMENT_PATH --clear
& ".\$_ENVIRONMENT_PATH\Scripts\activate"

& {
    pip install wheel
    pip install -r player\requirements.txt
    pip install player\
}

deactivate
