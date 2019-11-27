Set-Location $PSScriptRoot
$_SRC_PATH = "..\..\"
Set-Location $_SRC_PATH
$_ENVIRONMENT_PATH = "venv"
& ".\$_ENVIRONMENT_PATH\Scripts\activate"

tablut_player.exe $Args[0] -t $Args[1] -s $Args[2]

deactivate
