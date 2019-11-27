$_SRC_PATH="..\..\"
$_ENVIRONMENT_PATH="venv"
$_SRC_PATH$_ENVIRONMENT_PATH\Scripts\activate

tablut_player.exe $Args[0] -t $Args[1] -s $Args[2]

deactivate
