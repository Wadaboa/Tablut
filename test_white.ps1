
ant compile -f .\Server\Tablut\
Start-Process ant server -f .\Server\Tablut\
.\Client\venv\Scripts\activate
if (Get-VirtualEnvName == 'venv') {
    pip install .\Client\player\
    Start-Process tablut_player White
    Start-Process tablut_player Black
}
