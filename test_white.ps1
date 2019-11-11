
ant compile -f .\Server\Tablut\
ant server -f .\Server\Tablut\ &
.\Client\venv\Scripts\activate
if (Get-VirtualEnvName == 'venv') {
    pip install .\Client\player\
    tablut_player White &
    tablut_player Black &
}
