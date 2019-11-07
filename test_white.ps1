
cd Server\Tablut\
ant clean
ant compile
Start-Process ant server
cd ..\..\Client\
venv\Scripts\activate
python -m pip install --user player\

$White = @"
"White" "-d"
"@

$Black = @""Black""@

Start-Process tablut_player -ArgumentList $White
Start-Process tablut_player -ArgumentList $Black

cd ..\Server\Tablut\



