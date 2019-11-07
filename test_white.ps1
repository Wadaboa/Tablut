
cd Server\Tablut\
ant clean
ant compile
Start-Process ant server
cd ..\..\Client\
venv\Scripts\activate
python -m pip install 
$Args = @"
"White" "-d"
"@

python -m pip install --user player\
Start-Process tablut_player -ArgumentList $Args
cd ..\
cd ..\Server\Tablut\



