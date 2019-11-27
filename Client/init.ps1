py -3.7 -m venv venv --clear
.\venv\Scripts\activate

pip install wheel
pip install -r .\player\requirements.txt
pip install .\player\

deactivate
