#!/bin/bash

cd Server/Tablut/
ant clean
ant compile
ant server &

cd ../../Client/
source venv/bin/activate
pip install player/
tablut_player White &

cd ../Server/Tablut/
ant randomblack > /dev/null &