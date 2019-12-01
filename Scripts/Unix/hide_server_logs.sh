#!/bin/bash

logs='logs/'
garbage='garbage/'
txt='*.txt'
root='../../'
exc="$root".git/modules/Server/info/exclude
first_line=$(head -n 1 "$exc")
if [ "$first_line" != "$garbage" ]; then
	>"$exc"
	echo -e "${txt}\n" > "$exc"
	value=$(<"$exc")
	echo -e "${logs}\n${value}" > "$exc"
	value=$(<"$exc")
	echo -e "${garbage}\n${value}" > "$exc"
	echo "Server logs hidden."
else
	echo "Server logs already hidden."
fi
