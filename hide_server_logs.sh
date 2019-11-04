#!/bin/bash

logs='logs/'
garbage='garbage/'
first_line=$(head -n 1 .git/modules/Server/info/exclude)
if [ "$first_line" != "$garbage" ]; then
	>.git/modules/Server/info/exclude
	value=$(<.git/modules/Server/info/exclude)
	echo -e "${logs}\n${value}" > .git/modules/Server/info/exclude
	value=$(<.git/modules/Server/info/exclude)
	echo -e "${garbage}\n${value}" > .git/modules/Server/info/exclude
	echo "Server logs hidden."
else
	echo "Server logs already hidden."
fi
