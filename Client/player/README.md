# Tablut player
This Tablut player is a `pip` package and it is written in `Python 3.7.4`.\
Please, make sure to install this specific Python version on your system, to ensure compatibility.

## Installation

In order to correctly install this Tablut player follow the guidelines specified 
in the `Client/` parent directory.\
If everything went fine, now you should be able to execute the player 
by running `tablut_player` on your terminal.

## Usage
This Tablut player requires the 3 mandatory parameters upon execution, in the following order:
1. `role`: Tablut player role, `White` or `Black`
2. `timeout`: Integer value, which is the given time to compute each move
3. `server_sock`: String representing the socket (ip, port) where the server is running

It is also possible to specify the following optional parameters:
1. `--debug` or `-d`: Run the command in verbose mode

If you need further help, you can type `tablut_player -h` or `tablut_player --help` in a command line.
