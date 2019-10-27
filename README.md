# Tablut board game
Software for the Tablut Students Competition at University of Bologna.\
The `Server` side is written in `Java 8`, so you'll need `JDK 8` to use it.\
The `Client` side is written in `Python 3`, so you'll need to install it properly on your system.	

## Installation
To install the client/server softwares and their dependencies follow the guidelines specified in the sub-directories [`Client`](Client/README.md) and [`Server`](https://github.com/AGalassi/TablutCompetition/README.md).

## Server execution
If you have `ANT` installed on your system, just run:
1. `cd` inside `Server/Tablut/`
2. `ant clean`
3. `ant compile`
4. `ant server`

## Client execution
Run the following commands:
1. `cd` into `Client/`
2. Run [`./CalbiFalai.sh`](Client/CalbiFalai.sh), also giving the mandatory parameters
