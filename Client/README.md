# Tablut client

## Installation

In order to correctly install this Tablut client, do the following:
1. Make sure your current working directory is the `Client/` directory
2. Run `chmod +x init.sh && ./init.sh`
3. Test if the Python virtual environment is active, by running `pip -V`. The correct output should be `Tablut/Client/venv/lib/python3.7/site-packages/pip (python 3.7)`
   * If the check above is not ok, run `source venv/bin/activate`
4. Run `chmod +x CalbiFalai.sh`

## Usage
Run [`./CalbiFalai.sh`](CalbiFalai.sh) with the following 3 mandatory parameters:
1. Role: `White` or `Black`
2. Timeout: Given time to compute each move
3. Server address: IP address where the server is running

For example, you could execute the client with a command like `./CalbiFalai.sh White 60 localhost:8080`.
