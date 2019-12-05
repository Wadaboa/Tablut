# Tablut client

## Installation

### Linux / MacOS
In order to correctly install this Tablut client, do the following:
1. Make sure your current working directory is the `Client/scripts/unix/` directory
2. Run `chmod +x init.sh && ./init.sh`
3. Run `chmod +x CalbiFalai.sh`

### Windows
In order to correctly install this Tablut client, do the following:
1. Make sure your current working directory is the `Client/scripts/win/` directory
2. Launch a powershell as administrator
3. Run `get-executionpolicy -list` to identify the current execution policy settings
4. Run `set-executionpolicy unrestricted`, if not already set as `Unrestricted`
5. Run `./init.ps1`
6. Run `set-executionpolicy commandlet` to restore execution policy settings

This script has been tested as working with `Powershell 6`.

## Usage
Make sure your current working directory is the `Client/scripts/unix/` (`Client/scripts/win/`) directory and run [`./CalbiFalai.sh`](CalbiFalai.sh) ([`.\CalbiFalai.ps1`](CalbiFalai.ps1)) with the following 3 positional arguments:
1. Role: `White` or `Black`
2. Timeout: Given time to compute each move
3. Server address: IP address where the server is running

If you are on Windows you might need to do the same execution policy setting and restore, as specified above, before and after running the script.

For example, you could execute the client with a command like `./CalbiFalai.sh White 60 127.0.0.1` (`.\CalbiFalai.ps1 White 60 127.0.0.1`).
