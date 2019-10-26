import socket
import struct
import json
from .data_output_stream import DataOutputStream
from .data_input_stream import DataInputStream

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 5800        # The port used by the server


def main():

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        name = 'CalbiFala'
        s.sendall(struct.pack('>i', len(name.encode('utf-8'))))
        s.sendall(name.encode('utf-8'))

        # Number of bytes to read for the board state
        header = bytes()
        for x in range(0, 4):
            header += s.recv(1)
        l = int.from_bytes(header, byteorder='big')
        # Reading the board state
        data = ''
        for x in range(0, l):
            data += s.recv(1).decode('utf-8')
        print(data)
