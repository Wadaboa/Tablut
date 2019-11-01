import socket
import struct
import json

import tablut_player


def test():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((tablut_player.SERVER_IP, 5800))
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
