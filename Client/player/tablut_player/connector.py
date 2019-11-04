'''
Module that helps connecting to Tablut server
'''


import struct
import json


def connect(sock, ip_addr, port):
    '''
    Bind the given socket to (ip_addr, port)
    '''
    sock.connect((ip_addr, port))


def send_name(sock, name):
    '''
    Send the given player name to the given server socket
    '''
    return send_str(sock, name)


def send_action(sock, action, turn):
    '''
    Send the given action to the server
    '''
    from_action, to_action = action
    action_dict = {
        "from": from_action,
        "to": to_action,
        "turn": turn[0]
    }
    json_str = json.dumps(action_dict)
    return send_str(sock, json_str)


def receive_state(sock):
    json_str = receive_str(sock)
    json_obj = json.loads(json_str)
    return json_obj["board"], json_obj["turn"]


def receive_int(sock):
    '''
    Read a 4 bytes integer from the given socket
    '''
    header = bytes()
    for _ in range(0, 4):
        header += sock.recv(1)
    return int.from_bytes(header, byteorder='big')


def receive_str(sock):
    '''
    Read a string from the given socket, in UTF-8 format
    '''
    length = receive_int(sock)
    data = ''
    for _ in range(0, length):
        data += sock.recv(1).decode('utf-8')
    return data


def send_int(sock, integer):
    '''
    Send a 4 bytes integer to the given socket
    '''
    bytes_sent = sock.send(struct.pack('>i', integer))
    return bytes_sent == 4


def send_str(sock, string):
    '''
    Send a string to the given socket, in UTF-8 format
    '''
    string = string.encode('utf-8')
    length = len(string)
    if send_int(sock, length):
        bytes_sent = sock.send(string)
        return bytes_sent == length
    return False
