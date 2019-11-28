'''
Module that helps connecting to Tablut server
'''

import struct
import json
import socket
from multiprocessing import Process

import tablut_player.game_utils as gutils


class Connector(Process):
    '''
    Create a connector process to handle send and receive
    between player and server
    '''

    def __init__(self, ip_addr, port, player_name,
                 state_queue, action_queue, is_black=False):
        Process.__init__(self, name='ConnectorProcess')
        self.ip_addr = ip_addr
        self.port = port
        self.player_name = player_name
        self.state_queue = state_queue
        self.action_queue = action_queue
        self.is_black = is_black
        self.sock = None

    def run(self):
        try:
            self.sock = self._connect()
            self.send_name()
            state = self.read_state()
            self._put_state(state)
            if self.is_black:
                state = self.read_state()
                self._put_state(state)
            while is_socket_valid(self.sock):
                action = self._get_action()
                self.write_action(action)
                state = self.read_state()
                self._put_state(state)
                if not is_socket_valid(self.sock):
                    break
                state = self.read_state()
                self._put_state(state)
            self.sock.close()
        except ConnectionRefusedError as cre:
            self.state_queue.put(cre)
            self.state_queue.join()
        except Exception as exc:
            self.sock.close()
            self.state_queue.put(exc)
            self.state_queue.join()
        finally:
            self.action_queue.close()
            self.state_queue.close()

    def _connect(self):
        '''
        Estabilish a TCP connection with the server
        '''
        return connect(self.ip_addr, self.port)

    def read_state(self):
        '''
        Read and convert the current game state from the server
        '''
        board, to_move = self.receive_state()
        return gutils.from_server_state_to_pawns(board, to_move)

    def write_action(self, action):
        '''
        Convert and send the player move to the server
        '''
        move, to_move = action
        action = gutils.from_move_to_server_action(move)
        self.send_action(action, gutils.from_player_type_to_role(to_move))

    def send_name(self):
        '''
        Send the given player name to the given server socket
        '''
        return send_str(self.sock, self.player_name)

    def send_action(self, action, turn):
        '''
        Send the given action to the server
        '''
        from_action, to_action = action
        action_dict = {
            "from": from_action,
            "to": to_action,
            "turn": turn
        }
        json_str = json.dumps(action_dict)
        return send_str(self.sock, json_str)

    def receive_state(self):
        '''
        Receive the current state of the game from the server
        '''
        json_str = receive_str(self.sock)
        json_obj = json.loads(json_str)
        return json_obj["board"], json_obj["turn"]

    def _get_action(self):
        '''
        Wait until there is an action in the queue and return it
        '''
        action = self.action_queue.get()
        self.action_queue.task_done()
        return action

    def _put_state(self, state):
        '''
        Wait until there is a free slot in the state queue and put the new state
        '''
        self.state_queue.put(state)
        self.state_queue.join()


def connect(ip_addr, port):
    '''
    Bind a TCP socket to (ip_addr, port)
    '''
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect((ip_addr, port))
    return sock


def is_socket_valid(sock):
    '''
    Return True if this socket is connected
    '''
    if not sock:
        return False

    if sock.fileno() == -1:
        return False

    return True


def receive_int(sock):
    '''
    Read a 4 bytes integer from the given socket
    '''
    size = 4
    header = b''
    while len(header) < size:
        data = sock.recv(size - len(header))
        if not data:
            break
        header += data
    return struct.unpack("!i", header)[0]


def receive_str(sock):
    '''
    Read a string from the given socket, in UTF-8 format
    '''
    length = receive_int(sock)
    return sock.recv(length, socket.MSG_WAITALL).decode('utf-8')


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
