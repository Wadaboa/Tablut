'''
Tablut player entry point
'''


import argparse
import socket
import random

import tablut_player.config as conf
import tablut_player.connector as conn
import tablut_player.game_utils as gutils
from tablut_player.game import TablutGame


PLAYER_NAME = 'CalbiFalai'


def parse_args():
    '''
    Parse standard input arguments
    '''
    parser = argparse.ArgumentParser(description='Tablut client player')
    parser.add_argument(
        dest='role',
        choices={conf.WHITE_ROLE, conf.BLACK_ROLE},
        help='tablut player role'
    )
    parser.add_argument(
        '-t', '--timeout', dest='timeout', action='store',
        help='given time to compute each move',
        default=conf.MOVE_TIMEOUT
    )
    parser.add_argument(
        '-s', '--server-ip', dest='server_ip', action='store',
        help='tablut server ip address', default=conf.SERVER_IP
    )
    parser.add_argument(
        '-d', '--debug', dest='debug', action='store_true',
        help='run the command in debug mode'
    )
    args = parser.parse_args()
    conf.TIMEOUT = args.timeout
    conf.SERVER_IP = args.server_ip
    conf.DEBUG = args.debug
    conf.PLAYER_ROLE = args.role
    if args.role == conf.BLACK_ROLE:
        conf.PLAYER_SERVER_PORT = conf.BLACK_SERVER_PORT


def main():
    parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect(
        sock,
        conf.SERVER_IP,
        conf.PLAYER_SERVER_PORT
    )
    conn.send_name(sock, PLAYER_NAME)
    pawns, to_move = read_state(sock)
    if conf.PLAYER_ROLE == conf.BLACK_ROLE:
        pawns, to_move = read_state(sock)
    game = TablutGame(initial_pawns=pawns, to_move=to_move)
    game_state = game.initial
    while True:
        my_move = random.choice(game_state.moves)  # strategia
        game_state = game.result(game_state, my_move)
        print(f'MY MOVE: {my_move}')
        write_action(sock, my_move, game_state.to_move)
        # game.display(game_state)
        if game.terminal_test(game_state):
            break
        _, _ = read_state(sock)
        new_pawns, _ = read_state(sock)
        enemy_move = gutils.from_pawns_to_move(
            game_state.pawns, new_pawns, game_state.to_move
        )
        print(f'ENEMY MOVE: {enemy_move}')
        game_state = game.result(game_state, enemy_move)
        # game.display(game_state)
        if game.terminal_test(game_state):
            break
    win = game.utility(
        game_state, gutils.from_player_role_to_type(conf.PLAYER_ROLE)
    )
    print(win)


def read_state(sock):
    board, turn = conn.receive_state(sock)
    return gutils.from_server_state_to_pawns(board, turn)


def write_action(sock, move, turn):
    action = gutils.from_move_to_server_action(move)
    conn.send_action(sock, action, gutils.from_player_type_to_role(turn))
