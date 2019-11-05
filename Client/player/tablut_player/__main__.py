'''
Tablut player entry point
'''


import argparse
import socket
import sys
import threading

from PyQt5 import QtWidgets

import tablut_player.config as conf
import tablut_player.connector as conn
import tablut_player.game_utils as gutils
import tablut_player.utils as utils
from tablut_player.game import TablutGame
from tablut_player.board import TablutBoardGUIScene


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


def entry():
    parse_args()
    if conf.DEBUG:
        app = QtWidgets.QApplication(sys.argv)
        gui_scene = TablutBoardGUIScene()
        gui_view = QtWidgets.QGraphicsView()
        gui_view.setScene(gui_scene)
        gui_view.show()
        thr = threading.Thread(target=play, args=(gui_scene,))
        thr.daemon = True
        thr.start()
        app.exec_()
    else:
        play()


def play(gui_scene=None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect(
        sock,
        conf.SERVER_IP,
        conf.PLAYER_SERVER_PORT
    )
    conn.send_name(sock, PLAYER_NAME)
    pawns, to_move = read_state(sock)
    if conf.DEBUG:
        gui_scene.set_pawns(pawns)
    if conf.PLAYER_ROLE == conf.BLACK_ROLE:
        pawns, to_move = read_state(sock)
        if conf.DEBUG:
            gui_scene.set_pawns(pawns)
    game = TablutGame(initial_pawns=pawns, to_move=to_move)
    game_state = game.initial
    while True:
        my_move = utils.get_from_set(game_state.moves)  # strategia
        game_state = game.result(game_state, my_move)
        write_action(sock, my_move, game_state.to_move)
        game.display(game_state)
        _, turn = read_state(sock)
        if conf.DEBUG:
            gui_scene.set_pawns(game_state.pawns)
        if turn is None or game.terminal_test(game_state):
            break
        new_pawns, turn = read_state(sock)
        if conf.DEBUG:
            gui_scene.set_pawns(new_pawns)
        enemy_move = gutils.from_pawns_to_move(
            game_state.pawns, new_pawns, game_state.to_move
        )
        game_state = game.result(game_state, enemy_move)
        game.display(game_state)
        if turn is None or game.terminal_test(game_state):
            break
    win = game.utility(
        game_state, gutils.from_player_role_to_type(conf.PLAYER_ROLE)
    )
    print(win)
    sock.close()
    sys.exit()


def read_state(sock):
    board, turn = conn.receive_state(sock)
    return gutils.from_server_state_to_pawns(board, turn)


def write_action(sock, move, turn):
    action = gutils.from_move_to_server_action(move)
    conn.send_action(sock, action, gutils.from_player_type_to_role(turn))
