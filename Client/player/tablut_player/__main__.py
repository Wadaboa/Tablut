'''
Tablut player entry point
'''


import argparse
import socket
import sys
import threading
import _thread

import tablut_player.config as conf
import tablut_player.connector as conn
import tablut_player.game_utils as gutils
import tablut_player.strategy as strat
import tablut_player.utils as utils
from tablut_player.board import TablutBoardGUI
from tablut_player.game import TablutGame
from tablut_player.strategy import get_move

from PyQt5 import QtCore, QtWidgets

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
    sock = connect()
    if conf.DEBUG:
        app = QtWidgets.QApplication(sys.argv)
        gui_scene = TablutBoardGUI()
        gui_view = QtWidgets.QGraphicsView()
        gui_view.setWindowTitle('Tablut')
        gui_view.setScene(gui_scene)
        gui_view.show()
        thr = threading.Thread(target=play, args=(sock, gui_scene))
        thr.start()
        app.exec_()
        thr.join()
        del gui_view
        del gui_scene
    else:
        play(sock)


def connect():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        conn.connect(
            sock,
            conf.SERVER_IP,
            conf.PLAYER_SERVER_PORT
        )
    except ConnectionRefusedError:
        print('Server is not running. Please, start the server and try again.')
        sys.exit()
    return sock


def play(sock, gui_scene=None):
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
        game.inc_turn()
        print(f'Turn {game.turn}')
        my_move = get_move(game, game_state, conf.MOVE_TIMEOUT - 5)
        game_state = game.result(game_state, my_move)
        print(f'My move: {my_move}')
        print(f'King Heu:{strat.king_moves_to_goals_count(game_state.pawns)}')
        print(f'White Heu:{strat.white_heuristic(game.turn,game_state)}')
        print(f'Black Heu:{strat.black_heuristic(game.turn,game_state)}')
        write_action(sock, my_move, game_state.to_move)
        game.display(game_state)
        _, to_move = read_state(sock)
        if conf.DEBUG:
            gui_scene.set_pawns(game_state.pawns)
        if to_move is None or game.terminal_test(game_state):
            break
        new_pawns, to_move = read_state(sock)
        if conf.DEBUG:
            gui_scene.set_pawns(new_pawns)
        enemy_move = gutils.from_pawns_to_move(
            game_state.pawns, new_pawns, game_state.to_move
        )
        game_state = game.result(game_state, enemy_move)
        print(f'Enemy move: {enemy_move}')
        print(f'King Heu:{strat.king_moves_to_goals_count(game_state.pawns)}')
        print(f'White Heu:{strat.white_heuristic(game.turn,game_state)}')
        print(f'Black Heu:{strat.black_heuristic(game.turn,game_state)}')
        game.display(game_state)
        if to_move is None or game.terminal_test(game_state):
            break
    win = game.utility(
        game_state, gutils.from_player_role_to_type(conf.PLAYER_ROLE)
    )
    print(win)
    sock.close()


def read_state(sock):
    board, to_move = conn.receive_state(sock)
    return gutils.from_server_state_to_pawns(board, to_move)


def write_action(sock, move, to_move):
    action = gutils.from_move_to_server_action(move)
    conn.send_action(sock, action, gutils.from_player_type_to_role(to_move))
