'''
Tablut player entry point
'''


import argparse
import sys
import threading
import queue
import traceback
import time

from PyQt5 import QtCore, QtWidgets
from concurrent.futures import ThreadPoolExecutor

import tablut_player.config as conf
import tablut_player.game_utils as gutils
import tablut_player.strategy as strat
import tablut_player.utils as utils
import tablut_player.heuristic as heu
from tablut_player.board import TablutBoardGUI, TablutBoard
from tablut_player.game import TablutGame
from tablut_player.strategy import get_move
from tablut_player.connector import (Connector, is_socket_valid)


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
    parser.add_argument(
        '-a', '--autoplay', dest='autoplay', action='store_true',
        help="avoid connecting to the server and play locally with both roles"
    )
    args = parser.parse_args()
    conf.MOVE_TIMEOUT = int(args.timeout)
    conf.SERVER_IP = args.server_ip
    conf.DEBUG = args.debug
    conf.AUTOPLAY = args.autoplay
    conf.PLAYER_ROLE = args.role
    if args.role == conf.BLACK_ROLE:
        conf.PLAYER_SERVER_PORT = conf.BLACK_SERVER_PORT


def entry():
    parse_args()
    if conf.AUTOPLAY:
        app = QtWidgets.QApplication(sys.argv)
        gui_scene = TablutBoardGUI()
        gui_view = QtWidgets.QGraphicsView()
        gui_view.setWindowTitle('Tablut')
        gui_view.setScene(gui_scene)
        gui_view.show()
        thr = threading.Thread(
            target=autoplay,
            args=(gui_scene,),
            name='GUIGameManager'
        )
        thr.daemon = True
        thr.start()
        app.exec_()
        del gui_view
        del gui_scene
    else:
        thr = threading.Thread(target=play, name='GameManager')
        thr.start()
        thr.join()
    sys.exit()


def autoplay(gui):
    game = TablutGame()
    game_state = game.initial
    update_gui(gui, game_state.pawns)
    while not game.terminal_test(game_state):
        game.inc_turn()
        white_move = get_move(
            game, game_state, conf.MOVE_TIMEOUT - conf.MOVE_TIME_OVERHEAD
        )
        game_state = game.result(game_state, white_move)
        update_gui(gui, game_state.pawns)
        if game.terminal_test(game_state):
            break
        black_move = get_move(
            game, game_state, conf.MOVE_TIMEOUT - conf.MOVE_TIME_OVERHEAD,
            prev_move=white_move
        )
        game_state = game.result(game_state, black_move)
        update_gui(gui, game_state.pawns)
        # game.display(game_state)
        time.sleep(3)
    winner = game.utility(
        game_state, gutils.from_player_role_to_type(conf.PLAYER_ROLE)
    )
    print(winner)


def play():
    game = TablutGame()
    game_state = game.initial
    try:
        state_queue = queue.Queue(1)
        action_queue = queue.Queue(1)
        exception_queue = queue.Queue(1)
        conn = Connector(
            conf.SERVER_IP,
            conf.PLAYER_SERVER_PORT,
            conf.PLAYER_NAME,
            state_queue,
            action_queue,
            exception_queue,
            gutils.is_black(conf.PLAYER_ROLE)
        )
        conn.start()
        get_state(state_queue, exception_queue)
        if gutils.is_black(conf.PLAYER_ROLE):
            pawns, _ = get_state(state_queue, exception_queue)
        while not game.terminal_test(game_state):
            game.inc_turn()
            print(f'Turn {game.turn}')
            # Not working
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    get_move, game, game_state,
                    conf.MOVE_TIMEOUT - conf.MOVE_TIME_OVERHEAD
                )
                my_move = future.result()
            game_state = game.result(game_state, my_move)

            game.display(game_state)
            print(f'My move: {my_move}')

            action_queue.put((my_move, game_state.to_move))
            get_state(state_queue, exception_queue)  # Get my move
            if game.terminal_test(game_state):
                break

            pawns, _ = get_state(
                state_queue, exception_queue)  # Get enemy move
            enemy_move = gutils.from_pawns_to_move(
                game_state.pawns, pawns, game_state.to_move
            )
            game_state = game.result(game_state, enemy_move)

            game.display(game_state)
            print(f'Enemy move: {enemy_move}')

    except Exception:
        print(traceback.format_exc())
    finally:
        conn.join()
    winner = game.utility(
        game_state, gutils.from_player_role_to_type(conf.PLAYER_ROLE)
    )
    print(winner)
    print('-' * 50)


def update_gui(gui, pawns):
    if gui is not None:
        gui.set_pawns(pawns)


def get_state(state_queue, exception_queue):
    while exception_queue.empty():
        try:
            state = state_queue.get_nowait()
            return state
        except queue.Empty:
            pass
    raise exception_queue.get_nowait()


def test_state():
    '''
    State used to test the game
    '''
    initial_pawns = {
        gutils.TablutPawnType.WHITE: {
            gutils.TablutBoardPosition(6, 4),
            gutils.TablutBoardPosition(5, 4),
            gutils.TablutBoardPosition(8, 2),
            gutils.TablutBoardPosition(4, 5),
            gutils.TablutBoardPosition(2, 0),
            gutils.TablutBoardPosition(0, 6),
            gutils.TablutBoardPosition(4, 2),
            gutils.TablutBoardPosition(3, 4)
        },
        gutils.TablutPawnType.BLACK: {
            gutils.TablutBoardPosition(1, 7),
            gutils.TablutBoardPosition(5, 6),
            gutils.TablutBoardPosition(7, 2),
            gutils.TablutBoardPosition(1, 5),
            gutils.TablutBoardPosition(0, 3),
            gutils.TablutBoardPosition(2, 6),
            gutils.TablutBoardPosition(8, 5),
            gutils.TablutBoardPosition(5, 8),
            gutils.TablutBoardPosition(4, 0),
            gutils.TablutBoardPosition(3, 6),
            gutils.TablutBoardPosition(8, 1),
            gutils.TablutBoardPosition(6, 1),
            gutils.TablutBoardPosition(7, 4),
            gutils.TablutBoardPosition(0, 4),
            gutils.TablutBoardPosition(4, 1),
            gutils.TablutBoardPosition(8, 4)
        },
        gutils.TablutPawnType.KING: {gutils.TablutBoardPosition(4, 4)}
    }
    player = gutils.TablutPlayerType.WHITE
    return gutils.TablutGameState(
        player,
        0,
        initial_pawns,
        moves=TablutGame.player_moves(initial_pawns, player)
    )
