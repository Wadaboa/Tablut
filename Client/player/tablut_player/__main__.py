'''
Tablut player entry point
'''


import argparse
import sys
import threading
import traceback
import time
from multiprocessing import JoinableQueue

from PyQt5 import QtWidgets

import tablut_player.config as conf
import tablut_player.game_utils as gutils
import tablut_player.strategy as strat
import tablut_player.genetic as gen
from tablut_player.board import TablutBoardGUI
from tablut_player.game import TablutGame
from tablut_player.strategy import get_move
from tablut_player.connector import Connector


PLAYERS = {
    'random': strat.random_player,
    'minimax': strat.minimax_player,
    'negamax': strat.negamax_player,
    'negascout': strat.negascout_player,
    'montecarlo': strat.monte_carlo_player
}


def required_length(nmin, nmax):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not nmin <= len(values) <= nmax:
                msg = (
                    f'{self.dest} requires between {nmin} and {nmax} arguments'
                )
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength


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
        '-a', '--autoplay', dest='autoplay', action='store_true',
        help="avoid connecting to the server"
    )
    parser.add_argument(
        '-p', '--players', dest='players', action=required_length(1, 2),
        choices=PLAYERS.keys(), nargs='+', default=['minimax'],
        help="choose the players you want to play with", type=str.lower
    )
    parser.add_argument(
        '-g', '--genetic', dest='genetic', nargs=2, type=int,
        help=(
            "train tablut player using a genetic algorithm, "
            "given the number of generations and that of players, in order"
        )
    )
    parser.add_argument(
        '-d', '--debug', dest='debug', action='store_true',
        help='run the command in debug mode'
    )
    args = parser.parse_args()
    conf.MOVE_TIMEOUT = int(args.timeout)
    conf.SERVER_IP = args.server_ip
    conf.DEBUG = args.debug
    conf.TRAIN = False if args.genetic is None else True
    if conf.TRAIN:
        conf.GEN_GENERATIONS = args.genetic[0]
        conf.GEN_POPULATION = args.genetic[1]
    conf.AUTOPLAY = args.autoplay
    conf.WHITE_PLAYER = PLAYERS[args.players[0]]
    conf.MY_PLAYER = conf.WHITE_PLAYER
    conf.BLACK_PLAYER = (
        PLAYERS[args.players[1]] if len(args.players) > 1 else conf.WHITE_PLAYER
    )
    conf.PLAYER_ROLE = args.role
    if args.role == conf.BLACK_ROLE:
        conf.PLAYER_SERVER_PORT = conf.BLACK_SERVER_PORT
        conf.BLACK_PLAYER = PLAYERS[args.players[0]]
        conf.MY_PLAYER = conf.BLACK_PLAYER
        conf.WHITE_PLAYER = (
            PLAYERS[args.players[1]] if len(
                args.players) > 1 else conf.WHITE_PLAYER
        )


def entry():
    '''
    Tablut package entry point
    '''
    parse_args()
    strat.init_openings()
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
    elif conf.TRAIN:
        gen.genetic_algorithm(
            ngen=conf.GEN_GENERATIONS, pop_number=conf.GEN_POPULATION
        )
    else:
        play()
    sys.exit()


def autoplay(gui):
    '''
    Play a game with the specified players and view it in the given GUI
    '''
    game = TablutGame()
    game_state = game.initial
    update_gui(gui, game_state.pawns)
    white_ttable = strat.TT()
    black_ttable = strat.TT()
    while not game.terminal_test(game_state):
        if game.turn % 10 == 0:
            black_ttable.clear()
            white_ttable.clear()
        game.inc_turn()
        print(f'Turn {game.turn}')
        white_move = get_move(
            game, game_state, conf.WHITE_PLAYER, prev_move=None,
            timeout=conf.MOVE_TIMEOUT - conf.MOVE_TIME_OVERHEAD,
            max_depth=4, tt=white_ttable, max_it=1000
        )
        game_state = game.result(game_state, white_move)
        update_gui(gui, game_state.pawns)
        if game.terminal_test(game_state):
            break
        black_move = get_move(
            game, game_state, conf.BLACK_PLAYER, prev_move=None,
            timeout=conf.MOVE_TIMEOUT - conf.MOVE_TIME_OVERHEAD,
            max_depth=4, tt=black_ttable, max_it=1000
        )
        game_state = game.result(game_state, black_move)
        update_gui(gui, game_state.pawns)
    winner = game.utility(
        game_state, gutils.from_player_role_to_type(conf.PLAYER_ROLE)
    )
    print('WIN' if winner == 1 else 'LOSE' if winner == -1 else 'DRAW')


def play():
    '''
    Play a game connecting to the server
    '''
    game = TablutGame()
    game_state = game.initial
    ttable = strat.TT()
    enemy_move = None
    try:
        state_queue = JoinableQueue(2)
        action_queue = JoinableQueue(1)
        conn = Connector(
            conf.SERVER_IP,
            conf.PLAYER_SERVER_PORT,
            conf.PLAYER_NAME,
            state_queue, action_queue,
            gutils.is_black(conf.PLAYER_ROLE)
        )
        conn.start()
        get_state(state_queue)
        if gutils.is_black(conf.PLAYER_ROLE):
            pawns, _ = get_state(state_queue)
            enemy_move = gutils.from_pawns_to_move(
                game_state.pawns, pawns, game_state.to_move
            )
            game_state = game.result(game_state, enemy_move)
        while not game.terminal_test(game_state):
            game.inc_turn()
            print(f'Turn {game.turn}')
            my_move = get_move(
                game, game_state, conf.MY_PLAYER, prev_move=None,
                timeout=conf.MOVE_TIMEOUT - conf.MOVE_TIME_OVERHEAD,
                max_depth=4, tt=ttable, max_it=1000
            )
            game_state = game.result(game_state, my_move)
            action_queue.put((my_move, game_state.to_move))
            action_queue.join()
            get_state(state_queue)
            if game.terminal_test(game_state):
                break
            pawns, _ = get_state(state_queue)
            enemy_move = gutils.from_pawns_to_move(
                game_state.pawns, pawns, game_state.to_move
            )
            game_state = game.result(game_state, enemy_move)
    except Exception:
        print(traceback.format_exc())
    finally:
        conn.terminate()
        conn.join()
    winner = game.utility(
        game_state, gutils.from_player_role_to_type(conf.PLAYER_ROLE)
    )
    print('WIN' if winner == 1 else 'LOSE' if winner == -1 else 'DRAW')


def update_gui(gui, pawns):
    if gui is not None:
        gui.set_pawns(pawns)


def get_state(queue):
    elem = queue.get()
    queue.task_done()
    if not isinstance(elem, tuple):
        raise elem
    return elem


def test_state():
    '''
    State used to test the game
    '''
    initial_pawns = {
        gutils.TablutPawnType.WHITE: {
            gutils.TablutBoardPosition.create(6, 1),
            gutils.TablutBoardPosition.create(7, 1),
            gutils.TablutBoardPosition.create(7, 8),
            gutils.TablutBoardPosition.create(2, 4),
            gutils.TablutBoardPosition.create(3, 4),
            gutils.TablutBoardPosition.create(4, 5),
            gutils.TablutBoardPosition.create(4, 6),
            gutils.TablutBoardPosition.create(5, 4)
        },
        gutils.TablutPawnType.BLACK: {
            gutils.TablutBoardPosition.create(5, 2),
            gutils.TablutBoardPosition.create(6, 3),
            gutils.TablutBoardPosition.create(6, 4),
            gutils.TablutBoardPosition.create(6, 5),
            gutils.TablutBoardPosition.create(6, 7),
            gutils.TablutBoardPosition.create(7, 6),
            gutils.TablutBoardPosition.create(8, 5),
            gutils.TablutBoardPosition.create(8, 4),
            gutils.TablutBoardPosition.create(7, 4),
            gutils.TablutBoardPosition.create(8, 8),
            gutils.TablutBoardPosition.create(4, 8),
            gutils.TablutBoardPosition.create(3, 8),
            gutils.TablutBoardPosition.create(1, 2),
            gutils.TablutBoardPosition.create(0, 4),
            gutils.TablutBoardPosition.create(4, 1),
            gutils.TablutBoardPosition.create(0, 2)
        },
        gutils.TablutPawnType.KING: {gutils.TablutBoardPosition.create(4, 4)}
    }
    player = gutils.TablutPlayerType.WHITE
    return gutils.TablutGameState(
        player,
        0,
        initial_pawns,
        moves=TablutGame.player_moves(initial_pawns, player)
    )
