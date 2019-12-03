'''
Tablut player entry point
'''


import argparse
import sys
import threading
import traceback
import timeit
import time
from multiprocessing import JoinableQueue

from PyQt5 import QtWidgets

import tablut_player.config as conf
import tablut_player.game_utils as gutils
import tablut_player.strategy as strat
import tablut_player.genetic as gen
import tablut_player.heuristic as heu
from tablut_player.board import TablutBoardGUI, TablutBoard
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
        default=conf.GIVEN_MOVE_TIMEOUT
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
    conf.GIVEN_MOVE_TIMEOUT = int(args.timeout)
    conf.MOVE_TIMEOUT = conf.GIVEN_MOVE_TIMEOUT - conf.MOVE_TIME_OVERHEAD
    conf.SERVER_IP = args.server_ip
    conf.DEBUG = args.debug
    conf.TRAIN = args.genetic is not None
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
    if conf.DEBUG:
        print(game_state)
        heu.print_heuristic(game, game_state)
    update_gui(gui, game_state.pawns)
    white_ttable = strat.TT()
    black_ttable = strat.TT()
    heu_tt = strat.TT()
    while not game_state.is_terminal:
        if game.turn % 10 == 0:
            heu_tt.clear()
        if game.turn % 5 == 0:
            black_ttable.clear()
            white_ttable.clear()
        game.inc_turn()
        if conf.DEBUG:
            print(f'Turn {game.turn}')
        white_move = get_move(
            game, game_state, conf.WHITE_PLAYER, prev_move=None,
            timeout=conf.MOVE_TIMEOUT, max_depth=4, tt=white_ttable,
            heu_tt=heu_tt, max_it=1000
        )
        if conf.DEBUG:
            print(f'White move: {white_move}')
        game_state = game.result(game_state, white_move)
        if conf.DEBUG:
            print(game_state)
            heu.print_heuristic(game, game_state)
        update_gui(gui, game_state.pawns)
        if game_state.is_terminal:
            break
        black_move = get_move(
            game, game_state, conf.WHITE_PLAYER, prev_move=None,
            timeout=conf.MOVE_TIMEOUT, max_depth=4, tt=black_ttable,
            heu_tt=heu_tt, max_it=1000
        )
        if conf.DEBUG:
            print(f'Black move: {black_move}')
        game_state = game.result(game_state, black_move)
        if conf.DEBUG:
            print(game_state)
            heu.print_heuristic(game, game_state)
        update_gui(gui, game_state.pawns)
    if game_state.is_terminal:
        winner = game.utility(
            game_state, gutils.from_player_role_to_type(conf.PLAYER_ROLE)
        )
        print('WIN' if winner == 1 else 'LOSE' if winner == -1 else 'DRAW')
    else:
        print('ERROR')


def play():
    '''
    Play a game connecting to the server
    '''
    game = TablutGame()
    game_state = game.initial
    if conf.DEBUG:
        print(game_state)
        heu.print_heuristic(game, game_state)
    ttable = strat.TT()
    heu_tt = strat.TT()
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
            if conf.DEBUG:
                print(f'Enemy move: {enemy_move}')
            game_state = game.result(game_state, enemy_move)
            if conf.DEBUG:
                print(game_state)
                heu.print_heuristic(game, game_state)
        elapsed_time = 0
        while not game_state.is_terminal:
            if game.turn % 10 == 0:
                heu_tt.clear()
            if game.turn % 5 == 0:
                ttable.clear()
            game.inc_turn()
            if conf.DEBUG:
                print(f'Turn {game.turn}')
            conf.MOVE_TIMEOUT = (
                conf.GIVEN_MOVE_TIMEOUT - conf.MOVE_TIME_OVERHEAD - elapsed_time
            )
            my_move = get_move(
                game, game_state, conf.MY_PLAYER, prev_move=None,
                timeout=conf.MOVE_TIMEOUT, max_depth=4, tt=ttable,
                heu_tt=heu_tt, max_it=1000
            )
            if conf.DEBUG:
                print(f'My move: {my_move}')
            start_time = timeit.default_timer()
            action_queue.put((my_move, game_state.to_move))
            action_queue.join()
            get_state(state_queue)
            game_state = game.result(game_state, my_move)
            elapsed_time = timeit.default_timer() - start_time
            if conf.DEBUG:
                print(game_state)
                heu.print_heuristic(game, game_state)
            if game_state.is_terminal:
                break
            pawns, _ = get_state(state_queue)
            enemy_move = gutils.from_pawns_to_move(
                game_state.pawns, pawns, game_state.to_move
            )
            if conf.DEBUG:
                print(f'Enemy move: {enemy_move}')
            game_state = game.result(game_state, enemy_move)
            if conf.DEBUG:
                print(game_state)
                heu.print_heuristic(game, game_state)
    except Exception:
        print(traceback.format_exc())
    finally:
        conn.terminate()
        conn.join()

    if game_state.is_terminal:
        winner = game.utility(
            game_state, gutils.from_player_role_to_type(conf.PLAYER_ROLE)
        )
        if conf.DEBUG:
            print('WIN' if winner == 1 else 'LOSE' if winner == -1 else 'DRAW')
    else:
        if conf.DEBUG:
            print('ERROR')


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
            gutils.TablutBoardPosition.create(6, 4),
            gutils.TablutBoardPosition.create(5, 4),
            gutils.TablutBoardPosition.create(5, 6),
            gutils.TablutBoardPosition.create(4, 3),
            gutils.TablutBoardPosition.create(4, 2),
            gutils.TablutBoardPosition.create(2, 5),

        },
        gutils.TablutPawnType.BLACK: {
            gutils.TablutBoardPosition.create(3, 0),
            gutils.TablutBoardPosition.create(3, 8),
            gutils.TablutBoardPosition.create(8, 5),

        },
        gutils.TablutPawnType.KING: {gutils.TablutBoardPosition.create(4, 4)}
    }
    player = gutils.TablutPlayerType.WHITE
    return gutils.TablutGameState(
        player,
        0,
        initial_pawns,
        is_terminal=False,
        moves=TablutGame.player_moves(initial_pawns, player),
        old_state=None
    )
