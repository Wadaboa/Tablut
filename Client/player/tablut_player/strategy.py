'''
Tablut players strategies
'''


import math
import time
import random

import tablut_player.game_utils as gutils
import tablut_player.utils as utils
from tablut_player.board import TablutBoard
from tablut_player.game_utils import (
    TablutBoardPosition,
    TablutPawnType,
    TablutPlayerType
)
from tablut_player.utils import INF


BLACK_BEST_POSITIONS = {
    TablutBoardPosition(row=2, col=1),
    TablutBoardPosition(row=1, col=2),
    TablutBoardPosition(row=1, col=6),
    TablutBoardPosition(row=2, col=7),
    TablutBoardPosition(row=6, col=1),
    TablutBoardPosition(row=7, col=2),
    TablutBoardPosition(row=6, col=7),
    TablutBoardPosition(row=7, col=6)
}

# ______________________________________________________________________________
# AlphaBetaTree Search


def alphabeta_cutoff_search(state, game, eval_fn, cutoff_test, timeout, max_depth=4):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    player = game.to_move(state)

    # Functions used by alphabeta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(game, state, depth, max_depth, start_time, timeout):
            return eval_fn(game.turn, state, gutils.other_player(state.to_move)), 1
        total = 0
        for a in game.actions(state):
            new_alpha, index = min_value(game.result(state, a),
                                         alpha, beta, depth + 1)
            total += index
            alpha = max(alpha, new_alpha)
            if alpha >= beta:
                return beta, total
            #alpha = max(alpha, v)
        # print(f'Analized {index} moves')
        # print(depth)
        return alpha, total

    def min_value(state, alpha, beta, depth):
        if cutoff_test(game, state, depth, max_depth, start_time, timeout):
            return eval_fn(game.turn, state, gutils.other_player(state.to_move)), 1
        total = 0
        for a in game.actions(state):
            new_beta, index = max_value(game.result(state, a),
                                        alpha, beta, depth + 1)
            total += index
            beta = min(beta, new_beta)
            if alpha >= beta:
                return alpha, total
            #beta = min(beta, v)
        # print(f'Analized {total} moves')
        # print(depth)
        return beta, total

    # Body of alphabeta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    start_time = time.time()
    best_score = -INF
    beta = INF
    best_action = None
    big_total = 0
    print(f'Ricerca Mosse:')
    for a in game.actions(state):
        v, total = max_value(game.result(state, a), best_score, beta, 1)
        big_total += total
        print(f'Mossa analizzata {a}')
        if v > best_score:
            print(
                f'Old best score {best_score}, new best score {v}\n'
                f'Old best action {best_action}, new best action {a}\n'
                f'Total moves Analized: {total} \n'
            )
            best_score = v
            best_action = a
    print(f' The total amount of moves analized is {big_total}')
    return best_action

# ______________________________________________________________________________
# Monte Carlo Tree Search


def monte_carlo_tree_search(game, state, eval_fn, cutoff_test, timeout, itmax=1000):
    def select(node):
        '''
        Select a leaf node in the tree
        '''
        if node.children:
            return select(max(node.children.keys(), key=ucb))
        else:
            return node

    def expand(node):
        '''
        Expand the leaf node by adding all its children states
        '''
        if not node.children and not game.terminal_test(node.state):
            node.children = {
                MCTNode(state=game.result(node.state, action), parent=node):
                action for action in game.actions(node.state)
            }
        return select(node)

    def simulate(game, state):
        '''
        Simulate the utility of current state by random picking a step
        '''
        player = game.to_move(state)
        while not game.terminal_test(state):
            print(state)
            action = random.choice(list(game.actions(state)))
            state = game.result(state, action)
        return -eval_fn(game.turn, state, player)

    def backprop(node, value):
        '''
        Passing the utility back to all parent nodes
        '''
        if value > 0:
            node.U += value
        # if utility == 0:
        #     n.U += 0.5
        node.N += 1
        if node.parent is not None:
            backprop(node.parent, -value)

    start_time = time.time()
    root = MCTNode(state=state)
    for it in range(itmax):
        if not cutoff_test(start_time, timeout, it, itmax):
            leaf = select(root)
            child = expand(leaf)
            result = simulate(game, child.state)
            backprop(child, result)

    max_state = max(root.children, key=lambda p: p.N)
    return root.children.get(max_state)

# ______________________________________________________________________________
# Monte Carlo tree node and ucb function


class MCTNode:
    '''
    Node in the Monte Carlo search tree, keeps track of the children states
    '''

    def __init__(self, state=None, parent=None, U=0, N=0):
        self.__dict__.update(state=state, parent=parent, U=U, N=N)
        self.children = {}
        self.actions = None


def ucb(node, C=1.4):
    return (
        INF if node.N == 0
        else node.U / node.N + C * math.sqrt(math.log(node.parent.N) / node.N)
    )

# ______________________________________________________________________________


def get_move(game, state, timeout, max_depth=3):
    #move = None
    #move = alphabeta_player(game, state, timeout, max_depth)
    move = monte_carlo_player(game, state, timeout)
    if move is None:
        print('Alphabeta failure')
        move = random_player(state)
    return move


def random_player(state):
    '''
    Return a random legal move
    '''
    return utils.get_from_set(state.moves)


def alphabeta_player(game, state, timeout, max_depth=2):
    '''
    Computes the alphabeta search best move
    '''
    return alphabeta_cutoff_search(
        state, game, heuristic, alphabeta_cutoff_test, timeout, max_depth
    )


def monte_carlo_player(game, state, timeout, itmax=1000):
    return monte_carlo_tree_search(
        game, state, heuristic, montecarlo_cutoff_test, timeout, itmax
    )


def alphabeta_cutoff_test(game, state, depth, max_depth, start_time, timeout):
    '''
    Check if the given state in the search tree is a final state,
    if the search can continue or must end because the given timeout expired,
    if the depth of the tree as reached the given maximum
    '''
    return (
        depth > max_depth or
        game.terminal_test(state) or
        not in_time(start_time, timeout)
    )


def montecarlo_cutoff_test(start_time, timeout, it, itmax=1000):
    return (
        it > itmax or
        not in_time(start_time, timeout)
    )


def in_time(start_time, timeout):
    return time.time() - start_time < timeout


def heuristic(turn, state, player):
    if player == TablutPlayerType.WHITE:
        return white_heuristic(turn, state)
    return black_heuristic(turn, state)


def dumb_heuristic(state, player):
    '''
    Return only the piece difference count between the players
    '''
    return TablutBoard.piece_difference_count(
        state.pawns, player
    )


def black_heuristic(turn, state):
    '''
    Black player heuristic function
    '''
    value = blocking_escapes_positions_count(state.pawns)
    if turn < 40:
        value -= king_moves_to_goals_count(state.pawns)
        value += TablutBoard.piece_difference_count(
            state.pawns, TablutPlayerType.BLACK
        )
        value += potential_king_killers_count(state.pawns)
    elif turn < 70:
        value += 2 * TablutBoard.piece_difference_count(
            state.pawns, TablutPlayerType.BLACK
        )
        value -= 1.5 * king_moves_to_goals_count(state.pawns)
        value += 6 * potential_king_killers_count(state.pawns)
    else:
        value += 3 * TablutBoard.piece_difference_count(
            state.pawns, TablutPlayerType.BLACK
        )
        value -= 1.5 * king_moves_to_goals_count(state.pawns)
        value += 12 * potential_king_killers_count(state.pawns)
    return value + random_perturbation()


def white_heuristic(turn, state):
    '''
    White player heuristic function
    '''
    value = TablutBoard.piece_difference_count(
        state.pawns, TablutPlayerType.WHITE
    )
    if turn < 40:
        value += king_moves_to_goals_count(state.pawns)
    elif turn < 70:
        value += 2 * king_moves_to_goals_count(state.pawns)
    else:
        value += 3 * king_moves_to_goals_count(state.pawns)
    return value + random_perturbation()


def king_moves_to_goals_count(pawns):
    '''
    Return a value representing the number of king moves to every goal.
    Given a state, it checks the min number of moves to each goal,
    and return a positive value if we are within 1-2 moves
    to a certain goal, and an even higher value
    if we are within 1-2 moves to more than one corner
    '''
    king = TablutBoard.king_position(pawns)
    if king is None:
        return -INF
    total = 0.0
    for goal in TablutBoard.WHITE_GOALS:
        distance = TablutBoard.simulate_distance(pawns, king, goal)
        if distance == 0:
            total += INF
        elif distance == 1:
            total += 15
        elif distance == 2:
            total += 3  # 1
    return total


def blocking_escapes_positions_count(pawns):
    '''
    Return a value representing the number of black pawns in
    each of the two board spaces diagonally closest to each corner
    '''
    total = 0.0
    inc = 1 / len(BLACK_BEST_POSITIONS)
    for pawn in pawns[TablutPawnType.BLACK]:
        if pawn in BLACK_BEST_POSITIONS:
            total += inc
    return total


def potential_king_killers_count(pawns):
    '''
    Return a value representing the number of black pawns,
    camps and castle around the king.
    '''
    if TablutBoard.king_position(pawns) is None:
        return INF
    killers = TablutBoard.potential_king_killers(pawns)
    return killers * 0.25


def board_coverage_count(state):
    '''
    Return a value representing the total number of moves available
    from the new position. Moves that increase board coverage and
    moves that decrease enemy's board coverage are favored.
    '''
    pass


def random_perturbation(r=0.1):
    '''
    Return a random number in the symmetric interval [-r, r]
    '''
    return random.uniform(-r, r)
