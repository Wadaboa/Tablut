'''
Tablut states evaluation functions
'''


import random

import tablut_player.game_utils as gutils
from tablut_player.game_utils import (
    TablutBoardPosition,
    TablutPawnType,
    TablutPlayerType
)
from tablut_player.board import TablutBoard
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


def heuristic(turn, state):
    player = gutils.other_player(state.to_move)
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
        value += TablutBoard.piece_difference_count(
            state.pawns, TablutPlayerType.BLACK
        )
        value -= king_moves_to_goals_count(state.pawns)
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
