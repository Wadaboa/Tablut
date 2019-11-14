'''
Tablut states evaluation functions
'''


import random

import tablut_player.game_utils as gutils
from tablut_player.game_utils import (
    TablutBoardPosition,
    TablutPawnType,
    TablutPlayerType,
    TablutPawnDirection
)
from tablut_player.board import TablutBoard


CORNERS = {
    (TablutPawnDirection.UP, TablutPawnDirection.LEFT),
    (TablutPawnDirection.UP, TablutPawnDirection.RIGHT),
    (TablutPawnDirection.DOWN, TablutPawnDirection.LEFT),
    (TablutPawnDirection.DOWN, TablutPawnDirection.RIGHT)
}
GOALS = {
    (TablutPawnDirection.UP, TablutPawnDirection.LEFT): [
        TablutBoardPosition(row=2, col=0), TablutBoardPosition(row=0, col=2),
        TablutBoardPosition(row=1, col=0), TablutBoardPosition(row=0, col=1)
    ],
    (TablutPawnDirection.UP, TablutPawnDirection.RIGHT): [
        TablutBoardPosition(row=0, col=6), TablutBoardPosition(row=0, col=7),
        TablutBoardPosition(row=1, col=8), TablutBoardPosition(row=2, col=8)
    ],
    (TablutPawnDirection.DOWN, TablutPawnDirection.LEFT): [
        TablutBoardPosition(row=6, col=0), TablutBoardPosition(row=7, col=0),
        TablutBoardPosition(row=8, col=1), TablutBoardPosition(row=8, col=2)
    ],
    (TablutPawnDirection.DOWN, TablutPawnDirection.RIGHT): [
        TablutBoardPosition(row=8, col=6), TablutBoardPosition(row=8, col=7),
        TablutBoardPosition(row=6, col=8), TablutBoardPosition(row=7, col=8)
    ]
}
BEST_BLOCKING_POSITIONS = {
    (TablutPawnDirection.UP, TablutPawnDirection.LEFT): [
        TablutBoardPosition(row=2, col=1), TablutBoardPosition(row=1, col=2),
    ],
    (TablutPawnDirection.UP, TablutPawnDirection.RIGHT): [
        TablutBoardPosition(row=1, col=6), TablutBoardPosition(row=2, col=7)
    ],
    (TablutPawnDirection.DOWN, TablutPawnDirection.LEFT): [
        TablutBoardPosition(row=6, col=1), TablutBoardPosition(row=7, col=2)
    ],
    (TablutPawnDirection.DOWN, TablutPawnDirection.RIGHT): [
        TablutBoardPosition(row=6, col=7), TablutBoardPosition(row=7, col=6)
    ]
}
OUTER_CORNERS = {
    (TablutPawnDirection.UP, TablutPawnDirection.LEFT): [
        TablutBoardPosition(row=1, col=1)
    ],
    (TablutPawnDirection.UP, TablutPawnDirection.RIGHT): [
        TablutBoardPosition(row=1, col=7)
    ],
    (TablutPawnDirection.DOWN, TablutPawnDirection.LEFT): [
        TablutBoardPosition(row=7, col=1)
    ],
    (TablutPawnDirection.DOWN, TablutPawnDirection.RIGHT): [
        TablutBoardPosition(row=7, col=7)
    ]
}


def heuristic(turn, state):
    '''
    Game state evaluation function
    '''
    values = [blocked_goals(state), piece_difference(
        state), king_moves_to_goals(state), king_killers(state)]
    for value in values:
        if value == -1 or value == 1:
            return value
    if turn < 20:
        value = 3 * blocked_goals(state)
        value += piece_difference(state)
        value += 2 * king_moves_to_goals(state)
        value += 2 * king_killers(state)
        value = value/8
    elif turn < 40:
        value = 2 * blocked_goals(state)
        value += 2 * piece_difference(state)
        value += 3 * king_moves_to_goals(state)
        value += 3 * king_killers(state)
        value = value/10
    else:
        value = blocked_goals(state)
        value += 3 * piece_difference(state)
        value += 6 * king_moves_to_goals(state)
        value += 6 * king_killers(state)
        value = value/16
    #value += random_perturbation()
    return value


def king_moves_to_goals(state):
    '''
    Return a value representing the number of king moves to every goal.
    Given a state, it checks the min number of moves to each goal,
    and return a positive value if we are within 1 to 3 moves
    to a certain goal, and an even higher value if we are within
    1 to 3 moves to more than one corner, in range [-1, 1]
    '''
    max_moves = 4
    upper_bound = 0.8
    distances = []
    player = gutils.other_player(state.to_move)
    value = -(1 / 4)
    check = False
    if TablutBoard.is_king_dead(state.pawns):
        value = -1
    else:
        king = TablutBoard.king_position(state.pawns)
        for goal in TablutBoard.WHITE_GOALS:
            distance = TablutBoard.simulate_distance(
                state.pawns, king, goal, n_moves=0, max_moves=max_moves
            )
            if distance == 0:
                value = 1
                check = True
                break
            if distance < max_moves:
                distances.append(distance)
        if len(distances) > 0 and not check:
            value = 0
            distances.sort()
            for ind, distance in enumerate(distances):
                tmp = (2 ** (-ind - 1)) / distance
                if value + tmp > upper_bound:
                    break
                value += tmp
    if player == TablutPlayerType.BLACK:
        value = -value
    return value


def blocked_goals(state):
    '''
    Return a value representing the number of blocked white goals
    for each corner, in range [-1, 1]
    '''
    total = 0.0
    player = gutils.other_player(state.to_move)
    black_pawns = state.pawns[TablutPawnType.BLACK]
    for corner in CORNERS:
        value = 0.0
        left_goals = set(GOALS[corner])
        for blocking_position in BEST_BLOCKING_POSITIONS[corner]:
            if blocking_position in black_pawns:
                value += 1
                for direction in corner:
                    left_goals.discard(
                        TablutBoard.from_direction_to_pawn(
                            blocking_position, direction
                        )
                    )
        if value == 2:
            total += 4
            continue
        for outer_corner in OUTER_CORNERS[corner]:
            if outer_corner in black_pawns:
                value += 2
                for direction in corner:
                    left_goals.discard(
                        TablutBoard.from_direction_to_pawn(
                            outer_corner, direction
                        )
                    )
        for left_goal in left_goals:
            if left_goal in black_pawns:
                value += 1
        total += value
    if player == TablutPlayerType.WHITE:
        total = -total
    return total * (1 / 16)


def king_killers(state):
    '''
    Return a value representing the number of black pawns,
    camps and castle around the king, in range [-1, 1]
    '''
    value = 0.0
    player = gutils.other_player(state.to_move)
    if TablutBoard.is_king_dead(state.pawns):
        value = 1
    elif TablutBoard.is_king_in_castle(state.pawns):
        value = TablutBoard.potential_king_killers(state.pawns) * (1 / 5)
    elif TablutBoard.is_king_near_castle(state.pawns):
        value = TablutBoard.potential_king_killers(state.pawns) * (1 / 4)
    else:
        value = TablutBoard.potential_king_killers(state.pawns) * (1 / 3)
    if player == TablutPlayerType.WHITE:
        value = -value
    return value


def board_coverage_count(state):
    '''
    Return a value representing the total number of moves available
    from the new position. Moves that increase board coverage and
    moves that decrease enemy's board coverage are favored.
    '''
    pass


def random_perturbation(radius=0.001):
    '''
    Return a random number in the symmetric interval [-radius, radius]
    '''
    return random.uniform(-radius, radius)


def piece_difference(state):
    '''
    Return an evaluation of the pawn difference, in range [-1, 1]
    '''
    player = gutils.other_player(state.to_move)
    diff = (
        (1 / 2) * len(state.pawns[TablutPawnType.BLACK]) -
        len(state.pawns[TablutPawnType.WHITE])
    )
    if player == TablutPlayerType.WHITE:
        diff = -diff
    return diff * (1 / 8)
