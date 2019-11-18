'''
Tablut states evaluation functions
'''


import random

import tablut_player.utils as utils
import tablut_player.game_utils as gutils
from tablut_player.game_utils import (
    TablutBoardPosition,
    TablutPawnType,
    TablutPlayerType,
    TablutPawnDirection
)
from tablut_player.board import TablutBoard
from tablut_player.game import TablutGame


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
    Game state evaluation function in range [-100,100]
    with 1000, -1000 winning and losing scores
    '''
    values = [
        blocked_goals(state),
        piece_difference(state),
        potential_kills(state),
        king_moves_to_goals(state),
        king_killers(state)
    ]
    if turn < 10:
        weigths = [3, 1, 0.25, 2, 2]
    elif turn < 20:
        weigths = [2, 2, 0.3, 3, 3]
    else:
        weigths = [1, 3, 0.6, 6, 6]
    good_weights = 0
    score = 0
    #print([value*100 for value in values])
    # print(weigths)
    # print()
    for value, weigth in zip(values, weigths):
        if value == -1 or value == 1:
            return value*1000
        if value != 0:
            good_weights += weigth
        score += value * weigth
    # value += random_perturbation()
    return int((score * 100) / good_weights)


def potential_kills(state):
    '''
    '''
    def count_dead(moves, potential_killers, potential_victims):
        count = 0
        for _, to in moves:
            killables = TablutBoard.orthogonal_k_neighbors(to, k=1)
            killers = TablutBoard.orthogonal_k_neighbors(to, k=2)
            for victim, killer in zip(killables, killers):
                if victim in potential_victims and killer in potential_killers:
                    count += 1
        return count

    player = gutils.other_player(state.to_move)
    white_pawns = state.pawns[TablutPawnType.WHITE]
    black_pawns = state.pawns[TablutPawnType.BLACK]
    if state.to_move == TablutPlayerType.WHITE:
        white_moves = state.moves
        black_moves = TablutGame.player_moves(
            state.pawns, TablutPlayerType.BLACK)
    else:
        black_moves = state.moves
        white_moves = TablutGame.player_moves(
            state.pawns, TablutPlayerType.WHITE)
    black_dead = count_dead(white_moves, set(white_pawns).union(
        {TablutBoard.CASTLE}, TablutBoard.OUTER_CAMPS), black_pawns)
    white_dead = count_dead(black_moves, set(black_pawns).union(
        {TablutBoard.CASTLE}, TablutBoard.OUTER_CAMPS), white_pawns)
    value = black_dead - white_dead
    if value < 0:
        value = -1-(1/value)
    elif value > 0:
        value = 1-(1/value)
    if player == TablutPlayerType.BLACK:
        value = -value
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
    value = -(1 / 2)
    check = False
    if TablutBoard.is_king_dead(state.pawns):
        value = -1
    else:
        king = TablutBoard.king_position(state.pawns)
        for goal in TablutBoard.WHITE_GOALS:
            distance = TablutBoard.simulate_distance(
                state.pawns, king, goal,
                n_moves=0, max_moves=max_moves,
                unwanted_positions=TablutBoard.WHITE_GOALS
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
            # print(distances)
            for ind, distance in enumerate(distances):
                tmp = (2 ** (-ind - 1)) / distance
                if value + tmp > upper_bound:
                    break
                value += tmp
    if player == TablutPlayerType.BLACK:
        value = -value
    return value


def black_chain(state):
    '''
    '''
    black_pawns = set(state.pawns[TablutPawnType.BLACK])
    player = gutils.other_player(state.to_move)
    chains = []
    while len(black_pawns) > 0:
        camps_found = 0
        available_camps = set(TablutBoard.CAMPS)
        pawn = black_pawns.pop()
        chain, camps_found, _ = find_chain(
            pawn, black_pawns, available_camps
        )
        if camps_found == 2:
            chains.append(chain)
    print(chains)


def find_chain(pawn, black_pawns, available_camps, camps_found=0, chain=set()):
    '''
    '''
    neighbors = set(TablutBoard.full_k_neighbors(pawn, k=1))
    if not available_camps.isdisjoint(neighbors):
        chain.add(pawn)
        camps = available_camps.intersection(neighbors)
        camp = utils.get_from_set(camps)
        camps.update(
            TablutBoard.full_k_neighbors(camp, k=1),
            TablutBoard.full_k_neighbors(camp, k=2)
        )
        available_camps.difference_update(camps)
        camps_found += 1
    good_neighbors = black_pawns.intersection(neighbors)
    new_camps_found = camps_found
    for neighbor in good_neighbors:
        if neighbor in black_pawns:
            black_pawns.remove(neighbor)
        chain, new_camps_found, _ = find_chain(
            neighbor, black_pawns, available_camps, new_camps_found, chain
        )
        if new_camps_found > camps_found:
            chain.add(pawn)
    return chain, new_camps_found, black_pawns


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
    black_moves = state.moves
    if state.to_move == TablutPlayerType.WHITE:
        black_moves = TablutGame.player_moves(
            state.pawns, TablutPlayerType.BLACK)
    free_positions = []
    if TablutBoard.is_king_dead(state.pawns):
        value = 1
    else:
        if TablutBoard.is_king_in_castle(state.pawns) or TablutBoard.is_king_near_castle(state.pawns):
            value, free_positions = TablutBoard.potential_king_killers(
                state.pawns)
            if value == 3:
                weight = 1 / 5
            else:
                weight = 1 / 10
            value *= (1 / 5)
        else:
            value, free_positions = TablutBoard.potential_king_killers(
                state.pawns)
            if value == 1:
                weight = 1 / 3
            else:
                weight = 1 / 8
            value *= (1 / 3)

        for free_position in free_positions:
            for _, to in black_moves:
                if free_position == to:
                    value += weight
        if value >= 1:
            value = 0.99

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
        ((1 / 2) * len(state.pawns[TablutPawnType.BLACK])) -
        len(state.pawns[TablutPawnType.WHITE])
    )
    if player == TablutPlayerType.WHITE:
        diff = -diff
    return diff * (1 / 8)
