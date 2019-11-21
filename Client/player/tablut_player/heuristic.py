'''
Tablut states evaluation functions
'''

import random

import tablut_player.utils as utils
import tablut_player.game_utils as gutils
import tablut_player.config as conf
from tablut_player.game import TablutGame
from tablut_player.board import TablutBoard
from tablut_player.game_utils import (
    TablutBoardPosition as TBPos,
    TablutPawnType as TPawnType,
    TablutPlayerType as TPlayerType,
    TablutPawnDirection as TPawnDir
)


def heuristic(turn, state):
    '''
    Game state evaluation function, in range [-100, 100].
    Values 1000 and -1000 are used as winning and losing scores
    '''
    values = [
        blocked_goals(state),
        piece_difference(state),
        potential_kills(state),
        king_moves_to_goals(state),
        king_killers(state),
        black_blocking_chains(state)
    ]
    if turn < 10:
        weigths = [2, 2, 0.5, 1, 2, 5]
    elif turn < 20:
        weigths = [2, 4, 0.3, 5, 6, 2]
    else:
        weigths = [1.5, 5, 0.6, 6, 8, 2]
    good_weights = 0
    score = 0
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
    Return a value representing the number of potential player pawns killers,
    in range [-1, 1]
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
    white_pawns = state.pawns[TPawnType.WHITE]
    black_pawns = state.pawns[TPawnType.BLACK]
    if state.to_move == TPlayerType.WHITE:
        white_moves = state.moves
        black_moves = TablutGame.player_moves(
            state.pawns, TPlayerType.BLACK
        )
    else:
        black_moves = state.moves
        white_moves = TablutGame.player_moves(
            state.pawns, TPlayerType.WHITE
        )
    black_dead = count_dead(
        white_moves,
        set(white_pawns).union({TablutBoard.CASTLE}, TablutBoard.OUTER_CAMPS),
        black_pawns
    )
    white_dead = count_dead(
        black_moves,
        set(black_pawns).union({TablutBoard.CASTLE}, TablutBoard.OUTER_CAMPS),
        white_pawns
    )
    value = black_dead - white_dead
    if value < 0:
        value = -1 - (1 / value)
    elif value > 0:
        value = 1 - (1 / value)
    if player == TPlayerType.BLACK:
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
        if distances.count(1) > 1:
            value = 0.99
            check = True
        if len(distances) > 0 and not check:
            value = 0
            distances.sort()
            for ind, distance in enumerate(distances):
                tmp = (2 ** (-ind - 1)) / distance
                if value + tmp > upper_bound:
                    break
                value += tmp
    if player == TPlayerType.BLACK:
        value = -value
    return value


def black_blocking_chains(state):
    '''
    Return a value representing the number of chained black pawns,
    that are blocking goal positions, in range [-1, 1]
    '''
    player = gutils.other_player(state.to_move)
    chains = black_chains(state)
    value = 0
    blocked_whites = 0
    blocked_blacks = 0
    blocked_corners = 0
    for chain in chains:
        whites_found, blacks_found, corners_found = blocked_chain_pawns(
            state, chain)
        if corners_found == 0:
            value = 0.3-(1 / 12)*(blacks_found)
            if player == TPlayerType.BLACK:
                value = -value
            return value
        blocked_corners += corners_found
        blocked_whites += whites_found
        blocked_blacks += blacks_found

    value = -(blocked_corners * (4 / 17)) + (blocked_whites * (1 / 9))
    if player == TPlayerType.BLACK:
        value = -value
    return value


def blocked_chain_pawns(state, chain):
    '''
    Return the number of white and black pawns blocked from a chain of black
    pawns
    '''
    white_pawns = TablutBoard.player_pawns(state.pawns, TPlayerType.WHITE)
    black_pawns = TablutBoard.player_pawns(state.pawns, TPlayerType.BLACK)
    near_corners = find_near_corners(chain)
    whites_found = 0
    blacks_found = 0
    king_found = False
    for corner in near_corners:
        corner_whites = 0
        corner_blacks = 0
        neighbor = corner
        neighbors = TablutBoard.unique_orthogonal_k_neighbors(neighbor, k=1)
        visited_neighbors = set()
        while len(neighbors) > 0:
            visited_neighbors.add(neighbor)
            if neighbor not in chain:
                if neighbor in white_pawns:
                    if neighbor == TablutBoard.king_position(state.pawns):
                        king_found = True
                    corner_whites += 1
                elif neighbor in black_pawns:
                    corner_blacks += 1
                current_neighbors = TablutBoard.unique_orthogonal_k_neighbors(
                    neighbor, k=1
                )
                neighbors.update(
                    current_neighbors.difference(
                        visited_neighbors, TablutBoard.CAMPS
                    )
                )
            neighbor = neighbors.pop()
        if king_found:
            return corner_whites, corner_blacks, 0
        whites_found += corner_whites
        blacks_found += corner_blacks

    if len(near_corners) > 1:
        camps = set()
        for a_corner in near_corners:
            for b_corner in near_corners:
                if a_corner != b_corner:
                    camp = a_corner.middle_position(b_corner)
                    if camp is not None:
                        camps.add(camp)
                        camps.update(
                            TablutBoard.unique_orthogonal_k_neighbors(
                                camp, k=1
                            )
                        )
        for camp in camps:
            if camp in black_pawns:
                blacks_found += 1

    return whites_found, blacks_found, len(near_corners)


def find_near_corners(chain):
    extreme_indexes = [0, conf.BOARD_SIZE - 1]
    extreme_corners = [
        TBPos(i, j) for i in extreme_indexes for j in extreme_indexes
    ]
    near_corners = set()
    for pawn in chain:
        for corner in extreme_corners:
            if corner.distance(pawn) < 6:
                near_corners.add(corner)
    return near_corners


def black_chains(state):
    '''
    Return every chains of black pawns connecting two or more camps groups
    '''
    chains = []
    black_pawns = set(state.pawns[TPawnType.BLACK])
    black_pawns.difference_update(TablutBoard.CAMPS)
    while len(black_pawns) > 0:
        camps_found = 0
        available_camps = set(TablutBoard.CAMPS)
        pawn = black_pawns.pop()
        chain, camps_found, _ = find_chain(
            pawn, black_pawns, available_camps, camps_found=0, chain=set()
        )
        if camps_found > 1:
            chains.append(chain)
    return chains


def find_chain(pawn, black_pawns, available_camps, camps_found=0, chain=set()):
    '''
    Find a chain of black pawns connecting two or more camps groups,
    starting from the given coordinates
    '''
    neighbors = TablutBoard.unique_full_k_neighbors(pawn, k=1)
    if not available_camps.isdisjoint(neighbors):
        chain.add(pawn)
        camps = available_camps.intersection(neighbors)
        camp = utils.get_from_set(camps)
        camps.update(
            TablutBoard.unique_full_k_neighbors(camp, k=1),
            TablutBoard.unique_full_k_neighbors(camp, k=2)
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
    value = 0.0
    player = gutils.other_player(state.to_move)
    black_pawns = state.pawns[TPawnType.BLACK]
    white_pawns = state.pawns[TPawnType.WHITE]
    free_goals = set(TablutBoard.WHITE_GOALS)

    for pos in TablutBoard.OUTER_CORNERS:
        if pos in black_pawns:
            value -= (1 / 9)
            free_goals.difference_update(
                TablutBoard.unique_orthogonal_k_neighbors(pos))
        elif pos in white_pawns:
            value -= (1 / 16)

    for goal in free_goals:
        if goal in black_pawns:
            value -= (1 / 17)
        elif goal in white_pawns:
            value -= (1 / 20)

    if player == TPlayerType.BLACK:
        value = -value
    return value


def king_killers(state):
    '''
    Return a value representing the number of black pawns,
    camps and castle around the king, in range [-1, 1]
    '''
    value = 0.0
    player = gutils.other_player(state.to_move)
    black_moves = state.moves
    if state.to_move == TPlayerType.WHITE:
        black_moves = TablutGame.player_moves(
            state.pawns, TPlayerType.BLACK)
    free_positions = []
    if TablutBoard.is_king_dead(state.pawns):
        value = 1
    else:
        if (TablutBoard.is_king_in_castle(state.pawns) or
                TablutBoard.is_king_near_castle(state.pawns)):
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

    if player == TPlayerType.WHITE:
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
        ((1 / 2) * len(state.pawns[TPawnType.BLACK])) -
        len(state.pawns[TPawnType.WHITE])
    )
    if player == TPlayerType.WHITE:
        diff = -diff
    return diff * (1 / 8)
