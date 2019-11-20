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


CORNERS = {
    (TPawnDir.UP, TPawnDir.LEFT),
    (TPawnDir.UP, TPawnDir.RIGHT),
    (TPawnDir.DOWN, TPawnDir.LEFT),
    (TPawnDir.DOWN, TPawnDir.RIGHT)
}
GOALS = {
    (TPawnDir.UP, TPawnDir.LEFT): [
        TBPos(row=2, col=0), TBPos(row=0, col=2),
        TBPos(row=1, col=0), TBPos(row=0, col=1)
    ],
    (TPawnDir.UP, TPawnDir.RIGHT): [
        TBPos(row=0, col=6), TBPos(row=0, col=7),
        TBPos(row=1, col=8), TBPos(row=2, col=8)
    ],
    (TPawnDir.DOWN, TPawnDir.LEFT): [
        TBPos(row=6, col=0), TBPos(row=7, col=0),
        TBPos(row=8, col=1), TBPos(row=8, col=2)
    ],
    (TPawnDir.DOWN, TPawnDir.RIGHT): [
        TBPos(row=8, col=6), TBPos(row=8, col=7),
        TBPos(row=6, col=8), TBPos(row=7, col=8)
    ]
}
BEST_BLOCKING_POSITIONS = {
    (TPawnDir.UP, TPawnDir.LEFT): [
        TBPos(row=2, col=1), TBPos(row=1, col=2),
    ],
    (TPawnDir.UP, TPawnDir.RIGHT): [
        TBPos(row=1, col=6), TBPos(row=2, col=7)
    ],
    (TPawnDir.DOWN, TPawnDir.LEFT): [
        TBPos(row=6, col=1), TBPos(row=7, col=2)
    ],
    (TPawnDir.DOWN, TPawnDir.RIGHT): [
        TBPos(row=6, col=7), TBPos(row=7, col=6)
    ]
}
OUTER_CORNERS = {
    (TPawnDir.UP, TPawnDir.LEFT): [
        TBPos(row=1, col=1)
    ],
    (TPawnDir.UP, TPawnDir.RIGHT): [
        TBPos(row=1, col=7)
    ],
    (TPawnDir.DOWN, TPawnDir.LEFT): [
        TBPos(row=7, col=1)
    ],
    (TPawnDir.DOWN, TPawnDir.RIGHT): [
        TBPos(row=7, col=7)
    ]
}


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
        weigths = [3, 1, 0.25, 2, 2, 4]
    elif turn < 20:
        weigths = [2, 2, 0.3, 3, 3, 3]
    else:
        weigths = [1, 3, 0.6, 6, 6, 2]
    good_weights = 0
    score = 0
    print([value*100 for value in values])
    print(weigths)
    print()
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
    chains = black_chains(state)
    value = -(1 / 10) * len(chains)
    blocked_whites = 0
    blocked_blacks = 0
    for chain in chains:
        whites_found, blacks_found = blocked_chain_pawns(state, chain)
        blocked_whites += whites_found
        blocked_blacks += blacks_found
    player = gutils.other_player(state.to_move)
    value += (1 / 17) * blocked_blacks + (1 / 10) * blocked_whites
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
    extreme_indexes = [0, conf.BOARD_SIZE - 1]
    extreme_corners = [
        TBPos(i, j) for i in extreme_indexes for j in extreme_indexes
    ]
    whites_found = [0] * len(extreme_corners)
    blacks_found = [0] * len(extreme_corners)
    for i, corner in enumerate(extreme_corners):
        chain_count = 0
        neighbor = corner
        neighbors = set()
        visited_chain = set()
        visited_neighbors = set()
        while chain_count != len(chain):
            visited_neighbors.add(neighbor)
            if neighbor in chain and neighbor not in visited_chain:
                visited_chain.add(neighbor)
                chain_count += 1
            elif neighbor in white_pawns:
                whites_found[i] += 1
            elif neighbor in black_pawns:
                blacks_found[i] += 1
            neighbors.update(
                TablutBoard.unique_full_k_neighbors(neighbor, k=1).difference(
                    visited_neighbors
                )
            )
            neighbor = neighbors.pop()
    return min(whites_found), min(blacks_found)


def black_chains(state):
    '''
    Return every chains of black pawns connecting two or more camps groups
    '''
    chains = []
    black_pawns = set(state.pawns[TPawnType.BLACK])
    while len(black_pawns) > 0:
        camps_found = 0
        available_camps = set(TablutBoard.CAMPS)
        pawn = black_pawns.pop()
        chain, camps_found, _ = find_chain(
            pawn, black_pawns, available_camps, camps_found=0, chain=set()
        )
        if camps_found == 2:
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
    total = 0.0
    player = gutils.other_player(state.to_move)
    black_pawns = state.pawns[TPawnType.BLACK]
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
    if player == TPlayerType.WHITE:
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
