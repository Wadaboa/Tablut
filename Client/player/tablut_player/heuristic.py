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
    TablutPlayerType as TPlayerType
)


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
    upper_bound = 0.6
    distances = [
        dis for dis in GOALS_DISTANCES.values() if dis < MAX_KING_MOVES
    ]
    player = gutils.other_player(state.to_move)
    value = -0.3
    check = False
    if distances.count(1) > 1:
        value = 0.99
        check = True
    elif distances.count(1) == 1:
        value = 0.9
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
    weights = [4/10, 6/10, 8/10, 9/10]
    upper_bound = 0.9
    for chain in chains:
        whites_found, blacks_found, corners_found = blocked_chain_pawns(
            state, chain)
        if corners_found == 0:
            value = -0.9 + (1 / 12)*(blacks_found)
            if player == TPlayerType.WHITE:
                value = -value
            return value
        chain_value = corners_found * weights[corners_found-1]
        if whites_found > 1:
            chain_value /= whites_found
        if blacks_found > 2:
            chain_value -= chain_value/blacks_found
        value += chain_value

    if value > upper_bound:
        value = upper_bound
    if player == TPlayerType.WHITE:
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
    '''
    Return all the probable corners blocked by the given chain
    '''
    near_corners = set()
    for pawn in chain:
        for corner in CORNERS:
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
    free_goals = {
        pos for pos, dis in GOALS_DISTANCES.items() if dis < MAX_KING_MOVES + 1
    }
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
    value = 0
    player = gutils.other_player(state.to_move)
    black_moves = state.moves
    if state.to_move == TPlayerType.WHITE:
        black_moves = TablutGame.player_moves(
            state.pawns, TPlayerType.BLACK
        )
    free_positions = []
    killer_positions = []
    killers, free_positions, killer_positions = (
        TablutBoard.potential_king_killers(state.pawns)
    )

    possible_killers_count = _reachable_positions(killer_positions, black_moves)

    occupable_free_positions = _reachable_positions(free_positions, black_moves)

    values = [killers, occupable_free_positions, possible_killers_count]
    weights = [0, 1/32, 0]
    if (TablutBoard.is_king_in_castle(state.pawns) or
            TablutBoard.is_king_near_castle(state.pawns)):
        if killers >= 3:
            value = 0.7
            if possible_killers_count != 0:
                value = 0.9
        elif killers == 2:
            weights = [1/6, 1/10, 1/10]
        elif killers == 1:
            weights = [1/8, 1/25, 1/10]
    else:
        if killers >= 1 and possible_killers_count != 0:
            value = 0.9
        elif killers >= 3:
            value = 0.7
        elif killers == 2:
            weights = [1/4, 1/10, 0]
        elif killers == 1:
            weights = [1/3, 1/10, 0]
    if value == 0:
        for val, weight in zip(values, weights):
            value += val*weight
    if player == TPlayerType.WHITE:
        value = -value
    return value


def _reachable_positions(positions, moves):
    count = 0
    for position in positions:
        for _, to in moves:
            if position == to:
                count += 1
                break
    return count


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


def pawns_in_corners(state):
    '''
    '''
    player = gutils.other_player(state.to_move)
    value = 0
    player_pawns = TablutBoard.player_pawns(state.pawns, player)
    enemy_pawns = TablutBoard.player_pawns(state.pawns, state.to_move)
    for corner in CORNERS:
        if corner in player_pawns:
            value -= 1/20
        elif corner in enemy_pawns:
            value += 1/30
    return value


def _init_goals_distances(state):
    king = TablutBoard.king_position(state.pawns)
    for goal in TablutBoard.WHITE_GOALS:
        GOALS_DISTANCES[goal] = TablutBoard.simulate_distance(
            state.pawns, king, goal, max_moves=MAX_KING_MOVES,
            unwanted_positions=TablutBoard.WHITE_GOALS
        )


HEURISTIC_WEIGHTS = {
    blocked_goals: 2,
    piece_difference: 2,
    potential_kills: 0.5,
    king_moves_to_goals: 1,
    king_killers: 2,
    black_blocking_chains: 10,
    pawns_in_corners: 1
}

MAX_KING_MOVES = 4
GOALS_DISTANCES = {goal: MAX_KING_MOVES + 1 for goal in TablutBoard.WHITE_GOALS}
CORNERS_INDEXES = [0, conf.BOARD_SIZE - 1]
CORNERS = [TBPos(i, j) for i in CORNERS_INDEXES for j in CORNERS_INDEXES]


def set_heuristic_weights(weights):
    '''
    Set given heuristic weights
    '''
    heus = HEURISTIC_WEIGHTS.keys()
    for heu, weight in zip(heus, weights):
        HEURISTIC_WEIGHTS[heu] = weight


def heuristic(state):
    '''
    Game state evaluation function, in range [-100, 100].
    Values 1000 and -1000 are used as winning and losing scores
    '''
    if TablutGame.terminal_test(state):
        if state.utility != 0:
            return state.utility * 1000
        return 0
    _init_goals_distances(state)
    good_weights = 0
    score = 0
    for heu, weigth in HEURISTIC_WEIGHTS.items():
        value = heu(state)
        if value != 0:
            good_weights += weigth
        score += value * weigth
    # value += random_perturbation()
    return int((score * 100) / good_weights)
