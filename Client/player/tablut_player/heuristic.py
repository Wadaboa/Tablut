'''
Tablut states evaluation functions
'''


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


def heuristic_pov(state, pov, value):
    '''
    Given an heuristic evaluation and a player on top of which the evaluation
    got computed, return the correct value
    '''
    return value if gutils.other_player(state.to_move) == pov else -value


def get_moves(state):
    '''
    Return legal white and black moves from the given state
    '''
    if state.to_move == TPlayerType.WHITE:
        return (
            state.moves, TablutGame.player_moves(state.pawns, TPlayerType.BLACK)
        )
    return (
        TablutGame.player_moves(state.pawns, TPlayerType.WHITE), state.moves
    )


def get_pawns(state, king=False):
    '''
    Return white and black pawns, with or without considering the king
    '''
    white_pawns = (
        TablutBoard.player_pawns(state.pawns, TPlayerType.WHITE) if king
        else state.pawns[TPawnType.WHITE]
    )
    return white_pawns, TablutBoard.player_pawns(state.pawns, TPlayerType.BLACK)


def potential_kills(state):
    '''
    Return a value representing the number of potential player pawns killers,
    in range [-1, 1]
    '''

    def count_dead(moves, potential_killers, potential_victims):
        count = 0
        for _, to_move in moves:
            killables = TablutBoard.orthogonal_k_neighbors(to_move, k=1)
            killers = TablutBoard.orthogonal_k_neighbors(to_move, k=2)
            for victim, killer in zip(killables, killers):
                if victim in potential_victims and killer in potential_killers:
                    count += 1
        return count

    white_pawns, black_pawns = get_pawns(state, king=False)
    white_moves, black_moves = get_moves(state)
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
    value = float(black_dead - white_dead)
    if value < 0:
        value = -1 - (1 / value)
    elif value > 0:
        value = 1 - (1 / value)
    return heuristic_pov(state, TPlayerType.WHITE, value)


def king_moves_to_goals(state):
    '''
    Return a value representing the number of king moves to every goal.
    Given a state, it checks the min number of moves to each goal,
    and return a positive value if we are within 1 to 3 moves
    to a certain goal, and an even higher value if we are within
    1 to 3 moves to more than one corner, in range [-1, 1]
    '''
    value = -0.3
    upper_bound = 0.6
    goals_distances = compute_goals_distances(state)
    distances = [
        dis for dis in goals_distances.values() if dis < MAX_KING_MOVES_GOALS
    ]
    check_distances = len(distances) > 0
    if distances.count(1) >= 1:
        value = 0.99 if distances.count(1) > 1 else 0.9
        check_distances = False
    if check_distances:
        value = 0.0
        distances.sort()
        for ind, distance in enumerate(distances):
            tmp = (2 ** (-ind - 1)) / distance
            if value + tmp > upper_bound:
                break
            value += tmp
    return heuristic_pov(state, TPlayerType.WHITE, value)


def black_blocking_chains(state):
    '''
    Return a value representing the number of chained black pawns,
    that are blocking goal positions, in range [-1, 1]
    '''
    value = 0.0
    upper_bound = 0.9
    corners_weights = [4 / 10, 6 / 10, 8 / 10, 9 / 10]
    chains = black_chains(state)
    for chain in chains:
        whites, blacks, corners = blocked_chain_pawns(state, chain)
        if corners == 0:
            value = -0.9 + (1 / 12) * blacks
            return heuristic_pov(state, TPlayerType.BLACK, value)
        chain_value = corners * corners_weights[corners - 1]
        if useful_chain(state, chain):
            if whites > 1:
                chain_value /= whites
            if blacks > 2:
                chain_value -= (chain_value / blacks)
            chain_weight = 1.0
        else:
            chain_weight = 1 / 4
        value += (chain_weight * chain_value)

    value = utils.clip(value, upper=upper_bound)
    return heuristic_pov(state, TPlayerType.BLACK, value)


def useful_chain(state, chain):
    '''
    Return if the given chain is useful in the sense that the king
    could reach the examined corner before the chain was present,
    and now it is blocked
    '''
    king = TablutBoard.king_position(state.pawns)
    tmp_pawns = gutils.clone_pawns(state.pawns)
    tmp_pawns[TPawnType.BLACK].difference_update(chain)
    reachable_corners = compute_reachable_corners(king, state.pawns)
    tmp_reachable_corners = compute_reachable_corners(king, tmp_pawns)
    for corner in CORNERS:
        if not reachable_corners[corner] and tmp_reachable_corners[corner]:
            return True
    return False


def blocked_chain_pawns(state, chain):
    '''
    Return the number of white pawns, black pawns and corners
    blocked from a chain of black pawns
    '''
    white_pawns, black_pawns = get_pawns(state, king=True)
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


def find_near_corners(chain, max_distance=6):
    '''
    Return all the probable corners blocked by the given chain
    '''
    near_corners = set()
    for pawn in chain:
        for corner in CORNERS:
            if corner.distance(pawn) < max_distance:
                near_corners.add(corner)
    return near_corners


def black_chains(state):
    '''
    Return every chains of black pawns connecting two or more camps groups
    '''
    chains = []
    black_pawns = set(state.pawns[TPawnType.BLACK]).difference(
        TablutBoard.CAMPS
    )
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
        camp = utils.get_rand(camps)
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


def white_barriers(state):
    '''
    Return a value representing the number of corners reachable by the king
    and not by the black pawns, in range [-1, 1]
    '''
    value = 0.0
    king = TablutBoard.king_position(state.pawns)
    tmp_pawns = gutils.clone_pawns(state.pawns)
    tmp_pawns[TPawnType.WHITE].difference_update(CORNERS)
    reachable_corners = compute_reachable_corners(king, state.pawns)
    for corner, king_reach in reachable_corners.items():
        barrier = True
        if king_reach:
            for black_pawn in tmp_pawns[TPawnType.BLACK]:
                black_reach = TablutBoard.dfs(tmp_pawns, black_pawn, corner)
                if black_reach:
                    barrier = False
                    break
            if barrier:
                value = 0.99
                break
    return heuristic_pov(state, TPlayerType.WHITE, value)


def blocked_goals(state):
    '''
    Return a value representing the number of blocked white goals
    for each corner, in range [-1, 1]
    '''
    value = 0.0
    white_pawns, black_pawns = get_pawns(state, king=False)
    free_goals = compute_reachable_goals(state)
    for pos in TablutBoard.OUTER_CORNERS:
        if pos in black_pawns:
            value -= (1 / 9)
            free_goals.difference_update(
                TablutBoard.unique_orthogonal_k_neighbors(pos)
            )
        elif pos in white_pawns:
            value -= (1 / 16)

    for goal in free_goals:
        if goal in black_pawns:
            value -= (1 / 17)
        elif goal in white_pawns:
            value -= (1 / 20)

    return heuristic_pov(state, TPlayerType.WHITE, value)


def king_killers(state):
    '''
    Return a value representing the number of black pawns,
    camps and castle around the king, in range [-1, 1]
    '''
    _, black_moves = get_moves(state)
    free_positions = []
    killer_positions = []
    killers, free_positions, killer_positions = (
        TablutBoard.potential_king_killers(state.pawns)
    )
    possible_killers_count = _reachable_positions(killer_positions, black_moves)
    occupable_free_positions = _reachable_positions(free_positions, black_moves)
    white_neighbors = (
        4 - (killers + len(free_positions) + len(killer_positions))
    )
    value = 0.0
    values = [killers, occupable_free_positions, possible_killers_count]
    weights = [0, 1 / 32, 0]
    if (TablutBoard.is_king_in_castle(state.pawns) or
            TablutBoard.is_king_near_castle(state.pawns)):
        if killers >= 3:
            value = 0.7
            if possible_killers_count != 0:
                value = 0.9
        elif killers == 2:
            weights = [1 / 6, 1 / 10, 1 / 10]
        elif killers == 1:
            weights = [1 / 8, 1 / 25, 1 / 10]
    else:
        if killers >= 1 and possible_killers_count != 0:
            value = 0.9
        elif killers >= 3:
            value = 0.7
        elif killers == 2:
            weights = [1 / 4, 1 / 10, 0]
        elif killers == 1:
            weights = [1 / 3, 1 / 10, 0]
    if value == 0.0:
        value = -white_neighbors * (3 / 40)
        for val, weight in zip(values, weights):
            value += (val * weight)
    return heuristic_pov(state, TPlayerType.BLACK, value)


def _reachable_positions(positions, moves):
    '''
    Return the number of reachable positions with the given pawns and moves
    '''
    count = 0
    for position in positions:
        for _, to_move in moves:
            if position == to_move:
                count += 1
                break
    return count


def piece_difference(state):
    '''
    Return an evaluation of the pawn difference, in range [-1, 1]
    '''
    value = (
        ((1 / 2) * len(state.pawns[TPawnType.BLACK])) -
        len(state.pawns[TPawnType.WHITE])
    ) * (1 / 8)
    return heuristic_pov(state, TPlayerType.BLACK, value)


def pawns_in_corners(state):
    '''
    Return a value representing the number of player and enemy pawns
    in the extreme corners, in range [-1, 1]
    '''
    value = 0.0
    player_pawns = TablutBoard.player_pawns(
        state.pawns, gutils.other_player(state.to_move)
    )
    enemy_pawns = TablutBoard.player_pawns(state.pawns, state.to_move)
    for corner in CORNERS:
        if corner in player_pawns:
            value -= 1 / 20
        elif corner in enemy_pawns:
            value += 1 / 30
    return value


def compute_goals_distances(state):
    '''
    Compute king distances to goals
    '''
    king = TablutBoard.king_position(state.pawns)
    goals_distances = {}
    for goal in TablutBoard.WHITE_GOALS:
        goals_distances[goal] = TablutBoard.simulate_distance(
            state.pawns, king, goal, max_moves=MAX_KING_MOVES_GOALS,
            unwanted_positions=TablutBoard.WHITE_GOALS
        )
    return goals_distances


def compute_reachable_goals(state):
    '''
    Compute the goals that the king can reach
    '''
    king = TablutBoard.king_position(state.pawns)
    reachable_goals = set()
    for goal in TablutBoard.WHITE_GOALS:
        if TablutBoard.dfs(state.pawns, king, goal):
            reachable_goals.add(goal)
    return reachable_goals


def compute_reachable_corners(pawn, pawns):
    '''
    Compute pawn distances to goals
    '''
    reachable_corners = {corner: False for corner in CORNERS}
    for corner in reachable_corners:
        if corner not in pawns[TPawnType.BLACK]:
            pawns[TPawnType.WHITE].discard(corner)
            reachable_corners[corner] = TablutBoard.dfs(pawns, pawn, corner)
    return reachable_corners


HEURISTICS = {
    blocked_goals: 1.5,
    piece_difference: 3,
    potential_kills: 0.5,
    king_moves_to_goals: 2,
    king_killers: 1,
    black_blocking_chains: 6,
    pawns_in_corners: 0.5,
    white_barriers: 10
}

MAX_KING_MOVES_GOALS = 4
CORNERS_INDEXES = [0, conf.BOARD_SIZE - 1]
CORNERS = [TBPos(i, j) for i in CORNERS_INDEXES for j in CORNERS_INDEXES]


def set_heuristic_weights(weights):
    '''
    Set given heuristic weights
    '''
    heus = HEURISTICS.keys()
    for heu, weight in zip(heus, weights):
        HEURISTICS[heu] = weight


def print_heuristic(game, state):
    print(f'VALUE: {heuristic(game, state)}')
    for heur in HEURISTICS:
        print(f'{heur.__name__}: {heur(state)}')


def heuristic(game, state, color=None):
    '''
    Game state evaluation function, in range [-100, 100].
    Values 1000 and -1000 are used as winning and losing scores
    '''
    if color is None:
        color = gutils.other_player(state.to_move)
    if state.is_terminal:
        if state.utility != 0:
            return game.utility(state, color) * 1000
        return -999
    good_weights = 0
    score = 0.0
    for heu, weigth in HEURISTICS.items():
        value = heu(state)
        if value != 0.0:
            good_weights += weigth
        score += value * weigth
    return (
        int((score * 100) / good_weights) if good_weights > 0
        else int(score * 100)
    )
