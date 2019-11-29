'''
Tablut players search strategies
'''


import math
import timeit
import sys

import tablut_player.config as conf
import tablut_player.game_utils as gutils
import tablut_player.utils as utils
import tablut_player.heuristic as heu
from tablut_player.utils import INF
from tablut_player.game_utils import TablutBoardPosition as TBPos


THIS = sys.modules[__name__]
BEST_MOVE = None


# ______________________________________________________________________________
# Opening moves

WHITE_OPENINGS = [
    (TBPos.create(row=4, col=2), TBPos.create(row=7, col=2)),
    (TBPos.create(row=4, col=2), TBPos.create(row=8, col=2))
]
BLACK_OPENINGS = {
    (TBPos.create(row=4, col=2), TBPos.create(row=5, col=2)): [
        (TBPos.create(row=5, col=8), TBPos.create(row=5, col=7))
    ],
    (TBPos.create(row=4, col=2), TBPos.create(row=6, col=2)): [
        (TBPos.create(row=3, col=0), TBPos.create(row=3, col=2))
    ],
    (TBPos.create(row=4, col=2), TBPos.create(row=7, col=2)): [
        (TBPos.create(row=3, col=0), TBPos.create(row=3, col=2))
    ],
    (TBPos.create(row=4, col=2), TBPos.create(row=8, col=2)): [
        (TBPos.create(row=4, col=1), TBPos.create(row=8, col=1)),
        (TBPos.create(row=3, col=0), TBPos.create(row=3, col=2))
    ],
    (TBPos.create(row=4, col=3), TBPos.create(row=5, col=3)): [
        (TBPos.create(row=0, col=3), TBPos.create(row=4, col=3))
    ],
    (TBPos.create(row=4, col=3), TBPos.create(row=6, col=3)): [
        (TBPos.create(row=0, col=3), TBPos.create(row=4, col=3))
    ],
    (TBPos.create(row=4, col=3), TBPos.create(row=7, col=3)): [
        (TBPos.create(row=0, col=3), TBPos.create(row=4, col=3))
    ]
}


def init_white_openings():
    '''
    Compute white player first move
    '''
    moves = WHITE_OPENINGS
    for i in range(len(moves)):
        from_move, to_move = moves[i]
        moves.append((from_move, to_move.horizontal_mirroring()))
    for i in range(len(moves)):
        from_move, to_move = moves[i]
        moves.append((
            from_move.vertical_mirroring(), to_move.vertical_mirroring()
        ))
        moves.append((
            from_move.diagonal_mirroring(diag=1),
            to_move.diagonal_mirroring(diag=1)
        ))
        moves.append((
            from_move.diagonal_mirroring(diag=-1),
            to_move.diagonal_mirroring(diag=-1)
        ))


def init_black_openings():
    '''
    Compute black player first move
    '''
    initial_white_keys = list(BLACK_OPENINGS.keys())
    white_keys = list(initial_white_keys)
    for white_key in initial_white_keys:
        from_white, to_white = white_key
        white_keys.append(white_key)
        for black_move in BLACK_OPENINGS[white_key]:
            from_black, to_black = black_move
            key = (from_white, to_white.horizontal_mirroring())
            white_keys.append(key)
            BLACK_OPENINGS.setdefault(key, []).append((
                from_black.horizontal_mirroring(),
                to_black.horizontal_mirroring()
            ))
    for white_key in white_keys:
        from_white, to_white = white_key
        for black_move in BLACK_OPENINGS[white_key]:
            from_black, to_black = black_move
            key = (
                from_white.vertical_mirroring(),
                to_white.vertical_mirroring()
            )
            BLACK_OPENINGS.setdefault(key, []).append((
                from_black.vertical_mirroring(),
                to_black.vertical_mirroring()
            ))
            key = (
                from_white.diagonal_mirroring(diag=1),
                to_white.diagonal_mirroring(diag=1)
            )
            BLACK_OPENINGS.setdefault(key, []).append((
                from_black.diagonal_mirroring(diag=1),
                to_black.diagonal_mirroring(diag=1)
            ))
            key = (
                from_white.diagonal_mirroring(diag=-1),
                to_white.diagonal_mirroring(diag=-1)
            )
            BLACK_OPENINGS.setdefault(key, []).append((
                from_black.diagonal_mirroring(diag=-1),
                to_black.diagonal_mirroring(diag=-1)
            ))


def init_openings():
    '''
    Compute first turn moves
    '''
    init_white_openings()
    init_black_openings()


def first_move(state, prev_move=None):
    '''
    First opening move
    '''
    utils.set_random_seed()
    return (
        utils.get_rand(WHITE_OPENINGS) if (
            state.to_move == gutils.TablutPlayerType.WHITE
        )
        else utils.get_rand(BLACK_OPENINGS[prev_move]) if prev_move is not None
        else None
    )


# ______________________________________________________________________________
# Transposition table

class TTEntry:
    '''
    Transposition table entry
    '''

    def __init__(self, state_key, player_key, moves, value, depth):
        self.state_key = state_key
        self.player_key = player_key
        self.moves = moves
        self.value = value
        self.depth = depth

    def __repr__(self):
        return (
            f'State key: {self.state_key}\n'
            f'Player key: {self.player_key}\n'
            f'Moves: {self.moves}\n'
            f'Value: {self.value}\n'
            f'Depth: {self.depth}'
        )


class TT:
    '''
    Transposition table
    '''

    def __init__(self):
        self.table = {}

    def get_entry(self, state):
        '''
        Return the transposition table entry of the given state,
        if it exists
        '''
        player_key = state.ZOBRIST_KEYS.to_move[state.to_move]
        state_key = hash(state)
        entry = self.table.get((state_key, player_key))
        if (entry is not None and
                entry.state_key == state_key and
                entry.player_key == player_key):
            return entry
        return None

    def get_moves(self, state):
        '''
        Return the transposition table entry moves of the given state,
        if it exists
        '''
        entry = self.get_entry(state)
        return entry.moves if entry is not None else None

    def get_value(self, state):
        '''
        Return the transposition table entry value of the given state,
        if it exists
        '''
        entry = self.get_entry(state)
        return entry.value if entry is not None else None

    def store_entry(self, entry):
        '''
        Store the given entry in the transposition table
        '''
        self.table[(entry.state_key, entry.player_key)] = entry

    def store_exp_entry(self, state, moves, value, depth):
        '''
        Store the given entry in the transposition table
        '''
        entry = TTEntry(
            hash(state), state.ZOBRIST_KEYS.to_move[state.to_move],
            moves, value, depth
        )
        self.store_entry(entry)

    def clear(self):
        '''
        Delete every entry from the transposition table
        '''
        self.table = {}


# ______________________________________________________________________________
# Minimax algorithm

def minimax_alphabeta(game, state, timeout, max_depth, tt):
    '''
    Iterative deepening minimax with alpha-beta pruning and transposition tables
    '''

    def max_value(state, depth, alpha, beta):
        entry = tt.get_entry(state)
        state.moves = (
            entry.moves if entry is not None and entry.moves is not None
            else game.player_moves(state.pawns, state.to_move)
        )
        moves = state.moves

        if alphabeta_cutoff(game, state, depth, start_time, timeout):
            if entry is not None and entry.depth == depth:
                value = entry.value
            else:
                value = heu.heuristic(state)
                tt.store_exp_entry(state, moves, value, depth)
            return value
        v = -INF
        for move in moves:
            value = min_value(
                game.result(state, move, compute_moves=False), depth - 1,
                alpha=alpha, beta=beta
            )
            if value > v:
                v = value
                moves.remove(move)
                moves.insert(0, move)
            if v >= beta:
                tt.store_exp_entry(state, moves, v, depth)
                return v
            alpha = max(alpha, v)
        tt.store_exp_entry(state, moves, v, depth)
        return v

    def min_value(state, depth, alpha, beta):
        entry = tt.get_entry(state)
        state.moves = (
            entry.moves if entry is not None and entry.moves is not None
            else game.player_moves(state.pawns, state.to_move)
        )
        moves = state.moves

        if alphabeta_cutoff(game, state, depth, start_time, timeout):
            if entry is not None and entry.depth == depth:
                value = entry.value
            else:
                value = heu.heuristic(state)
                tt.store_exp_entry(state, moves, value, depth)
            return value
        v = INF
        for move in moves:
            value = max_value(
                game.result(state, move, compute_moves=False), depth - 1,
                alpha=alpha, beta=beta
            )
            if value < v:
                v = value
                moves.remove(move)
                moves.insert(0, move)
            if v <= alpha:
                tt.store_exp_entry(state, moves, v, depth)
                return v
            beta = min(beta, v)
        tt.store_exp_entry(state, moves, v, depth)
        return v

    start_time = timeit.default_timer()
    best_score = -INF
    beta = INF
    best_move = None
    moves = state.moves
    for current_depth in range(0, max_depth + 1, 2):
        for move in moves:
            v = min_value(
                game.result(state, move, compute_moves=False),
                current_depth, alpha=best_score, beta=beta
            )
            if v > best_score:
                best_score = v
                best_move = move
                THIS.BEST_MOVE = move
                if best_score == 1000:
                    return best_move
                moves.remove(best_move)
                moves.insert(0, best_move)
    return best_move


# ______________________________________________________________________________
# Negamax algorithm

def failsoft_negamax_alphabeta(game, state, timeout, max_depth,
                               alpha=-INF, beta=INF):
    '''
    Negamax with alpha-beta pruning, implemented in a fail-soft way, which
    means that if it fails it still returns the best result found so far.
    The fail-hard version would only return either alpha or beta.
    '''

    def negamax(state, depth, alpha, beta):
        if alphabeta_cutoff(game, state, depth, start_time, timeout):
            return heu.heuristic(state), None
        best_move = None
        best_value = -INF
        for move in game.actions(state):
            new_state = game.result(state, move)
            recursed_value, _ = negamax(
                state=new_state,
                depth=depth - 1,
                alpha=-beta,
                beta=-max(alpha, best_value)
            )
            current_value = -recursed_value
            if current_value > best_value:
                best_value = current_value
                best_move = move
                THIS.BEST_MOVE = move
                if best_value >= beta:
                    return best_value, best_move
        return best_value, best_move

    start_time = timeit.default_timer()
    return negamax(state, max_depth, alpha=alpha, beta=beta)


# ______________________________________________________________________________
# Aspiration search

def aspiration(game, state, timeout, previous, window_size, max_depth):
    '''
    Aspiration search
    '''
    alpha = previous - window_size
    beta = previous + window_size
    while True:
        result, move = failsoft_negamax_alphabeta(
            game, state, timeout, max_depth, alpha=alpha, beta=beta
        )
        if result <= alpha:
            alpha = -INF
        elif result >= beta:
            beta = INF
        else:
            return result, move


# ______________________________________________________________________________
# Negascout algorithm

def negascout_alphabeta(game, state, timeout, max_depth, alpha=-INF, beta=INF):
    '''
    Negascout with alpha-beta pruning
    '''

    def negascout(state, depth, alpha, beta):
        if alphabeta_cutoff(game, state, depth, start_time, timeout):
            return heu.heuristic(state), None
        best_move = None
        best_value = -INF
        adaptive_beta = beta
        for move in game.moves(state):
            new_state = game.result(state, move)
            recursed_value, _ = negascout(
                state=new_state,
                depth=depth - 1,
                alpha=-adaptive_beta,
                beta=-max(alpha, best_value)
            )
            current_value = -recursed_value
            if current_value > best_value:
                if adaptive_beta == beta or depth < 3 or current_value >= beta:
                    best_value = current_value
                    best_move = move
                    THIS.BEST_MOVE = move
                else:
                    negative_best_value, _ = negascout(
                        state=new_state,
                        depth=depth - 1,
                        alpha=-beta,
                        beta=-current_value
                    )
                    best_value = -negative_best_value
                if best_value >= beta:
                    return best_value, best_move
                adaptive_beta = max(alpha, best_value) + 1
        return best_value, best_move

    start_time = timeit.default_timer()
    return negascout(state, max_depth, alpha=alpha, beta=beta)


# ______________________________________________________________________________
# Monte Carlo tree search

def monte_carlo(game, state, timeout, max_it):
    '''
    Monte Carlo tree search
    '''

    def select(node):
        '''
        Select a leaf node in the tree
        '''
        return (
            select(max(node.children.keys(), key=ucb)) if node.children
            else node
        )

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
            action = get_random_move(state)
            state = game.result(state, action)
        return -game.utility(state, player)

    def backprop(node, utility):
        '''
        Passing the utility back to all parent nodes
        '''
        if utility > 0:
            node.win += utility
        node.visit += 1
        if node.parent is not None:
            backprop(node.parent, -utility)

    start_time = timeit.default_timer()
    root = MCTNode(state=state)
    for curr_it in range(max_it):
        if not monte_carlo_cutoff(start_time, timeout, curr_it, max_it):
            leaf = select(root)
            child = expand(leaf)
            result = simulate(game, child.state)
            backprop(child, result)
    max_state = max(root.children, key=lambda p: p.visit)
    return root.children.get(max_state)


class MCTNode:
    '''
    Node in the Monte Carlo search tree, to keep track of the children states
    '''

    def __init__(self, state=None, parent=None, win=0, visit=0):
        self.__dict__.update(state=state, parent=parent, win=win, visit=visit)
        self.children = {}


def ucb(node, const=math.sqrt(2)):
    '''
    Upper Confidence Bounds function
    '''
    return (
        INF if node.visit == 0
        else (
            node.win / node.visit +
            const * math.sqrt(math.log(node.parent.visit) / node.visit)
        )
    )

# ______________________________________________________________________________
# Cutoff and utility functions


def in_time(start_time, timeout):
    '''
    Check if there's no time left
    '''
    return timeit.default_timer() - start_time < timeout


def get_random_move(state, seed=None):
    '''
    Return a random legal move
    '''
    utils.set_random_seed(seed)
    return utils.get_rand(state.moves)


def alphabeta_cutoff(game, state, depth, start_time, timeout):
    '''
    Cut the search when the given state in the search tree is a final state,
    when there's no time left, or when the search depth has reached
    the given maximum
    '''
    return (
        depth == 0 or
        game.terminal_test(state) or
        not in_time(start_time, timeout)
    )


def monte_carlo_cutoff(start_time, timeout, curr_it, max_it):
    '''
    Cut the search when the maximum number of iterations is reached
    or when there's no time left
    '''
    return (
        curr_it > max_it or
        not in_time(start_time, timeout)
    )


# ______________________________________________________________________________
# Players

def random_player(game, state, **kwargs):
    '''
    Return a random legal move
    '''
    return get_random_move(state, seed=game.turn)


def minimax_player(game, state, timeout, max_depth, tt, **kwargs):
    '''
    Iterative deepening minimax player with alpha-beta pruning and
    transposition tables
    '''
    return minimax_alphabeta(game, state, timeout, max_depth, tt)


def negamax_player(game, state, timeout, max_depth=4, **kwargs):
    '''
    Negamax player with alpha-beta pruning
    '''
    return failsoft_negamax_alphabeta(game, state, timeout, max_depth)


def negascout_player(game, state, timeout, max_depth=4, **kwargs):
    '''
    Negascout player with alpha-beta pruning
    '''
    return negascout_alphabeta(game, state, timeout, max_depth)


def monte_carlo_player(game, state, timeout, max_it=1000, **kwargs):
    '''
    Monte Carlo player
    '''
    return monte_carlo(game, state, timeout, max_it)


def get_move(game, state, player, prev_move=None, **kwargs):
    '''
    Compute a move with the given player
    '''
    THIS.BEST_MOVE = None
    try:
        player = utils.timeout(conf.MOVE_TIMEOUT)(player)
        move = (
            first_move(state, prev_move) if game.turn < 2
            else player(game, state, **kwargs)
        )
    except TimeoutError:
        move = BEST_MOVE
    return move if move is not None else random_player(game, state)
