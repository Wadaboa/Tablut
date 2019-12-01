'''
Tablut players search strategies
'''


import math
import timeit

import tablut_player.config as conf
import tablut_player.game_utils as gutils
import tablut_player.utils as utils
import tablut_player.heuristic as heu
from tablut_player.board import TablutBoard
from tablut_player.utils import INF
from tablut_player.game_utils import TablutBoardPosition as TBPos


BEST_MOVE = None


# ______________________________________________________________________________
# Opening moves

WHITE_OPENINGS = [
    (TBPos.create(row=4, col=2), TBPos.create(row=7, col=2))
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

    def __init__(self, state_key, player_key, value, moves=None, height=None):
        self.state_key = state_key
        self.player_key = player_key
        self.value = value
        self.moves = moves
        self.height = height

    def __repr__(self):
        return (
            f'State key: {self.state_key}\n'
            f'Player key: {self.player_key}\n'
            f'Value: {self.value}\n'
            f'Moves: {self.moves}\n'
            f'Height: {self.height}'
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

    def store_exp_entry(self, state, value, moves=None, height=None):
        '''
        Store the given entry in the transposition table
        '''
        entry = TTEntry(
            hash(state), state.ZOBRIST_KEYS.to_move[state.to_move],
            value, moves, height
        )
        self.store_entry(entry)

    def clear(self):
        '''
        Delete every entry from the transposition table
        '''
        self.table = {}


# ______________________________________________________________________________
# Minimax algorithm

def minimax_alphabeta(kill, game, state, max_depth, tt, heu_tt):
    '''
    Iterative deepening minimax with alpha-beta pruning and transposition tables
    '''

    def max_value(state, height, depth, alpha, beta):
        entry = tt.get_entry(state)
        state.moves = (
            entry.moves if entry is not None and entry.moves is not None
            else game.player_moves(state.pawns, state.to_move)
        )
        max_moves = state.moves

        if alphabeta_cutoff(kill, state, depth):
            heu_entry = heu_tt.get_entry(state)
            value = (
                heu_entry.value if heu_entry is not None
                else heu.heuristic(game, state, color)
            )
            print(f'HEU_MAX:{value}')
            heu_tt.store_exp_entry(state, value)
            return value
        elif entry is not None and entry.height == depth:
            value = entry.value
            print(f'TT_MAX:{value}')
            return value
        v = -INF
        for i in range(len(max_moves)):
            print(max_moves[i])
            value = min_value(
                game.result(state, max_moves[i], compute_moves=False), height,
                depth - 1, alpha=alpha, beta=beta
            )
            if value > v:
                v = value
                max_moves[:] = (
                    [max_moves[i]] + max_moves[:i] + max_moves[i + 1:]
                )
                if v >= beta:
                    return v
            alpha = max(alpha, v)
            if kill.is_set():
                break
        tt.store_exp_entry(state, v, max_moves, height)
        return v

    def min_value(state, height, depth, alpha, beta):
        entry = tt.get_entry(state)
        state.moves = (
            entry.moves if entry is not None and entry.moves is not None
            else game.player_moves(state.pawns, state.to_move)
        )
        min_moves = state.moves

        if alphabeta_cutoff(kill, state, depth):
            heu_entry = heu_tt.get_entry(state)
            value = (
                heu_entry.value if heu_entry is not None
                else heu.heuristic(game, state, color)
            )
            print(f'HEU_MIN:{value}')
            heu_tt.store_exp_entry(state, value)
        elif entry is not None and entry.height == depth:
            value = entry.value
            print(f'TT_MIN:{value}')
            return value
        v = INF
        for i in range(len(min_moves)):
            print(min_moves[i])
            value = max_value(
                game.result(state, min_moves[i], compute_moves=False), height,
                depth - 1, alpha=alpha, beta=beta
            )
            if value == -1000 and depth == 1:
                losing_moves.append(min_moves[i])
            if value < v:
                v = value
                min_moves[:] = (
                    [min_moves[i]] + min_moves[:i] + min_moves[i + 1:]
                )
                if v <= alpha:
                    return v
            beta = min(beta, v)
            if kill.is_set():
                break
        tt.store_exp_entry(state, v, min_moves, height)
        return v

    global BEST_MOVE
    best_move = None
    moves = state.moves
    color = state.to_move
    for current_depth in range(0, 1, 1):
        if kill.is_set():
            print(f'QUITTING MINIMAX AT DEPTH {current_depth}')
            break
        losing_moves = []
        best_score = -INF
        beta = INF
        for i in range(len(moves)):
            print(moves[i])
            v = min_value(
                game.result(state, moves[i], compute_moves=False),
                current_depth, current_depth, alpha=best_score, beta=beta
            )
            if v > best_score:
                best_score = v
                best_move = moves[i]
                BEST_MOVE = moves[i]
                if current_depth == 0 and best_score == 1000:
                    print('QUITTING MINIMAX, FOUND WINNING MOVE')
                    return best_move
                if not kill.is_set():
                    moves[:] = [moves[i]] + moves[:i] + moves[i + 1:]
            if kill.is_set():
                print(
                    f'QUITTING MINIMAX AT DEPTH {current_depth}, '
                    f'WITHOUT FINISHING MOVES ANALYSIS'
                )
                break
        print(f'fine depth: {current_depth}')
        print(losing_moves)
        print()
        moves = [m for m in moves if m not in losing_moves]

    return best_move


# ______________________________________________________________________________
# Negamax algorithm

def failsoft_negamax_alphabeta(kill, game, state, max_depth,
                               alpha=-INF, beta=INF):
    '''
    Negamax with alpha-beta pruning, implemented in a fail-soft way, which
    means that if it fails it still returns the best result found so far.
    The fail-hard version would only return either alpha or beta.
    '''

    def negamax(state, depth, alpha, beta):
        if alphabeta_cutoff(kill, state, depth):
            return heu.heuristic(game, state, color), None
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
                BEST_MOVE = move
                if best_value >= beta:
                    return best_value, best_move
            if kill.is_set():
                break
        return best_value, best_move

    global BEST_MOVE
    color = state.to_move
    _, move = negamax(state, max_depth, alpha=alpha, beta=beta)
    return move


# ______________________________________________________________________________
# Negascout algorithm

def negascout_alphabeta(kill, game, state, max_depth, alpha=-INF, beta=INF):
    '''
    Negascout with alpha-beta pruning
    '''

    def negascout(state, depth, alpha, beta):
        if alphabeta_cutoff(kill, state, depth):
            return heu.heuristic(game, state, color), None
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
                    BEST_MOVE = move
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
            if kill.is_set():
                break
        return best_value, best_move

    global BEST_MOVE
    color = state.to_move
    _, move = negascout(state, max_depth, alpha=alpha, beta=beta)
    return move


# ______________________________________________________________________________
# Monte Carlo tree search

def monte_carlo(kill, game, state, max_it):
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
        if not node.children and not node.state.is_terminal:
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
        while not state.is_terminal:
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

    root = MCTNode(state=state)
    for curr_it in range(max_it):
        if not monte_carlo_cutoff(kill, curr_it, max_it):
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


def black_survival(state):
    '''
    Return a move that blocks white player winning situation
    '''
    king_moves = TablutBoard.legal_moves(
        state.pawns, TablutBoard.king_position(state.pawns)
    )
    for white_position in TablutBoard.WHITE_GOALS:
        for king_to_move in king_moves:
            if white_position == king_to_move:
                for black_from_move, black_to_move in state.moves:
                    if king_to_move == black_to_move:
                        return (black_from_move, black_to_move)
    return None


def white_survival(game, state):
    '''
    Return a move that blocks black player winning situation
    '''
    good_moves = [(m, 0) for m in game.actions(state)]
    initial_len = len(good_moves)
    for i, white_move in enumerate(game.actions(state)):
        new_state_by_white = game.result(state, white_move, compute_moves=True)
        for black_move in game.actions(new_state_by_white):
            new_state_by_black = game.result(new_state_by_white, black_move)
            if game.will_king_be_dead_by_move(new_state_by_black, black_move):
                del good_moves[i]
            else:
                value = heu.piece_difference(new_state_by_black)
                m, v = good_moves[i]
                good_moves[i] = (m, min(v, value))
    if len(good_moves) < initial_len:
        best, _ = min(good_moves, key=lambda tup: tup[1])
        return best
    return None


def alphabeta_cutoff(kill, state, depth):
    '''
    Cut the search when the given state in the search tree is a final state,
    when there's no time left, or when the search depth has reached
    the given maximum
    '''
    return (
        kill.is_set() or
        depth == 0 or
        state.is_terminal
    )


def monte_carlo_cutoff(kill, curr_it, max_it):
    '''
    Cut the search when the maximum number of iterations is reached
    or when there's no time left
    '''
    return (
        curr_it > max_it or
        kill.is_set()
    )


# ______________________________________________________________________________
# Players

def random_player(kill, game, state, **kwargs):
    '''
    Return a random legal move
    '''
    return get_random_move(state, seed=game.turn)


def minimax_player(kill, game, state, max_depth, tt, heu_tt, **kwargs):
    '''
    Iterative deepening minimax player with alpha-beta pruning and
    transposition tables
    '''
    return minimax_alphabeta(kill, game, state, max_depth, tt, heu_tt)


def negamax_player(kill, game, state, max_depth=4, **kwargs):
    '''
    Negamax player with alpha-beta pruning
    '''
    return failsoft_negamax_alphabeta(kill, game, state, max_depth)


def negascout_player(kill, game, state, max_depth=4, **kwargs):
    '''
    Negascout player with alpha-beta pruning
    '''
    return negascout_alphabeta(kill, game, state, max_depth)


def monte_carlo_player(kill, game, state, max_it=1000, **kwargs):
    '''
    Monte Carlo player
    '''
    return monte_carlo(kill, game, state, max_it)


def get_move(game, state, player, prev_move=None, **kwargs):
    '''
    Compute a move with the given player
    '''
    global BEST_MOVE
    BEST_MOVE = None
    try:
        player = utils.timeout(conf.MOVE_TIMEOUT)(player)
        move = (
            first_move(state, prev_move) if game.turn < 2
            else player(game, state, **kwargs)
        )
    except TimeoutError:
        move = BEST_MOVE
    return (
        move if move is not None else get_random_move(state, seed=game.turn)
    )
