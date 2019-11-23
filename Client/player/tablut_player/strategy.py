'''
Tablut players search strategies
'''


import math
import time
import random
from enum import Enum

import tablut_player.game_utils as gutils
import tablut_player.utils as utils
import tablut_player.heuristic as heu
from tablut_player.utils import INF
from tablut_player.game_utils import TablutBoardPosition as TBPos


WHITE_OPENINGS = [
    (TBPos(row=4, col=2), TBPos(row=7, col=2)),
    (TBPos(row=4, col=2), TBPos(row=8, col=2))
]
BLACK_OPENINGS = {
    (TBPos(row=4, col=2), TBPos(row=5, col=2)): [
        (TBPos(row=5, col=8), TBPos(row=5, col=7))
    ],
    (TBPos(row=4, col=2), TBPos(row=6, col=2)): [
        (TBPos(row=3, col=0), TBPos(row=3, col=2))
    ],
    (TBPos(row=4, col=2), TBPos(row=7, col=2)): [
        (TBPos(row=3, col=0), TBPos(row=3, col=2))
    ],
    (TBPos(row=4, col=2), TBPos(row=8, col=2)): [
        (TBPos(row=4, col=1), TBPos(row=8, col=1)),
        (TBPos(row=3, col=0), TBPos(row=3, col=2))
    ],
    (TBPos(row=4, col=3), TBPos(row=5, col=3)): [
        (TBPos(row=0, col=3), TBPos(row=4, col=3))
    ],
    (TBPos(row=4, col=3), TBPos(row=6, col=3)): [
        (TBPos(row=0, col=3), TBPos(row=4, col=3))
    ],
    (TBPos(row=4, col=3), TBPos(row=7, col=3)): [
        (TBPos(row=0, col=3), TBPos(row=4, col=3))
    ]
}

# ______________________________________________________________________________
# Alpha-Beta tree search


class TTEntry:

    def __init__(self, key, moves, value, depth):
        self.key = key
        self.moves = moves
        self.value = value
        self.depth = depth

    def __repr__(self):
        return (
            f'Key: {self.key}\n'
            f'Moves: {self.moves}\n'
            f'Value: {self.value}\n'
            f'Depth: {self.depth}'
        )


class TT:

    def __init__(self):
        self.table = {}

    def get_entry(self, state):
        hash_value = hash(state)
        entry = self.table.get(hash_value)
        if entry is not None and entry.key == hash_value:
            return entry
        return None

    def get_moves(self, state):
        entry = self.get_entry(state)
        return entry.moves if entry is not None else None

    def get_value(self, state):
        entry = self.get_entry(state)
        return entry.value if entry is not None else None

    def store_entry(self, entry):
        self.table[entry.key] = entry

    def clear(self):
        self.table = {}


def alphabeta_cutoff_search(state, game, timeout, d=2, tt=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    # Functions used by alphabeta
    def max_value(state, alpha, beta, depth):
        entry = tt.get_entry(state)
        if entry is not None and entry.moves is not None:
            state.moves = entry.moves
        else:
            state.moves = game.player_moves(state.pawns, state.to_move)

        moves = state.moves

        if alphabeta_cutoff(game, state, depth, start_time, timeout):
            if entry is not None and entry.depth == depth:
                value = entry.value
            else:
                value = heu.heuristic(game.turn, state)
                entry = TTEntry(hash(state), moves, value, depth)
                tt.store_entry(entry)
            # print(value)
            return value
        v = -INF
        for move in moves:
            value = min_value(game.result(state, move, compute_moves=False),
                              alpha, beta, depth - 1)
            if value > v:
                v = value
                moves.remove(move)
                moves.insert(0, move)
            if v >= beta:
                entry = TTEntry(hash(state), moves, v, depth)
                tt.store_entry(entry)
                return v
            alpha = max(alpha, v)
        entry = TTEntry(hash(state), moves, v, depth)
        tt.store_entry(entry)
        return v

    def min_value(state, alpha, beta, depth):
        entry = tt.get_entry(state)
        if entry is not None and entry.moves is not None:
            state.moves = entry.moves
        else:
            state.moves = game.player_moves(state.pawns, state.to_move)

        moves = state.moves

        if alphabeta_cutoff(game, state, depth, start_time, timeout):
            if entry is not None and entry.depth == depth:
                value = entry.value
            else:
                value = heu.heuristic(game.turn, state)
                entry = TTEntry(hash(state), moves, value, depth)
                tt.store_entry(entry)
            # print(value)
            return value
        v = INF
        for move in moves:
            value = max_value(game.result(state, move, compute_moves=False),
                              alpha, beta, depth - 1)
            if value < v:
                v = value
                moves.remove(move)
                moves.insert(0, move)
            if v <= alpha:
                entry = TTEntry(hash(state), moves, v, depth)
                tt.store_entry(entry)
                return v
            beta = min(beta, v)
        entry = TTEntry(hash(state), moves, v, depth)
        tt.store_entry(entry)
        return v

    start_time = time.time()
    best_score = -INF
    beta = INF
    best_move = None
    moves = state.moves
    for current_depth in range(0, d + 1, 2):
        for move in moves:
            # print(move)
            v = min_value(game.result(state, move, compute_moves=False),
                          best_score, beta, current_depth)
            if v > best_score:
                best_score = v
                best_move = move
                if best_score == 1000:
                    return best_move
                moves.remove(best_move)
                moves.insert(0, best_move)
                #print(f'The best move is {best_move}')
        #print(f'END DEPTH {current_depth}')

    final_time = time.time() - start_time
    # print(
    #    f'Tempo di ricerca:{final_time} -heuristica migliore:{best_score} - mossa migliore:{best_move}')
    return best_move

# ______________________________________________________________________________
# Monte Carlo tree search


def monte_carlo_tree_search(game, state, eval_fn, cutoff, timeout, max_it=1000):
    '''
    Monte Carlo tree search
    '''

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
                action for action in game.moves(node.state)
            }
        return select(node)

    def simulate(game, state):
        '''
        Simulate the utility of current state by random picking a step
        '''
        player = game.to_move(state)
        while not game.terminal_test(state):
            action = random.choice(list(game.moves(state)))
            state = game.result(state, action)
        return -game.utility(state, player)

    def backprop(node, value):
        '''
        Passing the utility back to all parent nodes
        '''
        if value > 0:
            node.win += value
        # if utility == 0:
        #     n.win += 0.5
        node.visit += 1
        if node.parent is not None:
            backprop(node.parent, -value)

    start_time = time.time()
    root = MCTNode(state=state)
    for curr_it in range(max_it):
        if not cutoff(start_time, timeout, curr_it, max_it):
            leaf = select(root)
            child = expand(leaf)
            result = simulate(game, child.state)
            backprop(child, result)

    max_state = max(root.children, key=lambda p: p.visit)
    return root.children.get(max_state)


# ______________________________________________________________________________
# Monte Carlo tree node and ucb function


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


def get_move(game, state, timeout, max_depth=4, prev_move=None, tt=None):
    #move = None
    # move = alphabeta_player(game, state, timeout, max_depth)
    # move = monte_carlo_player(game, state, timeout)
    # value, move = iterative_deepening_negamax(
    #    game, state, depth=3, alpha=-INF, beta=INF
    # )

    # value, move = negascout_alphabeta(game, state, 1, -INF, INF)
    if game.turn < 2:
        move = first_move(state, prev_move)
    else:
        move = alphabeta_cutoff_search(state, game, timeout, 0, tt)
        # print(move)

    # print(value)

    if move is None:
        #print('Alphabeta failure')
        move = random_player(state)
    return move


def init_openings():
    # White openings
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

    # Black openings
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


def first_move(state, prev_move=None):
    '''
    First opening move
    '''
    if state.to_move == gutils.TablutPlayerType.WHITE:
        return utils.get_rand(WHITE_OPENINGS)
    elif prev_move is not None:
        return utils.get_rand(BLACK_OPENINGS[prev_move])
    return None


def random_player(state):
    '''
    Return a random legal move
    '''
    return utils.get_from_set(state.moves)


def alphabeta_player(game, state, timeout, max_depth):
    '''
    Alphabeta player
    '''
    return alphabeta_search(
        game, state, heu.heuristic, alphabeta_cutoff, timeout, max_depth
    )


def monte_carlo_player(game, state, timeout, max_it=1000):
    '''
    MCTS player
    '''
    return monte_carlo_tree_search(
        game, state, heu.heuristic, montecarlo_cutoff, timeout, max_it
    )


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


def montecarlo_cutoff(start_time, timeout, curr_it, max_it=1000):
    '''
    Cut the search when the maximum number of iterations is reached
    or when there's no time left
    '''
    return (
        curr_it > max_it or
        not in_time(start_time, timeout)
    )


def in_time(start_time, timeout):
    '''
    Check if there's no time left
    '''
    return time.time() - start_time < timeout


def failsoft_negamax_alphabeta(game, state, depth, alpha, beta,
                               moves=None, tt=None):
    '''
    Negamax with alpha-beta pruning, implemented in a fail-soft way, which
    means that if it fails it still returns the best result found so far.
    The fail-hard version would only return either alpha or beta.
    '''
    '''
    # Lookup transposition table
    if tt is not None:
        entry = tt.get_entry(state)
        if entry is not None and entry.depth >= depth:
            if entry.entry_type == TTEntryType.EXACT:
                print('get1')
                return entry.best_move
            elif (entry.node_type == TTEntryType.LOWER and
                  entry.valued_action.value > alpha):
                print('get2')
                alpha = entry.value
            elif (entry.node_type == TTEntryType.UPPER and
                  entry.valued_action.value < beta):
                print('get3')
                beta = entry.value
            if alpha >= beta:
                print('get4')
                return entry.best_move
    '''
    random.seed(int(time.time()))
    # Negamax
    if game.terminal_test(state) or depth == 0:
        return heu.heuristic(game.turn, state), None
    if moves is None:
        moves = list(game.moves(state))
        random.shuffle(moves)
    best_move = None
    best_value = -INF
    for move in moves:
        new_state = game.result(state, move)
        recursed_value, _ = failsoft_negamax_alphabeta(
            game=game,
            state=new_state,
            depth=depth - 1,
            alpha=-beta,
            beta=-max(alpha, best_value),
            tt=tt
        )
        current_value = -recursed_value
        if current_value > best_value:
            best_value = current_value
            best_move = move
            print(
                f'Nuova best move per {state.to_move} :{best_move} Best-value:{best_value} - Depth: {depth}')
            if best_value >= beta:
                print(f'Taglio effettuato Beta:{beta} Best-value:{best_value}')
                return best_value, best_move
    '''
    # Update transposition table
    if tt is not None:
        entry = TTEntry(
            key=hash(state),
            entry_type=None,
            value=best_value,
            depth=depth,
            best_move=best_move
        )
        if best_value <= alpha:
            entry.entry_type = TTEntryType.LOWER
        elif best_value >= beta:
            entry.entry_type = TTEntryType.UPPER
        else:
            entry.entry_type = TTEntryType.EXACT
        tt.store_entry(entry)
    '''
    print(
        f'Nuova mossa per {state.to_move} : {best_move} - Heu: {best_value} - Depth: {depth}')
    return best_value, best_move


def iterative_deepening_negamax(game, state, depth, alpha, beta):
    '''
    Fail-soft negamax with alpha-beta pruning and iterative deepening
    '''
    best_value = alpha
    best_move = None
    moves = list(state.moves)
    random.shuffle(moves)
    tt = TT()
    for d in range(1, depth + 1):
        value, move = failsoft_negamax_alphabeta(
            game, state, d, best_value, beta, moves, tt
        )
        if move is not None and value > best_value:
            best_value = value
            best_move = move
            moves.remove(best_move)
            moves.insert(0, best_move)
            print(f'The best move is {best_move}')
        print(f'END DEPTH {d}')
    tt.clear()
    return best_value, best_move


def aspiration(game, state, depth, alpha, beta, previous, window_size):
    '''
    Aspiration search
    '''
    alpha = previous - window_size
    beta = previous + window_size
    while True:
        result, move = failsoft_negamax_alphabeta(
            game, state, depth, alpha, beta
        )
        if result <= alpha:
            alpha = -INF
        elif result >= beta:
            beta = INF
        else:
            return result, move


def negascout_alphabeta(game, state, depth, alpha, beta):
    '''
    Negascout with alpha-beta pruning
    '''
    if game.terminal_test(state) or depth == 0:
        return heu.heuristic(game.turn, state), None
    best_move = None
    best_value = -INF
    adaptive_beta = beta
    for move in game.moves(state):
        new_state = game.result(state, move)
        recursed_value, _ = negascout_alphabeta(
            game=game,
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
            else:
                negative_best_value, _ = negascout_alphabeta(
                    game=game,
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
