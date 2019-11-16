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


class TranspositionTableEntryType(Enum):
    '''
    Search strategy node alpha-beta type. It could represent
    the exact value of a position, its lower bound (beta),
    or its upper bound (alpha).
    '''

    EXACT, LOWER, UPPER = range(3)

    def __str__(self):
        return f'{self.name}'

    def __repr__(self):
        return f'Type: {self.name}'


class TranspositionTableEntry:

    def __init__(self, valued_action, depth, node_type, best_move):
        self.valued_action = valued_action
        self.depth = depth
        self.node_type = node_type
        self.best_move = best_move

    def __repr__(self):
        return (
            f'Action: {self.valued_action}\n'
            f'Depth: {self.depth}\n'
            f'Type: {self.node_type}\n'
            f'Best move: {self.best_move}'
        )


class TranspositionTable:

    def __init__(self):
        self.table = {}

    def get_action(self, state):
        entry = self.get_entry(state)
        return entry.valued_action if entry is not None else None

    def get_entry(self, state):
        return self.table.get(hash(state))

    def get_value(self, state, depth, alpha, beta):
        entry = self.table.get(hash(state))
        value = None
        if entry is not None and entry.depth >= depth:
            if entry.node_type == TranspositionTableEntryType.EXACT:
                value = entry.valued_action.value
            elif (entry.node_type == TranspositionTableEntryType.LOWER and
                  entry.valued_action.value > alpha):
                alpha = entry.valued_action.value
            elif (entry.node_type == TranspositionTableEntryType.UPPER and
                  entry.valued_action.value < beta):
                beta = entry.valued_action.value
            if alpha >= beta:
                value = entry.valued_action.value
        return value, alpha, beta

    def put_action(self, valued_action, depth, best_move, node_type):
        self.table[hash(valued_action.state)] = TranspositionTableEntry(
            valued_action=valued_action,
            depth=depth,
            node_type=node_type,
            best_move=best_move
        )

    def clear(self):
        self.table = {}

# ______________________________________________________________________________
# Alpha-Beta tree search


TABLE = TranspositionTable()


def alphabeta_search(game, state, eval_fn, cutoff, timeout, max_depth):
    '''
    Minimax with alpha-beta pruning search strategy
    '''

    def max_value(action, alpha, beta, depth, max_d):
        state = action.state
        value, alpha, beta = TABLE.get_value(state, depth, alpha, beta)
        if value is not None:
            return value, 1
        if cutoff(game, state, depth, start_time, timeout):
            value = eval_fn(game.turn, action.state)
            node_type = TranspositionTableEntryType.EXACT
            if value <= alpha:
                node_type = TranspositionTableEntryType.LOWER
            elif value >= beta:
                node_type = TranspositionTableEntryType.UPPER
            TABLE.put_action(
                gutils.TablutValuedAction.from_action(action, value),
                max_d, None, node_type
            )
            return value, 1
        value = -INF
        total = 0
        for new_action in game.ordered_valued_actions(state):
            new_value, index = min_value(
                new_action, alpha, beta, depth - 1, max_d
            )
            value = max(value, new_value)
            total += index
            if value >= beta:
                '''
                TABLE.put_action(
                    gutils.TablutValuedAction.from_action(action, value),
                    max_d, TranspositionTableEntryType.UPPER
                )
                '''
                return value, total
            alpha = max(alpha, value)

        node_type = TranspositionTableEntryType.EXACT
        if value <= alpha:
            node_type = TranspositionTableEntryType.LOWER
        elif value >= beta:
            node_type = TranspositionTableEntryType.UPPER
        TABLE.put_action(
            gutils.TablutValuedAction.from_action(action, value),
            max_d, None, node_type
        )
        return value, total

    def min_value(action, alpha, beta, depth, max_d):
        state = action.state
        value, alpha, beta = TABLE.get_value(state, depth, alpha, beta)
        if value is not None:
            return value, 1
        if cutoff(game, state, depth, start_time, timeout):
            value = eval_fn(game.turn, action.state)
            value = eval_fn(game.turn, action.state)
            node_type = TranspositionTableEntryType.EXACT
            if value <= alpha:
                node_type = TranspositionTableEntryType.LOWER
            elif value >= beta:
                node_type = TranspositionTableEntryType.UPPER
            TABLE.put_action(
                gutils.TablutValuedAction.from_action(action, value),
                max_d, None, node_type
            )
            return value, 1
        value = INF
        total = 0
        for new_action in game.ordered_valued_actions(state):
            new_value, index = max_value(
                new_action, alpha, beta, depth - 1, max_d
            )
            value = min(value, new_value)
            total += index
            if value <= alpha:
                '''
                TABLE.put_action(
                    gutils.TablutValuedAction.from_action(action, value),
                    best_move, max_d, TranspositionTableEntryType.UPPER
                )
                '''
                return value, total
            beta = min(beta, value)

        node_type = TranspositionTableEntryType.EXACT
        if value <= alpha:
            node_type = TranspositionTableEntryType.LOWER
        elif value >= beta:
            node_type = TranspositionTableEntryType.UPPER
        TABLE.put_action(
            gutils.TablutValuedAction.from_action(action, value),
            max_d, None, node_type
        )
        return value, total

    start_time = time.time()
    best_score = -INF
    beta = INF
    best_action = None
    big_total = 0
    print(f'Ricerca Mosse:')
    for max_d in range(max_depth, -1, -1):
        actions = game.ordered_valued_actions(state)
        for action in actions:
            move = action.move
            value, total = min_value(
                action, best_score, beta, max_depth - max_d, max_d
            )
            big_total += total
            print(f'Mossa analizzata {move}')
            if value > best_score:
                print(
                    f'Old best score {best_score}, new best score {value}\n'
                    f'Old best action {best_action.move if best_action is not None else "None"}, new best action {move}\n'
                    f'Total moves Analized: {total} \n'
                )
                best_score = value
                best_action = action
            if not in_time(start_time, timeout):
                break
        print(f'FINITO DEPTH {max_depth - max_d}')
    print(f' The total amount of moves analized is {big_total}')
    print(TABLE.table[hash(best_action.state)])
    return best_action.move


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


def get_move(game, state, timeout, max_depth=4):
    #move = None
    # move = alphabeta_player(game, state, timeout, max_depth)
    # move = monte_carlo_player(game, state, timeout)
    # value, move = iterative_deepening_negamax(
    #    game, state, depth=3, alpha=-INF, beta=INF
    # )
    # value, move = failsoft_negamax_alphabeta(game, state, 2, -INF, INF)
    # value, move = negascout_alphabeta(game, state, 1, -INF, INF)
    if game.turn < 3:
        move = random_player(state)
    else:
        move = alphabeta_cutoff_search(state, game, d=0, timeout=60)
    # print(value)
    print(move)
    if move is None:
        print('Alphabeta failure')
        move = random_player(state)
    return move


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
    # or not in_time(start_time, timeout)
    return (
        depth == 0 or
        game.terminal_test(state)
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


class TTEntryType(Enum):
    '''
    Search strategy node alpha-beta type. It could represent
    the exact value of a position, its lower bound (beta),
    or its upper bound (alpha).
    '''

    EXACT, LOWER, UPPER = range(3)

    def __str__(self):
        return f'{self.name}'

    def __repr__(self):
        return f'Type: {self.name}'


class TTEntry:

    def __init__(self, key, entry_type, value, depth, best_move):
        self.key = key
        self.entry_type = entry_type
        self.value = value
        self.depth = depth
        self.best_move = best_move

    def __repr__(self):
        return (
            f'Key: {self.key}\n'
            f'Type: {self.entry_type}\n'
            f'Value: {self.value}\n'
            f'Depth: {self.depth}\n'
            f'Best move: {self.best_move}'
        )


class TT:

    TABLE_SIZE = 4000

    def __init__(self):
        self.table = {}

    def get_entry(self, state):
        hash_value = hash(state)
        entry = self.table.get(hash_value % self.TABLE_SIZE)
        if entry is not None and entry.key == hash_value:
            return entry
        return None

    def get_value(self, state):
        entry = self.get_entry(state)
        return entry.value if entry is not None else None

    def store_entry(self, entry):
        self.table[entry.key % self.TABLE_SIZE] = entry

    def clear(self):
        self.table = {}


def alphabeta_cutoff_search(state, game, d=4, timeout=60):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    player = game.to_move(state)

    # Functions used by alphabeta
    def max_value(state, alpha, beta, depth):
        if alphabeta_cutoff(game, state, depth, start_time, timeout):
            return heu.heuristic(game.turn, state)
        v = -INF
        moves = list(game.moves(state))
        random.seed(time.time())
        random.shuffle(moves)
        for move in moves:
            v = max(v, min_value(game.result(state, move),
                                 alpha, beta, depth - 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if alphabeta_cutoff(game, state, depth, start_time, timeout):
            return heu.heuristic(game.turn, state)
        v = INF
        moves = list(game.moves(state))
        random.seed(time.time())
        random.shuffle(moves)
        for move in moves:
            v = min(v, max_value(game.result(state, move),
                                 alpha, beta, depth - 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    start_time = time.time()
    best_score = -INF
    beta = INF
    best_action = None
    moves = list(game.moves(state))
    random.seed(time.time())
    random.shuffle(moves)
    for move in moves:
        print(move)
        v = min_value(game.result(state, move), best_score, beta, d)
        if v > best_score:
            best_score = v
            best_action = move
    final_time = time.time()-start_time
    print(
        f'Tempo di ricerca:{final_time} -heuristica migliore:{best_score} - mossa migliore:{best_action}')
    return best_action
