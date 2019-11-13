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
    Search strategy node alpha-beta type.
    It could represent the exact value of a position, its lower bound (beta)
    or its upper bound (alpha).
    '''

    EXACT, LOWER, UPPER = range(3)


class TranspositionTableEntry:

    def __init__(self, valued_action, depth, node_type):
        self.__dict__.update(
            valued_action=valued_action, depth=depth, node_type=node_type
        )


class TranspositionTable:

    def __init__(self):
        self.table = {}

    def get_action(self, state):
        entry = self.table.get(hash(state))
        return entry.valued_action if entry is not None else None

    def get_value(self, state, depth, alpha, beta):
        entry = self.table.get(hash(state))
        if entry is not None and entry.depth <= depth:
            if entry.node_type == TranspositionTableEntryType.EXACT:
                return entry.valued_action.value
            elif (entry.node_type == TranspositionTableEntryType.UPPER and
                  entry.valued_action.value <= alpha):
                return alpha
            elif (entry.node_type == TranspositionTableEntryType.LOWER and
                  entry.valued_action.value >= beta):
                return beta
        return None

    def put_action(self, valued_action, depth, node_type):
        self.table[hash(valued_action.state)] = TranspositionTableEntry(
            valued_action=valued_action,
            depth=depth,
            node_type=node_type
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
        value = TABLE.get_value(state, depth, alpha, beta)
        if value is not None:
            return value, 1
        if cutoff(game, state, depth, max_d, start_time, timeout):
            value = eval_fn(game.turn, action.state)
            TABLE.put_action(
                gutils.TablutValuedAction.from_action(action, value),
                depth, TranspositionTableEntryType.EXACT
            )
            return value, 1
        value = -INF
        total = 0
        for new_action in game.actions(state):
            new_value, index = min_value(
                new_action, alpha, beta, depth + 1, max_d
            )
            value = max(value, new_value)
            total += index
            if value >= beta:
                TABLE.put_action(
                    gutils.TablutValuedAction.from_action(new_action, value),
                    depth, TranspositionTableEntryType.LOWER
                )
                return value, total
            alpha = max(alpha, value)

        TABLE.put_action(
            gutils.TablutValuedAction.from_action(action, value),
            depth, TranspositionTableEntryType.EXACT
        )
        return value, total

    def min_value(action, alpha, beta, depth, max_d):
        state = action.state
        value = TABLE.get_value(state, depth, alpha, beta)
        if value is not None:
            return value, 1
        if cutoff(game, state, depth, max_d, start_time, timeout):
            value = eval_fn(game.turn, action.state)
            TABLE.put_action(
                gutils.TablutValuedAction.from_action(action, value),
                depth, TranspositionTableEntryType.EXACT
            )
            return value, 1
        value = INF
        total = 0
        for new_action in game.actions(state):
            new_value, index = max_value(
                new_action, alpha, beta, depth + 1, max_d
            )
            value = min(value, new_value)
            total += index
            if value <= alpha:
                TABLE.put_action(
                    gutils.TablutValuedAction.from_action(new_action, value),
                    depth, TranspositionTableEntryType.UPPER
                )
                return value, total
            beta = min(beta, value)

        TABLE.put_action(
            gutils.TablutValuedAction.from_action(action, value),
            depth, TranspositionTableEntryType.EXACT
        )
        return value, total

    start_time = time.time()
    best_score = -INF
    beta = INF
    best_action = None
    big_total = 0
    print(f'Ricerca Mosse:')
    for max_d in range(1, max_depth + 1):
        for action in game.actions(state):
            move = action.move
            value, total = min_value(
                action, best_score, beta, 1, max_d
            )
            big_total += total
            print(f'Mossa analizzata {move}')
            if value > best_score:
                print(
                    f'Old best score {best_score}, new best score {value}\n'
                    f'Old best action {best_action}, new best action {move}\n'
                    f'Total moves Analized: {total} \n'
                )
                best_score = value
                best_action = action
            if not in_time(start_time, timeout):
                break
        print(f'FINITO DEPTH {max_d}')
    print(f' The total amount of moves analized is {big_total}')
    TABLE.clear()
    return best_action.move

# ______________________________________________________________________________
# Negascout tree search


def negascout_search(game, state, eval_fn, depth, alpha, beta):
    '''
    Negascout search
    '''

    def failsoft_alphabeta(state, depth, alpha, beta):
        best_move = None
        current = -INF
        if game.terminal_test(state) or depth <= 0:
            return eval_fn(game.turn, state)
        for move, new_state, _ in game.valued_actions(state):
            score = -failsoft_alphabeta(new_state, depth - 1, -beta, -alpha)
            if score >= current:
                current = score
                best_move = move
                if score >= alpha:
                    alpha = score
                if score >= beta:
                    break
        return current

    def negascout(game, state, eval_fn, depth, alpha, beta):
        if game.terminal_test(state) or depth <= 0:
            return eval_fn(game.turn, state)
        moves = game.valued_actions(state)
        best_move, new_state, _ = moves[0]
        current = -failsoft_alphabeta(new_state, depth - 1, -beta, -alpha)
        for move, new_state, _ in moves[1:]:
            score = -failsoft_alphabeta(new_state,
                                        depth - 1, -alpha - 1, -alpha)
            if score > alpha and score < beta:
                score = -failsoft_alphabeta(new_state, depth - 1, -beta, -alpha)
            if score >= current:
                current = score
                best_move = move
                if score >= alpha:
                    alpha = score
                if score >= beta:
                    break
        return best_move

    return negascout(game, state, eval_fn, depth, alpha, beta)


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
    # move = None
    move = alphabeta_player(game, state, timeout, max_depth)
    # move = negascout_player(game, state, max_depth)
    # move = monte_carlo_player(game, state, timeout)
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


def negascout_player(game, state, max_depth):
    '''
    Negascout player
    '''
    return negascout_search(
        game, state, heu.heuristic, max_depth, -INF, INF
    )


def monte_carlo_player(game, state, timeout, max_it=1000):
    '''
    MCTS player
    '''
    return monte_carlo_tree_search(
        game, state, heu.heuristic, montecarlo_cutoff, timeout, max_it
    )


def alphabeta_cutoff(game, state, depth, max_depth, start_time, timeout):
    '''
    Cut the search when the given state in the search tree is a final state,
    when there's no time left, or when the search depth has reached
    the given maximum
    '''
    if not in_time(start_time, timeout):
        print('TIMEOUT')
        print(depth)
    return (
        depth >= max_depth or
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
