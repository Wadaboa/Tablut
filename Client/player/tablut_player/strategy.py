'''
Tablut players strategies
'''


from tablut_player.utils import INF
from tablut_player.board import TablutBoard
from tablut_player.game_utils import (
    TablutBoardPosition,
    TablutPawnType,
    TablutPlayerType
)


BLACK_BEST_POSITIONS = {
    TablutBoardPosition(row=2, col=1),
    TablutBoardPosition(row=1, col=2),
    TablutBoardPosition(row=1, col=6),
    TablutBoardPosition(row=2, col=7),
    TablutBoardPosition(row=6, col=1),
    TablutBoardPosition(row=7, col=2),
    TablutBoardPosition(row=6, col=7),
    TablutBoardPosition(row=7, col=6)
}


def minimax_decision(state, game):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -INF
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = INF
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minimax_decision:
    return max(game.actions(state),
               key=lambda a: min_value(game.result(state, a)))


def minimax_player(game, state):
    return minimax_decision(state, game)


def alphabeta_cutoff_search(state, game, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    player = game.to_move(state)

    # Functions used by alphabeta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(game.turn, state)
        v = -INF
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a),
                                 alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(game.turn, state)
        v = INF
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a),
                                 alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alphabeta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or
                   (lambda state, depth: depth > d or
                    game.terminal_test(state)))
    eval_fn = eval_fn or (lambda state: game.utility(state, player))
    best_score = -INF
    beta = INF
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action


def alphabeta_player(game, state, d=2):
    return alphabeta_cutoff_search(state, game, d, eval_fn=white_heuristic)


def black_heuristic(state):
    pass


def white_heuristic(turn, state):
    '''
    White player heuristic function
    '''
    value = TablutBoard.piece_difference_count(
        state.pawns, TablutPlayerType.WHITE
    )
    if turn < 40:
        value += king_moves_to_goals_count(state.pawns)
    elif turn < 70:
        value += 2 * king_moves_to_goals_count(state.pawns)
    else:
        value += 3 * king_moves_to_goals_count(state.pawns)
    return value


def king_moves_to_goals_count(pawns):
    '''
    Return a value representing the number of king moves to every goal.
    Given a state, it checks the min number of moves to each goal,
    and return a positive value if we are within 1-2 moves
    to a certain goal, and an even higher value
    if we are within 1-2 moves to more than one corner
    '''
    king = TablutBoard.king_position(pawns)
    total = 0.0
    for goal in TablutBoard.WHITE_GOALS:
        distance = TablutBoard.simulate_distance(pawns, king, goal)
        if distance == 0:
            total += INF
        elif distance == 1:
            total += 15
        elif distance == 2:
            total += 1
    return total


def blocking_escapes_positions_count(pawns):
    '''
    Return a value representing the number of black pawns in
    each of the two board spaces diagonally closest to each corner
    '''
    total = 0.0
    inc = 1 / len(BLACK_BEST_POSITIONS)
    for pawn in pawns[TablutPawnType.BLACK]:
        if pawn in BLACK_BEST_POSITIONS:
            total += inc
    return total


def potential_king_killers_count(pawns):
    '''
    Return a value representing the number of black pawns,
    camps and castle around the king.
    '''
    killers = TablutBoard.potential_king_killers(pawns)
    return killers * 0.25
