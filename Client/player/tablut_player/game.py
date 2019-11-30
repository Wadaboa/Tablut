'''
Module containing a generic game representation,
partially taken from AIMA library.
'''


from typing import overload
import random

import tablut_player.game_utils as gutils
import tablut_player.utils as utils
import tablut_player.config as conf
from tablut_player.game_utils import (
    TablutBoardPosition,
    TablutPawnType,
    TablutPlayerType,
    TablutGameState
)
from tablut_player.board import TablutBoard


class Game:
    '''
    A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor.
    '''

    def actions(self, state):
        '''
        Return a list of the allowable moves at this point.
        '''
        raise NotImplementedError

    def result(self, state, move):
        '''
        Return the state that results from making a move from a state.
        '''
        raise NotImplementedError

    def utility(self, state, player):
        '''
        Return the value of this final state to player.
        '''
        raise NotImplementedError

    def terminal_test(self, state):
        '''
        Return True if this is a final state for the game.
        '''
        raise NotImplementedError

    def to_move(self, state):
        '''
        Return the player whose move it is in this state.
        '''
        return state.to_move

    def display(self, state):
        '''
        Print or otherwise display the state.
        '''
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        '''
        Play an n-person, move-alternating game.
        '''
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))


class TablutGame(Game):
    '''
    Tablut game representation
    '''

    MAX_REPEATED_STATES = 4

    def __init__(self, initial_pawns=None, to_move=TablutPlayerType.WHITE):
        if initial_pawns is None:
            initial_pawns = self._init_pawns()
        self.initial = TablutGameState(
            to_move=to_move,
            utility=0,
            is_terminal=False,
            pawns=initial_pawns,
            moves=self.player_moves(initial_pawns, to_move),
            old_state=None
        )
        self._init_zobrist()
        self.turn = 0

    def _init_pawns(self):
        '''
        Pawns initial values
        '''
        pawns = {}
        white_pawns_positions = {
            TablutBoardPosition.create(row=2, col=4),
            TablutBoardPosition.create(row=3, col=4),
            TablutBoardPosition.create(row=5, col=4),
            TablutBoardPosition.create(row=6, col=4),
            TablutBoardPosition.create(row=4, col=2),
            TablutBoardPosition.create(row=4, col=3),
            TablutBoardPosition.create(row=4, col=5),
            TablutBoardPosition.create(row=4, col=6)
        }
        pawns[TablutPawnType.WHITE] = white_pawns_positions
        pawns[TablutPawnType.BLACK] = TablutBoard.CAMPS
        pawns[TablutPawnType.KING] = {TablutBoard.CASTLE}
        return pawns

    def _init_zobrist(self):
        '''
        Generate zobrist random bitstrings
        '''
        pawn_types = TablutPawnType.values()
        player_types = TablutPlayerType.values()
        is_terminal = [True, False]
        dim = (
            ((conf.BOARD_SIZE ** 2) * len(pawn_types)) +
            len(player_types) + len(is_terminal)
        )
        keys = set()
        utils.set_random_seed(dim)
        while len(keys) < dim:
            keys.add(random.getrandbits(64))
        for i in range(conf.BOARD_SIZE):
            for j in range(conf.BOARD_SIZE):
                pos = TablutBoardPosition.create(row=i, col=j)
                for pawn_type in TablutPawnType.values():
                    key = keys.pop()
                    TablutGameState.ZOBRIST_KEYS.board.setdefault(
                        pos, {}
                    )[pawn_type] = key
        for player_type in player_types:
            TablutGameState.ZOBRIST_KEYS.to_move[player_type] = keys.pop()
        for terminal in is_terminal:
            TablutGameState.ZOBRIST_KEYS.is_terminal[terminal] = keys.pop()

    @classmethod
    def _draw(cls, state):
        '''
        Check if there is a draw, based on the number of repeated states
        '''
        pawn_difference = {}
        first = state
        second = state.old_state
        for i in range(0, int(cls.MAX_REPEATED_STATES)):
            if second is None:
                return False
            for pawn_type in TablutPawnType.values():
                if len(first.pawns[pawn_type]) != len(second.pawns[pawn_type]):
                    return False
                if i < cls.MAX_REPEATED_STATES / 2:
                    pawn_difference.setdefault(pawn_type, set()).update(
                        first.pawns[pawn_type].symmetric_difference(
                            second.pawns[pawn_type]
                        )
                    )
                else:
                    pawn_difference[pawn_type].difference_update(
                        second.pawns[pawn_type].symmetric_difference(
                            first.pawns[pawn_type]
                        )
                    )
            first = second
            second = second.old_state
        for pawn_diff in pawn_difference.values():
            if len(pawn_diff) > 0:
                return False
        return True

    def inc_turn(self):
        '''
        Next turn
        '''
        self.turn += 1

    def actions(self, state):
        '''
        Return every possible move from the given state
        '''
        return state.moves

    def next_states(self, state):
        '''
        Return the next states from the current possible actions
        '''
        return [
            self.result(state, move)
            for move in self.actions(state)
        ]

    def will_king_be_dead(self, state):
        '''
        Check if the king will be dead in the next possible states
        '''
        return any([
            TablutBoard.king_position(new_state.pawns) is None
            for new_state in self.next_states(state)
        ])

    def result(self, state, move, compute_moves=True):
        '''
        Return the next state with the given move and
        compute the new state moves, if specified
        '''
        pawns = TablutBoard.move(state.pawns, state.to_move, move)
        to_move = gutils.other_player(state.to_move)
        res = TablutGameState(
            to_move=to_move,
            utility=self._compute_utility(pawns, state.to_move),
            is_terminal=True,
            pawns=pawns,
            moves=[],
            old_state=state
        )
        if not self.terminal_test(res):
            res.is_terminal = False
            if compute_moves:
                res.moves = self.player_moves(pawns, to_move)
        return res

    def utility(self, state, player):
        '''
        Return 1 if the given player is white and white wins,
        return -1 if the given player is black and black wins,
        return 0 otherwise
        '''
        return -state.utility if player == state.to_move else state.utility

    def _compute_utility(self, pawns, player):
        '''
        Compute utility value for the given player, with the given pawns
        '''
        return (
            0 if not self._goal_state(pawns)
            else 1 if (
                (player == TablutPlayerType.WHITE and
                 TablutBoard.king_position(pawns) in TablutBoard.WHITE_GOALS) or
                (player == TablutPlayerType.BLACK and
                 TablutBoard.king_position(pawns) is None)
            )
            else -1
        )

    @classmethod
    def terminal_test(cls, state):
        '''
        A state is terminal if one player wins or the game is stuck
        '''
        return state.utility != 0 or cls._draw(state)

    def _goal_state(self, pawns):
        '''
        A state is a goal state if either the white or the black player wins
        '''
        return (
            TablutBoard.king_position(pawns) is None or
            TablutBoard.king_position(pawns) in TablutBoard.WHITE_GOALS
        )

    @overload
    @classmethod
    def all_moves(cls, pawns):
        '''
        Return a list of tuples of coordinates representing every possibile
        new position for each pawn
        '''
        moves = []
        moves.extend(cls.player_moves(pawns, TablutPlayerType.WHITE))
        moves.extend(cls.player_moves(pawns, TablutPlayerType.BLACK))

    @classmethod
    def player_moves(cls, pawns, player_type):
        '''
        Return a list of tuples of coordinates representing every possibile
        new position for each pawn of the given player
        '''
        moves = []
        pawn_types = gutils.from_player_to_pawn_types(player_type)
        for pawn_type in pawn_types:
            for pawn in pawns[pawn_type]:
                moves.extend(TablutBoard.moves(pawns, pawn))
        return moves
