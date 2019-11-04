'''
Module containing a generic game representation,
partially taken from AIMA library.
'''


from typing import overload

import tablut_player.game_utils as gutils
from tablut_player.game_utils import (
    TablutBoardPosition,
    TablutPawnType,
    TablutPlayerType,
    GameState
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

    WHITE_GOAL = [
        TablutBoardPosition(row=0, col=1),
        TablutBoardPosition(row=0, col=2),
        TablutBoardPosition(row=0, col=6),
        TablutBoardPosition(row=0, col=7),
        TablutBoardPosition(row=7, col=1),
        TablutBoardPosition(row=7, col=2),
        TablutBoardPosition(row=7, col=6),
        TablutBoardPosition(row=7, col=7),
        TablutBoardPosition(row=1, col=0),
        TablutBoardPosition(row=2, col=0),
        TablutBoardPosition(row=6, col=0),
        TablutBoardPosition(row=7, col=0),
        TablutBoardPosition(row=1, col=7),
        TablutBoardPosition(row=2, col=7),
        TablutBoardPosition(row=6, col=7),
        TablutBoardPosition(row=7, col=7)
    ]

    def __init__(self, initial_pawns=None, to_move=TablutPlayerType.WHITE):
        if initial_pawns is None:
            initial_pawns = self._init_pawns()
        self.initial = GameState(
            to_move=to_move,
            utility=0,
            pawns=initial_pawns,
            moves=self._moves(initial_pawns, to_move)
        )

    def _init_pawns(self):
        '''
        Pawns initial values
        '''
        pawns = {}

        # White pawns
        white_pawns_positions = [
            TablutBoardPosition(row=2, col=4),
            TablutBoardPosition(row=3, col=4),
            TablutBoardPosition(row=5, col=4),
            TablutBoardPosition(row=6, col=4),
            TablutBoardPosition(row=4, col=2),
            TablutBoardPosition(row=4, col=3),
            TablutBoardPosition(row=4, col=5),
            TablutBoardPosition(row=4, col=6)
        ]
        pawns[TablutPawnType.WHITE] = white_pawns_positions

        # Black pawns
        pawns[TablutPawnType.BLACK] = TablutBoard.CAMPS

        # King
        pawns[TablutPawnType.KING] = [TablutBoard.CASTLE]

        return pawns

    def actions(self, state):
        return state.moves

    def result(self, state, move):
        pawns = TablutBoard.move(state.pawns, state.to_move, move)
        to_move = gutils.other_player(state.to_move)
        return GameState(
            to_move=to_move,
            utility=self._compute_utility(pawns, state.to_move),
            pawns=pawns,
            moves=self._moves(pawns, to_move)
        )

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
                 TablutBoard.king_position(pawns) in self.WHITE_GOAL) or
                (player == TablutPlayerType.BLACK and
                 TablutBoard.king_position(pawns) is None)
            )
            else -1
        )

    def terminal_test(self, state):
        '''
        A state is terminale if one player wins
        '''
        return state.utility != 0

    def _goal_state(self, pawns):
        '''
        A state is a goal state if either the white or the black player wins
        '''
        return (
            TablutBoard.king_position(pawns) is None or
            TablutBoard.king_position(pawns) in self.WHITE_GOAL
        )

    @overload
    def _moves(self, pawns):
        '''
        Return a list of tuples of coordinates representing every possibile
        new position for each pawn
        '''
        moves = []
        moves.extend(self._moves(pawns, TablutPlayerType.WHITE))
        moves.extend(self._moves(pawns, TablutPlayerType.BLACK))

    def _moves(self, pawns, player_type):
        '''
        Return a list of tuples of coordinates representing every possibile
        new position for each pawn of the given player
        '''
        moves = []
        pawn_types = gutils.from_player_to_pawn_types(player_type)
        for pawn_type in pawn_types:
            for pawn in pawns[pawn_type]:
                moves.extend(TablutBoard.moves(pawns, pawn_type, pawn))
        return moves
