'''
Search strategies.
Taken from AIMA library.
'''


from collections import namedtuple
from enum import Enum
from typing import overload

import tablut_player
from .utils import *

inf = float('inf')
GameState = namedtuple('GameState', 'to_move, utility, board, moves')


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
        return not self.actions(state)

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


class TablutPawnType(Enum):
    '''
    Tablut game pawn types
    '''

    WHITE, BLACK, KING = range(3)


class TablutPlayerType(Enum):
    '''
    Tablut player types
    '''

    WHITE, BLACK = range(2)


class TablutPawnDirection(Enum):
    '''
    Tablut game pawn directions
    '''
    UP, DOWN, LEFT, RIGHT = range(4)


class TablutBoardPosition:
    '''
    Tablut board cell
    '''

    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __eq__(self, position):
        return (
            isinstance(position, TablutBoardPosition) and
            position.row == self.row and position.col == self.col
        )


def from_player_to_pawn_type(player_type):
    '''
    Get pawn type from player type
    '''
    return (
        TablutPawnType.WHITE if player_type == TablutPlayerType.WHITE
        else TablutPawnType.BLACK
    )


def from_pawn_to_player_type(pawn_type):
    '''
    Get player type from pawn type
    '''
    return (
        TablutPlayerType.WHITE if pawn_type in (
            TablutPawnType.WHITE, TablutPawnType.KING
        )
        else TablutPlayerType.BLACK
    )


def other_player(player_type):
    '''
    Return other player
    '''
    return (
        TablutPlayerType.WHITE if player_type == TablutPlayerType.BLACK
        else TablutPlayerType.BLACK
    )


class TablutBoard():

    SIZE = 9
    CASTLE = TablutBoardPosition(row=4, col=4)
    CAMPS = [
        TablutBoardPosition(row=3, col=0),
        TablutBoardPosition(row=4, col=0),
        TablutBoardPosition(row=5, col=0),
        TablutBoardPosition(row=4, col=1),
        TablutBoardPosition(row=0, col=3),
        TablutBoardPosition(row=0, col=4),
        TablutBoardPosition(row=0, col=5),
        TablutBoardPosition(row=1, col=4),
        TablutBoardPosition(row=4, col=7),
        TablutBoardPosition(row=7, col=4),
        TablutBoardPosition(row=3, col=8),
        TablutBoardPosition(row=4, col=8),
        TablutBoardPosition(row=5, col=8),
        TablutBoardPosition(row=8, col=3),
        TablutBoardPosition(row=8, col=4),
        TablutBoardPosition(row=8, col=5)
    ]

    def __init__(self, pawns=None):
        if pawns is None:
            self.pawns = self._init_pawns()
        else:
            self.pawns = pawns

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
        pawns[TablutPawnType.BLACK] = self.CAMPS

        # King
        pawns[TablutPawnType.KING] = self.CASTLE

        return pawns

    @overload
    def moves(self):
        '''
        Return a list of tuples of coordinates representing every possibile
        new position for each pawn
        '''
        moves = []
        moves.append(self.moves(TablutPlayerType.WHITE))
        moves.append(self.moves(TablutPlayerType.BLACK))

    @overload
    def moves(self, player_type):
        '''
        Return a list of tuples of coordinates representing every possibile
        new position for each pawn of the given player
        '''
        moves = []
        pawn_type = from_player_to_pawn_type(player_type)
        for p in self.pawns[pawn_type]:
            moves.append(
                self.moves(pawn_type, p)
            )
        if pawn_type == TablutPawnType.WHITE:
            moves.extend(self.moves(pawn_type, self.pawns[TablutPawnType.KING]))
        return moves

    @overload
    def moves(self, pawn_type, pawn_coords):
        '''
        Return a list of tuples of coordinates representing every possibile
        new position of the given pawn
        '''
        positions = []
        for i in range(self.SIZE):
            positions.append(
                TablutBoardPosition(row=i, col=pawn_coords.col)
            )
            positions.append(
                TablutBoardPosition(row=pawn_coords.col, col=i)
            )

        unwanted_positions = list(self.pawns.values())
        if pawn_type in (TablutPawnType.WHITE, TablutPawnType.KING):
            unwanted_positions.extend(self.CAMPS)
        else:
            unwanted_positions.append(self.CASTLE)

        positions = self.reachable_positions(
            pawn_coords, unwanted_positions, positions
        )
        moves = []
        for p in positions:
            moves.append((pawn_coords, p))
        return moves

    def pawn_direction(self, initial_pawn_coords, final_pawn_coords):
        '''
        Given two pawn coordinates, return its move direction
        '''
        if initial_pawn_coords.row == final_pawn_coords.row:
            return (
                TablutPawnDirection.LEFT if (
                    final_pawn_coords.col < initial_pawn_coords.col
                ) else TablutPawnDirection.RIGHT
            )
        elif initial_pawn_coords.col == final_pawn_coords.col:
            return (
                TablutPawnDirection.UP if (
                    final_pawn_coords.row < initial_pawn_coords.row
                ) else TablutPawnDirection.DOWN
            )
        return None

    def blocked_positions(self, pawn_coords, pawn_direction):
        '''
        Given a pawn position and a pawn direction, return every
        unreachable board position
        '''
        unreachable = []
        if pawn_direction == TablutPawnDirection.LEFT:
            for j in range(pawn_coords.col):
                unreachable.append(
                    TablutBoardPosition(row=pawn_coords.row, col=j)
                )
        elif pawn_direction == TablutPawnDirection.RIGHT:
            for j in range(pawn_coords.col + 1, self.SIZE):
                unreachable.append(
                    TablutBoardPosition(row=pawn_coords.row, col=j)
                )
        elif pawn_direction == TablutPawnDirection.UP:
            for i in range(pawn_coords.row):
                unreachable.append(
                    TablutBoardPosition(row=i, col=pawn_coords.col)
                )
        elif pawn_direction == TablutPawnDirection.DOWN:
            for i in range(pawn_coords.row + 1, self.SIZE):
                unreachable.append(
                    TablutBoardPosition(row=i, col=pawn_coords.col)
                )
        return unreachable

    def reachable_positions(self, pawn_coords, unwanted_positions, moves):
        '''
        Return all the valid moves available, starting from the given
        pawn position
        '''
        unreachable = unwanted_positions
        for u in unwanted_positions:
            pawn_direction = self.pawn_direction(pawn_coords, u)
            if pawn_direction is not None:
                unreachable.append(
                    self.blocked_positions(pawn_coords, pawn_direction)
                )
        return remove_unwanted_seq(moves, unreachable)

    def move(self, player_type, move):
        '''
        Apply the given move
        '''
        pawn_type = from_player_to_pawn_type(player_type)
        from_move, to_move = move
        found = False
        for i, pawn in enumerate(self.pawns[pawn_type]):
            if pawn == from_move:
                self.pawns[pawn_type][i] = to_move
                found = True
                break
        if not found and player_type == TablutPlayerType.WHITE:
            pawn_type = TablutPawnType.KING
            if self.pawns[pawn_type] == from_move:
                self.pawns[pawn_type] = to_move

    def player_pawns(self, player_type):
        pawn_type = from_player_to_pawn_type(player_type)
        pawns = self.pawns[pawn_type]
        if pawn_type == TablutPawnType.WHITE:
            pawns.extend(self.pawns[TablutPawnType.KING])
        return pawns

    def is_hungry(self, player_type, pawn_index, to_move):

        def neighboring_pawns(pawn, pawns):
            left_pawn = TablutBoardPosition(row=pawn.row, col=pawn.col - 1)
            right_pawn = TablutBoardPosition(row=pawn.row, col=pawn.col + 1)
            up_pawn = TablutBoardPosition(row=pawn.row - 1, col=pawn.col)
            down_pawn = TablutBoardPosition(row=pawn.row + 1, col=pawn.col)
            neighbors = [left_pawn, right_pawn, up_pawn, down_pawn]
            return [p for p in pawns if p in neighbors]

        other_player_pawns = self.player_pawns(other_player(player_type))
        player_pawns = self.player_pawns(player_type)
        neighbors = neighboring_pawns(to_move, other_player_pawns)
        for n in neighbors:
            pass
            # TODO


class TablutGame(Game):
    '''
    Tablut game representation
    '''

    def __init__(self):
        self.board = TablutBoard()
        self.state = GameState(
            to_move=TablutPlayerType.WHITE,
            utility=0,
            board=self.board,
            move=self.board.moves(TablutPlayerType.WHITE)
        )

    def actions(self, state):
        return state.move

    def result(self, state, action):
        pass

    def utility(self, state, player):
        pass

    def terminal_test(self, state):
        pass

    def value(self, state):
        pass
