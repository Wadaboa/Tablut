'''
'''

from collections import namedtuple
from enum import Enum

import tablut_player

__all__ = [
    'GameState', 'TablutPawnType', 'TablutPlayerType', 'TablutPawnDirection',
    'TablutBoardPosition', 'from_player_to_pawn_types',
    'from_pawn_to_player_type', 'other_player'
]

GameState = namedtuple('GameState', 'to_move, utility, pawns, moves')


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


def from_player_to_pawn_types(player_type):
    '''
    Get pawn type from player type
    '''
    return (
        [TablutPawnType.WHITE, TablutPawnType.KING]
        if player_type == TablutPlayerType.WHITE
        else [TablutPawnType.BLACK]
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
