'''
Helper module, containing shared game utility functions
'''


from enum import Enum
import tablut_player.config as conf


class TablutGameState:
    '''
    Tablut game state
    '''

    def __init__(self, to_move, utility, pawns, moves, old_state=None):
        self.to_move = to_move
        self.utility = utility
        self.pawns = pawns
        self.moves = moves
        self.old_state = old_state

    def __eq__(self, other):
        return self.pawns == other.pawns and self.to_move == other.to_move

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        rep = ''
        if self.utility == 0:
            rep += "Game in progress\n"
        else:
            rep += "Game ended\t Winner -> "
            if self.utility == 1:
                rep += str(other_player(self.to_move))
            else:
                rep += str(self.to_move)
            rep += "\n"
        for pawn_type, pawns in self.pawns.items():
            rep += f'{pawn_type} pawns: {pawns}\n'
        rep += f'{self.to_move} player moves: {self.moves}\n'
        rep += '-' * 100
        return rep


class TablutPawnType(Enum):
    '''
    Tablut game pawn types
    '''

    WHITE = 'White'
    BLACK = 'Black'
    KING = 'King'

    @staticmethod
    def value_of(value):
        for _, pawn_type in TablutPawnType.__members__.items():
            if pawn_type.value == value.capitalize():
                return pawn_type
        return None

    def __str__(self):
        return f'{self.value}'

    def __repr__(self):
        return f'Pawn: {self.value}'


class TablutPlayerType(Enum):
    '''
    Tablut player types
    '''

    WHITE = 'White'
    BLACK = 'Black'

    @staticmethod
    def value_of(value):
        for _, player_type in TablutPlayerType.__members__.items():
            if player_type.value == value.capitalize():
                return player_type
        return None

    def __str__(self):
        return f'{self.value}'

    def __repr__(self):
        return f'Player: {self.value}'


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

    def distance(self, other):
        '''
        Manhattan distance
        '''
        return abs(self.row - other.row) + abs(self.col - other.col)

    def __eq__(self, position):
        return (
            isinstance(position, TablutBoardPosition) and
            position.row == self.row and position.col == self.col
        )

    def __ne__(self, position):
        return not self.__eq__(position)

    def __hash__(self):
        return hash((self.row, self.col))

    def __repr__(self):
        return f'({self.row}, {self.col})'


def is_black(player_role):
    '''
    Check if the given player is black
    '''
    return player_role == conf.BLACK_ROLE


def from_player_to_pawn_types(player_type):
    '''
    Get pawn type from player type
    '''
    return (
        [TablutPawnType.KING, TablutPawnType.WHITE]
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


def from_player_role_to_type(player_role):
    '''
    Convert a player role string to its respective player type
    '''
    return TablutPlayerType.value_of(player_role)


def from_player_type_to_role(player_type):
    '''
    Convert a player type to its respective player role
    '''
    return player_type.value.capitalize()


def other_player(player_type):
    '''
    Return other player
    '''
    return (
        TablutPlayerType.WHITE if player_type == TablutPlayerType.BLACK
        else TablutPlayerType.BLACK
    )


def from_server_state_to_pawns(board, turn):
    '''
    Convert the given server state to pawns dictionary
    '''
    to_move = TablutPlayerType.value_of(turn)
    pawns = {}
    for i, row in enumerate(board):
        for j, elem in enumerate(row):
            pawn_type = TablutPawnType.value_of(elem)
            if pawn_type is not None:
                pawns.setdefault(pawn_type, set()).add(
                    TablutBoardPosition(row=i, col=j)
                )
    return pawns, to_move


def from_pawns_to_move(old_pawns, new_pawns, player_type):
    '''
    Extract the performed move from the given pawns positions
    '''
    pawn_types = from_player_to_pawn_types(player_type)
    for pawn_type in pawn_types:
        from_move = old_pawns[pawn_type].difference(new_pawns[pawn_type])
        if len(from_move) > 0:
            to_move = new_pawns[pawn_type].difference(old_pawns[pawn_type])
    return (from_move.pop(), to_move.pop())


def from_move_to_server_action(move):
    '''
    Convert the given move to the action expected by the server
    '''
    from_move, to_move = move
    from_action = f'{chr(ord("`") + (from_move.col + 1))}{from_move.row + 1}'
    to_action = f'{chr(ord("`") + (to_move.col + 1))}{to_move.row + 1}'
    return from_action, to_action
