'''
Board game representations module
'''


import tablut_player.utils as utils
import tablut_player.game_utils as gutils
from tablut_player.game_utils import (
    TablutBoardPosition,
    TablutPawnDirection,
    TablutPawnType,
    TablutPlayerType
)


class TablutBoard():

    SIZE = 9
    CASTLE = TablutBoardPosition(row=4, col=4)
    INNER_CAMPS = {
        TablutBoardPosition(row=4, col=0),
        TablutBoardPosition(row=0, col=4),
        TablutBoardPosition(row=8, col=4),
        TablutBoardPosition(row=4, col=8)
    }
    OUTER_CAMPS = {
        TablutBoardPosition(row=3, col=0),
        TablutBoardPosition(row=5, col=0),
        TablutBoardPosition(row=4, col=1),
        TablutBoardPosition(row=0, col=3),
        TablutBoardPosition(row=0, col=5),
        TablutBoardPosition(row=1, col=4),
        TablutBoardPosition(row=3, col=8),
        TablutBoardPosition(row=4, col=7),
        TablutBoardPosition(row=5, col=8),
        TablutBoardPosition(row=8, col=3),
        TablutBoardPosition(row=7, col=4),
        TablutBoardPosition(row=8, col=5)
    }
    CAMPS = INNER_CAMPS.union(OUTER_CAMPS)

    @classmethod
    def moves(cls, pawns, pawn_coords):
        '''
        Return a list of tuples of coordinates representing every possibile
        new position of the given pawn
        '''
        positions = set()
        for i in range(cls.SIZE):
            positions.add(
                TablutBoardPosition(row=i, col=pawn_coords.col)
            )
            positions.add(
                TablutBoardPosition(row=pawn_coords.row, col=i)
            )

        unwanted_positions = set()
        for sub in pawns.values():
            unwanted_positions.update(sub)
        unwanted_positions.add(cls.CASTLE)
        bad_camps = utils.copy(cls.CAMPS)
        if pawn_coords in cls.CAMPS:
            near_camps = set(cls.k_neighbors(pawn_coords, k=1))
            near_camps.update(set(cls.k_neighbors(pawn_coords, k=2)))
            bad_camps = cls.CAMPS.difference(near_camps)
        unwanted_positions.update(bad_camps)

        positions = cls._reachable_positions(
            pawn_coords, unwanted_positions, positions
        )
        moves = set()
        for p in positions:
            moves.add((pawn_coords, p))
        return moves

    @classmethod
    def _pawn_direction(cls, initial_pawn_coords, final_pawn_coords):
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

    @classmethod
    def _blocked_positions(cls, pawn_coords, pawn_direction):
        '''
        Given a pawn position and a pawn direction, return every
        unreachable board position
        '''
        unreachables = set()
        if pawn_direction == TablutPawnDirection.LEFT:
            for j in range(pawn_coords.col):
                unreachables.add(
                    TablutBoardPosition(row=pawn_coords.row, col=j)
                )
        elif pawn_direction == TablutPawnDirection.RIGHT:
            for j in range(pawn_coords.col + 1, cls.SIZE):
                unreachables.add(
                    TablutBoardPosition(row=pawn_coords.row, col=j)
                )
        elif pawn_direction == TablutPawnDirection.UP:
            for i in range(pawn_coords.row):
                unreachables.add(
                    TablutBoardPosition(row=i, col=pawn_coords.col)
                )
        elif pawn_direction == TablutPawnDirection.DOWN:
            for i in range(pawn_coords.row + 1, cls.SIZE):
                unreachables.add(
                    TablutBoardPosition(row=i, col=pawn_coords.col)
                )
        return unreachables

    @classmethod
    def _reachable_positions(cls, pawn_coords, unwanted_positions, moves):
        '''
        Return all the valid moves available, starting from the given
        pawn position
        '''
        unreachables = utils.copy(unwanted_positions)
        for u in unwanted_positions:
            pawn_direction = cls._pawn_direction(pawn_coords, u)
            if pawn_direction is not None:
                unreachables.update(
                    cls._blocked_positions(u, pawn_direction)
                )
        return moves.difference(unreachables)

    @classmethod
    def move(cls, pawns, player_type, move):
        '''
        Apply the given move
        '''
        new_pawns = utils.copy(pawns)
        pawn_types = gutils.from_player_to_pawn_types(player_type)
        from_move, to_move = move
        for pawn_type in pawn_types:
            try:
                new_pawns[pawn_type].remove(from_move)
                new_pawns[pawn_type].add(to_move)
                break
            except KeyError:
                pass
        return cls._remove_dead_pawns(new_pawns, player_type, to_move)

    @classmethod
    def player_pawns(cls, pawns, player_type):
        '''
        '''
        pawn_types = gutils.from_player_to_pawn_types(player_type)
        player_pawns = set()
        for pawn_type in pawn_types:
            player_pawns.update(pawns[pawn_type])
        return player_pawns

    @classmethod
    def _remove_pawns(cls, pawns, player_type, to_remove):
        '''
        '''
        pawn_types = gutils.from_player_to_pawn_types(player_type)
        for pawn_type in pawn_types:
            pawns[pawn_type] = pawns[pawn_type].difference(to_remove)
        return pawns

    @classmethod
    def king_position(cls, pawns):
        for king in pawns[TablutPawnType.KING]:
            return king
        return None

    @classmethod
    def k_neighbors(cls, pawn, k=1):
        '''
        '''
        left_pawn = TablutBoardPosition(row=pawn.row, col=pawn.col - k)
        right_pawn = TablutBoardPosition(row=pawn.row, col=pawn.col + k)
        up_pawn = TablutBoardPosition(row=pawn.row - k, col=pawn.col)
        down_pawn = TablutBoardPosition(row=pawn.row + k, col=pawn.col)
        return [left_pawn, right_pawn, up_pawn, down_pawn]

    @classmethod
    def _remove_dead_pawns(cls, pawns, my_type, moved_pawn):
        '''
        '''

        def dead_pawns(pawn, enemy_pawns, my_pawns):
            '''
            '''
            one_neighbors = cls.k_neighbors(pawn, k=1)
            two_neighbors = cls.k_neighbors(pawn, k=2)
            dead = set()
            pawns = set()
            pawns.update(my_pawns, cls.OUTER_CAMPS, {cls.CASTLE})
            for op, tp in zip(one_neighbors, two_neighbors):
                if op in enemy_pawns and tp in pawns:
                    dead.add(op)
            return dead

        def king_capture(pawns, enemy_pawns, my_pawns):
            king = cls.king_position(pawns)
            king_neighbors = cls.k_neighbors(king, k=1)
            if cls.CASTLE in king_neighbors or king == cls.CASTLE:
                enemy_pawns.remove(king)
                if cls.CASTLE in king_neighbors:
                    king_neighbors.remove(cls.CASTLE)
                if all(p in my_pawns for p in king_neighbors):
                    pawns[TablutPawnType.KING] = set()
            return pawns, enemy_pawns

        enemy_type = gutils.other_player(my_type)
        enemy_pawns = cls.player_pawns(pawns, enemy_type)
        my_pawns = cls.player_pawns(pawns, my_type)
        if enemy_type == TablutPlayerType.WHITE:
            pawns, enemy_pawns = king_capture(pawns, enemy_pawns, my_pawns)
        dead = dead_pawns(moved_pawn, enemy_pawns, my_pawns)
        return cls._remove_pawns(pawns, enemy_type, dead)
