'''
Board game representations module
'''


from PyQt5 import QtWidgets
from PyQt5.QtCore import QEvent, QRectF, Qt
from PyQt5.QtGui import QColor, QPen

import tablut_player.game_utils as gutils
import tablut_player.config as conf
from tablut_player.game_utils import (
    TablutBoardPosition,
    TablutPawnDirection,
    TablutPawnType,
    TablutPlayerType
)


class TablutBoard():
    '''
    Tablut board rules and interaction
    '''

    CASTLE = TablutBoardPosition.create(row=4, col=4)
    INNER_CAMPS = {
        TablutBoardPosition.create(row=4, col=0),
        TablutBoardPosition.create(row=0, col=4),
        TablutBoardPosition.create(row=8, col=4),
        TablutBoardPosition.create(row=4, col=8)
    }
    OUTER_CAMPS = {
        TablutBoardPosition.create(row=3, col=0),
        TablutBoardPosition.create(row=5, col=0),
        TablutBoardPosition.create(row=4, col=1),
        TablutBoardPosition.create(row=0, col=3),
        TablutBoardPosition.create(row=0, col=5),
        TablutBoardPosition.create(row=1, col=4),
        TablutBoardPosition.create(row=3, col=8),
        TablutBoardPosition.create(row=4, col=7),
        TablutBoardPosition.create(row=5, col=8),
        TablutBoardPosition.create(row=8, col=3),
        TablutBoardPosition.create(row=7, col=4),
        TablutBoardPosition.create(row=8, col=5)
    }
    CAMPS = INNER_CAMPS.union(OUTER_CAMPS)
    WHITE_GOALS = {
        TablutBoardPosition.create(row=0, col=1),
        TablutBoardPosition.create(row=0, col=2),
        TablutBoardPosition.create(row=0, col=6),
        TablutBoardPosition.create(row=0, col=7),
        TablutBoardPosition.create(row=8, col=1),
        TablutBoardPosition.create(row=8, col=2),
        TablutBoardPosition.create(row=8, col=6),
        TablutBoardPosition.create(row=8, col=7),
        TablutBoardPosition.create(row=1, col=0),
        TablutBoardPosition.create(row=2, col=0),
        TablutBoardPosition.create(row=6, col=0),
        TablutBoardPosition.create(row=7, col=0),
        TablutBoardPosition.create(row=1, col=8),
        TablutBoardPosition.create(row=2, col=8),
        TablutBoardPosition.create(row=6, col=8),
        TablutBoardPosition.create(row=7, col=8)
    }
    OUTER_CORNERS = {
        TablutBoardPosition.create(row=1, col=1),
        TablutBoardPosition.create(row=1, col=7),
        TablutBoardPosition.create(row=7, col=1),
        TablutBoardPosition.create(row=7, col=7)
    }

    @classmethod
    def moves(cls, pawns, pawn_coords):
        '''
        Return a set of tuples of coordinates representing every possibile
        new position of the given pawn
        '''
        moves = []
        positions = cls.legal_moves(pawns, pawn_coords)
        for pos in positions:
            moves.append((pawn_coords, pos))
        return moves

    @classmethod
    def legal_moves(cls, pawns, pawn_coords):
        '''
        Return a set of TablutBoardPosition representing every possibile
        new position of the given pawn
        '''
        return cls._reachable_positions(
            pawn_coords,
            cls._unallowed_positions(pawns, pawn_coords),
            cls._possible_positions(pawn_coords)
        )

    @classmethod
    def _possible_positions(cls, pawn_coords):
        '''
        Computes every row and column pawn coordinates, from the given pawn
        '''
        positions = set()
        for i in range(conf.BOARD_SIZE):
            positions.add(
                TablutBoardPosition.create(row=i, col=pawn_coords.col)
            )
            positions.add(
                TablutBoardPosition.create(row=pawn_coords.row, col=i)
            )
        return positions

    @classmethod
    def _unallowed_positions(cls, pawns, pawn_coords):
        '''
        Computes not allowed pawns positions
        '''
        unallowed_positions = set()
        for sub in pawns.values():
            unallowed_positions.update(sub)
        unallowed_positions.add(cls.CASTLE)
        bad_camps = set(cls.CAMPS)
        if pawn_coords in cls.CAMPS:
            near_camps = set(cls.orthogonal_k_neighbors(pawn_coords, k=1))
            near_camps.update(set(cls.orthogonal_k_neighbors(pawn_coords, k=2)))
            bad_camps = cls.CAMPS.difference(near_camps)
        unallowed_positions.update(bad_camps)
        return unallowed_positions

    @classmethod
    def _pawn_direction(cls, initial_pawn_coords, final_pawn_coords):
        '''
        Given two pawn coordinates, return its move direction
        '''
        if (initial_pawn_coords.row == final_pawn_coords.row and
                initial_pawn_coords.col == final_pawn_coords.col):
            return None
        elif initial_pawn_coords.row == final_pawn_coords.row:
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
    def from_direction_to_pawn(cls, initial_pawn_coords, pawn_direction, k=1):
        '''
        Return the k-distant cell from the given pawn, in the given direction
        '''
        if pawn_direction == TablutPawnDirection.UP:
            return TablutBoardPosition.create(
                row=initial_pawn_coords.row - k, col=initial_pawn_coords.col
            )
        elif pawn_direction == TablutPawnDirection.DOWN:
            return TablutBoardPosition.create(
                row=initial_pawn_coords.row + k, col=initial_pawn_coords.col
            )
        elif pawn_direction == TablutPawnDirection.LEFT:
            return TablutBoardPosition.create(
                row=initial_pawn_coords.row, col=initial_pawn_coords.col - k
            )
        elif pawn_direction == TablutPawnDirection.RIGHT:
            return TablutBoardPosition.create(
                row=initial_pawn_coords.row, col=initial_pawn_coords.col + k
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
                    TablutBoardPosition.create(row=pawn_coords.row, col=j)
                )
        elif pawn_direction == TablutPawnDirection.RIGHT:
            for j in range(pawn_coords.col + 1, conf.BOARD_SIZE):
                unreachables.add(
                    TablutBoardPosition.create(row=pawn_coords.row, col=j)
                )
        elif pawn_direction == TablutPawnDirection.UP:
            for i in range(pawn_coords.row):
                unreachables.add(
                    TablutBoardPosition.create(row=i, col=pawn_coords.col)
                )
        elif pawn_direction == TablutPawnDirection.DOWN:
            for i in range(pawn_coords.row + 1, conf.BOARD_SIZE):
                unreachables.add(
                    TablutBoardPosition.create(row=i, col=pawn_coords.col)
                )
        return unreachables

    @classmethod
    def _reachable_positions(cls, pawn_coords, unallowed_positions, moves):
        '''
        Return all the valid moves available, starting from the given
        pawn position
        '''
        unreachables = set(unallowed_positions)
        for pos in unallowed_positions:
            pawn_direction = cls._pawn_direction(pawn_coords, pos)
            if pawn_direction is not None:
                unreachables.update(
                    cls._blocked_positions(pos, pawn_direction)
                )
        return moves.difference(unreachables)

    @classmethod
    def move(cls, pawns, player_type, move):
        '''
        Apply the given move
        '''
        new_pawns = dict()
        for pawn_type in pawns:
            new_pawns[pawn_type] = set(pawns[pawn_type])
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
        Return the pawns associated with the given player type
        '''
        pawn_types = gutils.from_player_to_pawn_types(player_type)
        player_pawns = set()
        for pawn_type in pawn_types:
            player_pawns.update(pawns[pawn_type])
        return player_pawns

    @classmethod
    def _remove_pawns(cls, pawns, player_type, to_remove):
        '''
        Remove the given pawns from the all player pawns
        '''
        pawn_types = gutils.from_player_to_pawn_types(player_type)
        for pawn_type in pawn_types:
            pawns[pawn_type] = pawns[pawn_type].difference(to_remove)
        return pawns

    @classmethod
    def king_position(cls, pawns):
        '''
        Return the king position in the board
        '''
        for king in pawns[TablutPawnType.KING]:
            return king
        return None

    @classmethod
    def is_king_dead(cls, pawns):
        '''
        Check if the king is dead or alive
        '''
        return cls.king_position(pawns) is None

    @classmethod
    def total_pawns(cls, pawns):
        '''
        Total number of pawns on the board
        '''
        total = 0
        for player_type in TablutPlayerType:
            total += cls.total_player_pawns(pawns, player_type)
        return total

    @classmethod
    def total_player_pawns(cls, pawns, player_type):
        '''
        Total number of pawns of the given player
        '''
        return len(cls.player_pawns(pawns, player_type))

    @classmethod
    def piece_difference_count(cls, pawns, player_type):
        '''
        Return the total number of player pawns minus
        the total number of opponent pawns
        '''
        return (
            cls.total_player_pawns(pawns, player_type) -
            cls.total_player_pawns(pawns, gutils.other_player(player_type))
        )

    @classmethod
    def simulate_distance(cls, pawns, initial_coords, final_coords,
                          max_moves, unwanted_positions=[], n_moves=0):
        '''
        Compute a simulation of the minimum number of moves required
        to reach the given final coordinates, by ignoring oppenent moves
        and applying the given maximum number of moves
        '''
        if n_moves == max_moves or initial_coords == final_coords:
            return n_moves
        if initial_coords in unwanted_positions:
            return max_moves + 1
        moves = cls.legal_moves(pawns, initial_coords)
        if len(moves) <= 0:
            return max_moves + 1
        moves_counter = []
        for move in moves:
            if (initial_coords.distance(final_coords) >
                    move.distance(final_coords)):
                moves_counter.append(
                    cls.simulate_distance(
                        pawns,
                        move,
                        final_coords,
                        max_moves,
                        unwanted_positions,
                        n_moves + 1
                    )
                )
        min_moves = max_moves + 1
        if len(moves_counter) > 0:
            min_moves = min(moves_counter)
        return min_moves

    @classmethod
    def orthogonal_k_neighbors(cls, pawn, k=1):
        '''
        Return the k-level neighbors of the given pawn
        '''
        left_pawn = TablutBoardPosition.create(row=pawn.row, col=pawn.col - k)
        right_pawn = TablutBoardPosition.create(row=pawn.row, col=pawn.col + k)
        up_pawn = TablutBoardPosition.create(row=pawn.row - k, col=pawn.col)
        down_pawn = TablutBoardPosition.create(row=pawn.row + k, col=pawn.col)
        return [up_pawn, left_pawn, right_pawn, down_pawn]

    @classmethod
    def unique_orthogonal_k_neighbors(cls, pawn, k=1):
        '''
        Return the valid k-level orthogonal neighbors of the given pawn
        '''
        return {
            pos for pos in cls.orthogonal_k_neighbors(pawn, k)
            if pos is not None
        }

    @classmethod
    def diagonal_k_neighbors(cls, pawn, k=1):
        '''
        Return the k-level diagonal neighbors of the given pawn
        '''
        l_up_pawn = TablutBoardPosition.create(
            row=pawn.row - k, col=pawn.col - k
        )
        r_up_pawn = TablutBoardPosition.create(
            row=pawn.row - k, col=pawn.col + k
        )
        r_down_pawn = TablutBoardPosition.create(
            row=pawn.row + k, col=pawn.col + k
        )
        l_down_pawn = TablutBoardPosition.create(
            row=pawn.row + k, col=pawn.col - k
        )
        return [l_up_pawn, r_up_pawn, r_down_pawn, l_down_pawn]

    @classmethod
    def unique_diagonal_k_neighbors(cls, pawn, k=1):
        '''
        Return the valid k-level diagonal neighbors of the given pawn
        '''
        return {
            pos for pos in cls.diagonal_k_neighbors(pawn, k) if pos is not None
        }

    @classmethod
    def full_k_neighbors(cls, pawn, k=1):
        '''
        Return the k-level orthogonal and diagonal neighbors of the given pawn
        '''
        neighbors = cls.orthogonal_k_neighbors(pawn, k)
        neighbors.extend(cls.diagonal_k_neighbors(pawn, k))
        return neighbors

    @classmethod
    def unique_full_k_neighbors(cls, pawn, k=1):
        '''
        Return the valid k-level orthogonal and diagonal neighbors
        of the given pawn
        '''
        return {
            pos for pos in cls.full_k_neighbors(pawn, k) if pos is not None
        }

    @classmethod
    def is_king_in_castle(cls, pawns):
        '''
        Return True if the king is in the castle, False otherwise
        '''
        return cls.king_position(pawns) == cls.CASTLE

    @classmethod
    def is_king_near_castle(cls, pawns):
        '''
        Return True if the king is one cell away from the castle,
        False otherwise
        '''
        return cls.king_position(pawns) in (
            cls.orthogonal_k_neighbors(cls.CASTLE, k=1)
        )

    @classmethod
    def potential_king_killers(cls, pawns):
        '''
        Return the number of enemy pawns, camps and castle around the king
        and the remaining positions to kill him
        '''

        def is_black(neighbor):
            '''
            Check if the given neighbor is a black pawn, an outer camp
            or the castle
            '''
            return (
                neighbor in cls.OUTER_CAMPS or neighbor == cls.CASTLE or
                neighbor in pawns[TablutPawnType.BLACK]
            )

        def is_white(neighbor):
            '''
            Check if the given neighbor is a white pawn
            '''
            return neighbor in pawns[TablutPawnType.WHITE]

        king = cls.king_position(pawns)
        left_pawn = TablutBoardPosition.create(row=king.row, col=king.col - 1)
        right_pawn = TablutBoardPosition.create(row=king.row, col=king.col + 1)
        up_pawn = TablutBoardPosition.create(row=king.row - 1, col=king.col)
        down_pawn = TablutBoardPosition.create(row=king.row + 1, col=king.col)
        king_neighbors = [(left_pawn, right_pawn), (up_pawn, down_pawn)]
        free_neighbors = []
        killer_neighbors = []
        killers = 0
        for neighbor_one, neighbor_two in king_neighbors:
            if is_black(neighbor_one) and is_black(neighbor_two):
                killers += 2
            elif is_black(neighbor_one) or is_black(neighbor_two):
                killers += 1
                if is_black(neighbor_one) and not is_white(neighbor_two):
                    killer_neighbors.append(neighbor_two)
                elif is_black(neighbor_two) and not is_white(neighbor_one):
                    killer_neighbors.append(neighbor_one)
            if (not is_white(neighbor_one) and
                    neighbor_one not in killer_neighbors):
                free_neighbors.append(neighbor_one)
            if (not is_white(neighbor_two) and
                    neighbor_two not in killer_neighbors):
                free_neighbors.append(neighbor_two)
        return killers, free_neighbors, killer_neighbors

    @classmethod
    def _remove_dead_pawns(cls, pawns, my_type, moved_pawn):
        '''
        Compute pawns to remove from the board, after the given move
        '''

        def dead_pawns(pawn, enemy_pawns, my_pawns):
            '''
            Compute captured pawns
            '''
            one_neighbors = cls.orthogonal_k_neighbors(pawn, k=1)
            two_neighbors = cls.orthogonal_k_neighbors(pawn, k=2)
            dead = set()
            pawns = set()
            pawns.update(my_pawns, cls.OUTER_CAMPS, {cls.CASTLE})
            for one_pawn, two_pawn in zip(one_neighbors, two_neighbors):
                if one_pawn in enemy_pawns and two_pawn in pawns:
                    dead.add(one_pawn)
            return dead

        def king_capture(pawns, enemy_pawns, my_pawns):
            '''
            Check if the king is dead or alive, after the given move
            '''
            king = cls.king_position(pawns)
            king_neighbors = cls.orthogonal_k_neighbors(king, k=1)
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


class TablutBoardGUI(QtWidgets.QGraphicsScene):
    '''
    Tablut board GUI, used for debugging purposes
    '''

    CELL_SIZE = 40
    _COLOR_FROM_PAWN = {
        TablutPawnType.WHITE: Qt.white,
        TablutPawnType.BLACK: Qt.black,
        TablutPawnType.KING: Qt.magenta
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lines = []
        self.markers = []
        self.draw_grid()
        self.draw_special_cells()
        self.setBackgroundBrush(QColor(255, 233, 127))
        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        '''
        Avoid mouse click events
        '''
        if event.type() in (QEvent.MouseButtonPress,
                            QEvent.MouseButtonDblClick):
            return False
        return super(TablutBoardGUI, self).eventFilter(obj, event)

    def draw_grid(self):
        '''
        Draw Tablut grid
        '''
        offset_factor = 1.5
        offset = int(self.CELL_SIZE / offset_factor)
        width = conf.BOARD_SIZE * self.CELL_SIZE
        height = conf.BOARD_SIZE * self.CELL_SIZE
        self.setSceneRect(
            -offset, -offset,
            width + self.CELL_SIZE, height + self.CELL_SIZE
        )
        self.setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)
        pen = QPen(Qt.black, 2, Qt.SolidLine)

        for index in range(0, conf.BOARD_SIZE + 1):
            coord = index * self.CELL_SIZE
            self.lines.append(self.add_line(coord, 0, coord, height, pen))
            self.lines.append(self.add_line(0, coord, width, coord, pen))
            if index != 0:
                self.add_text(str(index - 1), - offset, coord - offset)
                self.add_text(str(index - 1), coord - offset, - offset)

    def add_line(self,
                 x_top_left, y_top_left,
                 x_bottom_right, y_bottom_right,
                 pen, z_value=-1):
        '''
        Add a grid line
        '''
        line = self.addLine(
            x_top_left, y_top_left,
            x_bottom_right, y_bottom_right,
            pen
        )
        line.setZValue(z_value)
        return line

    def add_text(self, text, x_value, y_value, z_value=1, color=Qt.black):
        '''
        Add the given text in the specified position in the grid
        '''
        text = self.addText(text)
        text.setZValue(z_value)
        text.setDefaultTextColor(color)
        text.setPos(x_value, y_value)
        return text

    def set_visible(self, visible=True):
        '''
        Show/hide grid lines
        '''
        for line in self.lines:
            line.setVisible(visible)

    def delete_grid(self):
        '''
        Delete grid lines
        '''
        for line in self.lines:
            self.removeItem(line)
        del self.lines[:]

    def set_opacity(self, opacity):
        '''
        Set grid lines opacity
        '''
        for line in self.lines:
            line.setOpacity(opacity)

    def draw_ellipse(self, coords, z_value=0, color=Qt.white):
        '''
        Add an ellipsoidal marker in the given position
        '''
        row, col = coords
        top_left_x = col * self.CELL_SIZE
        top_left_y = row * self.CELL_SIZE
        ellipse = self.addEllipse(
            QRectF(
                top_left_x, top_left_y,
                self.CELL_SIZE, self.CELL_SIZE
            ),
            color, color
        )
        ellipse.setZValue(z_value)
        return ellipse

    def draw_rect(self, coords, z_value=-2, color=Qt.white):
        '''
        Add a rectangular marker in the given position
        '''
        row, col = coords
        top_left_x = col * self.CELL_SIZE
        top_left_y = row * self.CELL_SIZE
        rect = self.addRect(
            QRectF(
                top_left_x, top_left_y,
                self.CELL_SIZE, self.CELL_SIZE
            ),
            color, color
        )
        rect.setZValue(z_value)
        return rect

    def delete_markers(self):
        '''
        Delete grid markers
        '''
        for marker in self.markers:
            self.removeItem(marker)
        del self.markers[:]

    def draw_special_cells(self):
        '''
        Draw camps and castle
        '''
        for camp in TablutBoard.CAMPS:
            self.draw_rect(
                coords=(camp.row, camp.col),
                color=QColor(128, 128, 128)
            )
        for goal in TablutBoard.WHITE_GOALS:
            self.draw_rect(
                coords=(goal.row, goal.col),
                color=QColor(90, 250, 250)
            )
        self.draw_rect(
            coords=(TablutBoard.CASTLE.row, TablutBoard.CASTLE.col),
            color=QColor(255, 128, 0)
        )

    def draw_pawns(self, pawns):
        '''
        Add the given pawns to the grid
        '''
        for pawn_type in pawns:
            color = self._COLOR_FROM_PAWN[pawn_type]
            for pawn in pawns[pawn_type]:
                self.markers.append(
                    self.draw_ellipse(
                        coords=(pawn.row, pawn.col),
                        color=color
                    )
                )

    def set_pawns(self, pawns):
        '''
        Update the grid with the given pawns
        '''
        self.delete_markers()
        self.draw_pawns(pawns)
        self.update()
