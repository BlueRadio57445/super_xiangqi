#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
import time
from collections import namedtuple
from itertools import count

###############################################################################
# Piece-Square tables. Tune these to change sunfish's behaviour
###############################################################################
uni_pieces = {'R': '🩤', 'H': '🩣', 'E': '🩢', 'A': '🩡', 'K': '🩠', 'C': '🩥', 'P': '🩦',
              'r': '🩫', 'h': '🩪', 'e': '🩩', 'a': '🩨', 'k': '🩧', 'c': '🩬', 'p': '🩭', '.': '·'}
chinese_pieces = {'R': '车', 'H': '马', 'E': '相', 'A': '仕', 'K': '帅', 'C': '炮', 'P': '兵',
                  'r': '車', 'h': '马', 'e': '象', 'a': '士', 'k': '将', 'c': '砲', 'p': '卒', '.': '· '}

CHESS_ROW = 10
CHESS_COLUMN = 9
BOARD_ROW = CHESS_ROW+4
BOARD_COLUMN = CHESS_COLUMN+2
# king, advisor, elephant, horse, rook, pawn, cannon


###############################################################################
# Global constants
###############################################################################

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
initial = (
    '          \n'
    '          \n'
    ' rheakaehr\n'
    ' .........\n'
    ' .c.....c.\n'
    ' p.p.p.p.p\n'
    ' .........\n'
    # river
    ' .........\n'
    ' P.P.P.P.P\n'
    ' .C.....C.\n'
    ' .........\n'
    ' RHEAKAEHR\n'
    '          \n'
    '          \n'
)

# Lists of possible moves for each piece type.
N, E, S, W = -BOARD_COLUMN, 1, BOARD_COLUMN, -1
directions = {
    'P': (N, W, E),
    'H': ((N, N + E), (N, N + W), (S, S + E), (S, S + W), (E, E + N), (E, E + S), (W, W + N), (W, W + S)),
    'E': ((N + E, N + E), (S + E, S + E), (S + W, S + W), (N + W, N + W)),
    'A': (N + E, S + E, S + W, N + W),
    'R': (N, E, S, W),
    'C': (N, E, S, W),
    'K': (N, E, S, W)
}



###############################################################################
# Chess logic
###############################################################################

class BoardState(namedtuple('BoardState', 'board')):
    """ A state of a chess game
    board -- a BOARD_ROW*BOARD_COLUMN char representation of the board
    """

    def gen_moves(self, piece_pos=None):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as horse.

        moves = []
        pieces = [piece_pos] if piece_pos is not None else [
            i for i, p in enumerate(self.board) if p.isupper()
        ]

        for i in pieces:
            p = self.board[i]
            if not p.isupper():
                continue
            for d in directions[p]:
                cannon_flag = False
                step = 0
                if isinstance(d, tuple):
                    step = d[0]
                    d = sum(d)
                for j in count(i + d, d):
                    q = self.board[j]
                    # inside the board
                    if q.isspace():
                        break
                    # friend chess
                    if q.isupper() and p != 'C':
                        break
                    if p == 'C':
                        if cannon_flag:
                            if q.islower():
                                moves.append((p, i, j))
                                break
                            elif q.isupper():
                                break
                            else:
                                continue
                        # cannon need a carriage to attack opponent
                        elif q.isalpha():
                            cannon_flag = True
                            continue
                    # horse and elephant leg should not be crappy
                    if p in ('H', 'E') and self.board[i + step] != '.':
                        break
                    # king and advisor should stay in palace
                    if p in ('A', 'K'):
                        row, column = j // BOARD_COLUMN, j % BOARD_COLUMN
                        if not (9 <= row <= BOARD_COLUMN and 4 <= column <= 6):
                            break
                    # elephant cannot go across river
                    if p == 'E' and not 6 <= j // BOARD_COLUMN <= BOARD_COLUMN:
                        break
                    # pawn can move east or west only after crossing river
                    if p == 'P' and j // BOARD_COLUMN > 6 and d in (E, W):
                        break
                    # two kings cannot see each other
                    black_king = self.board.index('k')
                    if p == 'K':
                        red_king = j
                    else:
                        red_king = self.board.index('K')
                    if black_king % BOARD_COLUMN == red_king % BOARD_COLUMN:
                        if not any(piece != '.' for piece in self.board[black_king+BOARD_COLUMN:red_king:BOARD_COLUMN]):
                            break

                    # Record Move
                    moves.append((p, i, j))
                    if p in 'HPEAK' or q.islower():
                        break
        return moves

    def rotate(self):
        """ Rotates the board"""
        return BoardState(self.board[::-1].swapcase())

    def put(self, board, i, p):
        return board[:i] + p + board[i + 1:]

    def move(self, move):
        _, i, j = move
        # Copy variables and reset ep and kp
        board = self.board
        # Actual move
        board = self.put(board, j, board[i])
        board = self.put(board, i, '.')
        return BoardState(board).rotate()
    
    def is_terminal(self):
        b = self.board
        if 'K' not in b:
            return -1  # 黑方勝
        if 'k' not in b:
            return 1   # 紅方勝
        # 檢查當前玩家（紅方）是否無合法移動
        legal_moves = self.gen_moves()
        if not legal_moves:
            # 紅方無合法移動，視為黑方勝
            return -1
        # 檢查黑方是否無合法移動
        black_board = self.rotate()  # 旋轉到黑方視角
        black_legal_moves = black_board.gen_moves()
        if not black_legal_moves:
            # 黑方無合法移動，視為紅方勝
            return 1
        return None