#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
import time
from collections import namedtuple, Counter
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

class BoardState(namedtuple('BoardState', 'board move_count counter')):
    """ A state of a chess game
    board -- a BOARD_ROW*BOARD_COLUMN char representation of the board
    move_count -- number of moves without capture (for 50-move rule equivalent)
    counter -- Counter object tracking board position repetitions
    """

    def __new__(cls, board, move_count=0, counter=None):
        if counter is None:
            counter = Counter()
            counter[board] = 1
        return super(BoardState, cls).__new__(cls, board, move_count, counter)

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
        return BoardState(self.board[::-1].swapcase(), self.move_count, self.counter)

    def put(self, board, i, p):
        return board[:i] + p + board[i + 1:]

    def move(self, move):
        _, i, j = move
        # Copy variables and reset ep and kp
        board = self.board
        # Actual move
        board = self.put(board, j, board[i])
        board = self.put(board, i, '.')

        rotated_board_state = BoardState(board).rotate()
        new_move_count = self.move_count + 1
        new_counter = self.counter.copy()
        new_counter[rotated_board_state.board] += 1

        return BoardState(rotated_board_state.board, new_move_count, new_counter)
    
    def is_terminal(self, owo=True):
        """
        只看當下我方是贏還是輸，不管我什麼顏色，先手或後手。
        """
        b = self.board

        # 和局判斷
        if self.counter[b] >= 3:
            if owo: print("和局")
            return 0
        # 普通勝負
        if 'K' not in b:
            if owo: print("輸出-1，最常")
            return -1  # 我方敗
        if 'k' not in b:
            if owo: print("不該")
            return 1   # 我方勝
        # 檢查當前玩家是否無合法移動
        legal_moves = self.gen_moves()
        if not legal_moves:
            # 我方無合法移動，視為我方敗
            if owo: print("困，負")
            return -1
        # 檢查敵方是否無合法移動
        black_board = self.rotate()  # 旋轉到敵方視角
        black_legal_moves = black_board.gen_moves()
        if not black_legal_moves:
            # 敵方無合法移動，視為我方勝
            if owo: print("困，1")
            return 1
        return None
    
def start_is_terminal(terminal_result, current_move_count, start_move_count=0):
    """
    用來判斷start那一回合是贏還是輸，不然整天搞那個太麻煩了
    輸入：
    terminal_result：1或-1或0
    current_move_count：整數
    start_move_count：整數
    """
    if terminal_result == 0:
        return 0 # 和局
    d_move_count = current_move_count - start_move_count
    start_wins = (terminal_result == 1 and d_move_count % 2 == 0) or (terminal_result == -1 and d_move_count % 2 == 1)
    if start_wins:
        return 1 # 贏
    else:
        return -1 # 輸



if __name__=="__main__":
    pass
