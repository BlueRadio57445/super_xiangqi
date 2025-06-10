import tkinter as tk
from tkinter import messagebox, ttk
import time
import threading
from collections import namedtuple
from itertools import count
import re

uni_pieces = {'R': '🩤', 'H': '🩣', 'E': '🩢', 'A': '🩡', 'K': '🩠', 'C': '🩥', 'P': '🩦',
              'r': '🩫', 'h': '🩪', 'e': '🩩', 'a': '🩨', 'k': '🩧', 'c': '🩬', 'p': '🩭', '.': '·'}
chinese_pieces = {'R': '車', 'H': '馬', 'E': '相', 'A': '仕', 'K': '帥', 'C': '炮', 'P': '兵',
                  'r': '车', 'h': '马', 'e': '象', 'a': '士', 'k': '将', 'c': '砲', 'p': '卒', '.': ''}

CHESS_ROW = 10
CHESS_COLUMN = 9
BOARD_ROW = CHESS_ROW+4
BOARD_COLUMN = CHESS_COLUMN+2
piece = {'K': 6000, 'A': 120, 'E': 120, 'H': 270, 'R': 600, 'P': 30, 'C': 285}
pst = {
    'K': (
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, -9, -9, -9, 0, 0, 0,
        0, 0, 0, -8, -8, -8, 0, 0, 0,
        0, 0, 0, 1, 5, 1, 0, 0, 0,
    ),
    'A': (
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, -1, 0, -1, 0, 0, 0,
        0, 0, 0, 0, 3, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    ),
    'E': (
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, -1, 0, 0, 0, -1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        -2, 0, 0, 0, 3, 0, 0, 0, -2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    ),
    'H': (
        4, 8, 16, 12, 4, 12, 16, 8, 4,
        4, 10, 28, 16, 8, 16, 28, 10, 4,
        12, 14, 16, 20, 18, 20, 16, 14, 12,
        8, 24, 18, 24, 20, 24, 18, 24, 8,
        6, 16, 14, 18, 16, 18, 14, 16, 6,
        4, 12, 16, 14, 12, 14, 16, 12, 4,
        2, 6, 8, 6, 10, 6, 8, 6, 2,
        4, 2, 8, 8, 6, 8, 8, 2, 4,
        0, 2, 4, 4, -10, 4, 4, 2, 0,
        0, -4, 0, 0, 0, 0, 0, -4, 0,

    ),
    'R': (
        14, 14, 12, 18, 16, 18, 12, 14, 14,
        16, 20, 18, 24, 26, 24, 18, 20, 16,
        12, 12, 12, 18, 18, 18, 12, 12, 12,
        12, 18, 16, 22, 22, 22, 16, 18, 12,
        12, 14, 12, 18, 18, 18, 12, 14, 12,
        12, 16, 14, 20, 20, 20, 14, 16, 12,
        6, 10, 8, 14, 14, 14, 8, 10, 6,
        4, 8, 6, 14, 12, 14, 6, 8, 4,
        8, 4, 8, 16, 8, 16, 8, 4, 8,
        -2, 10, 6, 14, 12, 14, 6, 10, -2,
    ),
    'C': (
        6, 4, 0, -10, -12, -10, 0, 4, 6,
        2, 2, 0, -4, -14, -4, 0, 2, 2,
        2, 2, 0, -10, -8, -10, 0, 2, 2,
        0, 0, -2, 4, 10, 4, -2, 0, 0,
        0, 0, 0, 2, 8, 2, 0, 0, 0,
        -2, 0, 4, 2, 6, 2, 4, 0, -2,
        0, 0, 0, 2, 4, 2, 0, 0, 0,
        4, 0, 8, 6, 19, 6, 8, 0, 4,
        0, 2, 4, 6, 6, 6, 4, 2, 0,
        0, 0, 2, 6, 6, 6, 2, 0, 0,
    ),
    'P': (
        0, 3, 6, 9, 12, 9, 6, 3, 0,
        18, 36, 56, 80, 120, 80, 56, 36, 18,
        14, 26, 42, 60, 80, 60, 42, 26, 14,
        10, 20, 30, 34, 40, 34, 30, 20, 10,
        6, 12, 18, 18, 20, 18, 18, 12, 6,
        2, 0, 8, 0, 8, 0, 8, 0, 2,
        0, 0, -2, 0, 4, 0, -2, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    )
}

def padrow(row):
    result = (0,) + tuple(x + piece[k] for x in row) + (0,)
    return result

for k, table in pst.items():
    pst[k] = sum((padrow(table[i * CHESS_COLUMN:(i+1)*CHESS_COLUMN])
                  for i in range(CHESS_ROW)), ())
    pst[k] = (0,) * 22 + pst[k] + (0,) * 22
    pst[k] = pst[k]

initial = (
    '          \n'
    '          \n'
    ' rheakaehr\n'
    ' .........\n'
    ' .c.....c.\n'
    ' p.p.p.p.p\n'
    ' .........\n'
    ' .........\n'
    ' P.P.P.P.P\n'
    ' .C.....C.\n'
    ' .........\n'
    ' RHEAKAEHR\n'
    '          \n'
    '          \n'
)

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

MATE_LOWER = piece['K'] - 2 * (piece['R'] + piece['H'] +
                               piece['C'] + piece['A'] + piece['E'] + 2.5 * piece['P'])
MATE_UPPER = piece['K'] + 2 * (piece['R'] + piece['H'] +
                               piece['C'] + piece['A'] + piece['E'] + 2.5 * piece['P'])

TABLE_SIZE = 1e7
QS_LIMIT = 219
EVAL_ROUGHNESS = 13
DRAW_TEST = True


class Position(namedtuple('Position', 'board score')):
    def gen_moves(self):
        for i, p in enumerate(self.board):
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
                    if q.isspace():
                        break
                    if q.isupper() and p != 'C':
                        break
                    if p == 'C':
                        if cannon_flag:
                            if q.islower():
                                pass
                            elif q.isupper():
                                break
                            else:
                                continue
                        elif q.isalpha():
                            cannon_flag = True
                            continue
                    if p in ('H', 'E') and self.board[i + step] != '.':
                        break
                    if p in ('A', 'K'):
                        row, column = j // BOARD_COLUMN, j % BOARD_COLUMN
                        if not (9 <= row <= BOARD_COLUMN and 4 <= column <= 6):
                            break
                    if p == 'E' and not 6 <= j // BOARD_COLUMN <= BOARD_COLUMN:
                        break
                    if p == 'P' and j // BOARD_COLUMN > 6 and d in (E, W):
                        break
                    black_king = self.board.index('k')
                    if p == 'K':
                        red_king = j
                    else:
                        red_king = self.board.index('K')
                    if black_king % BOARD_COLUMN == red_king % BOARD_COLUMN:
                        if not any(piece != '.' for piece in self.board[black_king+BOARD_COLUMN:red_king:BOARD_COLUMN]):
                            break

                    yield i, j
                    if p in 'HPEAK' or q.islower():
                        break

    def rotate(self):
        return Position(self.board[::-1].swapcase(), -self.score)

    def put(self, board, i, p):
        return board[:i] + p + board[i + 1:]

    def move(self, move):
        i, j = move
        board = self.board
        score = self.score + self.value(move)
        board = self.put(board, j, board[i])
        board = self.put(board, i, '.')
        return Position(board, score).rotate()

    def value(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        MOVE_COST = 0
        score = pst[p][j] - pst[p][i] - MOVE_COST
        if q.islower():
            score += pst[q.upper()][BOARD_ROW * BOARD_COLUMN-1 - j]
        return score


Entry = namedtuple('Entry', 'lower upper')


class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()

    def bound(self, pos, mid, depth, root=True):
        depth = max(depth, 0)

        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        if DRAW_TEST:
            if not root and pos in self.history:
                return 0

        entry = self.tp_score.get(
            (pos, depth, root), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= mid and (not root or self.tp_move.get(pos) is not None):
            return entry.lower
        if entry.upper < mid:
            return entry.upper

        def moves():
            if depth > 0 and not root and any(c in pos.board for c in 'RHCP'):
                yield None, -self.bound(pos.rotate(), 1 - mid, depth - 3, root=False)
            if depth == 0:
                yield None, pos.score
            killer = self.tp_move.get(pos)
            if killer and (depth > 0 or pos.value(killer) >= QS_LIMIT):
                yield killer, -self.bound(pos.move(killer), 1 - mid, depth - 1, root=False)
            for move in sorted(pos.gen_moves(), key=pos.value, reverse=True):
                if depth > 0 or pos.value(move) >= QS_LIMIT:
                    yield move, -self.bound(pos.move(move), 1 - mid, depth - 1, root=False)

        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= mid:
                if len(self.tp_move) > TABLE_SIZE:
                    self.tp_move.clear()
                self.tp_move[pos] = move
                break

        if best < mid and best < 0 and depth > 0:
            def is_dead(pos): return any(pos.value(m) >=
                                         MATE_LOWER for m in pos.gen_moves())
            if all(is_dead(pos.move(m)) for m in pos.gen_moves()):
                in_check = is_dead(pos.rotate())
                best = -MATE_UPPER if in_check else 0

        if len(self.tp_score) > TABLE_SIZE:
            self.tp_score.clear()
        if best >= mid:
            self.tp_score[pos, depth, root] = Entry(best, entry.upper)
        if best < mid:
            self.tp_score[pos, depth, root] = Entry(entry.lower, best)

        return best

    def search(self, pos, history=()):
        if DRAW_TEST:
            self.history = set(history)
            self.tp_score.clear()

        for depth in range(1, 1000):
            lower, upper = -MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                mid = (lower + upper + 1) // 2
                score = self.bound(pos, mid, depth)
                if score >= mid:
                    lower = score
                if score < mid:
                    upper = score
            self.bound(pos, lower, depth)
            yield depth, self.tp_move.get(pos), self.tp_score.get((pos, depth, True)).lower


A1 = CHESS_ROW*BOARD_COLUMN+1


def parse(c):
    fil, rank = ord(c[0]) - ord('a'), int(c[1]) - 1
    return A1 + fil - BOARD_COLUMN * rank


def render(i):
    rank, fil = divmod(i - A1, BOARD_COLUMN)
    return chr(fil + ord('a')) + str(-rank + 1)


class XiangqiGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("中國象棋")
        self.root.geometry("480x576")
        self.root.configure(bg='#8B4513')
        
        # 遊戲狀態
        self.hist = [Position(initial, 0)]
        self.searcher = Searcher()
        self.selected_pos = None
        self.valid_moves = []
        self.game_over = False
        self.ai_thinking = False
        
        # 創建介面
        self.create_widgets()
        self.update_board()
        
    def create_widgets(self):
        # 狀態欄
        self.status_frame = tk.Frame(self.root, bg='#8B4513')
        self.status_frame.pack(pady=4)
        
        # 字體大小調整
        self.status_label = tk.Label(self.status_frame, text="輪到紅方走棋", 
                                   font=('SimHei', 12), bg='#8B4513', fg='white')
        self.status_label.pack()
        
        # 棋盤框架
        self.board_frame = tk.Frame(self.root, bg='#DEB887', relief='raised', bd=2)
        self.board_frame.pack(pady=6, padx=12)
        
        # 創建棋盤按鈕
        self.buttons = []
        for row in range(10):
            button_row = []
            for col in range(9):
                btn = tk.Button(self.board_frame, width=4, height=1,
                              font=('SimHei', 13, 'bold'),
                              command=lambda r=row, c=col: self.on_square_click(r, c))
                btn.grid(row=row, column=col, padx=0, pady=0)
                button_row.append(btn)
            self.buttons.append(button_row)
            
        # 控制按鈕
        self.control_frame = tk.Frame(self.root, bg='#8B4513')
        self.control_frame.pack(pady=6)
        
        # 控制按鈕字體
        self.new_game_btn = tk.Button(self.control_frame, text="新遊戲", 
                                    font=('SimHei', 11),
                                    command=self.new_game)
        self.new_game_btn.pack(side=tk.LEFT, padx=6)
        
        self.hint_btn = tk.Button(self.control_frame, text="提示", 
                                font=('SimHei', 11),
                                command=self.show_hint)
        self.hint_btn.pack(side=tk.LEFT, padx=6)
        
    def board_pos_to_index(self, row, col):
        """將棋盤座標轉換為内部索引"""
        return A1 + col - BOARD_COLUMN * (8 - row)
        
    def index_to_board_pos(self, index):
        """將内部索引轉換為棋盤座標"""
        rank, fil = divmod(index - A1, BOARD_COLUMN)
        return 8 + rank, fil
        
    def update_board(self):
        """更新棋盤顯示"""
        pos = self.hist[-1]
        
        for row in range(10):
            for col in range(9):
                btn = self.buttons[row][col]
                index = self.board_pos_to_index(row, col)
                piece = pos.board[index] if 0 <= index < len(pos.board) else '.'
                
                # 設置棋子文字
                if piece in chinese_pieces:
                    btn.config(text=chinese_pieces[piece])
                else:
                    btn.config(text='')
                
                # 設置顏色
                if piece.isupper():  # 紅方棋子
                    btn.config(fg='red', bg='#F5DEB3')
                elif piece.islower():  # 黑方棋子
                    btn.config(fg='black', bg='#F5DEB3')
                else:  # 空位
                    btn.config(fg='black', bg='#DEB887')
                
                # 高亮選中的棋子
                if self.selected_pos == (row, col):
                    btn.config(bg='yellow')
                
                # 高亮可移動位置
                elif (row, col) in [(self.index_to_board_pos(move[1])) for move in self.valid_moves]:
                    btn.config(bg='lightgreen')
                    
    def on_square_click(self, row, col):
        """處理棋盤點擊事件"""
        if self.game_over or self.ai_thinking:
            return
            
        index = self.board_pos_to_index(row, col)
        pos = self.hist[-1]
        piece = pos.board[index] if 0 <= index < len(pos.board) else '.'
        
        if self.selected_pos is None:
            # 選擇棋子
            if piece.isupper():  # 只能選擇紅方棋子
                self.selected_pos = (row, col)
                self.valid_moves = [move for move in pos.gen_moves() 
                                  if move[0] == index]
                self.update_board()
        else:
            # 移動棋子或重新選擇
            if (row, col) == self.selected_pos:
                # 取消選擇
                self.selected_pos = None
                self.valid_moves = []
                self.update_board()
            elif piece.isupper():
                # 重新選擇紅方棋子
                self.selected_pos = (row, col)
                self.valid_moves = [move for move in pos.gen_moves() 
                                  if move[0] == index]
                self.update_board()
            else:
                # 嘗試移動
                from_index = self.board_pos_to_index(*self.selected_pos)
                to_index = index
                move = (from_index, to_index)
                
                if move in pos.gen_moves():
                    self.make_move(move)
                else:
                    messagebox.showwarning("無效移動", "這個移動不合法！")
                
                self.selected_pos = None
                self.valid_moves = []
                self.update_board()
                
    def make_move(self, move):
        """執行移動"""
        self.hist.append(self.hist[-1].move(move))
        self.update_board()
        
        # 檢查遊戲結束
        if self.hist[-1].score <= -MATE_LOWER:
            self.game_over = True
            messagebox.showinfo("遊戲結束", "紅方獲勝！")
            return
            
        # AI思考
        self.status_label.config(text="AI思考中...")
        self.ai_thinking = True
        self.root.update()
        
        # 在新線程中運行AI
        threading.Thread(target=self.ai_move, daemon=True).start()
        
    def ai_move(self):
        """AI移動"""
        start = time.perf_counter()
        move = None
        score = 0
        
        for _, m, s in self.searcher.search(self.hist[-1], self.hist):
            move = m
            score = s
            if time.perf_counter() - start > 1:  # 限制思考時間為1秒
                break
                
        if move:
            # 在主線程中更新UI
            self.root.after(0, self.complete_ai_move, move, score)
        else:
            self.root.after(0, self.ai_no_move)
            
    def complete_ai_move(self, move, score):
        """完成AI移動"""
        self.hist.append(self.hist[-1].move(move))
        self.update_board()
        self.ai_thinking = False
        
        if score == MATE_UPPER:
            self.game_over = True
            messagebox.showinfo("遊戲結束", "將軍！")
            
        if self.hist[-1].score <= -MATE_LOWER:
            self.game_over = True
            messagebox.showinfo("遊戲結束", "黑方獲勝！")
        else:
            self.status_label.config(text="輪到紅方走棋")
            
    def ai_no_move(self):
        """AI無法移動"""
        self.ai_thinking = False
        self.game_over = True
        messagebox.showinfo("遊戲結束", "黑方無法移動，紅方獲勝！")
        
    def new_game(self):
        """開始新遊戲"""
        self.hist = [Position(initial, 0)]
        self.searcher = Searcher()
        self.selected_pos = None
        self.valid_moves = []
        self.game_over = False
        self.ai_thinking = False
        self.status_label.config(text="輪到紅方走棋")
        self.update_board()
        
    def show_hint(self):
        """顯示提示"""
        if self.game_over or self.ai_thinking:
            return
            
        pos = self.hist[-1]
        moves = list(pos.gen_moves())
        if moves:
            best_move = max(moves, key=pos.value)
            from_pos = self.index_to_board_pos(best_move[0])
            to_pos = self.index_to_board_pos(best_move[1])
            messagebox.showinfo("提示", f"建議移動: {chr(ord('a')+from_pos[1])}{10-from_pos[0]} 到 {chr(ord('a')+to_pos[1])}{10-to_pos[0]}")


def main():
    root = tk.Tk()
    game = XiangqiGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()