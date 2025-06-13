import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider
import torch
import argparse
from collections import Counter
import platform

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'Microsoft JhengHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 根據系統選擇合適的字體
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
elif platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'KaiTi']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei']

# 棋子對照表
piece_map = {0: 'K', 1: 'A', 2: 'E', 3: 'R', 4: 'H', 5: 'C', 6: 'P',
             7: 'k', 8: 'a', 9: 'e', 10: 'r', 11: 'h', 12: 'c', 13: 'p'}

# 棋子顯示符號
chinese_pieces = {'K': '帥', 'A': '仕', 'E': '相', 'R': '俥', 'H': '馬', 'C': '炮', 'P': '兵',
                  'k': '將', 'a': '士', 'e': '象', 'r': '車', 'h': '馬', 'c': '包', 'p': '卒', '.': '·'}

# 初始棋盤
INITIAL = (
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

# 動作解碼映射
action_names = {
    # 滑動動作 (0-35)
    **{i: f'N{i+1}' for i in range(9)},      # 北向 1-9 步
    **{i+9: f'S{i+1}' for i in range(9)},    # 南向 1-9 步  
    **{i+18: f'E{i+1}' for i in range(9)},   # 東向 1-9 步
    **{i+27: f'W{i+1}' for i in range(9)},   # 西向 1-9 步
    # 馬的動作 (36-43)
    36: 'H(-2,-1)', 37: 'H(-2,+1)', 38: 'H(-1,-2)', 39: 'H(-1,+2)',
    40: 'H(+1,-2)', 41: 'H(+1,+2)', 42: 'H(+2,-1)', 43: 'H(+2,+1)',
    # 象的動作 (44-47)
    44: 'E(-2,-2)', 45: 'E(-2,+2)', 46: 'E(+2,-2)', 47: 'E(+2,+2)',
    # 士的動作 (48-51)
    48: 'A(-1,-1)', 49: 'A(-1,+1)', 50: 'A(+1,-1)', 51: 'A(+1,+1)'
}

def encode_board_from_string(board_string):
    """從字串編碼棋盤為tensor格式"""
    piece_map_reverse = {'K':0, 'A':1, 'E':2, 'R':3, 'H':4, 'C':5, 'P':6,
                        'k':7, 'a':8, 'e':9, 'r':10, 'h':11, 'c':12, 'p':13}
    
    board_tensor = torch.zeros(14, 10, 9)
    BOARD_COLUMN = 11  # 根據原始碼的定義
    
    for i in range(len(board_string)):
        row = (i // BOARD_COLUMN) - 2
        col = (i % BOARD_COLUMN) - 1
        piece = board_string[i]
        
        if piece in piece_map_reverse:
            if 0 <= row < 10 and 0 <= col < 9:
                board_tensor[piece_map_reverse[piece], row, col] = 1
    
    return board_tensor

def decode_board_from_tensor(board_tensor):
    """從神經網路輸入tensor解碼出棋盤狀態"""
    # 取第一層（最新狀態）的前14個通道
    if len(board_tensor.shape) == 4:  # 如果有batch維度
        current_state = board_tensor[0, :14, :, :]
    else:
        current_state = board_tensor[:14, :, :]
    
    board = [['.' for _ in range(9)] for _ in range(10)]
    
    for channel in range(14):
        piece = piece_map[channel]
        positions = torch.nonzero(current_state[channel], as_tuple=False)
        for pos in positions:
            row, col = pos[0].item(), pos[1].item()
            board[row][col] = piece
    
    return board

def find_game_starts(data):
    """找到每場遊戲的開始位置"""
    initial_tensor = encode_board_from_string(INITIAL)
    game_starts = []
    
    for i, (state_tensor, policy_tensor, value) in enumerate(data):
        # 取最新狀態（第一個時間步）
        if len(state_tensor.shape) == 4:
            current_state = state_tensor[0, :14, :, :]
        else:
            current_state = state_tensor[:14, :, :]
        
        # 檢查是否與初始狀態相符
        if torch.allclose(current_state, initial_tensor, atol=1e-6):
            game_starts.append(i)
    
    return game_starts

def group_data_by_games(data):
    """將資料按遊戲分組"""
    game_starts = find_game_starts(data)
    games = []
    
    for i, start_idx in enumerate(game_starts):
        if i < len(game_starts) - 1:
            end_idx = game_starts[i + 1]
            games.append(data[start_idx:end_idx])
        else:
            games.append(data[start_idx:])
    
    return games

class GameViewer:
    def __init__(self, games):
        self.games = games
        self.current_game = 0
        self.current_move = 0
        
        # 控制元件引用，避免重複創建
        self.game_slider = None
        self.move_slider = None
        self.buttons = {}
        
        # 創建圖形界面
        self.fig = plt.figure(figsize=(18, 10))
        
        # 棋盤區域
        self.ax_board = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
        
        # 策略分佈區域
        self.ax_policy = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        
        # 控制區域
        self.ax_controls = plt.subplot2grid((3, 3), (2, 2))
        
        # 滑桿軸區域（固定位置）
        self.ax_slider_game = None
        self.ax_slider_move = None
        self.ax_buttons = {}
        
        self.setup_controls()
        self.update_display()
        
    def clear_controls(self):
        """清理現有控制元件"""
        # 清理滑桿
        if self.game_slider is not None:
            try:
                self.game_slider.disconnect_events()
            except:
                pass
            self.game_slider = None
            
        if self.move_slider is not None:
            try:
                self.move_slider.disconnect_events()
            except:
                pass
            self.move_slider = None
            
        # 清理按鈕
        for button in self.buttons.values():
            try:
                button.disconnect_events()
            except:
                pass
        self.buttons.clear()
        
    def setup_controls(self):
        """設置控制按鈕和滑桿"""
        # 清理資訊顯示區域
        self.ax_controls.clear()
        self.ax_controls.set_xlim(0, 10)
        self.ax_controls.set_ylim(0, 10)
        self.ax_controls.axis('off')
        
        # 創建滑桿軸（只在第一次創建）
        if self.ax_slider_game is None and len(self.games) > 1:
            self.ax_slider_game = plt.axes([0.7, 0.25, 0.25, 0.03])
            self.game_slider = Slider(self.ax_slider_game, 'Game', 0, len(self.games)-1, 
                                    valinit=self.current_game, valfmt='%d')
            self.game_slider.on_changed(self.update_game)
        
        # 更新或創建 move 滑桿
        max_moves = len(self.games[self.current_game]) - 1
        if max_moves > 0:
            if self.ax_slider_move is None:
                self.ax_slider_move = plt.axes([0.7, 0.20, 0.25, 0.03])
                self.move_slider = Slider(self.ax_slider_move, 'Move', 0, max_moves,
                                        valinit=self.current_move, valfmt='%d')
                self.move_slider.on_changed(self.update_move)
            else:
                # 更新現有滑桿的範圍
                if self.move_slider is not None:
                    # 暫時斷開事件連接
                    cid = self.move_slider.on_changed(lambda x: None)
                    try:
                        self.move_slider.valmax = max_moves
                        self.move_slider.ax.set_xlim(0, max_moves)
                        self.move_slider.poly.xy[2] = max_moves, 1
                        self.move_slider.poly.xy[3] = max_moves, 0
                        self.move_slider.set_val(min(self.current_move, max_moves))
                        self.move_slider.ax.figure.canvas.draw_idle()
                    except:
                        pass
                    # 重新連接事件
                    self.move_slider.on_changed(self.update_move)
        
        # 創建按鈕（只在第一次創建）
        if not self.buttons:
            button_configs = [
                ('prev', [0.7, 0.15, 0.05, 0.04], 'Prev', self.prev_move),
                ('next', [0.76, 0.15, 0.05, 0.04], 'Next', self.next_move),
                ('first', [0.82, 0.15, 0.05, 0.04], 'First', self.first_move),
                ('last', [0.88, 0.15, 0.05, 0.04], 'Last', self.last_move)
            ]
            
            for name, pos, label, callback in button_configs:
                self.ax_buttons[name] = plt.axes(pos)
                self.buttons[name] = Button(self.ax_buttons[name], label)
                self.buttons[name].on_clicked(callback)
        
        # 顯示資訊
        self.update_info_display()
        
    def update_info_display(self):
        """更新資訊顯示"""
        # 清除之前的文字
        for text in self.ax_controls.texts:
            text.remove()
            
        game_info = f"Game {self.current_game + 1}/{len(self.games)}"
        move_info = f"Move {self.current_move + 1}/{len(self.games[self.current_game])}"
        current_player = "紅方" if self.current_move % 2 == 0 else "黑方"
        value = self.games[self.current_game][self.current_move][2]
        
        self.ax_controls.text(5, 8, game_info, ha='center', fontsize=12, fontweight='bold')
        self.ax_controls.text(5, 7, move_info, ha='center', fontsize=12)
        self.ax_controls.text(5, 6, f"當前: {current_player}", ha='center', fontsize=12)
        self.ax_controls.text(5, 5, f"價值: {value:.3f}", ha='center', fontsize=12)
        
    def visualize_board(self, board, title=""):
        """視覺化棋盤"""
        self.ax_board.clear()
        self.ax_board.set_xlim(-0.5, 8.5)
        self.ax_board.set_ylim(-0.5, 9.5)
        self.ax_board.set_aspect('equal')
        self.ax_board.invert_yaxis()
        
        # 畫棋盤線
        for i in range(10):
            self.ax_board.axhline(i, color='black', linewidth=1)
        for j in range(9):
            self.ax_board.axvline(j, color='black', linewidth=1)
        
        # 畫河界
        self.ax_board.axhline(4.5, color='red', linewidth=3, alpha=0.7)
        
        # 畫九宮格
        self.ax_board.plot([3, 5], [0, 2], 'k-', linewidth=1)
        self.ax_board.plot([3, 5], [2, 0], 'k-', linewidth=1)
        self.ax_board.plot([3, 5], [7, 9], 'k-', linewidth=1)
        self.ax_board.plot([3, 5], [9, 7], 'k-', linewidth=1)
        
        # 放置棋子
        for i in range(10):
            for j in range(9):
                piece = board[i][j]
                if piece != '.':
                    color = 'lightcoral' if piece.isupper() else 'lightblue'
                    circle = plt.Circle((j, i), 0.3, color=color, alpha=0.8)
                    self.ax_board.add_patch(circle)
                    
                    self.ax_board.text(j, i, chinese_pieces[piece], ha='center', va='center', 
                                     fontsize=12, fontweight='bold', 
                                     color='white' if piece.isupper() else 'black')
        
        self.ax_board.set_title(title, fontsize=14, fontweight='bold')
        self.ax_board.set_xticks(range(9))
        self.ax_board.set_yticks(range(10))
        self.ax_board.grid(True, alpha=0.3)
        
    def visualize_policy(self, policy_tensor, title="", top_k=8):
        """視覺化策略分佈"""
        self.ax_policy.clear()
        
        policy_flat = policy_tensor.view(-1)
        top_probs, top_indices = torch.topk(policy_flat, top_k)
        
        actions_info = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            if prob.item() > 0:
                from_row = idx.item() // (9 * 52)
                from_col = (idx.item() % (9 * 52)) // 52
                action_type = idx.item() % 52
                
                action_name = action_names.get(action_type, f"A{action_type}")
                actions_info.append({
                    'rank': i+1,
                    'from_pos': f"({from_row},{from_col})",
                    'action': action_name,
                    'prob': prob.item()
                })
        
        if actions_info:
            ranks = [info['rank'] for info in actions_info]
            probs = [info['prob'] for info in actions_info]
            labels = [f"{info['from_pos']}\n{info['action']}" for info in actions_info]
            
            bars = self.ax_policy.bar(ranks, probs, color='skyblue', alpha=0.7)
            self.ax_policy.set_xlabel('Action Rank')
            self.ax_policy.set_ylabel('Probability')
            self.ax_policy.set_title(f'{title}\nTop {top_k} Actions')
            self.ax_policy.set_xticks(ranks)
            self.ax_policy.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                self.ax_policy.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                                  f'{prob:.3f}', ha='center', va='bottom', fontsize=7)
        
    def update_display(self):
        """更新顯示"""
        if not self.games or self.current_game >= len(self.games):
            return
            
        game = self.games[self.current_game]
        if self.current_move >= len(game):
            self.current_move = len(game) - 1
            
        state_tensor, policy_tensor, value = game[self.current_move]
        board = decode_board_from_tensor(state_tensor)
        
        current_player = "紅方" if self.current_move % 2 == 0 else "黑方"
        title = f"Game {self.current_game + 1} - Move {self.current_move} - {current_player} - Value: {value:.3f}"
        
        if self.current_move % 2 == 0:
            pass # 紅方不轉
        else:
            # 黑方轉
            board.reverse()
            for i in range(10):
                board[i].reverse()
            for i in range(10):
                for j in range(9):
                    board[i][j] = board[i][j].swapcase()
                    pass

        self.visualize_board(board, title)
        self.visualize_policy(policy_tensor, f"策略分佈")
        
        # 只更新資訊顯示，不重新創建控制元件
        self.update_info_display()
        plt.draw()
        
    def update_game(self, val):
        """更新遊戲選擇"""
        new_game = int(self.game_slider.val)
        if new_game != self.current_game:
            self.current_game = new_game
            self.current_move = 0
            # 更新 move 滑桿的範圍
            self.setup_controls()
            self.update_display()
        
    def update_move(self, val):
        """更新回合選擇"""
        new_move = int(self.move_slider.val)
        if new_move != self.current_move:
            self.current_move = new_move
            self.update_display()
        
    def prev_move(self, event):
        """上一步"""
        if self.current_move > 0:
            self.current_move -= 1
            if self.move_slider:
                self.move_slider.set_val(self.current_move)
            self.update_display()
            
    def next_move(self, event):
        """下一步"""
        if self.current_move < len(self.games[self.current_game]) - 1:
            self.current_move += 1
            if self.move_slider:
                self.move_slider.set_val(self.current_move)
            self.update_display()
            
    def first_move(self, event):
        """第一步"""
        self.current_move = 0
        if self.move_slider:
            self.move_slider.set_val(self.current_move)
        self.update_display()
        
    def last_move(self, event):
        """最後一步"""
        self.current_move = len(self.games[self.current_game]) - 1
        if self.move_slider:
            self.move_slider.set_val(self.current_move)
        self.update_display()

def visualize_games_interactive(pkl_file):
    """互動式遊戲視覺化"""
    print(f"載入訓練資料: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"總共有 {len(data)} 筆訓練資料")
    
    # 按遊戲分組
    games = group_data_by_games(data)
    print(f"找到 {len(games)} 場遊戲")
    
    for i, game in enumerate(games):
        print(f"遊戲 {i+1}: {len(game)} 步")
    
    # 創建互動式檢視器
    viewer = GameViewer(games)
    plt.show()

def analyze_dataset_statistics(pkl_file):
    """分析資料集統計資訊"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"=== 資料集統計 ===")
    print(f"總資料筆數: {len(data)}")
    
    # 價值分佈
    values = [item[2] for item in data]
    value_counts = Counter(values)
    print(f"價值分佈: {dict(value_counts)}")
    
    # 按遊戲分組
    games = group_data_by_games(data)
    print(f"\n=== 遊戲統計 ===")
    print(f"總遊戲數: {len(games)}")
    
    game_lengths = [len(game) for game in games]
    print(f"遊戲長度 - 平均: {np.mean(game_lengths):.1f}, 最短: {min(game_lengths)}, 最長: {max(game_lengths)}")
    
    # 結果統計
    game_results = []
    for game in games:
        final_value = game[-1][2]  # 最後一步的價值
        if abs(final_value) < 0.1:
            game_results.append('平局')
        elif final_value > 0:
            game_results.append('紅勝')
        else:
            game_results.append('黑勝')
    
    result_counts = Counter(game_results)
    print(f"遊戲結果: {dict(result_counts)}")
    
    # 繪製統計圖
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(value_counts.keys(), value_counts.values(), color=['red', 'gray', 'blue'])
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Value Distribution')
    
    plt.subplot(1, 3, 2)
    plt.hist(game_lengths, bins=20, alpha=0.7, color='green')
    plt.xlabel('Game Length')
    plt.ylabel('Frequency')
    plt.title('Game Length Distribution')
    
    plt.subplot(1, 3, 3)
    colors = ['red', 'gray', 'blue']
    plt.bar(result_counts.keys(), result_counts.values(), color=colors[:len(result_counts)])
    plt.xlabel('Game Result')
    plt.ylabel('Count')
    plt.title('Game Results')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='象棋訓練資料視覺化工具')
    parser.add_argument('pkl_file', help='訓練資料pkl檔案路徑')
    parser.add_argument('--stats', action='store_true', help='只顯示統計資訊')
    
    args = parser.parse_args()
    
    if args.stats:
        analyze_dataset_statistics(args.pkl_file)
    else:
        visualize_games_interactive(args.pkl_file)