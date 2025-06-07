import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import prodigyopt
import random
from new_xiangqi import BoardState
import math
import tqdm
from tqdm import trange
import copy
import os
import pickle
from collections import deque, Counter

# 全局常數
CHESS_ROW = 10
CHESS_COLUMN = 9
BOARD_ROW = CHESS_ROW + 4
BOARD_COLUMN = CHESS_COLUMN + 2
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
N_SIMULATIONS = 10  # MCTS 模擬次數
BATCH_SIZE = 1024  # mini-batch 大小
EPOCHS = 20 # 一份訓練資料要訓練幾個 epoch
ITERATIONS = 100  # 訓練幾代模型
SELF_PLAY_GAMES = 1 # 自我對弈的場數
EVAL_GAMES = 20  # 評估遊戲數量
WIN_THRESHOLD = 0.55  # 勝率閾值
DIRICHLET_ALPHA = 0.3

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class ChessNet(nn.Module):
    def __init__(self, num_res_blocks=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=112, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(num_res_blocks)
        ])
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10*9, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 52, kernel_size=1, bias=False),
            nn.BatchNorm2d(52),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(52*10*9, 10*9*52)
        )

    def forward(self, x):
        x = self.conv1(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        policy = self.policy_head(x)
        policy = policy.view(x.size(0), 10, 9, 52)
         
        value = self.value_head(x)
        
        return policy, value
    
class MCTSNode:
    def __init__(self, board_state:BoardState, parent=None, prior = 0):
        self.board_state = board_state
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.is_expanded = False

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0
    
def encode_board(board_state:BoardState):
    piece_map = {'K':0, 'A':1, 'E':2, 'R':3, 'H':4, 'C':5, 'P':6,
                 'k':7, 'a':8, 'e':9, 'r':10, 'h':11, 'c':12, 'p':13}
    
    board_tensor = torch.zeros(14, 10, 9)
    board = board_state.board
    for i in range(len(board)):
            row = (i // BOARD_COLUMN) - 2
            col = (i % BOARD_COLUMN) - 1
            piece = board[i]

            if piece in piece_map:
                if 0 <= row < CHESS_ROW and 0 <= col < CHESS_COLUMN:
                    board_tensor[piece_map[piece], row, col] = 1
    return board_tensor

def encode_board_from_node_and_list(node:MCTSNode=None, history_list:list=None):
    tensor_list = []
    current_node = node
    for _ in range(8):
        if current_node is not None:
            board_tensor_small = encode_board(current_node.board_state)
            tensor_list.append(board_tensor_small)
            current_node = current_node.parent
        else:
            break
    
    if history_list:
        for t, board_state in enumerate(reversed(history_list)):
            if len(tensor_list) < 8:
                board_tensor_small = encode_board(board_state)
                tensor_list.append(board_tensor_small)
            else:
                break

    while len(tensor_list) < 8:
        tensor_list.append(torch.zeros(14, 10, 9))

    board_tensor = torch.cat(tensor_list, dim=0)
    return board_tensor.unsqueeze(0)

def encode_action(move):
    piece, i, j = move
    from_row, from_col = (i // BOARD_COLUMN) - 2, (i % BOARD_COLUMN) - 1
    to_row, to_col = (j // BOARD_COLUMN) - 2, (j % BOARD_COLUMN) - 1

    drow = to_row - from_row
    dcol = to_col - from_col
    
    # 1) 馬/象/士 的 Δ→channel 對照表
    special_map = {
        'H': {(-2,-1):36, (-2,+1):37, (-1,-2):38, (-1,+2):39,
              (+1,-2):40, (+1,+2):41, (+2,-1):42, (+2,+1):43},
        'E': {(-2,-2):44, (-2,+2):45, (+2,-2):46, (+2,+2):47},
        'A': {(-1,-1):48, (-1,+1):49, (+1,-1):50, (+1,+1):51},
    }
    if piece in special_map:
        try:
            channel = special_map[piece][(drow,dcol)]
        except KeyError:
            raise ValueError(f"Invalid move for {piece}: Δ=({drow},{dcol})")
        return (from_row, from_col, channel)
    
    # 2) 車/炮/將/兵 都屬於「正交滑動」0..35
    #    North:  drow<0, dcol==0 → channel = (-drow)-1
    #    South:  drow>0, dcol==0 → channel =  9+(drow-1)
    #    East:   dcol>0, drow==0 → channel = 18+(dcol-1)
    #    West:   dcol<0, drow==0 → channel = 27+(-dcol-1)
    if dcol == 0 and drow != 0:
        dist = abs(drow)
        if not (1 <= dist <= 9):
            raise ValueError(f"Invalid slide distance: {dist}")
        channel = ( -drow - 1 if drow < 0 else 9 + (drow - 1) )
    elif drow == 0 and dcol != 0:
        dist = abs(dcol)
        if not (1 <= dist <= 9):
            raise ValueError(f"Invalid slide distance: {dist}")
        channel = ( 18 + (dcol - 1) if dcol > 0 else 27 + (-dcol - 1) )
    else:
        raise ValueError(f"Invalid sliding move for {piece}: Δ=({drow},{dcol})")

    return (from_row, from_col, channel)

def decode_action(board_state:BoardState, policy:torch.Tensor):
    move_probs = {}
    legal_moves = board_state.gen_moves()
    
    for move in legal_moves:
        from_row, from_col, channel = encode_action(move)
        move_probs[move] = policy[from_row][from_col][channel]

    # 確保機率和是1
    total_prob = sum(move_probs.values())
    if total_prob > 0:
        for move in move_probs:
            move_probs[move] /= total_prob
    return move_probs

class MCTS:
    def __init__(self, net:ChessNet):
        self.net = net
        self.c_puct = 1.5
        self.device = next(net.parameters()).device

    def puct(self, node:MCTSNode, child:MCTSNode):
        return child.value() + self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)

    def select(self, node:MCTSNode):
        while (node.board_state.is_terminal() is None) and node.is_expanded :
            node = max(node.children.values(), key=lambda c: self.puct(node, c))
        return node

    def expand(self, node:MCTSNode, p:torch.Tensor):
        if node.board_state.is_terminal() is not None:
            return
        move_probs = decode_action(node.board_state, p)
        for move, prob in move_probs.items():
            new_board_state = node.board_state.move(move)
            node.children[move] = MCTSNode(new_board_state, node, prob)
        node.is_expanded = True
        pass

    def simulate(self, node:MCTSNode, v:torch.Tensor):
        # TODO:加入平局=0的判斷
        result = node.board_state.is_terminal()
        if result is not None:
            return result
        return v.item()

    def backpropagate(self, node:MCTSNode, value):
        while node:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def call_network(self, node:MCTSNode, history_list):
        board_tensor = encode_board_from_node_and_list(node=node, history_list=history_list).to(self.device)
        
        with torch.no_grad():
            p, v = self.net(board_tensor)
            batch_size = p.size(0)
            p_flat = p.view(batch_size, -1)
            p_flat = torch.softmax(p_flat, dim=1)
            p = p_flat.view(batch_size, 10, 9, 52)
        return p.squeeze(0), v

    def search(self, board_state:BoardState, n_simulations, history_list):
        root_node = MCTSNode(board_state)
        for _ in range(n_simulations):
            node = root_node
            node = self.select(node)
            p, v = self.call_network(node, history_list)
            if not node.is_expanded:
                self.expand(node, p)
            value = self.simulate(node, v)
            self.backpropagate(node, value)

        pi = {}
        pi_vector = torch.zeros(10, 9, 52)
        total_visits = sum(child.visit_count for child in root_node.children.values())
        for move, child in root_node.children.items():
            pi[move] = child.visit_count / total_visits if total_visits > 0 else 0
            from_row, from_col, channel = encode_action(move)
            pi_vector[from_row][from_col][channel] = child.visit_count / total_visits if total_visits > 0 else 0
        
        return pi, pi_vector

def self_play_game(net:ChessNet):
    board_state = BoardState(board=INITIAL)
    mcts = MCTS(net)
    last8 = deque(maxlen=8) # 純歷史，不包含當前狀態
    counts = Counter()
    counts[board_state.board] = 1
    history = []
    move_count = 0

    with tqdm.tqdm(desc="Self-Play Game",leave=False) as pbar:
        while True:

            # 勝負與和局
            value = board_state.is_terminal()
            if value is not None:
                break
            if counts[board_state.board] >=3:
                value = 0
                break
            
            # 搜尋
            pi, pi_vector = mcts.search(board_state, N_SIMULATIONS, history_list=list(last8))
            
            # 選取動作
            legal_moves = list(pi.keys())
            probs = np.array(list(pi.values()))
            action = random.choices(legal_moves, weights=probs)[0] # 待商榷

            # 紀錄訓練資料
            last8.append(board_state)
            board_tensor = encode_board_from_node_and_list(history_list=list(last8))
            history.append((board_tensor, pi_vector))

            # 分水嶺
            board_state = board_state.move(action)
            
            counts[board_state.board] += 1
            move_count += 1
            pbar.update(1)

    data = []
    for i, (s, p) in enumerate(history):
        z = value if i % 2 == 0 else -value
        data.append((s, p, z))

    # 根據終局結果判斷勝負平
    if value == 0:
        result = (0, 1, 0)  # (wins, draws, losses)
    elif (value == 1 and move_count % 2 == 0) or (value == -1 and move_count % 2 == 1):
        result = (1, 0, 0)  # 紅方勝
    else:
        result = (0, 0, 1)  # 黑方勝

    return data, result
    

def train(net:ChessNet, data, epochs=EPOCHS, batch_size=BATCH_SIZE):
    device = next(net.parameters()).device
    optimizer = prodigyopt.Prodigy(net.parameters())
    for epoch in tqdm.tqdm(range(epochs), desc="Training Epochs"):
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            if len(batch) < batch_size:
                continue
            states, pis, values = zip(*batch)
            s = torch.cat(states, dim=0).to(device)
            pi = torch.stack(pis).to(device)
            z = torch.tensor(values, dtype=torch.float32).unsqueeze(-1).to(device)
        
            p, v = net(s)

            batch_size = p.size(0)
            p_flat = p.view(batch_size, -1)
            pi_flat = pi.view(batch_size, -1)

            loss = F.cross_entropy(p_flat,pi_flat) + F.mse_loss(v,z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 顯示 batch 進度
            tqdm.tqdm.write(f"Epoch {epoch+1}, Batch {i//batch_size+1}, Loss: {loss.item():.4f}")
    pass

def evaluate(net_new: ChessNet, net_old: ChessNet, n_games=EVAL_GAMES):
    wins, draws, losses = 0, 0, 0
    mcts_new = MCTS(net_new)
    mcts_old = MCTS(net_old)

    with tqdm.tqdm(range(n_games), desc="Eval BIG") as pbar:
        for game_idx in range(n_games):
            board_state = BoardState(INITIAL)
            last8 = deque(maxlen=8) # 純歷史，不包含當前狀態
            counts = Counter()
            counts[board_state.board] = 1
            move_count = 0

            # 偶數場次：新模型先手 (紅方)
            # 奇數場次：舊模型先手 (紅方)
            new_model_plays_red = (game_idx % 2 == 0)

            with tqdm.tqdm(desc="Eval Game",leave=False) as pbar2:
                while True:

                    # 勝負與和局
                    terminal_result = board_state.is_terminal()
                    if terminal_result is not None:
                        break
                    if counts[board_state.board] >=3:
                        terminal_result = 0
                        break
                    
                    # 選擇模型：紅方 (move_count % 2 == 0) 時
                    if move_count % 2 == 0:  # 紅方回合
                        mcts = mcts_new if new_model_plays_red else mcts_old
                    else:  # 黑方回合
                        mcts = mcts_old if new_model_plays_red else mcts_new

                    # 搜尋
                    pi, _ = mcts.search(board_state, N_SIMULATIONS, history_list=list(last8))

                    # 選取動作
                    action = max(pi.items(), key=lambda x: x[1])[0]

                    # 記錄歷史
                    last8.append(board_state)

                    # 分水嶺
                    board_state = board_state.move(action)
                    
                    counts[board_state.board] += 1
                    move_count += 1
                    pbar2.update(1)

            # 計算結果 (從新模型的角度)
            if terminal_result == 0:
                draws += 1
            else:
                # 判斷新模型是否獲勝
                red_wins = (terminal_result == 1 and move_count % 2 == 0) or (terminal_result == -1 and move_count % 2 == 1)
                
                if (new_model_plays_red and red_wins) or (not new_model_plays_red and not red_wins):
                    wins += 1  # 新模型勝
                else:
                    losses += 1  # 舊模型勝

            pbar.set_postfix({"Wins": wins, "Draws": draws, "Losses": losses})
            pbar.update(1)

    win_rate = wins / n_games
    return win_rate, wins, draws, losses

def save_dataset_pickle(dataset, iteration, folder="training_data"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = f"{folder}/dataset_iter_{iteration:04d}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {filename}")

def load_dataset_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    net = ChessNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    best_net = copy.deepcopy(net)
    best_net.to(device)
    model_path = "best_model.pth"

    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        best_net.load_state_dict(torch.load(model_path))

    for iteration in tqdm.tqdm(range(ITERATIONS), desc="Training Iterations"):
        print(f"正在訓練第{iteration}的迭代")
        
        # 自我對弈
        dataset = []
        wins, draws, losses = 0, 0, 0
        with tqdm.tqdm(range(SELF_PLAY_GAMES), desc="Self-play BIG", leave=True) as pbar:
            for game_idx in range(SELF_PLAY_GAMES):
                game_data, (w, d, l) = self_play_game(net)
                dataset.extend(game_data)
                wins += w
                draws += d
                losses += l
                pbar.set_postfix({"Wins": wins, "Draws": draws, "Losses": losses})
                pbar.update(1)

        # 把費盡心思的訓練資料存下來
        save_dataset_pickle(dataset, iteration)

        train(net, dataset)

        win_rate, wins, draws, losses = evaluate(net, best_net)

        if win_rate > WIN_THRESHOLD:
            best_net = copy.deepcopy(net)
            torch.save(best_net.state_dict(), model_path)
            print(f"New model saved with win rate {win_rate:.3f}")

        # 再怎麼樣都加減存一下模型
        torch.save(net.state_dict(), f"model_{iteration}.pth")
    