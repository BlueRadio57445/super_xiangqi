# 這個版本原本以為可以加速訓練速度，但實際上反而更慢了
# 也許我哪天會需要這個版本吧？
import torch
import torch.jit as jit
import math
from mcts_cnn import ChessNet, MCTSNode, decode_action, encode_action, encode_board_from_node_and_list, DIRICHLET_ALPHA, INITIAL
from new_xiangqi import BoardState, start_is_terminal
from collections import deque
import tqdm
import numpy as np
import random

N_SIMULATIONS = 50

@jit.script
def fast_puct_calc(visit_counts: torch.Tensor, value_sums: torch.Tensor, 
                   priors: torch.Tensor, parent_visits: int, c_puct: float):
    values = torch.where(visit_counts > 0, value_sums / visit_counts, torch.zeros_like(visit_counts))
    exploration = c_puct * priors * math.sqrt(float(parent_visits)) / (1.0 + visit_counts)
    return values + exploration

class ChickenMCTS:
    def __init__(self, net:ChessNet):
        self.net = net
        self.c_puct = 1.5
        self.device = next(net.parameters()).device
        self.bonus = 10

    def puct(self, node:MCTSNode, child:MCTSNode):
        return child.value() + self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)

    def select(self, node: MCTSNode):
        while (node.board_state.is_terminal() is None) and node.is_expanded:
            children = list(node.children.values())
            if len(children) <= 1:
                if children:
                    node = children[0]
                break
                
            # 使用 JIT 編譯的函數
            visit_counts = torch.tensor([c.visit_count for c in children], dtype=torch.float32)
            value_sums = torch.tensor([c.value_sum for c in children], dtype=torch.float32)  
            priors = torch.tensor([c.prior for c in children], dtype=torch.float32)
            
            scores = fast_puct_calc(visit_counts, value_sums, priors, node.visit_count, self.c_puct)
            best_idx = torch.argmax(scores).item()
            node = children[best_idx]
            
        return node

    def expand(self, node:MCTSNode, p:torch.Tensor, is_root:bool=False, add_noise=False):
        if node.board_state.is_terminal() is not None:
            return
        move_probs = decode_action(node.board_state, p)
        if is_root and add_noise and len(move_probs) > 0: move_probs = self.add_dirichlet_noise(move_probs)
        for move, prob in move_probs.items():
            new_board_state = node.board_state.move(move)
            node.children[move] = MCTSNode(new_board_state, node, prob)
        node.is_expanded = True
        pass

    def simulate(self, node:MCTSNode, v:torch.Tensor):
        """代表當前node盤面的價值，還沒有到我擔心的事情"""
        result = node.board_state.is_terminal()
        if result is not None:
            self.bonus += 1
            return 10*result
        return v.item()

    def backpropagate(self, node:MCTSNode, value):
        value = -value # 處理我最擔心的事情（因為選擇階段是根據子節點的value選取，所以雖然在這裡看起來輸了，但反而是parent(對手)想選的）
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
    
    def batch_call_network(self, children:dict[tuple, MCTSNode], history_list):
        tensor_list = []
        for move, node in children.items():
            board_tensor = encode_board_from_node_and_list(node=node, history_list=history_list).to(self.device)
            tensor_list.append(board_tensor)
        batch_board_tensor = torch.cat(tensor_list)

        with torch.no_grad():
            p, v = self.net(batch_board_tensor)
            batch_size = p.size(0)
            p_flat = p.view(batch_size, -1)
            p_flat = torch.softmax(p_flat, dim=1)
            p = p_flat.view(batch_size, 10, 9, 52)

        p_dict = {}
        v_dict = {}
        for i, move in enumerate(children.keys()):
            p_dict[move] = p[i]
            v_dict[move] = v[i]
        return p_dict, v_dict

    def add_dirichlet_noise(self, move_probs: dict, epsilon=0.25, alpha=DIRICHLET_ALPHA):
        """使用PyTorch的Dirichlet噪聲混合函數"""
        if not move_probs:
            return move_probs
        
        moves = list(move_probs.keys())
        priors = torch.stack(list(move_probs.values()))
        device = priors.device
        
        # 生成Dirichlet noise (在同一設備上)
        dirichlet = torch.distributions.Dirichlet(torch.full((len(moves),), alpha, device=device))
        noise = dirichlet.sample()
        
        mixed_probs = (1 - epsilon) * priors + epsilon * noise
        
        return dict(zip(moves, mixed_probs))

    def search(self, board_state:BoardState, n_simulations, history_list, add_noise=False):
        root_node = MCTSNode(board_state)

        # 將軍檢查
        for action in root_node.board_state.gen_moves():
            if root_node.board_state.move(action).is_terminal() == -1:
                pi = {}
                pi_vector = torch.zeros(10, 9, 52)
                pi[action] = 1
                from_row, from_col, channel = encode_action(action)
                pi_vector[from_row][from_col][channel] = 1
                return pi, pi_vector

        # 先為根節點單獨做一次
        node = root_node
        is_root = (node == root_node)
        p, v = self.call_network(node, history_list)
        if not node.is_expanded:
            self.expand(node, p, is_root, add_noise)
        value = self.simulate(node, v)
        self.backpropagate(node, value)

        for _ in range(n_simulations):
            node = root_node
            node = self.select(node)
            if node.parent is not None: node = node.parent
            p_dict, v_dict = self.batch_call_network(node.children, history_list)
            for move, child in node.children.items():
                if not child.is_expanded:
                    self.expand(child, p_dict[move])
                value = self.simulate(child, v_dict[move])
                self.backpropagate(child, value)

        # 額外確認
        for _ in range(self.bonus):
            node = root_node
            node = self.select(node)
            if node.parent is not None: node = node.parent
            p_dict, v_dict = self.batch_call_network(node.children, history_list)
            for move, child in node.children.items():
                if not child.is_expanded:
                    self.expand(child, p_dict[move])
                value = self.simulate(child, v_dict[move])
                self.backpropagate(child, value)

        pi = {}
        pi_vector = torch.zeros(10, 9, 52)
        total_visits = sum(child.visit_count for child in root_node.children.values())
        for move, child in root_node.children.items():
            pi[move] = child.visit_count / total_visits if total_visits > 0 else 0
            from_row, from_col, channel = encode_action(move)
            pi_vector[from_row][from_col][channel] = child.visit_count / total_visits if total_visits > 0 else 0
        
        return pi, pi_vector
    
def chicken_self_play_game(net:ChessNet):
    board_state = BoardState(board=INITIAL)
    mcts = ChickenMCTS(net)
    last8 = deque(maxlen=8) # 純歷史，不包含當前狀態
    history = []

    with tqdm.tqdm(desc="Self-Play Game",leave=False) as pbar:
        while True:
            # 搜尋
            pi, pi_vector = mcts.search(board_state, N_SIMULATIONS, history_list=list(last8), add_noise=True)
            
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
            
            pbar.update(1)

            # 勝負與和局
            value = board_state.is_terminal()
            if value is not None:
                break

    move_count = board_state.move_count
    data = []
    for i, (s, p) in enumerate(history):
        z = start_is_terminal(value, i, move_count)
        data.append((s, p, z))

    # 根據終局結果判斷勝負平
    game_result = start_is_terminal(value, 0, move_count)
    if game_result == 0:
        result = (0, 1, 0)  # (wins, draws, losses)
    elif game_result == 1:
        result = (1, 0, 0)  # 紅方勝
    else:
        result = (0, 0, 1)  # 黑方勝

    return data, result
