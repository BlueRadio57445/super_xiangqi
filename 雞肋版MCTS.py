# 這個版本原本以為可以加速訓練速度，但實際上反而更慢了
# 也許我哪天會需要這個版本吧？

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
            return result
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

        pi = {}
        pi_vector = torch.zeros(10, 9, 52)
        total_visits = sum(child.visit_count for child in root_node.children.values())
        for move, child in root_node.children.items():
            pi[move] = child.visit_count / total_visits if total_visits > 0 else 0
            from_row, from_col, channel = encode_action(move)
            pi_vector[from_row][from_col][channel] = child.visit_count / total_visits if total_visits > 0 else 0
        
        return pi, pi_vector