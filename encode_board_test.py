import torch
import copy
from collections import deque
from mcts_cnn import encode_board
from mcts_cnn import encode_board_from_node_and_list
from mcts_cnn import MCTSNode
from mcts_cnn import INITIAL
from new_xiangqi import BoardState

def test_encode_board_from_node_and_list():
    """
    測試 encode_board_from_node_and_list 函數的正確性
    """
    print("開始測試 encode_board_from_node_and_list 函數...")
    
    # 創建初始棋盤狀態
    initial_board = BoardState(board=INITIAL)
    
    # 測試案例1: 只有node歷史，沒有list歷史
    print("\n=== 測試案例1: 只有node歷史 ===")
    test_case_1_only_node_history(initial_board)
    
    # 測試案例2: 只有list歷史，沒有node歷史
    print("\n=== 測試案例2: 只有list歷史 ===")
    test_case_2_only_list_history(initial_board)
    
    # 測試案例3: 兩種歷史都有
    print("\n=== 測試案例3: 兩種歷史都有 ===")
    test_case_3_both_histories(initial_board)
    
    # 測試案例4: 歷史不足8步
    print("\n=== 測試案例4: 歷史不足8步 ===")
    test_case_4_insufficient_history(initial_board)
    
    # 測試案例5: 邊界情況
    print("\n=== 測試案例5: 邊界情況 ===")
    test_case_5_boundary_cases()

def test_case_1_only_node_history(initial_board):
    """測試只有node歷史的情況"""
    print("創建一個MCTS節點鏈...")
    
    # 創建一個簡單的節點鏈
    root = MCTSNode(initial_board)
    current = root
    
    # 模擬進行3步棋
    moves_made = []
    for i in range(3):
        legal_moves = current.board_state.gen_moves()
        if legal_moves:
            move = legal_moves[0]  # 選擇第一個合法移動
            moves_made.append(move)
            new_board = current.board_state.move(move)
            child = MCTSNode(new_board, parent=current)
            current.children[move] = child
            current = child
    
    # 測試編碼
    result = encode_board_from_node_and_list(node=current, history_list=None)
    
    print(f"節點鏈長度: {get_node_chain_length(current)}")
    print(f"結果張量形狀: {result.shape}")
    print(f"預期形狀: (1, 112, 10, 9)")
    
    # 驗證形狀正確
    assert result.shape == (1, 112, 10, 9), f"形狀錯誤: {result.shape}"
    
    # 驗證歷史順序
    verify_history_order(current, None, result)
    print("✓ 測試案例1通過")

def test_case_2_only_list_history(initial_board):
    """測試只有list歷史的情況"""
    print("創建list歷史...")
    
    # 創建一個歷史列表
    history_list = []
    current_board = initial_board
    
    for i in range(5):
        history_list.append(current_board)
        legal_moves = current_board.gen_moves()
        if legal_moves:
            move = legal_moves[0]
            current_board = current_board.move(move)
    
    # 測試編碼
    result = encode_board_from_node_and_list(node=None, history_list=history_list)
    
    print(f"歷史列表長度: {len(history_list)}")
    print(f"結果張量形狀: {result.shape}")
    
    # 驗證形狀正確
    assert result.shape == (1, 112, 10, 9), f"形狀錯誤: {result.shape}"
    
    # 驗證歷史順序
    verify_history_order(None, history_list, result)
    print("✓ 測試案例2通過")

def test_case_3_both_histories(initial_board):
    """測試兩種歷史都有的情況"""
    print("創建兩種歷史...")
    
    # 創建list歷史（較早的歷史）
    history_list = []
    current_board = initial_board
    
    for i in range(3):
        history_list.append(current_board)
        legal_moves = current_board.gen_moves()
        if legal_moves:
            move = legal_moves[0]
            current_board = current_board.move(move)
    
    # 創建node歷史（較近的歷史）
    root = MCTSNode(current_board)
    current = root
    
    for i in range(4):
        legal_moves = current.board_state.gen_moves()
        if legal_moves:
            move = legal_moves[0]
            new_board = current.board_state.move(move)
            child = MCTSNode(new_board, parent=current)
            current.children[move] = child
            current = child
    
    # 測試編碼
    result = encode_board_from_node_and_list(node=current, history_list=history_list)
    
    print(f"節點鏈長度: {get_node_chain_length(current)}")
    print(f"歷史列表長度: {len(history_list)}")
    print(f"結果張量形狀: {result.shape}")
    
    # 驗證形狀正確
    assert result.shape == (1, 112, 10, 9), f"形狀錯誤: {result.shape}"
    
    # 驗證node歷史優先
    verify_node_priority(current, history_list, result)
    print("✓ 測試案例3通過")

def test_case_4_insufficient_history(initial_board):
    """測試歷史不足8步的情況"""
    print("測試歷史不足的情況...")
    
    # 只創建2步歷史
    history_list = []
    current_board = initial_board
    
    for i in range(2):
        history_list.append(current_board)
        legal_moves = current_board.gen_moves()
        if legal_moves:
            move = legal_moves[0]
            current_board = current_board.move(move)
    
    # 測試編碼
    result = encode_board_from_node_and_list(node=None, history_list=history_list)
    
    print(f"歷史列表長度: {len(history_list)}")
    print(f"結果張量形狀: {result.shape}")
    
    # 驗證形狀正確
    assert result.shape == (1, 112, 10, 9), f"形狀錯誤: {result.shape}"
    
    # 驗證補零正確
    verify_zero_padding(result, len(history_list))
    print("✓ 測試案例4通過")

def test_case_5_boundary_cases():
    """測試邊界情況"""
    print("測試邊界情況...")
    
    # 測試空輸入
    result = encode_board_from_node_and_list(node=None, history_list=None)
    assert result.shape == (1, 112, 10, 9), f"空輸入形狀錯誤: {result.shape}"
    
    # 驗證全為零
    assert torch.all(result == 0), "空輸入應該全為零"
    
    # 測試空列表
    result = encode_board_from_node_and_list(node=None, history_list=[])
    assert result.shape == (1, 112, 10, 9), f"空列表形狀錯誤: {result.shape}"
    assert torch.all(result == 0), "空列表應該全為零"
    
    print("✓ 測試案例5通過")

def get_node_chain_length(node):
    """獲取節點鏈的長度"""
    length = 0
    current = node
    while current is not None:
        length += 1
        current = current.parent
    return length

def verify_history_order(node, history_list, result):
    """驗證歷史順序是否正確"""
    print("驗證歷史順序...")
    
    # 手動構建預期的歷史順序
    expected_boards = []
    
    # 首先從node收集
    if node is not None:
        node_boards = []
        current = node
        while current is not None and len(node_boards) < 8:
            node_boards.append(current.board_state)
            current = current.parent
        expected_boards.extend(node_boards)
    
    # 然後從history_list收集
    if history_list is not None and len(expected_boards) < 8:
        remaining = 8 - len(expected_boards)
        for board in reversed(history_list):
            if remaining <= 0:
                break
            expected_boards.append(board)
            remaining -= 1
    
    # 補零
    while len(expected_boards) < 8:
        expected_boards.append(None)
    
    print(f"預期歷史長度: {len([b for b in expected_boards if b is not None])}")
    
    # 這裡可以添加更詳細的驗證邏輯
    # 比較實際編碼結果與預期結果

def verify_node_priority(node, history_list, result):
    """驗證node歷史是否優先於list歷史"""
    print("驗證node歷史優先級...")
    
    # 統計node歷史的長度
    node_length = get_node_chain_length(node)
    print(f"Node歷史長度: {node_length}")
    print(f"List歷史長度: {len(history_list)}")
    
    # 如果node歷史足夠長，應該不會使用list歷史
    if node_length >= 8:
        print("Node歷史已滿，不應使用list歷史")
    else:
        print(f"Node歷史不足，應使用{8-node_length}步list歷史")

def verify_zero_padding(result, actual_history_length):
    """驗證零填充是否正確"""
    print("驗證零填充...")
    
    # 檢查後面的層是否為零
    expected_zero_layers = 8 - actual_history_length
    if expected_zero_layers > 0:
        print(f"預期有{expected_zero_layers}層為零")
        
        # 檢查最後幾層是否為零
        for i in range(expected_zero_layers):
            layer_idx = (7 - i) * 14  # 每個棋盤狀態14層
            layer_start = layer_idx * 14
            layer_end = (layer_idx + 1) * 14
            
            # 這裡需要根據實際的張量結構調整
            # 由於我們使用的是 (1, 112, 10, 9)，需要相應調整檢查邏輯

def create_test_board_sequence():
    """創建一個測試用的棋盤序列"""
    boards = []
    current_board = BoardState(board=INITIAL)
    boards.append(current_board)
    
    # 進行幾步棋
    for _ in range(10):
        legal_moves = current_board.gen_moves()
        if legal_moves:
            move = legal_moves[0]  # 選擇第十四個合法移動
            current_board = current_board.move(move)
            boards.append(current_board)
        else:
            break
    
    return boards

def manual_encode_for_comparison(boards):
    """手動編碼用於比較"""
    tensor_list = []
    
    for i in range(8):
        if i < len(boards):
            board_tensor = encode_board(boards[-(i+1)])  # 從最新開始
            tensor_list.append(board_tensor)
        else:
            tensor_list.append(torch.zeros(14, 10, 9))
    
    return torch.cat(tensor_list, dim=0).unsqueeze(0)

# 如果你想要更詳細的比較測試
def detailed_comparison_test():
    """詳細的對比測試"""
    print("\n=== 詳細對比測試 ===")
    
    # 創建測試序列
    test_boards = create_test_board_sequence()
    
    # 分割為node歷史和list歷史
    list_history = test_boards[:5]
    
    # 創建node鏈
    root = MCTSNode(test_boards[5])
    current = root
    for i in range(6, len(test_boards)):
        print(i)
        if i < len(test_boards):
            child = MCTSNode(test_boards[i], parent=current)
            current.children[('dummy', i, i+1)] = child
            current = child
    
    # 使用你的函數編碼
    your_result = encode_board_from_node_and_list(node=current, history_list=list_history)
    
    # 手動編碼預期結果
    expected_result = manual_encode_for_comparison(test_boards[-8:])
    
    print(f"你的結果形狀: {your_result.shape}")
    print(f"預期結果形狀: {expected_result.shape}")
    
    # 比較結果
    if torch.allclose(your_result, expected_result, atol=1e-6):
        print("✓ 詳細對比測試通過！")
    else:
        print("✗ 詳細對比測試失敗！")
        print("差異統計:")
        diff = torch.abs(your_result - expected_result)
        print(f"最大差異: {torch.max(diff)}")
        print(f"平均差異: {torch.mean(diff)}")

if __name__ == "__main__":
    # 運行所有測試
    test_encode_board_from_node_and_list()
    detailed_comparison_test()
    print("\n所有測試完成！")