import torch
import numpy as np

def mirror_board_tensor(board_tensor):
    """
    將棋盤張量進行左右鏡像
    Args:
        board_tensor: shape (batch_size, 128, 10, 9) 或 (128, 10, 9)
    Returns:
        鏡像後的棋盤張量
    """
    # 沿著最後一個維度（列）進行翻轉
    return torch.flip(board_tensor, dims=[-1])

def mirror_policy_tensor(policy_tensor):
    """
    將策略張量進行左右鏡像
    Args:
        policy_tensor: shape (10, 9, 52)
    Returns:
        鏡像後的策略張量
    """
    # 翻轉空間維度
    mirrored_policy = torch.flip(policy_tensor, dims=[1])  # 翻轉列維度
    
    # 需要調整動作編碼中的方向
    # 創建一個新的張量來存儲調整後的動作
    adjusted_policy = mirrored_policy.clone()
    
    # 調整東西方向的滑動動作 (channels 18-35)
    # East (18-26) <-> West (27-35)
    east_channels = mirrored_policy[:, :, 18:27]  # 原本的東向
    west_channels = mirrored_policy[:, :, 27:36]  # 原本的西向
    
    adjusted_policy[:, :, 18:27] = west_channels  # 東向變西向
    adjusted_policy[:, :, 27:36] = east_channels  # 西向變東向
    
    # 調整馬的動作 (channels 36-43)
    # 馬的8個方向需要左右對稱調整
    horse_moves = mirrored_policy[:, :, 36:44]
    adjusted_horse = horse_moves.clone()
    
    # 馬的動作對應關係 (原始 -> 鏡像後)
    # (-2,-1):36 -> (-2,+1):37
    # (-2,+1):37 -> (-2,-1):36  
    # (-1,-2):38 -> (-1,+2):39
    # (-1,+2):39 -> (-1,-2):38
    # (+1,-2):40 -> (+1,+2):41
    # (+1,+2):41 -> (+1,-2):40
    # (+2,-1):42 -> (+2,+1):43
    # (+2,+1):43 -> (+2,-1):42
    
    adjusted_horse[:, :, 0] = horse_moves[:, :, 1]  # 36<->37
    adjusted_horse[:, :, 1] = horse_moves[:, :, 0]
    adjusted_horse[:, :, 2] = horse_moves[:, :, 3]  # 38<->39
    adjusted_horse[:, :, 3] = horse_moves[:, :, 2]
    adjusted_horse[:, :, 4] = horse_moves[:, :, 5]  # 40<->41
    adjusted_horse[:, :, 5] = horse_moves[:, :, 4]
    adjusted_horse[:, :, 6] = horse_moves[:, :, 7]  # 42<->43
    adjusted_horse[:, :, 7] = horse_moves[:, :, 6]
    
    adjusted_policy[:, :, 36:44] = adjusted_horse
    
    # 調整象的動作 (channels 44-47)
    # (-2,-2):44 -> (-2,+2):45
    # (-2,+2):45 -> (-2,-2):44
    # (+2,-2):46 -> (+2,+2):47  
    # (+2,+2):47 -> (+2,-2):46
    elephant_moves = mirrored_policy[:, :, 44:48]
    adjusted_elephant = elephant_moves.clone()
    
    adjusted_elephant[:, :, 0] = elephant_moves[:, :, 1]  # 44<->45
    adjusted_elephant[:, :, 1] = elephant_moves[:, :, 0]
    adjusted_elephant[:, :, 2] = elephant_moves[:, :, 3]  # 46<->47
    adjusted_elephant[:, :, 3] = elephant_moves[:, :, 2]
    
    adjusted_policy[:, :, 44:48] = adjusted_elephant
    
    # 調整士的動作 (channels 48-51)
    # (-1,-1):48 -> (-1,+1):49
    # (-1,+1):49 -> (-1,-1):48
    # (+1,-1):50 -> (+1,+1):51
    # (+1,+1):51 -> (+1,-1):50
    advisor_moves = mirrored_policy[:, :, 48:52]
    adjusted_advisor = advisor_moves.clone()
    
    adjusted_advisor[:, :, 0] = advisor_moves[:, :, 1]  # 48<->49
    adjusted_advisor[:, :, 1] = advisor_moves[:, :, 0]
    adjusted_advisor[:, :, 2] = advisor_moves[:, :, 3]  # 50<->51
    adjusted_advisor[:, :, 3] = advisor_moves[:, :, 2]
    
    adjusted_policy[:, :, 48:52] = adjusted_advisor
    
    return adjusted_policy

def mirror_training_data(training_data):
    """
    將訓練資料進行鏡像化擴增
    Args:
        training_data: list of (state_tensor, policy_tensor, value)
                      其中 state_tensor shape: (1, 128, 10, 9)
                           policy_tensor shape: (10, 9, 52)
                           value: float
    Returns:
        原始資料 + 鏡像資料的組合列表
    """
    mirrored_data = []
    
    for state_tensor, policy_tensor, value in training_data:
        # 鏡像棋盤狀態
        mirrored_state = mirror_board_tensor(state_tensor)
        
        # 鏡像策略
        mirrored_policy = mirror_policy_tensor(policy_tensor)
        
        # 價值保持不變
        mirrored_value = value
        
        mirrored_data.append((mirrored_state, mirrored_policy, mirrored_value))
    
    # 返回原始資料 + 鏡像資料
    return training_data + mirrored_data

def augment_dataset_with_mirror(dataset):
    """
    便利函數：為整個資料集加入鏡像化資料
    Args:
        dataset: 原始訓練資料集
    Returns:
        擴增後的資料集（大小翻倍）
    """
    print(f"原始資料集大小: {len(dataset)}")
    augmented_dataset = mirror_training_data(dataset)
    print(f"擴增後資料集大小: {len(augmented_dataset)}")
    return augmented_dataset

# 使用範例
if __name__ == "__main__":
    # 假設你有一個訓練資料集
    # dataset = [(state1, policy1, value1), (state2, policy2, value2), ...]
    
    # 在 self_play_game 函數後使用
    # augmented_dataset = augment_dataset_with_mirror(dataset)
    
    # 然後用擴增後的資料集進行訓練
    # train(net, augmented_dataset)
    
    pass