import os
from mcts_cnn import load_dataset_pickle, save_dataset_pickle

# 設定你的資料夾路徑
folder_path1 = 'owo/特別包1'


# 列出所有檔案的完整路徑（假設都是 .pkl 結尾的檔案）
file_list1 = [os.path.join(folder_path1, fname) 
             for fname in os.listdir(folder_path1) 
             if fname.endswith('.pkl')]



# 讀取所有檔案
big_data = []
print("----")
for f in file_list1:
    data = load_dataset_pickle(f)
    print(len(data))
    big_data.extend(data)



save_dataset_pickle(big_data, 66)