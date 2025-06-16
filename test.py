from mcts_cnn import load_dataset_pickle, save_dataset_pickle, evaluate, train, ChessNet
import torch
import copy
from mirror_reflect import mirror_training_data

data1 = load_dataset_pickle("babai_1_1.pkl")
data2 = load_dataset_pickle("babai_1_2.pkl")
data_sp = load_dataset_pickle("babai_1_sp.pkl")
data = []
data.extend(mirror_training_data(data1))
data.extend(mirror_training_data(data2))
data.extend(mirror_training_data(data_sp))
#data.extend(data1)
#data.extend(data2)
#data.extend(data_sp)

print(len(data))


net = ChessNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
best_net = copy.deepcopy(net)
best_net.to(device)

net.load_state_dict(torch.load("best_model_2.pth"))
best_net.load_state_dict(torch.load("best_model_2.pth"))

train(net, data, 40)

win_rate, wins, draws, losses, big_data = evaluate(net,best_net,10)
print(win_rate)
print(wins)
print(draws)
print(losses)

save_dataset_pickle(big_data, 87)
torch.save(net.state_dict(), f"best_model_3--.pth")