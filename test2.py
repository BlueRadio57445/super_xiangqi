from mcts_cnn import load_dataset_pickle, save_dataset_pickle, self_play_game, evaluate, train, ChessNet
import torch
import copy
import tqdm

SELF_PLAY_GAMES = 100

net = ChessNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
model_path = "best_model_2.pth"
net.load_state_dict(torch.load(model_path))

wins, draws, losses = 0, 0, 0
with tqdm.tqdm(range(SELF_PLAY_GAMES), desc="Self-play BIG", leave=True) as pbar:
    for game_idx in range(SELF_PLAY_GAMES):
        game_data, (w, d, l) = self_play_game(net)
        save_dataset_pickle(game_data, game_idx)
        wins += w
        draws += d
        losses += l
        pbar.set_postfix({"Wins": wins, "Draws": draws, "Losses": losses})
        pbar.update(1)