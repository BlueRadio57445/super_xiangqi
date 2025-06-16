import pygame
import torch
import sys
from mcts_cnn import ChessNet, MCTS
from new_xiangqi import BoardState, initial, chinese_pieces, start_is_terminal
from collections import deque

# 畫面與棋盤設定
WINDOW_WIDTH, WINDOW_HEIGHT = 567, 630
CELL_SIZE = 57
OFFSET_X, OFFSET_Y = 28, 90
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 40
BUTTON_COLOR = (200, 200, 200)
BUTTON_HOVER_COLOR = (170, 170, 170)
BUTTON_TEXT_COLOR = (0, 0, 0)
BUTTON_FONT_SIZE = 24
FPS = 30
N_SIMULATIONS = 50

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
RED = (220, 20, 60)

def load_model(model_path, device):
    net = ChessNet()
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    return net

def draw_board(screen, board_state, selected=None):
    screen.fill((245, 222, 179))  # 仿木頭背景色
    font = pygame.font.SysFont("mingliu", 36)
    turn_font = pygame.font.SysFont("mingliu", 28)
    river_font = pygame.font.SysFont("mingliu", 28)

    # # --- 畫九宮淡色底 ---
    # palace_rects = [
    #     pygame.Rect(OFFSET_X + 3 * CELL_SIZE, OFFSET_Y + 0 * CELL_SIZE, 3 * CELL_SIZE, 3 * CELL_SIZE),
    #     pygame.Rect(OFFSET_X + 3 * CELL_SIZE, OFFSET_Y + 7 * CELL_SIZE, 3 * CELL_SIZE, 3 * CELL_SIZE),
    # ]
    # for rect in palace_rects:
    #     pygame.draw.rect(screen, (255, 250, 205), rect)  # 淺棕底色

    # --- 畫格線 ---
    for row in range(10):
        y = OFFSET_Y + row * CELL_SIZE
        pygame.draw.line(screen, BLACK, (OFFSET_X, y), (OFFSET_X + 8 * CELL_SIZE, y), 1)

    for col in range(9):
        x = OFFSET_X + col * CELL_SIZE
        if col == 0 or col == 8:
            pygame.draw.line(screen, BLACK, (x, OFFSET_Y), (x, OFFSET_Y + 9 * CELL_SIZE), 1)
        else:
            pygame.draw.line(screen, BLACK, (x, OFFSET_Y), (x, OFFSET_Y + 4 * CELL_SIZE), 1)
            pygame.draw.line(screen, BLACK, (x, OFFSET_Y + 5 * CELL_SIZE), (x, OFFSET_Y + 9 * CELL_SIZE), 1)

    # --- 畫九宮交叉線 ---
    pygame.draw.line(screen, BLACK,
                     (OFFSET_X + 3 * CELL_SIZE, OFFSET_Y + 0 * CELL_SIZE),
                     (OFFSET_X + 5 * CELL_SIZE, OFFSET_Y + 2 * CELL_SIZE), 1)
    pygame.draw.line(screen, BLACK,
                     (OFFSET_X + 5 * CELL_SIZE, OFFSET_Y + 0 * CELL_SIZE),
                     (OFFSET_X + 3 * CELL_SIZE, OFFSET_Y + 2 * CELL_SIZE), 1)
    pygame.draw.line(screen, BLACK,
                     (OFFSET_X + 3 * CELL_SIZE, OFFSET_Y + 7 * CELL_SIZE),
                     (OFFSET_X + 5 * CELL_SIZE, OFFSET_Y + 9 * CELL_SIZE), 1)
    pygame.draw.line(screen, BLACK,
                     (OFFSET_X + 5 * CELL_SIZE, OFFSET_Y + 7 * CELL_SIZE),
                     (OFFSET_X + 3 * CELL_SIZE, OFFSET_Y + 9 * CELL_SIZE), 1)

    # --- 楚河漢界 ---
    chu = river_font.render("楚河", True, RED)
    han = river_font.render("漢界", True, BLACK)
    chu_rect = chu.get_rect(center=(OFFSET_X + 2 * CELL_SIZE, OFFSET_Y + 4.5 * CELL_SIZE))
    han_rect = han.get_rect(center=(OFFSET_X + 6 * CELL_SIZE, OFFSET_Y + 4.5 * CELL_SIZE))
    screen.blit(chu, chu_rect)
    screen.blit(han, han_rect)

    # --- 兵卒點點 ---
    def draw_corner_dots(screen, x, y, size=3, offset=6):
        for dx in [-offset, offset]:
            for dy in [-offset, offset]:
                pygame.draw.circle(screen, BLACK, (x + dx, y + dy), size)

    dot_positions = [
        (0, 3), (0, 6), (2, 1), (2, 7), (3, 0),
        (6, 0), (7, 1), (7, 7), (9, 3), (9, 6)
    ]
    for row, col in dot_positions:
        x = OFFSET_X + col * CELL_SIZE
        y = OFFSET_Y + row * CELL_SIZE
        draw_corner_dots(screen, x, y)

    # --- 顯示合法走法 ---
    legal_destinations = set()
    if selected:
        from_idx = (selected[0] + 2) * 11 + (selected[1] + 1)
        moves = board_state.gen_moves(piece_pos=from_idx)
        legal_destinations = {
            ((to_idx // 11) - 2, (to_idx % 11) - 1)
            for _, _, to_idx in moves
        }

    # --- 畫棋子 ---
    for row in range(10):
        for col in range(9):
            x = OFFSET_X + col * CELL_SIZE
            y = OFFSET_Y + row * CELL_SIZE
            idx = (row + 2) * 11 + (col + 1)
            piece = board_state.board[idx]

            # 合法落點圈圈
            if (row, col) in legal_destinations:
                pygame.draw.circle(screen, (0, 0, 255), (x, y), 6)

            if piece != '.':
                # 棋子底座
                pygame.draw.circle(screen, (230, 230, 230), (x, y), CELL_SIZE // 2 - 4)
                pygame.draw.circle(screen, BLACK, (x, y), CELL_SIZE // 2 - 4, 1)

                color = RED if piece.isupper() else BLACK
                label = chinese_pieces.get(piece, piece)
                text = font.render(label, True, color)
                text_rect = text.get_rect(center=(x, y))
                screen.blit(text, text_rect)

    # --- 畫選取框 ---
    if selected:
        sx, sy = selected
        x = OFFSET_X + sy * CELL_SIZE
        y = OFFSET_Y + sx * CELL_SIZE
        pygame.draw.rect(screen, RED, (x - CELL_SIZE // 2, y - CELL_SIZE // 2, CELL_SIZE, CELL_SIZE), 3)

    # --- 顯示行棋方 ---
    turn_text = "紅方行棋" if board_state.move_count % 2 == 0 else "黑方行棋"
    turn_color = RED if board_state.move_count % 2 == 0 else BLACK
    turn_surface = turn_font.render(turn_text, True, turn_color)
    screen.blit(turn_surface, (OFFSET_X, 10))


def draw_restart_button(screen):
    font = pygame.font.SysFont(None, BUTTON_FONT_SIZE)
    mouse_pos = pygame.mouse.get_pos()
    x = WINDOW_WIDTH - BUTTON_WIDTH - 10
    y = 10
    rect = pygame.Rect(x, y, BUTTON_WIDTH, BUTTON_HEIGHT)

    color = BUTTON_HOVER_COLOR if rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(screen, color, rect)
    pygame.draw.rect(screen, BLACK, rect, 2)

    text = font.render("New Game", True, BUTTON_TEXT_COLOR)
    text_rect = text.get_rect(center=rect.center)
    screen.blit(text, text_rect)

    return rect

def get_pos_from_mouse(pos, tolerance=CELL_SIZE // 3):
    mx, my = pos
    for row in range(10):
        for col in range(9):
            x = OFFSET_X + col * CELL_SIZE
            y = OFFSET_Y + row * CELL_SIZE
            if abs(mx - x) <= tolerance and abs(my - y) <= tolerance:
                return row, col
    return None

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("AlphaZero 中國象棋")
    clock = pygame.time.Clock()

    board_state = BoardState(board=initial)
    selected = None
    running = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_model("best_model_1.pth", device)
    mcts = MCTS(net)
    last8 = deque(maxlen=8)

    while running:
        draw_board(screen, board_state, selected)
        restart_button_rect = draw_restart_button(screen)  # 🔁 畫按鈕
        pygame.display.flip()
        clock.tick(FPS)

        if board_state.move_count % 2 == 0:  # 玩家紅方
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if restart_button_rect.collidepoint(pygame.mouse.get_pos()):
                        print("🔁 重開新遊戲（按鈕）")
                        board_state = BoardState(board=initial)
                        selected = None
                        last8.clear()
                        continue  # 跳過下棋邏輯

                    pos = get_pos_from_mouse(pygame.mouse.get_pos())
                    if pos:
                        row, col = pos
                        idx = (row + 2) * 11 + (col + 1)
                        piece = board_state.board[idx]
                        if selected:
                            from_idx = (selected[0] + 2) * 11 + (selected[1] + 1)
                            if piece.isupper():  # 換選另一顆己方棋子
                                selected = (row, col)
                            else:
                                move = (board_state.board[from_idx], from_idx, idx)
                                if move in board_state.gen_moves():
                                    board_state = board_state.move(move)
                                    last8.append(board_state)
                                    selected = None
                        else:
                            if piece.isupper():
                                selected = (row, col)

        else:
            print("AI 思考中...")
            pi, _ = mcts.search(board_state, N_SIMULATIONS, history_list=list(last8))
            action = max(pi.items(), key=lambda x: x[1])[0]
            board_state = board_state.move(action)
            last8.append(board_state)

        terminal_result = board_state.is_terminal()
        if terminal_result is not None:
            draw_board(screen, board_state)
            pygame.display.flip()
            pygame.time.delay(2000)

            start_result = start_is_terminal(
                terminal_result,
                start_move_count=board_state.move_count - 1,
                end_move_count=board_state.move_count
            )

            print("遊戲結束")
            if start_result == 1:
                print("你贏了！")
            elif start_result == -1:
                print("你輸了！")
            else:
                print("和局。")
            
            running = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
