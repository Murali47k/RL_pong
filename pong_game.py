# pong_game.py — Main game loop with RL agents

import pygame
import numpy as np
import sys
import cv2

from config import *
from rl_agent import RLAgent
from loading_screen import run_loading_screen

pygame.init()


#  Game objects

class Paddle:
    def __init__(self, x, color):
        self.rect  = pygame.Rect(x, HEIGHT // 2 - PADDLE_HEIGHT // 2,
                                 PADDLE_WIDTH, PADDLE_HEIGHT)
        self.color = color

    def move(self, up):
        if up and self.rect.top > 0:
            self.rect.y -= PADDLE_SPEED
        elif not up and self.rect.bottom < HEIGHT:
            self.rect.y += PADDLE_SPEED

    def draw(self, win):
        pygame.draw.rect(win, self.color, self.rect)


class Ball:
    BASE_SPEED = 5

    def __init__(self):
        self.reset()

    def reset(self):
        import random
        self.rect    = pygame.Rect(WIDTH // 2, HEIGHT // 2, BALL_SIZE, BALL_SIZE)
        self.speed_x = self.BASE_SPEED * random.choice([-1, 1])
        self.speed_y = self.BASE_SPEED * random.choice([-1, 1])

    def move(self):
        self.rect.x += int(self.speed_x)
        self.rect.y += int(self.speed_y)
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.speed_y *= -1

    def draw(self, win):
        pygame.draw.ellipse(win, WHITE, self.rect)


# Preprocessing 

def get_frame_gray(win):
    """Capture current pygame surface as grayscale numpy array."""
    raw   = pygame.surfarray.array3d(win)   # (W, H, 3)
    frame = np.transpose(raw, (1, 0, 2))    # (H, W, 3)
    gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return gray


# HUD drawing 

def draw_hud(win, score_l, score_r, game_num, total_games, ep_rwd_l, ep_rwd_r):
    font_big  = pygame.font.SysFont("Arial", 28, bold=True)
    font_small= pygame.font.SysFont("Arial", 18)

    # Centre score
    txt = font_big.render(f"{score_l}  :  {score_r}", True, WHITE)
    win.blit(txt, (WIDTH // 2 - txt.get_width() // 2, 8))

    # Game counter
    gc = font_small.render(f"Game {game_num}/{total_games}", True, GRAY)
    win.blit(gc, (WIDTH // 2 - gc.get_width() // 2, 42))

    # Per-agent reward
    r1 = font_small.render(f"P1 ep_rwd: {ep_rwd_l:.1f}", True, RED)
    r2 = font_small.render(f"P2 ep_rwd: {ep_rwd_r:.1f}", True, BLUE)
    win.blit(r1, (10, 10))
    win.blit(r2, (WIDTH - r2.get_width() - 10, 10))

    # Dividing line
    pygame.draw.aaline(win, (60, 60, 60), (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))


def draw_summary(win, agent_l, agent_r, total_games):
    win.fill(DARK)
    font_h = pygame.font.SysFont("Arial", 36, bold=True)
    font_b = pygame.font.SysFont("Arial", 22)

    title = font_h.render("Simulation Complete!", True, WHITE)
    win.blit(title, (WIDTH // 2 - title.get_width() // 2, 40))

    wins_l = sum(1 for r in agent_l.episode_rewards if r > 0)
    wins_r = sum(1 for r in agent_r.episode_rewards if r > 0)

    lines = [
        f"Total games: {total_games}",
        f"Player 1 (Red)  wins: {wins_l}   avg reward: {np.mean(agent_l.episode_rewards):.2f}",
        f"Player 2 (Blue) wins: {wins_r}   avg reward: {np.mean(agent_r.episode_rewards):.2f}",
        "",
        "Press Q or close window to quit.",
    ]
    for i, line in enumerate(lines):
        color = RED if "Player 1" in line else (BLUE if "Player 2" in line else GRAY)
        s = font_b.render(line, True, color)
        win.blit(s, (WIDTH // 2 - s.get_width() // 2, 130 + i * 40))

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return


# ── Main simulation ──────────────────────────────────────────────────────────

def run_simulation(params):
    num_games  = int(params["num_games"])
    fps        = int(params["fps"])

    agent_l = RLAgent(lr=params["lr1"], gamma=params["gamma1"],
                      hidden_dim=int(params["hid1"]), name="P1-Red")
    agent_r = RLAgent(lr=params["lr2"], gamma=params["gamma2"],
                      hidden_dim=int(params["hid2"]), name="P2-Blue")

    win   = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL Pong — Running")
    clock = pygame.time.Clock()

    total_score_l = 0
    total_score_r = 0

    for game_num in range(1, num_games + 1):
        left_paddle  = Paddle(20, RED)
        right_paddle = Paddle(WIDTH - 30, BLUE)
        ball         = Ball()

        score_l = score_r = 0
        MAX_POINTS_PER_GAME = 3  # rally to 3, then next game
        ep_rwd_l = ep_rwd_r = 0.0

        # We store last frame for frame-differencing (optional improvement)
        prev_gray = None

        while score_l < MAX_POINTS_PER_GAME and score_r < MAX_POINTS_PER_GAME:
            clock.tick(fps)

            # ── events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    agent_l.finish_episode()
                    agent_r.finish_episode()
                    draw_summary(win, agent_l, agent_r, game_num - 1)
                    pygame.quit(); sys.exit()

            # ── render frame (needed for state)
            win.fill(BLACK)
            left_paddle.draw(win)
            right_paddle.draw(win)
            ball.draw(win)
            draw_hud(win, score_l, score_r, game_num, num_games, ep_rwd_l, ep_rwd_r)
            pygame.display.flip()

            # ── capture state
            gray = get_frame_gray(win)
            if prev_gray is None:
                prev_gray = gray.copy()

            # Frame difference gives motion signal
            diff = np.clip(gray.astype(np.int16) - prev_gray.astype(np.int16),
                           -255, 255).astype(np.float32)
            prev_gray = gray.copy()

            state_l = agent_l.preprocess(gray)     # you can swap to diff
            state_r = agent_r.preprocess(gray)

            action_l = agent_l.select_action(state_l)   # 1=UP, 0=DOWN
            action_r = agent_r.select_action(state_r)

            left_paddle.move(up=(action_l == 1))
            right_paddle.move(up=(action_r == 1))

            # ── ball physics
            ball.move()

            # Paddle collision
            if ball.rect.colliderect(left_paddle.rect):
                ball.speed_x = abs(ball.speed_x)   # bounce right
            if ball.rect.colliderect(right_paddle.rect):
                ball.speed_x = -abs(ball.speed_x)  # bounce left

            # ── scoring
            reward_l = reward_r = 0.0

            if ball.rect.left <= 0:          # right scores
                score_r += 1
                reward_l = -1.0
                reward_r = +1.0
                ball.reset()
                prev_gray = None

            elif ball.rect.right >= WIDTH:   # left scores
                score_l += 1
                reward_l = +1.0
                reward_r = -1.0
                ball.reset()
                prev_gray = None

            agent_l.record_reward(reward_l)
            agent_r.record_reward(reward_r)
            ep_rwd_l += reward_l
            ep_rwd_r += reward_r

        # ── end of game → policy update
        agent_l.finish_episode()
        agent_r.finish_episode()
        total_score_l += score_l
        total_score_r += score_r

        print(f"[Game {game_num:>3}/{num_games}]  "
              f"P1 {score_l} — {score_r} P2  "
              f"| cumulative {total_score_l} — {total_score_r}")

    draw_summary(win, agent_l, agent_r, num_games)
    pygame.quit()


# Entry point 

if __name__ == "__main__":
    params = run_loading_screen()   # blocks until user clicks Start
    run_simulation(params)