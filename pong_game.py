import sys
import random

import pygame
import numpy as np
import cv2
import torch

from config import *
from rl_agent import RLAgent
from loading_screen import run_loading_screen


pygame.init()

cv2.setNumThreads(0)


# =========================================================
# Game Objects
# =========================================================

class Paddle:

    def __init__(self, x, color):

        self.rect = pygame.Rect(
            x,
            HEIGHT // 2 - PADDLE_HEIGHT // 2,
            PADDLE_WIDTH,
            PADDLE_HEIGHT
        )

        self.color = color

    def move(self, up):

        if up:

            if self.rect.top > 0:
                self.rect.y -= PADDLE_SPEED

        else:

            if self.rect.bottom < HEIGHT:
                self.rect.y += PADDLE_SPEED

    def draw(self, screen):

        pygame.draw.rect(
            screen,
            self.color,
            self.rect
        )


class Ball:

    BASE_SPEED = 5

    def __init__(self):
        self.reset()

    def reset(self):

        self.rect = pygame.Rect(
            WIDTH // 2,
            HEIGHT // 2,
            BALL_SIZE,
            BALL_SIZE
        )

        self.speed_x = (
            self.BASE_SPEED
            * random.choice([-1, 1])
        )

        self.speed_y = (
            self.BASE_SPEED
            * random.choice([-1, 1])
        )

    def move(self):

        self.rect.x += int(self.speed_x)
        self.rect.y += int(self.speed_y)

        if (
            self.rect.top <= 0
            or
            self.rect.bottom >= HEIGHT
        ):
            self.speed_y *= -1

    def draw(self, screen):

        pygame.draw.ellipse(
            screen,
            WHITE,
            self.rect
        )


# =========================================================
# Frame Capture
# =========================================================

def get_frame_gray(screen):

    pygame.event.pump()

    raw = pygame.surfarray.array3d(screen)

    frame = np.transpose(raw, (1, 0, 2))

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2GRAY
    )

    return gray


# =========================================================
# HUD
# =========================================================

def draw_hud(
    screen,
    score_l,
    score_r,
    game_num,
    total_games
):

    font_big = pygame.font.SysFont(
        "Arial",
        28,
        bold=True
    )

    font_small = pygame.font.SysFont(
        "Arial",
        18
    )

    score_text = font_big.render(
        f"{score_l} : {score_r}",
        True,
        WHITE
    )

    screen.blit(
        score_text,
        (
            WIDTH // 2 - score_text.get_width() // 2,
            8
        )
    )

    game_text = font_small.render(
        f"Game {game_num}/{total_games}",
        True,
        GRAY
    )

    screen.blit(
        game_text,
        (
            WIDTH // 2 - game_text.get_width() // 2,
            42
        )
    )

    pygame.draw.line(
        screen,
        (60, 60, 60),
        (WIDTH // 2, 0),
        (WIDTH // 2, HEIGHT)
    )


# =========================================================
# Summary Screen
# =========================================================

def show_summary(
    screen,
    agent_l,
    agent_r,
    total_games
):

    font_h = pygame.font.SysFont(
        "Arial",
        36,
        bold=True
    )

    font_b = pygame.font.SysFont(
        "Arial",
        22
    )

    wins_l = sum(
        1 for r in agent_l.episode_rewards
        if r > 0
    )

    wins_r = sum(
        1 for r in agent_r.episode_rewards
        if r > 0
    )

    while True:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                return

            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_q
            ):
                return

        screen.fill(DARK)

        title = font_h.render(
            "Simulation Complete",
            True,
            WHITE
        )

        screen.blit(
            title,
            (
                WIDTH // 2 - title.get_width() // 2,
                40
            )
        )

        lines = [

            f"Games Played: {total_games}",

            f"P1 Wins: {wins_l}",

            f"P2 Wins: {wins_r}",

            f"P1 Avg Reward: "
            f"{np.mean(agent_l.episode_rewards):.2f}",

            f"P2 Avg Reward: "
            f"{np.mean(agent_r.episode_rewards):.2f}",

            "",

            "Press Q to Quit"
        ]

        for i, line in enumerate(lines):

            color = WHITE

            if "P1" in line:
                color = RED

            if "P2" in line:
                color = BLUE

            surf = font_b.render(
                line,
                True,
                color
            )

            screen.blit(
                surf,
                (
                    WIDTH // 2 - surf.get_width() // 2,
                    130 + i * 40
                )
            )

        pygame.display.flip()


# =========================================================
# Main Simulation
# =========================================================

def run_simulation(params):

    num_games = int(params["num_games"])
    fps = int(params["fps"])

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT)
    )

    pygame.display.set_caption(
        "RL Pong"
    )

    clock = pygame.time.Clock()

    agent_l = RLAgent(
        lr=params["lr1"],
        gamma=params["gamma1"],
        hidden_dim=int(params["hid1"]),
        name="P1"
    )

    agent_r = RLAgent(
        lr=params["lr2"],
        gamma=params["gamma2"],
        hidden_dim=int(params["hid2"]),
        name="P2"
    )

    MAX_POINTS = 3

    for game_num in range(1, num_games + 1):

        left_paddle = Paddle(20, RED)

        right_paddle = Paddle(
            WIDTH - 30,
            BLUE
        )

        ball = Ball()

        score_l = 0
        score_r = 0

        prev_gray = None

        while (
            score_l < MAX_POINTS
            and
            score_r < MAX_POINTS
        ):

            clock.tick(fps)

            # ---------------------------------
            # events
            # ---------------------------------

            for event in pygame.event.get():

                if event.type == pygame.QUIT:

                    pygame.quit()
                    sys.exit()

            # ---------------------------------
            # render
            # ---------------------------------

            screen.fill(BLACK)

            left_paddle.draw(screen)
            right_paddle.draw(screen)
            ball.draw(screen)

            draw_hud(
                screen,
                score_l,
                score_r,
                game_num,
                num_games
            )

            pygame.display.flip()

            # ---------------------------------
            # capture state
            # ---------------------------------

            # gray = get_frame_gray(screen)

            # if prev_gray is None:
            #     prev_gray = gray.copy()

            # diff = gray.astype(np.float32) - prev_gray.astype(np.float32)

            # prev_gray = gray.copy()

            # # choose either gray or diff

            # state_l = agent_l.preprocess(gray)
            # state_r = agent_r.preprocess(gray)
            state_l = np.array([
                ball.rect.centerx / WIDTH,
                ball.rect.centery / HEIGHT,
                ball.speed_x / 10.0,
                ball.speed_y / 10.0,
                left_paddle.rect.centery / HEIGHT,
                right_paddle.rect.centery / HEIGHT
            ], dtype=np.float32)

            state_r = np.array([
                1.0 - (ball.rect.centerx / WIDTH),
                ball.rect.centery / HEIGHT,
                -ball.speed_x / 10.0,
                ball.speed_y / 10.0,
                right_paddle.rect.centery / HEIGHT,
                left_paddle.rect.centery / HEIGHT
            ], dtype=np.float32)

            state_l = torch.from_numpy(state_l)
            state_r = torch.from_numpy(state_r)

            action_l = agent_l.select_action(state_l)
            action_r = agent_r.select_action(state_r)

            left_paddle.move(action_l == 1)
            right_paddle.move(action_r == 1)

            # ---------------------------------
            # physics
            # ---------------------------------

            ball.move()

            if ball.rect.colliderect(left_paddle.rect):
                ball.speed_x = abs(ball.speed_x)

            if ball.rect.colliderect(right_paddle.rect):
                ball.speed_x = -abs(ball.speed_x)

            reward_l = 0.0
            reward_r = 0.0

            # ---------------------------------
            # scoring
            # ---------------------------------

            if ball.rect.left <= 0:

                score_r += 1

                reward_l = -1.0
                reward_r = +1.0

                ball.reset()

                prev_gray = None

            elif ball.rect.right >= WIDTH:

                score_l += 1

                reward_l = +1.0
                reward_r = -1.0

                ball.reset()

                prev_gray = None

            agent_l.record_reward(reward_l)
            agent_r.record_reward(reward_r)

        # ---------------------------------
        # end episode
        # ---------------------------------

        reward_sum_l = agent_l.finish_episode()
        reward_sum_r = agent_r.finish_episode()

        print(
            f"[Game {game_num}/{num_games}] "
            f"P1 {score_l} - {score_r} P2 "
            f"| rewards "
            f"{reward_sum_l:.1f} / {reward_sum_r:.1f}"
        )

    show_summary(
        screen,
        agent_l,
        agent_r,
        num_games
    )

    pygame.quit()


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":

    params = run_loading_screen()

    run_simulation(params)