import random
import pygame

from config import *

class Paddle:

    def __init__(self, x, color, spd=PADDLE_SPD):

        self.rect = pygame.Rect(
            x,
            GAME_H // 2 - PADDLE_H // 2,
            PADDLE_W,
            PADDLE_H
        )

        self.color = color
        self.spd = spd

        self._action = 1
        self._steps_left = 0

    def random_step(self):

        if self._steps_left <= 0:

            self._action = random.randint(0, 1)
            self._steps_left = random.randint(4, 14)

        self._steps_left -= 1

        if self._action == 1:

            if self.rect.top > 0:
                self.rect.y -= self.spd

        else:

            if self.rect.bottom < GAME_H:
                self.rect.y += self.spd

    def draw(self, surf, ox):

        pygame.draw.rect(
            surf,
            self.color,
            self.rect.move(ox, 0),
            border_radius=3
        )


class Ball:

    def __init__(self, spd=5):

        self.spd = spd
        self.reset()

    def reset(self):

        self.rect = pygame.Rect(
            GAME_W // 2,
            GAME_H // 2,
            BALL_SZ,
            BALL_SZ
        )

        # Randomise vertical angle so agents can't overfit to a fixed trajectory
        angle = random.uniform(0.3, 0.8)          # non-trivial vertical component
        vx    = self.spd * random.choice([-1, 1])
        vy    = self.spd * angle * random.choice([-1, 1])

        self.speed_x = vx
        self.speed_y = vy

    def move(self):

        self.rect.x += int(self.speed_x)
        self.rect.y += int(self.speed_y)

        if self.rect.top <= 0 or self.rect.bottom >= GAME_H:
            self.speed_y *= -1

    def draw(self, surf, ox):

        pygame.draw.ellipse(
            surf,
            WHITE,
            self.rect.move(ox, 0)
        )