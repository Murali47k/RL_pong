"""
main.py — Pong simulation, both players move randomly.
Whoever scores first wins the round.
"""

import sys
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "x11")

from config import *
from entities import Paddle, Ball
from ui import Slider


class App:

    def __init__(self):

        pygame.init()

        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Pong — Random Agents")

        self.clock = pygame.time.Clock()

        # fonts
        self.f_title = pygame.font.SysFont("Arial", 18, bold=True)
        self.f_sm = pygame.font.SysFont("Arial", 13)
        self.f_val = pygame.font.SysFont("Arial", 13, bold=True)
        self.f_hud = pygame.font.SysFont("Arial", 26, bold=True)
        self.f_sub = pygame.font.SysFont("Arial", 16)
        self.f_big = pygame.font.SysFont("Arial", 32, bold=True)
        self.f_med = pygame.font.SysFont("Arial", 20)

        SX = 14
        SW = SIDEBAR_W - SX * 2 - 34

        self.sliders = {
            "num_games": Slider("# Rounds", 10, 500, 20, int, 10),
            "fps": Slider("FPS", 10, 120, 60, int, 10),
            "ball_spd": Slider("Ball Speed", 2, 15, 5, int, 1),
            "paddle_spd": Slider("Paddle Speed", 2, 15, 6, int, 1),
        }

        y = 56

        for sl in self.sliders.values():
            sl.layout(SX, y, SW)
            y += Slider.H + 6

        self.btn = pygame.Rect(
            14,
            WIN_H - 58,
            SIDEBAR_W - 28,
            42
        )

        self.state = SETUP

        self.game_num = 0
        self.num_games = 20

        self.fps_val = 60

        self.total_wins = [0, 0]

        self.paddle_l = None
        self.paddle_r = None
        self.ball = None

        # stats
        self.rallies = []
        self.durations = []

        self._rally_count = 0
        self._game_start = 0.0

    # ─────────────────────────────────────────────
    # Sidebar
    # ─────────────────────────────────────────────
    def _draw_sidebar(self):

        sb = pygame.Surface((SIDEBAR_W, WIN_H))
        sb.fill(PANEL)

        title = self.f_title.render(
            "PONG — RANDOM",
            True,
            WHITE
        )

        sb.blit(
            title,
            (SIDEBAR_W // 2 - title.get_width() // 2, 10)
        )

        pygame.draw.line(
            sb,
            (60, 60, 70),
            (8, 32),
            (SIDEBAR_W - 8, 32)
        )

        sb.blit(
            self.f_sm.render("── SETTINGS ──", True, GRAY),
            (14, 48)
        )

        for sl in self.sliders.values():
            sl.draw(sb, self.f_sm, self.f_val)

        # stats
        if self.state in (RUNNING, SUMMARY):

            y0 = self.btn.top - 110

            pygame.draw.line(
                sb,
                (60, 60, 70),
                (8, y0 - 8),
                (SIDEBAR_W - 8, y0 - 8)
            )

            rows = [
                (f"Round {self.game_num}/{self.num_games}", WHITE),
                (f"P1 rounds won : {self.total_wins[0]}", RED),
                (f"P2 rounds won : {self.total_wins[1]}", BLUE),
            ]

            if self.rallies:

                avg_r = sum(self.rallies) / len(self.rallies)

                rows.append(
                    (f"Avg rallies : {avg_r:.1f}", GRAY)
                )

            if self.durations:

                avg_d = sum(self.durations) / len(self.durations)

                rows.append(
                    (f"Avg duration : {avg_d:.1f}s", GRAY)
                )

            for i, (txt, col) in enumerate(rows):

                sb.blit(
                    self.f_sm.render(txt, True, col),
                    (14, y0 + i * 20)
                )

        # button
        if self.state == SETUP:

            hov = self.btn.collidepoint(
                pygame.mouse.get_pos()
            )

            pygame.draw.rect(
                sb,
                (70, 210, 110) if hov else (45, 150, 75),
                self.btn,
                border_radius=8
            )

            bt = self.f_title.render(
                "▶ Start Simulation",
                True,
                WHITE
            )

        elif self.state == RUNNING:

            pygame.draw.rect(
                sb,
                (60, 60, 70),
                self.btn,
                border_radius=8
            )

            bt = self.f_sm.render(
                "Simulating…",
                True,
                GRAY
            )

        else:

            pygame.draw.rect(
                sb,
                (50, 80, 160),
                self.btn,
                border_radius=8
            )

            bt = self.f_title.render(
                "▶ Run Again",
                True,
                WHITE
            )

        sb.blit(
            bt,
            (
                self.btn.centerx - bt.get_width() // 2,
                self.btn.centery - bt.get_height() // 2
            )
        )

        self.screen.blit(sb, (0, 0))

    # ─────────────────────────────────────────────
    # Game Screen
    # ─────────────────────────────────────────────
    def _draw_game(self):

        ox = SIDEBAR_W

        pygame.draw.rect(
            self.screen,
            BLACK,
            (ox, 0, GAME_W, GAME_H)
        )

        # setup screen
        if self.state == SETUP:

            msg = self.f_big.render(
                "Configure & press Start →",
                True,
                (70, 70, 82)
            )

            self.screen.blit(
                msg,
                (
                    ox + GAME_W // 2 - msg.get_width() // 2,
                    GAME_H // 2 - msg.get_height() // 2
                )
            )

            return

        # summary
        if self.state == SUMMARY:
            self._draw_summary(ox)
            return

        # center divider
        for y in range(0, GAME_H, 20):

            pygame.draw.rect(
                self.screen,
                (45, 45, 45),
                (ox + GAME_W // 2 - 1, y, 2, 12)
            )

        self.paddle_l.draw(self.screen, ox)
        self.paddle_r.draw(self.screen, ox)

        self.ball.draw(self.screen, ox)

        # score = total wins
        sc = self.f_hud.render(
            f"{self.total_wins[0]} : {self.total_wins[1]}",
            True,
            WHITE
        )

        self.screen.blit(
            sc,
            (
                ox + GAME_W // 2 - sc.get_width() // 2,
                10
            )
        )

        gc = self.f_sub.render(
            f"Round {self.game_num} / {self.num_games}",
            True,
            GRAY
        )

        self.screen.blit(
            gc,
            (
                ox + GAME_W // 2 - gc.get_width() // 2,
                44
            )
        )

        self.screen.blit(
            self.f_sub.render("P1", True, RED),
            (ox + 24, 10)
        )

        self.screen.blit(
            self.f_sub.render("P2", True, BLUE),
            (ox + GAME_W - 40, 10)
        )

        rc = self.f_sm.render(
            f"rally hits: {self._rally_count}",
            True,
            (100, 100, 110)
        )

        self.screen.blit(
            rc,
            (
                ox + GAME_W // 2 - rc.get_width() // 2,
                GAME_H - 22
            )
        )

    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────
    def _draw_summary(self, ox):

        wl, wr = self.total_wins

        n = max(self.num_games, 1)

        avg_r = (
            sum(self.rallies) / len(self.rallies)
            if self.rallies else 0.0
        )

        avg_d = (
            sum(self.durations) / len(self.durations)
            if self.durations else 0.0
        )

        rows = [
            ("Simulation Complete", WHITE, self.f_big),
            ("", WHITE, self.f_med),
            (f"Rounds played : {self.num_games}", WHITE, self.f_med),
            (f"P1 rounds won : {wl} ({100 * wl // n}%)", RED, self.f_med),
            (f"P2 rounds won : {wr} ({100 * wr // n}%)", BLUE, self.f_med),
            (f"Avg rally hits : {avg_r:.1f}", GRAY, self.f_med),
            (f"Avg round length : {avg_d:.1f}s", GRAY, self.f_med),
            ("", WHITE, self.f_med),
            ('Press "Run Again" to go back', GRAY, self.f_sub),
        ]

        total_h = sum(f.get_height() + 8 for _, _, f in rows)

        y = GAME_H // 2 - total_h // 2

        for txt, col, font in rows:

            s = font.render(txt, True, col)

            self.screen.blit(
                s,
                (
                    ox + GAME_W // 2 - s.get_width() // 2,
                    y
                )
            )

            y += font.get_height() + 8

    # ─────────────────────────────────────────────
    # Start simulation
    # ─────────────────────────────────────────────
    def _start(self):

        p = {k: sl.val for k, sl in self.sliders.items()}

        self.num_games = int(p["num_games"])
        self.fps_val = int(p["fps"])

        self.total_wins = [0, 0]

        self.game_num = 0

        self.rallies = []
        self.durations = []

        self._ball_spd = int(p["ball_spd"])
        self._paddle_spd = int(p["paddle_spd"])

        self.state = RUNNING

        self._new_game()

    # ─────────────────────────────────────────────
    # New round
    # ─────────────────────────────────────────────
    def _new_game(self):

        self.game_num += 1

        self.paddle_l = Paddle(
            18,
            RED,
            self._paddle_spd
        )

        self.paddle_r = Paddle(
            GAME_W - 28,
            BLUE,
            self._paddle_spd
        )

        self.ball = Ball(self._ball_spd)

        self._rally_count = 0

        self._game_start = (
            pygame.time.get_ticks() / 1000.0
        )

    # ─────────────────────────────────────────────
    # Simulation
    # ─────────────────────────────────────────────
    def _sim_step(self):

        b = self.ball
        pl = self.paddle_l
        pr = self.paddle_r

        pl.random_step()
        pr.random_step()

        b.move()

        # collisions
        if b.rect.colliderect(pl.rect):

            b.rect.left = pl.rect.right

            b.speed_x = abs(b.speed_x)

            self._rally_count += 1

        if b.rect.colliderect(pr.rect):

            b.rect.right = pr.rect.left

            b.speed_x = -abs(b.speed_x)

            self._rally_count += 1

        game_over = False

        # P2 wins
        if b.rect.left <= 0:

            self.total_wins[1] += 1
            game_over = True

        # P1 wins
        elif b.rect.right >= GAME_W:

            self.total_wins[0] += 1
            game_over = True

        if game_over:

            duration = (
                pygame.time.get_ticks() / 1000.0
                - self._game_start
            )

            self.rallies.append(self._rally_count)
            self.durations.append(duration)

            print(
                f"[{self.game_num}/{self.num_games}] "
                f"P1 wins={self.total_wins[0]} "
                f"P2 wins={self.total_wins[1]} "
                f"rallies={self._rally_count} "
                f"t={duration:.1f}s"
            )

            if self.game_num >= self.num_games:
                self.state = SUMMARY
            else:
                self._new_game()

    # ─────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────
    def run(self):

        while True:

            self.clock.tick(
                self.fps_val if self.state == RUNNING else 60
            )

            for ev in pygame.event.get():

                if ev.type == pygame.QUIT:

                    pygame.quit()
                    sys.exit()

                if self.state == SETUP:

                    for sl in self.sliders.values():
                        sl.handle(ev)

                if (
                    ev.type == pygame.MOUSEBUTTONDOWN
                    and self.btn.collidepoint(ev.pos)
                ):

                    if self.state == SETUP:
                        self._start()

                    elif self.state == SUMMARY:
                        self.state = SETUP

            if self.state == RUNNING:
                self._sim_step()

            self._draw_sidebar()
            self._draw_game()

            pygame.display.flip()


if __name__ == "__main__":
    App().run()