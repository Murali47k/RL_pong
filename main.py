"""
main.py — Pong simulation, both players move randomly.
Whoever scores first wins the round.
"""

import sys
import os
import random

os.environ.setdefault("SDL_VIDEODRIVER", "x11")

import pygame

# ──────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────
SIDEBAR_W = 260
GAME_W = 800
GAME_H = 600
WIN_W = SIDEBAR_W + GAME_W
WIN_H = GAME_H

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 50, 50)
BLUE = (50, 100, 220)
GRAY = (180, 180, 180)
PANEL = (22, 22, 32)
GREEN = (50, 200, 100)
YELLOW = (240, 200, 60)

PADDLE_W = 10
PADDLE_H = 100
PADDLE_SPD = 6
BALL_SZ = 10

# each round ends after ONE score
MAX_PTS = 1


# ──────────────────────────────────────────────────────────────────────
# Slider widget
# ──────────────────────────────────────────────────────────────────────
class Slider:
    H = 44

    def __init__(self, label: str, lo, hi, default, dtype=float, step=None):
        self.label = label
        self.lo, self.hi = lo, hi
        self.val = default
        self.dtype = dtype
        self.step = step
        self.drag = False
        self.tx = self.ty = self.tw = 0

    def layout(self, x: int, y: int, w: int):
        self.tx, self.ty, self.tw = x, y + 28, w

    def _kx(self) -> int:
        r = (self.val - self.lo) / (self.hi - self.lo)
        return int(self.tx + r * self.tw)

    def handle(self, ev):
        kx = self._kx()
        knob = pygame.Rect(kx - 9, self.ty - 9, 18, 18)

        if ev.type == pygame.MOUSEBUTTONDOWN and knob.collidepoint(ev.pos):
            self.drag = True

        if ev.type == pygame.MOUSEBUTTONUP:
            self.drag = False

        if ev.type == pygame.MOUSEMOTION and self.drag:
            r = max(0.0, min(1.0, (ev.pos[0] - self.tx) / self.tw))
            v = self.lo + r * (self.hi - self.lo)

            if self.step:
                v = round(v / self.step) * self.step

            self.val = int(v) if self.dtype == int else round(float(v), 5)

    def draw(self, surf, fsm, fval):
        surf.blit(fsm.render(self.label, True, GRAY),
                  (self.tx, self.ty - 26))

        pygame.draw.rect(
            surf,
            (70, 70, 80),
            (self.tx, self.ty - 2, self.tw, 5),
            border_radius=3
        )

        kx = self._kx()

        if kx > self.tx:
            pygame.draw.rect(
                surf,
                GREEN,
                (self.tx, self.ty - 2, kx - self.tx, 5),
                border_radius=3
            )

        pygame.draw.circle(surf, WHITE, (kx, self.ty), 8)

        vs = str(self.val) if self.dtype == int else f"{self.val:.2f}"

        surf.blit(
            fval.render(vs, True, YELLOW),
            (self.tx + self.tw + 6, self.ty - 8)
        )


# ──────────────────────────────────────────────────────────────────────
# Paddle
# ──────────────────────────────────────────────────────────────────────
class Paddle:
    def __init__(self, x: int, color, spd: int = PADDLE_SPD):
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

    def draw(self, surf, ox: int):
        pygame.draw.rect(
            surf,
            self.color,
            self.rect.move(ox, 0),
            border_radius=3
        )


# ──────────────────────────────────────────────────────────────────────
# Ball
# ──────────────────────────────────────────────────────────────────────
class Ball:
    def __init__(self, spd: float = 5.0):
        self.spd = spd
        self.reset()

    def reset(self):
        self.rect = pygame.Rect(
            GAME_W // 2,
            GAME_H // 2,
            BALL_SZ,
            BALL_SZ
        )

        self.speed_x = self.spd * random.choice([-1, 1])
        self.speed_y = self.spd * random.choice([-1, 1])

    def move(self):
        self.rect.x += int(self.speed_x)
        self.rect.y += int(self.speed_y)

        if self.rect.top <= 0 or self.rect.bottom >= GAME_H:
            self.speed_y *= -1

    def draw(self, surf, ox: int):
        pygame.draw.ellipse(surf, WHITE, self.rect.move(ox, 0))


# ──────────────────────────────────────────────────────────────────────
# States
# ──────────────────────────────────────────────────────────────────────
SETUP = "setup"
RUNNING = "running"
SUMMARY = "summary"


# ──────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────
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

        self.btn = pygame.Rect(14, WIN_H - 58, SIDEBAR_W - 28, 42)

        self.state = SETUP

        self.game_num = 0
        self.num_games = 20
        self.fps_val = 60

        self.total_wins = [0, 0]

        self.paddle_l = None
        self.paddle_r = None
        self.ball = None

        self.rallies = []
        self.durations = []

        self._rally_count = 0
        self._game_start = 0.0

    # ──────────────────────────────────────────────────────────────────
    # Sidebar
    # ──────────────────────────────────────────────────────────────────
    def _draw_sidebar(self):
        sb = pygame.Surface((SIDEBAR_W, WIN_H))
        sb.fill(PANEL)

        t = self.f_title.render("PONG — RANDOM", True, WHITE)

        sb.blit(t, (SIDEBAR_W // 2 - t.get_width() // 2, 10))

        pygame.draw.line(sb, (60, 60, 70), (8, 32),
                         (SIDEBAR_W - 8, 32))

        sb.blit(
            self.f_sm.render("── SETTINGS ──", True, GRAY),
            (14, 48)
        )

        for sl in self.sliders.values():
            sl.draw(sb, self.f_sm, self.f_val)

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
                rows.append((f"Avg rallies : {avg_r:.1f}", GRAY))

            if self.durations:
                avg_d = sum(self.durations) / len(self.durations)
                rows.append((f"Avg duration : {avg_d:.1f}s", GRAY))

            for i, (txt, col) in enumerate(rows):
                sb.blit(
                    self.f_sm.render(txt, True, col),
                    (14, y0 + i * 20)
                )

        # button
        if self.state == SETUP:
            hov = self.btn.collidepoint(pygame.mouse.get_pos())

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

            bt = self.f_sm.render("Simulating…", True, GRAY)

        else:
            pygame.draw.rect(
                sb,
                (50, 80, 160),
                self.btn,
                border_radius=8
            )

            bt = self.f_title.render("▶ Run Again", True, WHITE)

        sb.blit(
            bt,
            (
                self.btn.centerx - bt.get_width() // 2,
                self.btn.centery - bt.get_height() // 2
            )
        )

        self.screen.blit(sb, (0, 0))

    # ──────────────────────────────────────────────────────────────────
    # Game canvas
    # ──────────────────────────────────────────────────────────────────
    def _draw_game(self):
        ox = SIDEBAR_W
        surf = self.screen

        pygame.draw.rect(surf, BLACK, (ox, 0, GAME_W, GAME_H))

        for y in range(0, GAME_H, 20):
            pygame.draw.rect(
                surf,
                (45, 45, 45),
                (ox + GAME_W // 2 - 1, y, 2, 12)
            )

        if self.state == SETUP:

            msg = self.f_big.render(
                "Configure & press Start →",
                True,
                (70, 70, 82)
            )

            surf.blit(
                msg,
                (
                    ox + GAME_W // 2 - msg.get_width() // 2,
                    GAME_H // 2 - msg.get_height() // 2
                )
            )

            return

        if self.state == SUMMARY:
            self._draw_summary(ox)
            return

        self.paddle_l.draw(surf, ox)
        self.paddle_r.draw(surf, ox)
        self.ball.draw(surf, ox)

        # show TOTAL ROUND WINS instead of points
        sc = self.f_hud.render(
            f"{self.total_wins[0]} : {self.total_wins[1]}",
            True,
            WHITE
        )

        surf.blit(
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

        surf.blit(
            gc,
            (
                ox + GAME_W // 2 - gc.get_width() // 2,
                44
            )
        )

        surf.blit(self.f_sub.render("P1", True, RED), (ox + 24, 10))

        surf.blit(
            self.f_sub.render("P2", True, BLUE),
            (ox + GAME_W - 40, 10)
        )

        rc = self.f_sm.render(
            f"rally hits: {self._rally_count}",
            True,
            (100, 100, 110)
        )

        surf.blit(
            rc,
            (
                ox + GAME_W // 2 - rc.get_width() // 2,
                GAME_H - 22
            )
        )

    # ──────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────
    def _draw_summary(self, ox: int):
        surf = self.screen

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

            surf.blit(
                s,
                (
                    ox + GAME_W // 2 - s.get_width() // 2,
                    y
                )
            )

            y += font.get_height() + 8

    # ──────────────────────────────────────────────────────────────────
    # Control
    # ──────────────────────────────────────────────────────────────────
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

        self._game_start = pygame.time.get_ticks() / 1000.0

    # ──────────────────────────────────────────────────────────────────
    # Simulation
    # ──────────────────────────────────────────────────────────────────
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

        # P2 wins round
        if b.rect.left <= 0:
            self.total_wins[1] += 1
            game_over = True

        # P1 wins round
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
                f"P1 wins={self.total_wins[0]}  "
                f"P2 wins={self.total_wins[1]}  "
                f"rallies={self._rally_count}  "
                f"t={duration:.1f}s"
            )

            if self.game_num >= self.num_games:
                self.state = SUMMARY
            else:
                self._new_game()

    # ──────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().run()