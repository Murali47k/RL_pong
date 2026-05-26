"""
main.py — Pong with DQN Reinforcement Learning agents.

Modes
-----
SETUP    : configure settings, choose Training or Eval
TRAINING : agents explore + learn (epsilon-greedy DQN)
EVAL     : trained policy runs greedily (no random actions)
SUMMARY  : statistics after all rounds complete

Frame pipeline
--------------
Every step the game area is captured as an 84×84 grayscale frame.
Each DQNAgent keeps a rolling stack of 4 frames and passes it through
a small CNN to choose an action (0=stay, 1=up, 2=down).
After every step a reward signal is sent back so the agents can learn.

Reward shaping
--------------
  +1   : ball hits YOUR paddle  (good defence)
  -1   : opponent scores        (you let the ball past)
   0   : everything else
"""

import sys
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "x11")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from config  import *
from entities import Paddle, Ball
from ui      import Slider

# ── RL ────────────────────────────────────────────────────────────────────
from rl import DQNAgent, preprocess, TORCH_OK

# Extra mode constants (SETUP / RUNNING / SUMMARY already live in config)
TRAINING = "training"
EVAL     = "eval"


# ──────────────────────────────────────────────────────────────────────────
class App:

    def __init__(self):

        pygame.init()

        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Pong — DQN RL Agents")

        self.clock = pygame.time.Clock()

        # fonts
        self.f_title = pygame.font.SysFont("Arial", 18, bold=True)
        self.f_sm    = pygame.font.SysFont("Arial", 13)
        self.f_val   = pygame.font.SysFont("Arial", 13, bold=True)
        self.f_hud   = pygame.font.SysFont("Arial", 26, bold=True)
        self.f_sub   = pygame.font.SysFont("Arial", 16)
        self.f_big   = pygame.font.SysFont("Arial", 32, bold=True)
        self.f_med   = pygame.font.SysFont("Arial", 20)

        SX = 14
        SW = SIDEBAR_W - SX * 2 - 34

        self.sliders = {
            "num_games":  Slider("# Rounds",     10, 500, 100, int, 10),
            "fps":        Slider("FPS",           10, 300,  60, int, 10),
            "ball_spd":   Slider("Ball Speed",     2,  15,   5, int,  1),
            "paddle_spd": Slider("Paddle Speed",   2,  15,   6, int,  1),
        }

        y = 56
        for sl in self.sliders.values():
            sl.layout(SX, y, SW)
            y += Slider.H + 6

        # Two buttons side-by-side at the bottom of the sidebar
        bw = (SIDEBAR_W - 28 - 6) // 2
        self.btn_train = pygame.Rect(14,          WIN_H - 58, bw, 42)
        self.btn_eval  = pygame.Rect(14 + bw + 6, WIN_H - 58, bw, 42)

        # Single "Run Again" button shown in SUMMARY
        self.btn_again = pygame.Rect(14, WIN_H - 58, SIDEBAR_W - 28, 42)

        # ── state ─────────────────────────────────────────────────────────
        self.state     = SETUP
        self.mode      = TRAINING   # last mode chosen (TRAINING or EVAL)

        self.game_num  = 0
        self.num_games = 100
        self.fps_val   = 60

        self.total_wins = [0, 0]
        self.rallies    = []
        self.durations  = []

        self.paddle_l  = None
        self.paddle_r  = None
        self.ball      = None

        self._rally_count = 0
        self._game_start  = 0.0

        # ── RL agents ─────────────────────────────────────────────────────
        self.agent_l = DQNAgent("P1-red")
        self.agent_r = DQNAgent("P2-blue")

        # Off-screen surface used to capture the game area for the CNN
        self._game_surf = pygame.Surface((GAME_W, GAME_H))

        # Pending rewards waiting to be delivered after act()
        self._reward_l = 0.0
        self._reward_r = 0.0

        # Whether agents should be in training mode
        self._rl_training = True

    # ══════════════════════════════════════════════════════════════════════
    # Frame capture helper
    # ══════════════════════════════════════════════════════════════════════

    def _capture_frame(self):
        """Render the game area to an off-screen surface and pre-process it."""
        self._game_surf.fill(BLACK)

        # centre divider
        for y in range(0, GAME_H, 20):
            pygame.draw.rect(
                self._game_surf, (45, 45, 45),
                (GAME_W // 2 - 1, y, 2, 12)
            )

        if self.paddle_l:
            self.paddle_l.draw(self._game_surf, 0)
        if self.paddle_r:
            self.paddle_r.draw(self._game_surf, 0)
        if self.ball:
            self.ball.draw(self._game_surf, 0)

        return preprocess(self._game_surf)   # numpy (84,84) float32

    # ══════════════════════════════════════════════════════════════════════
    # Sidebar
    # ══════════════════════════════════════════════════════════════════════

    def _draw_sidebar(self):

        sb = pygame.Surface((SIDEBAR_W, WIN_H))
        sb.fill(PANEL)

        title = self.f_title.render("PONG — DQN AGENTS", True, WHITE)
        sb.blit(title, (SIDEBAR_W // 2 - title.get_width() // 2, 10))

        pygame.draw.line(sb, (60, 60, 70), (8, 32), (SIDEBAR_W - 8, 32))

        sb.blit(self.f_sm.render("── SETTINGS ──", True, GRAY), (14, 48))

        for sl in self.sliders.values():
            sl.draw(sb, self.f_sm, self.f_val)

        # ── live stats ────────────────────────────────────────────────────
        if self.state in (TRAINING, EVAL, SUMMARY):

            y0 = self.btn_again.top - 148

            pygame.draw.line(
                sb, (60, 60, 70),
                (8, y0 - 8), (SIDEBAR_W - 8, y0 - 8)
            )

            rows = [
                (f"Round  {self.game_num}/{self.num_games}", WHITE),
                (f"P1 wins : {self.total_wins[0]}", RED),
                (f"P2 wins : {self.total_wins[1]}", BLUE),
            ]

            if self.rallies:
                avg_r = sum(self.rallies) / len(self.rallies)
                rows.append((f"Avg rallies : {avg_r:.1f}", GRAY))

            if self.durations:
                avg_d = sum(self.durations) / len(self.durations)
                rows.append((f"Avg duration : {avg_d:.1f}s", GRAY))

            # RL-specific stats
            if TORCH_OK:
                rows.append(("", GRAY))
                rows.append((
                    f"ε  {self.agent_l.epsilon:.3f} / {self.agent_r.epsilon:.3f}",
                    YELLOW,
                ))
                rows.append((
                    f"Loss P1 : {self.agent_l.avg_loss:.4f}",
                    (180, 180, 255),
                ))
                rows.append((
                    f"Loss P2 : {self.agent_r.avg_loss:.4f}",
                    (255, 180, 180),
                ))
                rows.append((
                    f"Steps   : {self.agent_l.steps}",
                    GRAY,
                ))

            for i, (txt, col) in enumerate(rows):
                sb.blit(
                    self.f_sm.render(txt, True, col),
                    (14, y0 + i * 18)
                )

        # ── buttons ───────────────────────────────────────────────────────
        mx, my = pygame.mouse.get_pos()

        if self.state == SETUP:

            for btn, label, color_idle, color_hov in [
                (self.btn_train, "▶ Train", (45, 150, 75),  (70, 210, 110)),
                (self.btn_eval,  "▶ Eval",  (50, 80, 160),  (80, 120, 220)),
            ]:
                hov = btn.collidepoint(mx, my)
                pygame.draw.rect(
                    sb,
                    color_hov if hov else color_idle,
                    btn, border_radius=8
                )
                bt = self.f_title.render(label, True, WHITE)
                sb.blit(bt, (
                    btn.centerx - bt.get_width() // 2,
                    btn.centery - bt.get_height() // 2,
                ))

            if not TORCH_OK:
                warn = self.f_sm.render("⚠ PyTorch not found", True, YELLOW)
                sb.blit(warn, (14, self.btn_train.top - 24))

        elif self.state in (TRAINING, EVAL):

            pygame.draw.rect(sb, (60, 60, 70), self.btn_again, border_radius=8)
            bt = self.f_sm.render("Simulating…", True, GRAY)
            sb.blit(bt, (
                self.btn_again.centerx - bt.get_width() // 2,
                self.btn_again.centery - bt.get_height() // 2,
            ))

        else:  # SUMMARY

            hov = self.btn_again.collidepoint(mx, my)
            pygame.draw.rect(
                sb,
                (70, 210, 110) if hov else (45, 150, 75),
                self.btn_again, border_radius=8
            )
            bt = self.f_title.render("▶ Run Again", True, WHITE)
            sb.blit(bt, (
                self.btn_again.centerx - bt.get_width() // 2,
                self.btn_again.centery - bt.get_height() // 2,
            ))

        self.screen.blit(sb, (0, 0))

    # ══════════════════════════════════════════════════════════════════════
    # Game area
    # ══════════════════════════════════════════════════════════════════════

    def _draw_game(self):

        ox = SIDEBAR_W

        pygame.draw.rect(self.screen, BLACK, (ox, 0, GAME_W, GAME_H))

        if self.state == SETUP:
            msg = self.f_big.render("Configure & press Train or Eval →", True, (70, 70, 82))
            self.screen.blit(msg, (
                ox + GAME_W // 2 - msg.get_width() // 2,
                GAME_H // 2 - msg.get_height() // 2,
            ))
            return

        if self.state == SUMMARY:
            self._draw_summary(ox)
            return

        # centre divider
        for y in range(0, GAME_H, 20):
            pygame.draw.rect(
                self.screen, (45, 45, 45),
                (ox + GAME_W // 2 - 1, y, 2, 12)
            )

        self.paddle_l.draw(self.screen, ox)
        self.paddle_r.draw(self.screen, ox)
        self.ball.draw(self.screen, ox)

        # score
        sc = self.f_hud.render(
            f"{self.total_wins[0]} : {self.total_wins[1]}", True, WHITE
        )
        self.screen.blit(sc, (
            ox + GAME_W // 2 - sc.get_width() // 2, 10
        ))

        gc = self.f_sub.render(
            f"Round {self.game_num} / {self.num_games}", True, GRAY
        )
        self.screen.blit(gc, (
            ox + GAME_W // 2 - gc.get_width() // 2, 44
        ))

        self.screen.blit(self.f_sub.render("P1", True, RED),  (ox + 24, 10))
        self.screen.blit(self.f_sub.render("P2", True, BLUE), (ox + GAME_W - 40, 10))

        mode_label = "TRAINING" if self.state == TRAINING else "EVAL"
        mode_col   = GREEN if self.state == TRAINING else YELLOW
        ml = self.f_sm.render(mode_label, True, mode_col)
        self.screen.blit(ml, (ox + GAME_W - ml.get_width() - 8, 44))

        rc = self.f_sm.render(
            f"rally hits: {self._rally_count}", True, (100, 100, 110)
        )
        self.screen.blit(rc, (
            ox + GAME_W // 2 - rc.get_width() // 2, GAME_H - 22
        ))

    # ══════════════════════════════════════════════════════════════════════
    # Summary screen
    # ══════════════════════════════════════════════════════════════════════

    def _draw_summary(self, ox):

        wl, wr = self.total_wins
        n      = max(self.num_games, 1)
        avg_r  = sum(self.rallies)   / len(self.rallies)   if self.rallies   else 0.0
        avg_d  = sum(self.durations) / len(self.durations) if self.durations else 0.0

        rows = [
            ("Simulation Complete",                   WHITE, self.f_big),
            ("",                                       WHITE, self.f_med),
            (f"Rounds played : {self.num_games}",      WHITE, self.f_med),
            (f"P1 wins : {wl}  ({100*wl//n}%)",        RED,   self.f_med),
            (f"P2 wins : {wr}  ({100*wr//n}%)",        BLUE,  self.f_med),
            (f"Avg rally hits   : {avg_r:.1f}",        GRAY,  self.f_med),
            (f"Avg round length : {avg_d:.1f}s",       GRAY,  self.f_med),
        ]

        if TORCH_OK:
            rows += [
                ("",                                   WHITE, self.f_med),
                (f"P1 training steps : {self.agent_l.steps}", (180,180,255), self.f_med),
                (f"P2 training steps : {self.agent_r.steps}", (255,180,180), self.f_med),
                (f"P1 final ε : {self.agent_l.epsilon:.3f}",  (180,180,255), self.f_sub),
                (f"P2 final ε : {self.agent_r.epsilon:.3f}",  (255,180,180), self.f_sub),
            ]

        rows += [
            ("", WHITE, self.f_med),
            ('Press "Run Again" to go back', GRAY, self.f_sub),
        ]

        total_h = sum(f.get_height() + 8 for _, _, f in rows)
        y = GAME_H // 2 - total_h // 2

        for txt, col, font in rows:
            s = font.render(txt, True, col)
            self.screen.blit(s, (
                ox + GAME_W // 2 - s.get_width() // 2, y
            ))
            y += font.get_height() + 8

    # ══════════════════════════════════════════════════════════════════════
    # Simulation control
    # ══════════════════════════════════════════════════════════════════════

    def _start(self, mode):
        """Begin a fresh simulation in TRAINING or EVAL mode."""
        p = {k: sl.val for k, sl in self.sliders.items()}

        self.num_games   = int(p["num_games"])
        self.fps_val     = int(p["fps"])
        self._ball_spd   = int(p["ball_spd"])
        self._paddle_spd = int(p["paddle_spd"])

        self.total_wins  = [0, 0]
        self.game_num    = 0
        self.rallies     = []
        self.durations   = []

        self.mode  = mode
        self.state = mode

        # configure agent training flags
        self._rl_training = (mode == TRAINING)
        self.agent_l.training = self._rl_training
        self.agent_r.training = self._rl_training

        self._new_game()

    def _new_game(self):
        self.game_num += 1

        self.paddle_l = Paddle(18,           RED,  self._paddle_spd)
        self.paddle_r = Paddle(GAME_W - 28,  BLUE, self._paddle_spd)
        self.ball     = Ball(self._ball_spd)

        self._rally_count = 0
        self._game_start  = pygame.time.get_ticks() / 1000.0
        self._reward_l    = 0.0
        self._reward_r    = 0.0

        # Give agents the first frame
        frame = self._capture_frame()
        self.agent_l.new_episode(frame)
        self.agent_r.new_episode(frame)

    # ══════════════════════════════════════════════════════════════════════
    # Simulation step  (called once per frame when state == TRAINING/EVAL)
    # ══════════════════════════════════════════════════════════════════════

    def _sim_step(self):

        # 1. Capture current frame
        frame = self._capture_frame()

        # 2. Agents choose actions
        action_l = self.agent_l.act(frame)
        action_r = self.agent_r.act(frame)

        # 3. Apply actions to paddles
        self._apply_action(self.paddle_l, action_l)
        self._apply_action(self.paddle_r, action_r)

        # 4. Move ball
        self.ball.move()

        # 5. Collision detection + reward shaping
        b  = self.ball
        pl = self.paddle_l
        pr = self.paddle_r

        reward_l = 0.0
        reward_r = 0.0
        done     = False

        if b.rect.colliderect(pl.rect):
            b.rect.left  = pl.rect.right
            b.speed_x    = abs(b.speed_x)
            self._rally_count += 1
            reward_l += 1.0   # P1 hit the ball — good

        if b.rect.colliderect(pr.rect):
            b.rect.right = pr.rect.left
            b.speed_x    = -abs(b.speed_x)
            self._rally_count += 1
            reward_r += 1.0   # P2 hit the ball — good

        # Scoring
        if b.rect.left <= 0:          # P1 missed → P2 scores
            self.total_wins[1] += 1
            reward_l -= 1.0
            reward_r += 1.0
            done = True

        elif b.rect.right >= GAME_W:  # P2 missed → P1 scores
            self.total_wins[0] += 1
            reward_l += 1.0
            reward_r -= 1.0
            done = True

        # 6. Capture next frame and store transitions
        next_frame = self._capture_frame()
        self.agent_l.remember(next_frame, reward_l, done)
        self.agent_r.remember(next_frame, reward_r, done)

        # 7. End of round
        if done:
            duration = pygame.time.get_ticks() / 1000.0 - self._game_start
            self.rallies.append(self._rally_count)
            self.durations.append(duration)

            print(
                f"[{self.game_num}/{self.num_games}]  "
                f"P1={self.total_wins[0]}  P2={self.total_wins[1]}  "
                f"rallies={self._rally_count}  t={duration:.1f}s  "
                f"ε_P1={self.agent_l.epsilon:.3f}  "
                f"loss_P1={self.agent_l.avg_loss:.4f}"
            )

            if self.game_num >= self.num_games:
                self.state = SUMMARY
                # auto-save trained weights
                if TORCH_OK and self._rl_training:
                    self.agent_l.save("agent_p1.pt")
                    self.agent_r.save("agent_p2.pt")
                    print("Weights saved → agent_p1.pt / agent_p2.pt")
            else:
                self._new_game()

    # ══════════════════════════════════════════════════════════════════════
    # Action → paddle movement
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _apply_action(paddle, action):
        """0=stay, 1=up, 2=down"""
        if action == 1 and paddle.rect.top > 0:
            paddle.rect.y -= paddle.spd
        elif action == 2 and paddle.rect.bottom < GAME_H:
            paddle.rect.y += paddle.spd

    # ══════════════════════════════════════════════════════════════════════
    # Main loop
    # ══════════════════════════════════════════════════════════════════════

    def run(self):

        while True:

            fps = self.fps_val if self.state in (TRAINING, EVAL) else 60
            self.clock.tick(fps)

            for ev in pygame.event.get():

                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # slider interaction only in SETUP
                if self.state == SETUP:
                    for sl in self.sliders.values():
                        sl.handle(ev)

                if ev.type == pygame.MOUSEBUTTONDOWN:

                    pos = ev.pos

                    if self.state == SETUP:
                        if self.btn_train.collidepoint(pos):
                            # Try to load previously saved weights for eval warmup
                            if TORCH_OK:
                                try:
                                    self.agent_l.load("agent_p1.pt")
                                    self.agent_r.load("agent_p2.pt")
                                    print("Loaded existing weights — continuing training.")
                                except Exception:
                                    pass
                            self._start(TRAINING)

                        elif self.btn_eval.collidepoint(pos):
                            if TORCH_OK:
                                try:
                                    self.agent_l.load("agent_p1.pt")
                                    self.agent_r.load("agent_p2.pt")
                                    print("Loaded weights for evaluation.")
                                except Exception:
                                    print("No saved weights found — agents will act randomly.")
                            self._start(EVAL)

                    elif self.state == SUMMARY:
                        if self.btn_again.collidepoint(pos):
                            self.state = SETUP

            if self.state in (TRAINING, EVAL):
                self._sim_step()

            self._draw_sidebar()
            self._draw_game()
            pygame.display.flip()


if __name__ == "__main__":
    App().run()