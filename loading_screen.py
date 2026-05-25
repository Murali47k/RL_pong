# loading_screen.py — Parameter configuration UI

import pygame
import sys
from config import *

pygame.init()


# helpers

def draw_label(surf, text, x, y, color=GRAY, size=22):
    f = pygame.font.SysFont("Arial", size)
    s = f.render(text, True, color)
    surf.blit(s, (x, y))

def draw_title(surf, text, y):
    f = pygame.font.SysFont("Arial", 36, bold=True)
    s = f.render(text, True, WHITE)
    surf.blit(s, (WIDTH // 2 - s.get_width() // 2, y))

def draw_button(surf, rect, text, hover=False):
    color = (80, 200, 120) if hover else (50, 140, 80)
    pygame.draw.rect(surf, color, rect, border_radius=8)
    f = pygame.font.SysFont("Arial", 24, bold=True)
    s = f.render(text, True, WHITE)
    surf.blit(s, (rect.centerx - s.get_width() // 2,
                  rect.centery - s.get_height() // 2))


# Slider widget 

class Slider:
    def __init__(self, x, y, w, min_v, max_v, default, label, dtype=float, step=None):
        self.x, self.y, self.w = x, y, w
        self.min_v, self.max_v = min_v, max_v
        self.value   = default
        self.label   = label
        self.dtype   = dtype
        self.step    = step
        self.dragging = False
        self.track = pygame.Rect(x, y + 20, w, 6)

    def _knob_x(self):
        ratio = (self.value - self.min_v) / (self.max_v - self.min_v)
        return int(self.x + ratio * self.w)

    def handle_event(self, event):
        kx = self._knob_x()
        knob = pygame.Rect(kx - 10, self.y + 10, 20, 20)
        if event.type == pygame.MOUSEBUTTONDOWN and knob.collidepoint(event.pos):
            self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging:
            ratio = max(0.0, min(1.0, (event.pos[0] - self.x) / self.w))
            raw   = self.min_v + ratio * (self.max_v - self.min_v)
            if self.step:
                raw = round(raw / self.step) * self.step
            self.value = self.dtype(round(raw, 4))

    def draw(self, surf):
        pygame.draw.rect(surf, (80, 80, 80), self.track, border_radius=3)
        kx = self._knob_x()
        pygame.draw.circle(surf, WHITE, (kx, self.y + 23), 10)
        draw_label(surf, self.label, self.x, self.y - 4)
        val_str = str(int(self.value)) if self.dtype == int else f"{self.value:.4f}"
        draw_label(surf, val_str, self.x + self.w + 12, self.y + 10,
                   color=GREEN, size=20)


# Main loading screen

def run_loading_screen():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL Pong — Settings")
    clock = pygame.time.Clock()

    # Sliders (shared params first, then per-agent) 
    sliders = {
        "num_games": Slider(80,  100, 500,   10, 500, 100,  "Number of Games",      int,  step=10),
        "fps":       Slider(80,  165, 500,   10, 240,  60,  "Simulation FPS",       int,  step=10),

        # Agent 1
        "lr1":       Slider(80,  250, 220, 1e-4, 1e-2, 1e-3, "P1  Learning Rate",  float),
        "gamma1":    Slider(380, 250, 220, 0.90, 0.999, 0.99, "P1  Gamma",         float),
        "hid1":      Slider(80,  320, 220,   64,  512,  256, "P1  Hidden Dim",     int,  step=64),

        # Agent 2
        "lr2":       Slider(80,  410, 220, 1e-4, 1e-2, 1e-3, "P2  Learning Rate",  float),
        "gamma2":    Slider(380, 410, 220, 0.90, 0.999, 0.99, "P2  Gamma",         float),
        "hid2":      Slider(80,  480, 220,   64,  512,  256, "P2  Hidden Dim",     int,  step=64),
    }

    btn_rect = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 60, 200, 44)

    while True:
        clock.tick(60)
        mouse = pygame.mouse.get_pos()
        win.fill(DARK)

        draw_title(win, "RL Pong Simulation — Setup", 20)
        draw_label(win, "── Global ──────────────────────────────────────────────────────", 60, 78, color=(100,100,100), size=16)
        draw_label(win, "── Player 1 (Red) ───────────────────────────────────────────", 60, 228, color=RED, size=16)
        draw_label(win, "── Player 2 (Blue) ──────────────────────────────────────────", 60, 390, color=BLUE, size=16)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            for sl in sliders.values():
                sl.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN and btn_rect.collidepoint(event.pos):
                pygame.display.quit()
                return {k: sl.value for k, sl in sliders.items()}

        for sl in sliders.values():
            sl.draw(win)

        hover = btn_rect.collidepoint(mouse)
        draw_button(win, btn_rect, "Start Simulation", hover)
        pygame.display.flip()


if __name__ == "__main__":
    params = run_loading_screen()
    print("Params chosen:", params)