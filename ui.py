import pygame

from config import *

class Slider:

    H = 44

    def __init__(self, label, lo, hi, default, dtype=float, step=None):

        self.label = label

        self.lo = lo
        self.hi = hi

        self.val = default

        self.dtype = dtype
        self.step = step

        self.drag = False

        self.tx = 0
        self.ty = 0
        self.tw = 0

    def layout(self, x, y, w):

        self.tx = x
        self.ty = y + 28
        self.tw = w

    def _kx(self):

        r = (self.val - self.lo) / (self.hi - self.lo)

        return int(self.tx + r * self.tw)

    def handle(self, ev):

        kx = self._kx()

        knob = pygame.Rect(
            kx - 9,
            self.ty - 9,
            18,
            18
        )

        if ev.type == pygame.MOUSEBUTTONDOWN:

            if knob.collidepoint(ev.pos):
                self.drag = True

        if ev.type == pygame.MOUSEBUTTONUP:
            self.drag = False

        if ev.type == pygame.MOUSEMOTION and self.drag:

            r = max(
                0.0,
                min(1.0, (ev.pos[0] - self.tx) / self.tw)
            )

            v = self.lo + r * (self.hi - self.lo)

            if self.step:
                v = round(v / self.step) * self.step

            self.val = int(v) if self.dtype == int else round(float(v), 5)

    def draw(self, surf, fsm, fval):

        surf.blit(
            fsm.render(self.label, True, GRAY),
            (self.tx, self.ty - 26)
        )

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

        pygame.draw.circle(
            surf,
            WHITE,
            (kx, self.ty),
            8
        )

        vs = str(self.val)

        surf.blit(
            fval.render(vs, True, YELLOW),
            (self.tx + self.tw + 6, self.ty - 8)
        )