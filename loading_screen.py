import sys
import pygame

from config import *

pygame.init()


# =========================================================
# Helpers
# =========================================================

def draw_text(surface, text, x, y,
              color=WHITE,
              size=22,
              bold=False):

    font = pygame.font.SysFont("Arial", size, bold=bold)
    surf = font.render(text, True, color)
    surface.blit(surf, (x, y))


def draw_center_text(surface, text, y,
                     color=WHITE,
                     size=36,
                     bold=True):

    font = pygame.font.SysFont("Arial", size, bold=bold)
    surf = font.render(text, True, color)

    surface.blit(
        surf,
        (WIDTH // 2 - surf.get_width() // 2, y)
    )


# =========================================================
# Slider Widget
# =========================================================

class Slider:

    def __init__(
        self,
        x,
        y,
        width,
        min_value,
        max_value,
        default_value,
        label,
        dtype=float,
        step=None
    ):

        self.x = x
        self.y = y
        self.width = width

        self.min_value = min_value
        self.max_value = max_value
        self.value = default_value

        self.label = label
        self.dtype = dtype
        self.step = step

        self.dragging = False

        self.track_rect = pygame.Rect(
            x,
            y + 20,
            width,
            6
        )

    def get_knob_x(self):

        ratio = (
            (self.value - self.min_value)
            /
            (self.max_value - self.min_value)
        )

        return int(self.x + ratio * self.width)

    def handle_event(self, event):

        knob_x = self.get_knob_x()

        knob_rect = pygame.Rect(
            knob_x - 10,
            self.y + 10,
            20,
            20
        )

        if event.type == pygame.MOUSEBUTTONDOWN:
            if knob_rect.collidepoint(event.pos):
                self.dragging = True

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION:

            if self.dragging:

                ratio = (
                    (event.pos[0] - self.x)
                    / self.width
                )

                ratio = max(0.0, min(1.0, ratio))

                raw_value = (
                    self.min_value
                    +
                    ratio * (self.max_value - self.min_value)
                )

                if self.step is not None:
                    raw_value = (
                        round(raw_value / self.step)
                        * self.step
                    )

                if self.dtype == int:
                    self.value = int(raw_value)
                else:
                    self.value = float(round(raw_value, 5))

    def draw(self, surface):

        pygame.draw.rect(
            surface,
            (80, 80, 80),
            self.track_rect,
            border_radius=3
        )

        knob_x = self.get_knob_x()

        pygame.draw.circle(
            surface,
            WHITE,
            (knob_x, self.y + 23),
            10
        )

        draw_text(
            surface,
            self.label,
            self.x,
            self.y - 6,
            color=GRAY,
            size=20
        )

        if self.dtype == int:
            value_str = str(self.value)
        else:
            value_str = f"{self.value:.4f}"

        draw_text(
            surface,
            value_str,
            self.x + self.width + 14,
            self.y + 10,
            color=GREEN,
            size=20
        )


# =========================================================
# Main Loading Screen
# =========================================================

def run_loading_screen():

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL Pong - Setup")

    clock = pygame.time.Clock()

    sliders = {

        "num_games":
            Slider(
                80, 100, 500,
                10, 500,
                100,
                "Number of Games",
                int,
                step=10
            ),

        "fps":
            Slider(
                80, 170, 500,
                10, 240,
                60,
                "Simulation FPS",
                int,
                step=10
            ),

        # Player 1

        "lr1":
            Slider(
                80, 260, 220,
                1e-4, 1e-2,
                1e-3,
                "P1 Learning Rate"
            ),

        "gamma1":
            Slider(
                380, 260, 220,
                0.90, 0.999,
                0.99,
                "P1 Gamma"
            ),

        "hid1":
            Slider(
                80, 330, 220,
                64, 512,
                256,
                "P1 Hidden Dim",
                int,
                step=64
            ),

        # Player 2

        "lr2":
            Slider(
                80, 430, 220,
                1e-4, 1e-2,
                1e-3,
                "P2 Learning Rate"
            ),

        "gamma2":
            Slider(
                380, 430, 220,
                0.90, 0.999,
                0.99,
                "P2 Gamma"
            ),

        "hid2":
            Slider(
                80, 500, 220,
                64, 512,
                256,
                "P2 Hidden Dim",
                int,
                step=64
            ),
    }

    start_button = pygame.Rect(
        WIDTH // 2 - 110,
        HEIGHT - 70,
        220,
        50
    )

    while True:

        clock.tick(60)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            for slider in sliders.values():
                slider.handle_event(event)

            if (
                event.type == pygame.MOUSEBUTTONDOWN
                and start_button.collidepoint(event.pos)
            ):

                params = {
                    key: slider.value
                    for key, slider in sliders.items()
                }

                return params

        screen.fill(DARK)

        draw_center_text(
            screen,
            "RL Pong Simulation",
            20
        )

        draw_text(
            screen,
            "Global Settings",
            60,
            70,
            color=GREEN,
            size=18,
            bold=True
        )

        draw_text(
            screen,
            "Player 1 (Red)",
            60,
            230,
            color=RED,
            size=18,
            bold=True
        )

        draw_text(
            screen,
            "Player 2 (Blue)",
            60,
            400,
            color=BLUE,
            size=18,
            bold=True
        )

        for slider in sliders.values():
            slider.draw(screen)

        mouse_pos = pygame.mouse.get_pos()

        hovered = start_button.collidepoint(mouse_pos)

        button_color = (
            (80, 200, 120)
            if hovered
            else
            (50, 140, 80)
        )

        pygame.draw.rect(
            screen,
            button_color,
            start_button,
            border_radius=8
        )

        draw_center_text(
            screen,
            "Start Simulation",
            HEIGHT - 62,
            size=24
        )

        pygame.display.flip()