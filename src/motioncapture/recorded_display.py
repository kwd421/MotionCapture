"""Main-thread display owners for the explicitly selected recorded preview."""
from __future__ import annotations

import threading

from motioncapture.wholebody_catalog import BenchmarkError


class SDLDisplay:
    """Synchronous BGR blit; never retains a packet or selects another backend."""

    def __init__(self, title):
        import pygame

        self.pg = pygame
        self.title = title
        self.surface = None
        self.open_attempted = False
        self.metadata = {"backend": "sdl", "pygame_version": pygame.version.ver,
                         "sdl_version": list(pygame.get_sdl_version()),
                         "driver": None, "vsync_requested": False}

    def _check_thread(self):
        if threading.current_thread() is not threading.main_thread():
            raise BenchmarkError("display_requires_main_thread")

    def show(self, image):
        self._check_thread()
        pg = self.pg
        if self.surface is None:
            self.open_attempted = True
            pg.display.init()
            driver = pg.display.get_driver()
            self.metadata["driver"] = driver
            if driver not in {"cocoa", "windows", "x11", "wayland"}:
                raise BenchmarkError("unsupported_sdl_display_driver")
            self.surface = pg.display.set_mode((image.shape[1], image.shape[0]))
            pg.display.set_caption(self.title)
        # frombuffer shares this fresh contiguous canvas until blit completes.
        view = pg.image.frombuffer(image, (image.shape[1], image.shape[0]), "BGR")
        self.surface.blit(view, (0, 0))
        pg.display.flip()

    def poll(self):
        self._check_thread()
        if self.surface is not None:
            for event in self.pg.event.get():
                if event.type == self.pg.QUIT or (
                    event.type == self.pg.KEYDOWN
                    and event.key in (self.pg.K_ESCAPE, self.pg.K_q)
                ):
                    raise KeyboardInterrupt

    def close(self):
        self._check_thread()
        if self.open_attempted:
            self.pg.display.quit()
            self.surface = None
