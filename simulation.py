from __future__ import print_function

import sys

import numpy as np
import pygame
import skimage as skimage
from ple.games import base
from pygame.constants import K_UP, K_LEFT, K_RIGHT, K_s
from pygame.surface import Surface
from skimage import transform, color, exposure


class Car(pygame.sprite.Sprite):

    def __init__(self, screen_width, screen_height, image_assets):
        pygame.sprite.Sprite.__init__(self)

        self.directions = {
            "up": "up",
            "left": "left",
            "right": "right"
        }
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.image_assets = image_assets

        self.image = None
        self.rect = None
        self.pos_x = None
        self.pos_y = None
        self.current_direction = None
        self.init()

        self.rect.center = (self.pos_x, self.pos_y)

    def init(self):
        self.current_direction = self.directions["up"]
        self.image = self.image_assets["car"]["up"]
        self.pos_x = int(self.screen_width * 0.6)
        self.pos_y = int(self.screen_height * 0.8)
        self.rect = self.image.get_rect()
        self.rect.center = (self.pos_x, self.pos_y)

    def move(self, direction):
        if direction == self.directions["up"]:
            self.pos_y -= 2
        if direction == self.directions["left"]:
            self.pos_x -= 2
        if direction == self.directions["right"]:
            self.pos_x += 2

    def change_direction(self, direction):
        self.image = self.image_assets["car"][direction]
        self.current_direction = self.directions[direction]

    def get_directions(self):
        return self.directions

    def update(self):
        self.rect = self.image.get_rect()
        self.rect.center = (self.pos_x, self.pos_y)

    def draw(self, screen):
        screen.blit(self.image, self.rect.center)

    def get_top_coords(self):
        pos_x = self.pos_x
        pos_y = self.pos_y
        if self.current_direction == self.directions["up"]:
            pos_x -= -20
        elif self.current_direction == self.directions["left"]:
            pos_y += 20
        elif self.current_direction == self.directions["right"]:
            pos_x += 49
            pos_y += 20

        return pos_x, pos_y


class CarSimulation(base.PyGameWrapper):

    def __init__(self, width=500, height=500, track_number=0):
        actions = {
            "up": K_UP,
            "left": K_LEFT,
            "right": K_RIGHT
        }

        base.PyGameWrapper.__init__(self, width, height, actions=actions)

        if 0 <= track_number <= 2:
            self.track_number = track_number
        else:
            raise ValueError("Invalid track number provided")

        # Needed to preload images
        pygame.display.set_mode((1, 1), pygame.NOFRAME)

        self.images = {}
        self._load_images()
        self.background = self.images["track"][self.track_number]

        self.car = None
        self.last_action_key = None
        self.current_episode = None
        self.current_epoch = None
        self.game_screen = Surface((width, height))

    def _load_images(self):
        self.images["car"] = {
            "up": pygame.image.load("assets/car.png").convert_alpha(),
            "left": pygame.transform.rotate(pygame.image.load("assets/car.png").convert_alpha(), 90),
            "right": pygame.transform.rotate(pygame.image.load("assets/car.png").convert_alpha(), -90)
        }

        self.images["track"] = [
            pygame.image.load("assets/background_1.png").convert_alpha(),
            pygame.image.load("assets/background_2.png").convert_alpha(),
            pygame.image.load("assets/background_3.png").convert_alpha()
        ]

    def init(self):
        self.score = 0
        self.lives = 1
        self.last_action_key = None

        if self.car is None:
            self.car = Car(
                self.width,
                self.height,
                self.images
            )

        self.car.init()

    def step(self, delta):
        self.handle_player_events()

        self.score += self.rewards["tick"]

        self.determine_reward()

        if self.lives == 0:
            self.score += self.rewards["loss"]
        if self.lives == 2:
            self.score += self.rewards["win"]
            self.lives = 0

        self.game_screen.blit(self.images["track"][self.track_number], (0, 0))
        self.car.update()
        self.car.draw(self.game_screen)
        self.screen.blit(self.game_screen, (0, 0))
        self.draw_text()

    def draw_text(self):
        font = pygame.font.SysFont(None, 25)
        label_coords = font.render("x: {}, y: {}".format(self.car.pos_x, self.car.pos_y), 1, (255, 255, 255))
        label_reward = font.render("reward: {}".format(self.score), 1, (255, 255, 255))
        label_epoch = font.render("epoch: {}".format(self.current_epoch), 1, (255, 255, 255))
        label_episode = font.render("episode: {}".format(self.current_episode), 1, (255, 255, 255))
        self.screen.blit(label_coords, (20, 20))
        self.screen.blit(label_reward, (20, 40))
        self.screen.blit(label_epoch, (150, 20))
        self.screen.blit(label_episode, (280, 20))

    def determine_reward(self):
        pixel_red_color = self.background.get_at(self.car.get_top_coords()).r
        if pixel_red_color == 0:
            self.lives = 0
        elif 255 > pixel_red_color > 0:
            self.lives = 2

    def getGameState(self):
        state = {
            "player_x": self.car.rect.center[0],
            "player_y": self.car.rect.center[1]
        }

        return state

    def set_agent_information(self, episode=None, epoch=None):
        self.current_episode = episode
        self.current_epoch = epoch

    def getScore(self):
        return self.score

    def game_over(self):
        return self.lives == 0

    def handle_player_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == K_s:
                    small_image = pygame.surfarray.make_surface(prepare_image(game.game_screen))
                    pygame.image.save(small_image, "grayscale.png")
                    pygame.image.save(self.game_screen, "screen.png")

                if key != self.last_action_key:
                    if key == self.actions['up']:
                        self.car.change_direction("up")
                    if key == self.actions['left']:
                        self.car.change_direction("left")
                    if key == self.actions['right']:
                        self.car.change_direction("right")
                else:
                    if key == self.actions['up']:
                        self.car.move("up")
                    if key == self.actions['left']:
                        self.car.move("left")
                    if key == self.actions['right']:
                        self.car.move("right")
                self.last_action_key = key
                # print(self.getGameState())


def prepare_image(screen):
    arr = get_screen_rgb_and_resize(screen)
    avgs = [[(r * 0.298 + g * 0.587 + b * 0.114) for (r, g, b) in col] for col in arr]
    arr = np.array([[[avg, avg, avg] for avg in col] for col in avgs])
    return arr


def get_screen_rgb_and_resize(screen):
    return pygame.surfarray.array3d(pygame.transform.scale(screen, (100, 100)))


def prepare_for_learning(screen):
    x_t1 = skimage.color.rgb2gray(screen_to_array3d(screen))
    x_t1 = skimage.transform.resize(x_t1, (100, 100))
    x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

    x_t1 = x_t1 / 255.0
    x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
    return x_t1


def screen_to_array3d(screen):
    return pygame.surfarray.array3d(screen)


if __name__ == '__main__':
    pygame.init()
    game = CarSimulation()
    game.screen = pygame.display.set_mode(game.getScreenDims())
    game.clock = pygame.time.Clock()
    game.init()

    while True:
        dt = game.clock.tick_busy_loop()
        if game.game_over():
            game.reset()
        game.step(dt)
        pygame.display.update()
