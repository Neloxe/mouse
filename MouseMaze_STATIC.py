from os import path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding


class MouseMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 5}

    hole_positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]

    def __init__(self, render_mode=None, size=8, slippery=False, epsilon=0.1):
        super(MouseMazeEnv, self).__init__()

        self.size = size
        self.slippery = slippery
        self.render_mode = render_mode
        self.epsilon = epsilon

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(size, size), dtype=np.int32
        )

        self.state = None
        self.goal_position = (size - 1, size - 1)
        self.start_position = (0, 0)

        self.lastaction = None
        self.path_length = 0
        self.optimal_path_length = 4 * size  # Adjust based on grid size

        self.reset()

        self.window_size = (min(64 * size, 512), min(64 * size, 512))
        self.cell_size = (self.window_size[0] // size, self.window_size[1] // size)
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.goal_img = None
        self.start_img = None
        self.elf_images = None

        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_position
        self.np_random, _ = seeding.np_random(seed)
        self.lastaction = None
        self.path_length = 0
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        if np.random.rand() < self.epsilon:
            action = self.action_space.sample()  # Explore: random action

        next_state = self._get_next_state(self.state, action)
        self.path_length += 1

        if self.path_length > self.optimal_path_length:
            reward = -0.5  # Reduced penalty for exceeding optimal path length
            terminated = True
        elif next_state in self.hole_positions:
            reward = -1
            terminated = True
        elif next_state == self.goal_position:
            reward = 10  # Increased reward for reaching the goal
            terminated = True
        else:
            reward = -0.05  # Reduced penalty for each step
            terminated = False

        self.state = next_state
        self.lastaction = action
        observation = self._get_observation()
        info = {}

        return observation, reward, terminated, False, info

    def _get_next_state(self, state, action):
        row, col = state
        if action == 0:
            col = max(col - 1, 0)
        elif action == 1:
            row = min(row + 1, self.size - 1)
        elif action == 2:
            col = min(col + 1, self.size - 1)
        elif action == 3:
            row = max(row - 1, 0)

        if self.slippery:
            possible_states = [(row, col)]
            if row > 0:
                possible_states.append((row - 1, col))
            if row < self.size - 1:
                possible_states.append((row + 1, col))
            if col > 0:
                possible_states.append((row, col - 1))
            if col < self.size - 1:
                possible_states.append((row, col + 1))
            next_state = tuple(self.np_random.choice(possible_states))
        else:
            next_state = (row, col)

        return next_state

    def _get_observation(self):
        observation = np.zeros((self.size, self.size), dtype=np.int32)
        observation[self.state] = 1  # Agent's position
        observation[self.goal_position] = 2  # Goal's position
        for hole in self.hole_positions:
            observation[hole] = -1  # Holes' positions
        return observation

    def render(self):
        if self.render_mode == "human":
            return self._render_gui()
        elif self.render_mode == "ansi":
            return self._render_text()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def _render_gui(self):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled("pygame is not installed")

        if self.window_surface is None:
            pygame.init()
            self.window_surface = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("MouseMaze")
            print("Pygame window initialized.")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.hole_img is None:
            self.hole_img = pygame.transform.scale(
                pygame.image.load(path.join(path.dirname(__file__), "img/hole.png")),
                self.cell_size,
            )
        if self.cracked_hole_img is None:
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(
                    path.join(path.dirname(__file__), "img/cracked_hole.png")
                ),
                self.cell_size,
            )
        if self.ice_img is None:
            self.ice_img = pygame.transform.scale(
                pygame.image.load(path.join(path.dirname(__file__), "img/ice.png")),
                self.cell_size,
            )
        if self.goal_img is None:
            self.goal_img = pygame.transform.scale(
                pygame.image.load(path.join(path.dirname(__file__), "img/goal.png")),
                self.cell_size,
            )
        if self.start_img is None:
            self.start_img = pygame.transform.scale(
                pygame.image.load(path.join(path.dirname(__file__), "img/stool.png")),
                self.cell_size,
            )
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "img/elf_left.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_up.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]

        self.window_surface.fill((255, 255, 255))

        for row in range(self.size):
            for col in range(self.size):
                pos = (col * self.cell_size[0], row * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if (row, col) in self.hole_positions:
                    self.window_surface.blit(self.hole_img, pos)
                elif (row, col) == self.goal_position:
                    self.window_surface.blit(self.goal_img, pos)
                elif (row, col) == self.start_position:
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        bot_row, bot_col = self.state
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        elf_img = self.elf_images[self.lastaction if self.lastaction is not None else 1]

        if (bot_row, bot_col) in self.hole_positions:
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        else:
            self.window_surface.blit(elf_img, cell_rect)

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def _render_text(self):
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[self.state] = "A"
        grid[self.goal_position] = "G"
        for hole in self.hole_positions:
            grid[hole] = "H"

        text = "\n".join(" ".join(row) for row in grid)
        return text

    def _render_rgb_array(self):
        rgb_array = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        rgb_array[self.state] = [0, 0, 255]
        rgb_array[self.goal_position] = [0, 255, 0]
        for hole in self.hole_positions:
            rgb_array[hole] = [255, 0, 0]

        return rgb_array

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            print("Pygame window closed.")
