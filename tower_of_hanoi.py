from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import wrappers


PEG_COUNT = 3

# Action map: directed peg moves
MOVE_MAP: List[Tuple[int, int]] = [
    (0, 1),  # 0: A -> B
    (0, 2),  # 1: A -> C
    (1, 0),  # 2: B -> A
    (1, 2),  # 3: B -> C
    (2, 0),  # 4: C -> A
    (2, 1),  # 5: C -> B
]


class HanoiEnv(AECEnv):
    """
    Tower of Hanoi (single-agent AEC environment).

    Observation (dict):
        observation:  (3, num_disks) int8 tower representation
        action_mask:  (6,) binary mask for legal actions

    Action space:
        Discrete(6) — directed peg moves defined by MOVE_MAP

    Rewards:
        -0.3   per step after exceeding optimal solution length
        -1.0   illegal action (no state change)
        +10.0  solving the puzzle

    Termination:
        - when puzzle is solved

    Truncation:
        - when max_cycles is reached (optional)
    """

    metadata = {
        "name": "hanoi_v0",
        "render_modes": ["human"],
        "is_parallelizable": False,
        "render_fps": 30,
    }

    def __init__(
        self,
        num_disks: int = 3,
        render_mode: Optional[str] = None,
        window_size: Tuple[int, int] = (640, 480),
        max_cycles: Optional[int] = None,
        manual_control: bool = False,
    ):
        super().__init__()

        assert num_disks >= 1, "num_disks must be >= 1"
        self.num_disks = int(num_disks)
        self.render_mode = render_mode
        self.window_size = window_size
        self.manual_control = manual_control

        optimal_steps = (2 ** self.num_disks) - 1
        self.max_cycles = max_cycles or (4 * optimal_steps)

        # --- single-agent bookkeeping ---------------------------------------
        self.possible_agents = ["player_0"]

        self.observation_spaces = {
            "player_0": spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0,
                        high=self.num_disks,
                        shape=(3, self.num_disks),
                        dtype=np.int8,
                    ),
                    "action_mask": spaces.MultiBinary(6),
                }
            )
        }

        self.action_spaces = {
            "player_0": spaces.Discrete(6)
        }

        # --- internal state ---------------------------------------------------
        self.state = np.zeros(self.num_disks, dtype=np.int8)
        self.steps = 0

        # RNG
        self.np_random, _ = seeding.np_random(None)

        # rendering
        self._pygame_inited = False
        self._screen = None
        self._clock = None
        self._surf = None
        self._selected_src = None

    # ------------------------------------------------------------------------
    # AEC API
    # ------------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        self.agents = self.possible_agents[:]
        self.agent_selection = self.agents[0]

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.state[:] = 0
        self.steps = 0
        self._selected_src = None

        if self.render_mode == "human":
            self._init_pygame()
            self.render()

    def step(self, action: int):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._clear_rewards()
        reward = 0.0

        optimal_steps = (2 ** self.num_disks) - 1
        legal_mask = self._legal_action_mask()

        # Illegal move
        if not legal_mask[action]:
            reward -= 1.0
            self.infos[agent]["illegal_action"] = True
        else:
            src, dst = MOVE_MAP[action]
            self._move(src, dst)

        self.steps += 1

        # Penalize only after exceeding optimal steps
        if self.steps > optimal_steps:
            reward -= 0.3

        # Check for success
        if self._is_solved():
            reward += 10.0
            self.terminations[agent] = True

        # Truncation
        if self.steps >= self.max_cycles and not self._is_solved():
            self.truncations[agent] = True

        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward

        if self.render_mode == "human":
            self.render()

    def observe(self, agent: str):
        return {
            "observation": self._get_tower_obs(),
            "action_mask": self._legal_action_mask().astype(np.int8),
        }

    def close(self):
        if self._pygame_inited:
            pygame.display.quit()
            pygame.quit()
            self._pygame_inited = False

    # ------------------------------------------------------------------------
    # Core game logic
    # ------------------------------------------------------------------------
    def _tops(self) -> List[Optional[int]]:
        tops = []
        for p in range(PEG_COUNT):
            disks = np.where(self.state == p)[0]
            tops.append(int(disks.min()) if disks.size else None)
        return tops

    def _legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(6, dtype=np.bool_)
        tops = self._tops()
        for i, (src, dst) in enumerate(MOVE_MAP):
            s, d = tops[src], tops[dst]
            mask[i] = s is not None and (d is None or s < d)
        return mask

    def _move(self, src: int, dst: int):
        top = int(np.where(self.state == src)[0].min())
        self.state[top] = dst

    def _is_solved(self) -> bool:
        return bool(np.all(self.state == PEG_COUNT - 1))

    def _get_tower_obs(self):
        towers = np.zeros((3, self.num_disks), dtype=np.int8)
        for peg in range(3):
            disks = sorted(np.where(self.state == peg)[0], reverse=True)
            for level, disk in enumerate(disks):
                towers[peg, level] = disk + 1
        return towers

    # ------------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return
        if self.render_mode == "human":
            self._render_pygame()

    def _init_pygame(self):
        if self._pygame_inited:
            return
        pygame.init()
        pygame.display.set_caption("Tower of Hanoi (PettingZoo)")
        self._screen = pygame.display.set_mode(self.window_size)
        self._clock = pygame.time.Clock()
        self._pygame_inited = True

    def _draw_surface(self):
        if self._surf is None:
            self._surf = pygame.Surface(self.window_size)

        W, H = self.window_size
        self._surf.fill((245, 245, 245))

        margin = 60
        base_y = H - 60
        peg_x = [int(W / 6), int(W / 2), int(5 * W / 6)]
        peg_h = int(H * 0.65)

        pygame.draw.rect(self._surf, (40, 40, 40), (margin, base_y, W - 2 * margin, 6))

        for i, x in enumerate(peg_x):
            color = (200, 160, 0) if self._selected_src == i else (80, 80, 80)
            pygame.draw.rect(
                self._surf,
                color,
                (x - 6, base_y - peg_h, 12, peg_h),
                border_radius=4,
            )

        max_w = int((W - 2 * margin) / 3 * 0.9)
        min_w = int(max_w * 0.3)
        disk_h = max(12, min(28, int((peg_h - 40) / self.num_disks)))

        for p in range(3):
            disks = sorted(np.where(self.state == p)[0], reverse=True)
            for i, d in enumerate(disks):
                w = int(min_w + (max_w - min_w) * (d + 1) / self.num_disks)
                x = peg_x[p] - w // 2
                y = base_y - (i + 1) * (disk_h + 4)
                c = (60 + 30 * d, 120 + 20 * d, 160 + 10 * d)
                pygame.draw.rect(self._surf, c, (x, y, w, disk_h), border_radius=6)

        font = pygame.font.SysFont(None, 22)
        txt = f"steps: {self.steps} | solved: {self._is_solved()}"
        self._surf.blit(font.render(txt, True, (20, 20, 20)), (10, 10))

        return self._surf

    def _render_pygame(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            if self.manual_control and event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_click(event.pos)

        self._screen.blit(self._draw_surface(), (0, 0))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def _handle_mouse_click(self, pos: Tuple[int, int]):
        W, _ = self.window_size
        peg_x = [int(W / 6), int(W / 2), int(5 * W / 6)]

        peg = int(np.argmin([abs(pos[0] - x) for x in peg_x]))

        if self._selected_src is None:
            if self._tops()[peg] is not None:
                self._selected_src = peg
        else:
            src = self._selected_src
            self._selected_src = None

            for a, (s, d) in enumerate(MOVE_MAP):
                if s == src and d == peg and self._legal_action_mask()[a]:
                    self.step(a)
                    break


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------
def raw_env(**kwargs) -> HanoiEnv:
    return HanoiEnv(**kwargs)


def env(**kwargs):
    base = raw_env(**kwargs)
    return wrappers.OrderEnforcingWrapper(base)
