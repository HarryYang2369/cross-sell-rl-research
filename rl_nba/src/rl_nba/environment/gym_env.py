"""Gymnasium environment wrapping the cross-sell simulator.

Turns :class:`CrossSellSimulator` into a standard Gymnasium ``Env`` so any RL
library (Stable-Baselines3, RLlib, CleanRL) can train on the cross-sell problem
through one interface. It sits on the **same DToC world** the bandits train on,
so the observation includes the journey features whenever ``dtoc.enabled``.

* **Observation** — the customer context vector (``Box``, ``context_dim``).
* **Action** — which product to offer (``Discrete``, one per catalog product).
* **Reward** — the configured ``conversion`` / ``ape`` / ``vnb`` value.

Three flavours plus an off switch, chosen by ``gym.mode`` in the config:

* ``none``    — disabled; constructing the env raises :class:`GymDisabledError`.
* ``bandit``  — every offer is its own one-step episode (``terminated`` each
  step); the faithful contextual-bandit view.
* ``rollout`` — ``gym.steps_per_episode`` offers per episode (``truncated`` at
  the limit, a fresh customer each step); the shape most deep-RL trainers want.
* ``masked``  — like ``rollout`` but eligibility is *hard*-enforced (an illegal
  offer is remapped to a legal one, so it can never happen); pairs with
  ``MaskablePPO``.

Eligibility (never re-offer an owned product) is exposed as an **action mask**,
both in ``info['action_mask']`` and via :meth:`CrossSellEnv.action_masks` for
sb3-contrib's ``MaskablePPO``. In the *soft* modes (``bandit`` / ``rollout``) an
unmasked agent that offers an owned product earns ``gym.illegal_reward``
(default 0) and the step is flagged ``info['illegal']``; in ``masked`` mode the
offer is remapped instead and flagged ``info['remapped']``.

    from rl_nba.config import load_config
    from rl_nba.environment.gym_env import CrossSellEnv

    env = CrossSellEnv(config=load_config("config/rl_nba_config.yml"))
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The Gym environment needs `gymnasium`. Install it with "
        "`pip install gymnasium` (or `pip install -r requirements.txt`)."
    ) from exc

from rl_nba.config import AppConfig, load_config
from rl_nba.run import prepare_experiment

ENV_ID = "rl_nba/CrossSell-v0"


class GymDisabledError(RuntimeError):
    """Raised when the Gym env is built while ``gym.mode: none``."""


class CrossSellEnv(gym.Env):
    """Gymnasium env for insurance cross-sell (see module docstring)."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: AppConfig | None = None,
        config_path: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            if config_path is None:
                raise ValueError("Provide either config or config_path")
            config = load_config(config_path)
        self.config = config

        gym_cfg = config.gym
        if not gym_cfg.enabled:
            raise GymDisabledError(
                "gym.mode is 'none' — the Gym env is disabled. Set gym.mode to "
                "'bandit', 'rollout', or 'masked' to use it."
            )

        prepared = prepare_experiment(config)
        self._sim = prepared.simulator
        self._n_customers = self._sim.n_customers
        self.n_actions = self._sim.n_actions

        self.mode = gym_cfg.mode
        # bandit == one decision per episode; rollout/masked == a batch.
        self.steps_per_episode = 1 if gym_cfg.single_step else gym_cfg.steps_per_episode
        self._hard_mask = gym_cfg.hard_mask
        self.illegal_reward = float(gym_cfg.illegal_reward)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(prepared.context_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        self._rng = np.random.default_rng(config.experiment.seed if seed is None else seed)
        self._customer = 0
        self._elapsed = 0
        self._mask = np.ones(self.n_actions, dtype=bool)

    # -- internal helpers -------------------------------------------------

    def _draw_customer(self) -> None:
        """Sample the next customer and refresh the eligibility mask."""
        self._customer = int(self._rng.integers(0, self._n_customers))
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[self._sim.eligible_actions(self._customer)] = True
        self._mask = mask

    def _obs(self) -> np.ndarray:
        return self._sim.context(self._customer).astype(np.float32)

    def _info(self, **extra: Any) -> dict[str, Any]:
        info: dict[str, Any] = {"action_mask": self._mask.copy(), "customer": self._customer}
        info.update(extra)
        return info

    def action_masks(self) -> np.ndarray:
        """Eligibility mask for the current customer (sb3-contrib calls this)."""
        return self._mask.copy()

    # -- Gymnasium API ----------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._elapsed = 0
        self._draw_customer()
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        legal = bool(self._mask[action])
        remapped = False
        if not legal and self._hard_mask:
            # masked mode: never let an illegal offer through — remap to a legal one.
            action = int(self._rng.choice(np.flatnonzero(self._mask)))
            legal, remapped = True, True

        if legal:
            reward = float(self._sim.step(self._customer, action, self._rng))
            expected = float(self._sim.expected_reward(self._customer, action))
        else:
            reward = self.illegal_reward
            expected = 0.0
        best = float(self._sim.best_expected_reward(self._customer))
        step_info = {
            "converted": bool(legal and reward > 0.0),
            "illegal": not legal,
            "remapped": remapped,
            "expected_reward": expected,
            "best_expected_reward": best,
            "regret": best - expected,
        }

        self._elapsed += 1
        terminated = self.mode == "bandit"
        truncated = (not terminated) and self._elapsed >= self.steps_per_episode

        # Advance to the next customer so the returned observation is the one the
        # next action will act on (in bandit mode this is the terminal obs the
        # caller discards before reset).
        self._draw_customer()
        return self._obs(), reward, terminated, truncated, self._info(**step_info)

    def render(self) -> None:
        return None


def make_env(
    config: AppConfig | None = None,
    config_path: str | None = None,
    seed: int | None = None,
) -> CrossSellEnv:
    """Convenience constructor. ``gym.make(ENV_ID, config=cfg)`` also works."""
    return CrossSellEnv(config=config, config_path=config_path, seed=seed)


# Register so `gymnasium.make("rl_nba/CrossSell-v0", config=cfg)` works. No
# ``max_episode_steps`` — the env ends episodes itself, so gym adds no TimeLimit.
if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="rl_nba.environment.gym_env:CrossSellEnv")
