"""Quickstart for the cross-sell Gym environment.

Runs a random, mask-respecting rollout to show the env works end to end, then —
*if* Stable-Baselines3 is installed (``pip install -e .[rl]``) — trains a small
MaskablePPO agent and reports its average reward, so a deep-RL number can sit
next to the bandit baselines from ``python -m rl_nba.run``.

Run it from the repo root::

    python rl_nba/examples/gym_quickstart.py
"""

from __future__ import annotations

import numpy as np

from rl_nba.config import load_config
from rl_nba.environment.gym_env import CrossSellEnv

CONFIG = "config/rl_nba_config.yml"


def random_rollout(env: CrossSellEnv, steps: int = 3000, seed: int = 0) -> float:
    """Mean reward of a policy that offers a uniformly random *eligible* product."""
    rng = np.random.default_rng(seed)
    _, info = env.reset(seed=seed)
    total = 0.0
    for _ in range(steps):
        eligible = np.flatnonzero(info["action_mask"])  # respect the mask
        _, reward, terminated, truncated, info = env.step(int(rng.choice(eligible)))
        total += reward
        if terminated or truncated:
            _, info = env.reset()
    return total / steps


def main() -> None:
    config = load_config(CONFIG)
    env = CrossSellEnv(config=config)
    print(
        f"Env ready: obs {env.observation_space.shape}, "
        f"{env.action_space.n} products, mode={env.mode}"
    )
    print(f"random (mask-respecting) mean reward: {random_rollout(env):,.1f}")

    # -- Optional deep-RL, only if the `rl` extra is installed --------------
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
    except ImportError:
        print("\n(Install `pip install -e .[rl]` to also train MaskablePPO here.)")
        return

    masked = ActionMasker(CrossSellEnv(config=config), lambda e: e.action_masks())
    model = MaskablePPO("MlpPolicy", masked, verbose=0)
    model.learn(total_timesteps=20_000)

    obs, info = masked.reset()
    total, n = 0.0, 3000
    for _ in range(n):
        action, _ = model.predict(
            obs, action_masks=info["action_mask"], deterministic=True
        )
        obs, reward, terminated, truncated, info = masked.step(int(action))
        total += reward
        if terminated or truncated:
            obs, info = masked.reset()
    print(f"MaskablePPO mean reward: {total / n:,.1f}")


if __name__ == "__main__":
    main()
