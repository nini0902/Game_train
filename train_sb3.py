import os
import time

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from game import HighwayPenaltyWrapper


TOTAL_TIMESTEPS = 100_000
RENDER_EVERY_N_STEPS = 1
RENDER_SLEEP_SECONDS = 0.03
MODEL_PATH = "highway_dqn"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FREQ = 1000


class RenderCallback(BaseCallback):
    def __init__(self, render_every_n_steps: int = 1, sleep_seconds: float = 0.03):
        super().__init__()
        self.render_every_n_steps = render_every_n_steps
        self.sleep_seconds = sleep_seconds

    def _on_step(self) -> bool:
        if self.n_calls % self.render_every_n_steps == 0:
            try:
                self.training_env.env_method("render")
            except Exception:
                pass
            time.sleep(self.sleep_seconds)
        return True


def make_env():
    def _init():
        env = gym.make(
            "highway-v0",
            render_mode="human",
            config={"observation": {"type": "Kinematics", "normalize": False}},
        )
        env = HighwayPenaltyWrapper(env)
        env = Monitor(env)
        return env

    return _init


def main():
    env = DummyVecEnv([make_env()])

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
    )

    callback = RenderCallback(
        render_every_n_steps=RENDER_EVERY_N_STEPS,
        sleep_seconds=RENDER_SLEEP_SECONDS,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="highway_dqn",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callbacks = CallbackList([callback, checkpoint_callback])

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
    model.save(MODEL_PATH)
    env.close()


if __name__ == "__main__":
    main()
