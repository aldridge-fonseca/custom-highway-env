import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from custom_highway_env import register_custom_env


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = REPO_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
MODEL_PATH = MODELS_DIR / "ID13_Custom_Env_Trained_Model.zip"
REWARD_PATH = RESULTS_DIR / "episode_rewards.npy"
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "300000"))
CHECKPOINT_FREQ = int(os.getenv("CHECKPOINT_FREQ", "50000"))

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

if "custom-highway-env-v0" not in gym.envs.registry:
    register_custom_env()


class RewardLoggerCallback(BaseCallback):
    def __init__(self, save_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_reward = 0.0
        self.save_path = save_path

    def _on_step(self) -> bool:
        self.current_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0
            if len(self.episode_rewards) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                print(f"Episode {len(self.episode_rewards)}: average reward over last 50 = {avg_reward:.2f}")
                np.save(self.save_path, np.array(self.episode_rewards))
        return True


def create_env():
    return gym.make("custom-highway-env-v0", render_mode=None)


def learning_rate_schedule(progress_remaining: float) -> float:
    return 3e-4 * (0.5 + 0.5 * progress_remaining)


policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256]))

env = DummyVecEnv([create_env])
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=learning_rate_schedule,
    n_steps=1024,
    batch_size=128,
    n_epochs=10,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=policy_kwargs,
    verbose=1,
    device="auto",
)

reward_callback = RewardLoggerCallback(REWARD_PATH)
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=str(CHECKPOINT_DIR),
    name_prefix="custom_ppo_checkpoint",
)

print(f"Training CustomHighwayEnv PPO for {TOTAL_TIMESTEPS:,} timesteps...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[reward_callback, checkpoint_callback])
model.save(str(MODEL_PATH))
env.close()

np.save(REWARD_PATH, np.array(reward_callback.episode_rewards))

rewards = reward_callback.episode_rewards
plt.figure(figsize=(10, 6))
if len(rewards) >= 50:
    rolling = np.convolve(rewards, np.ones(50) / 50, mode="valid")
    plt.plot(np.arange(50, len(rewards) + 1), rolling, "b-", linewidth=2, label="Rolling Mean (50 ep)")
plt.plot(rewards, color="lightblue", alpha=0.3, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Mean Episodic Reward (Return)")
plt.title("Custom Highway Environment - Learning Curve (ID 13)")
plt.legend()
plt.grid(True, alpha=0.3)
plot_path = PLOTS_DIR / "ID13_Custom_Env_Learning_Curve.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Model saved to {MODEL_PATH}")
print(f"Learning curve saved to {plot_path}")
