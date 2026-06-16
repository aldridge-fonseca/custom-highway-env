import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from custom_highway_env import register_custom_env


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "ID13_Custom_Env_Trained_Model.zip"
NUM_EPISODES = int(os.getenv("EVAL_EPISODES", "500"))

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

if "custom-highway-env-v0" not in gym.envs.registry:
    register_custom_env()


def create_env():
    return gym.make("custom-highway-env-v0", render_mode=None)


model = PPO.load(str(MODEL_PATH))
eval_env = create_env()
eval_rewards = []

print(f"Running CustomHighwayEnv performance test for {NUM_EPISODES} episodes...")
for episode in range(NUM_EPISODES):
    obs, info = eval_env.reset()
    done = truncated = False
    total_reward = 0.0

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        total_reward += reward

    eval_rewards.append(total_reward)

    if (episode + 1) % 100 == 0 or episode + 1 == NUM_EPISODES:
        print(f"Completed {episode + 1}/{NUM_EPISODES} | Mean: {np.mean(eval_rewards):.2f}")

eval_env.close()

mean_reward = float(np.mean(eval_rewards))
std_reward = float(np.std(eval_rewards))
median_reward = float(np.median(eval_rewards))

plt.figure(figsize=(8, 6))
parts = plt.violinplot(eval_rewards, positions=[1], showmeans=True, showmedians=True)
for body in parts["bodies"]:
    body.set_facecolor("steelblue")
    body.set_alpha(0.7)
plt.text(1.25, mean_reward, f"Mean: {mean_reward:.2f}\nStd: {std_reward:.2f}")
plt.xlabel("Custom Highway Env")
plt.ylabel("Episodic Reward (Return)")
plt.title("Performance Test - 500 Episodes (ID 14)")
plt.xticks([1], ["Custom Env"])
plt.grid(True, alpha=0.3, axis="y")
plot_path = PLOTS_DIR / "ID14_Custom_Env_Performance_Test.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

summary_path = PLOTS_DIR / "ID14_Custom_Env_Performance_Summary.md"
summary_path.write_text(
    "# Custom highway environment performance summary\n\n"
    f"- Episodes: {NUM_EPISODES}\n"
    f"- Mean reward: {mean_reward:.4f}\n"
    f"- Standard deviation: {std_reward:.4f}\n"
    f"- Median reward: {median_reward:.4f}\n"
    f"- Minimum reward: {float(np.min(eval_rewards)):.4f}\n"
    f"- Maximum reward: {float(np.max(eval_rewards)):.4f}\n",
    encoding="utf-8",
)
np.save(RESULTS_DIR / "ID14_Custom_Env_Episode_Rewards.npy", np.array(eval_rewards))

print(f"Performance plot saved to {plot_path}")
print(f"Summary saved to {summary_path}")
print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"Median Reward: {median_reward:.2f}")
