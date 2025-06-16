import os
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO
from sb3_contrib import QRDQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import ale_py

# Configuration
SEED = 42
ENV_ID = "ALE/MsPacman-v5"
MODEL_DIR = "./Models"
NUM_EPISODES = 5
ALGO = "ppo"  # Change to "dqn" or "qrdqn" as needed

# Algorithm lookup table
MODEL_CLASSES = {
    "dqn": DQN,
    "ppo": PPO,
    "qrdqn": QRDQN
}

# Load model
model_path = os.path.join(MODEL_DIR, f"{ALGO}_mspacman")
if not os.path.exists(model_path + ".zip"):
    raise FileNotFoundError(f"Model not found at {model_path}.zip")

model_class = MODEL_CLASSES.get(ALGO.lower())
if model_class is None:
    raise ValueError(f"Unsupported algorithm: {ALGO}")

print(f"Loading model from {model_path}...")
model = model_class.load(model_path)

# Create evaluation environment
def make_eval_env():
    env = gym.make(ENV_ID, render_mode=None)
    env = AtariWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    env.seed(SEED)
    return env

eval_env = make_eval_env()

# Evaluate the model
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=NUM_EPISODES,
    deterministic=True,
    render=False
)

# Prepare result text
output_text = (
    f"Evaluation Results for {ALGO.upper()} model over {NUM_EPISODES} episodes:\n"
    f"Mean Reward: {mean_reward:.2f}\n"
    f"Std Dev: {std_reward:.2f}\n"
)

print("\n" + output_text)

# Save results to text file (UTF-8 to support emojis)
result_path = os.path.join(MODEL_DIR, f"{ALGO}_eval_results.txt")
with open(result_path, "w", encoding="utf-8") as f:
    f.write(output_text)

print(f"Saved results to {result_path}")
