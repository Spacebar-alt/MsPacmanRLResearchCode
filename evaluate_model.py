import os
import numpy as np
import torch
import gymnasium as gym
from tqdm import trange

from stable_baselines3 import DQN, PPO
from sb3_contrib import QRDQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import ale_py

# Configuration
SEED = 42
ENV_ID = "ALE/MsPacman-v5"
ALGO = "qrdqn"  # Change to "ppo" or "qrdqn" as needed
MODEL_PATH = f"./Models/{ALGO}_mspacman_default_rerun"
N_EVAL_EPISODES = 1000

# Make evaluation environment
def make_eval_env():
    def _init():
        env = gym.make(ENV_ID, render_mode=None)
        env = AtariWrapper(env)
        env = Monitor(env)
        env.reset(seed=SEED + 999)
        return env
    env = DummyVecEnv([_init])
    env.seed(SEED + 999)
    env = VecFrameStack(env, n_stack=4)
    return env

# Load model
def load_model(algo_name: str, path: str, env):
    algo_name = algo_name.lower()
    if algo_name == "dqn":
        return DQN.load(path, env=env)
    elif algo_name == "ppo":
        return PPO.load(path, env=env)
    elif algo_name == "qrdqn":
        return QRDQN.load(path, env=env)
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

# Custom evaluation to record rewards and episode lengths
def evaluate_with_lengths(model, env, n_episodes=10):
    episode_rewards = []
    episode_lengths = []

    with torch.no_grad():
        for _ in trange(n_episodes, desc="Evaluating Episodes"):
            obs = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward[0]  # reward is a vector (1 value per env)
                steps += 1

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

    return episode_rewards, episode_lengths

# Main
def main():
    print(f"\nEvaluating {ALGO.upper()} model...\n")
    eval_env = make_eval_env()

    if not os.path.exists(MODEL_PATH + ".zip"):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}.zip")

    model = load_model(ALGO, MODEL_PATH, eval_env)
    rewards, lengths = evaluate_with_lengths(model, eval_env, n_episodes=N_EVAL_EPISODES)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)

    print(f"\n Final Evaluation Results for {ALGO.upper()}:")
    print(f" - Mean Reward:         {mean_reward:.2f}")
    print(f" - Std Dev Reward:      {std_reward:.2f}")
    print(f" - Mean Episode Length: {mean_length:.2f} steps")
    print(f" - Std Dev Length:      {std_length:.2f} steps\n")

    print("Sample Episode Results (first 20 shown):")
    for i, (r, l) in enumerate(zip(rewards[:20], lengths[:20])):
        print(f" - Episode {i+1:3d}: Reward = {r:.2f}, Length = {l} steps")

    # Optionally save results
    # np.savez(f"{ALGO}_eval_results.npz", rewards=rewards, lengths=lengths)

    eval_env.close()

if __name__ == "__main__":
    main()
