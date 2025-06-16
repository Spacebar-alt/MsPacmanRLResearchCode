import os
import numpy as np
import torch
import gymnasium as gym
import wandb
import ale_py

from stable_baselines3 import DQN, PPO
from sb3_contrib import QRDQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from wandb.integration.sb3 import WandbCallback

os.environ["TORCH_ENABLE_AUTO_MIXED_PRECISION"] = "1"
torch.backends.cudnn.benchmark = True
SEED = 42
ENV_ID = "ALE/MsPacman-v5"
TOTAL_TIMESTEPS = 10_000_000

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info
    

def make_env(rank: int):
    def _init():
        env = gym.make(ENV_ID, render_mode=None)
        env = NoopResetEnv(env, noop_max=30)
        env = AtariWrapper(env)
        env = Monitor(env)
        env.reset(seed=SEED + rank)
        return env
    return _init

def train_agent(algo: str):
    algo = algo.lower()
    print(f"\nTraining {algo.upper()}...\n")
    num_envs = 8

    run = wandb.init(
        project="mspacman",
        entity="pranav-mandalapu-university-of-technology-sydney",
        name=f"{algo.upper()}_run_rerun",
        config={
            "algo": algo,
            "timesteps": TOTAL_TIMESTEPS,
            "env_id": ENV_ID,
            "num_envs": num_envs,
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Create training environment
    train_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    train_env = VecFrameStack(train_env, n_stack=4)

    eval_env = VecFrameStack(SubprocVecEnv([make_env(999)]), n_stack=4)

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=50_000,
        best_model_save_path=f"./BestModels/{algo}/",
        log_path=f"./EvalLogs/{algo}/",
        deterministic=True,
        render=False,
        verbose=1
    )

    # Initialize model
    if algo == "dqn":
        model = DQN(
            "CnnPolicy",
            train_env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=100_000,
            batch_size=64,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            target_update_interval=10_000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            tensorboard_log=f"./Tensorboard/{algo}_rerun/",
            seed=SEED
        )
    elif algo == "ppo":
        model = PPO(
            "CnnPolicy",
            train_env,
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=128,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            tensorboard_log=f"./Tensorboard/{algo}_rerun/",
            seed=SEED
        )
    elif algo == "qrdqn":
        model = QRDQN(
            "CnnPolicy",
            train_env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=100_000,
            batch_size=64,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            target_update_interval=10_000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            tensorboard_log=f"./Tensorboard/{algo}_rerun/",
            seed=SEED
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    # Train and log
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[WandbCallback(log="all", verbose=1), eval_callback])

    model_path = f"./Models/{algo}_mspacman_default_rerun"
    model.save(model_path)
    print(f"Saved model to {model_path}")

    train_env.close()
    wandb.finish()
    print(f"Finished training {algo.upper()}\n")


if __name__ == "__main__":
    for algo in ["dqn"]:
        train_agent(algo)