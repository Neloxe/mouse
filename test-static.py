import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from MouseMaze_RND import MouseMazeEnv

gym.register(
    id="MouseMaze-v1",
    entry_point="MouseMaze_STATIC:MouseMazeEnv",
)

print("Creating environment...")
env = gym.make("MouseMaze-v1", render_mode="human")
print("Environment created.")

print("Loading model...")
model = DQN.load("dqn_frozen_static", env=env)
print("Model loaded.")

print("Evaluating policy...")
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
print("Starting the game loop...")
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
    if dones:
        print(f"Episode finished after {i+1} steps.")
print("Game loop finished.")
