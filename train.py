import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Import the MouseMazeEnv class from your module
from MouseMaze import MouseMazeEnv

# Register the environment
gym.register(
    id="MouseMaze-v1",
    entry_point="MouseMaze:MouseMazeEnv",
)

# Create the environment without the 'desc' parameter
env = gym.make("MouseMaze-v1", render_mode="human")

# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1)

# Train the agent and display a progress bar
model.learn(total_timesteps=250000, progress_bar=True)

# Save the agent
model.save("dqn_frozen")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
