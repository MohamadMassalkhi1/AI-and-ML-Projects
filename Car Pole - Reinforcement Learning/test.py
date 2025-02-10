import torch
import numpy as np
import gym
from agent import DQNAgent

# Load the trained model
env = gym.make("CartPole-v1")
agent = DQNAgent(env)
agent.model.load_state_dict(torch.load("models/dqn_model.pth"))
agent.model.eval()

# Test the agent
state = env.reset()
state = np.array(state)
done = False
total_reward = 0
while not done:
    action = agent.act(state)
    next_state, reward, done, _, _ = env.step(action)
    state = next_state
    total_reward += reward
    env.render()

print(f"Total Reward: {total_reward}")
env.close()
