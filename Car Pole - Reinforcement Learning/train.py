import torch
import numpy as np
from environment import create_cartpole_env
from agent import DQNAgent

# Create the CartPole environment and DQN agent
env = create_cartpole_env()
agent = DQNAgent(env)

# Training loop
episodes = 1000
for e in range(episodes):
    state, info = env.reset()  # Get state and info from reset
    print(f"State before conversion: {state}")  # Debugging line to check the state
    print(f"Shape of state: {np.shape(state)}")  # Debugging line to check shape
    state = np.array(state).flatten()  # Ensure it is a flat array (1D)
    
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Explicitly convert terminated and truncated flags to Python bool
        terminated = bool(terminated)
        truncated = bool(truncated)
        
        next_state = np.array(next_state).flatten()  # Flatten next_state too
        
        # Update the done condition to check both terminated and truncated
        done = terminated or truncated
        
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
    
    agent.update_target_model()
    print(f"Episode {e}/{episodes} - Total Reward: {total_reward}")
    if e % 100 == 0:
        torch.save(agent.model.state_dict(), f"models/dqn_model_{e}.pth")
