import gym

# Create the CartPole environment
def create_cartpole_env():
    env = gym.make("CartPole-v1")
    return env
