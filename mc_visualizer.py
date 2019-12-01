import gym
import script
import numpy as np

qtable = np.load("qvalues.npz")
print(list(qtable.keys())[2])
qtable = qtable[list(qtable.keys())[2]][1]
done = False
epsilon = 0.000001
mc_state_space = list(range(40*40))
mc_discretization = script.Discretization(mc_state_space, [np.linspace(-1.2 - epsilon, 0.6, num=41, endpoint=True), np.linspace(-0.07 - epsilon, 0.07, num=41, endpoint=True)])
max_steps = 200

env = gym.make("MountainCar-v0")
env._max_episode_steps = max_steps
rewards = []
for i in range(100):
    total_reward = 0
    the_state = env.reset()
    env.seed(5493)
    for j in range(max_steps):
        env.render()
        discrete_state = mc_discretization(the_state)
        #action = agent.decide(discrete_state)
        action = list(range(4))[np.argmax(qtable[discrete_state, :])]   # Greedy
        next_state, reward, done, info = env.step(action)
        discrete_next_state = mc_discretization(next_state)
        total_reward += reward
        the_state = next_state
        if done:
            break
    rewards.append(total_reward)
    print(total_reward, reward)
import matplotlib.pyplot as plt
plt.hist(rewards)

print(np.mean(rewards), np.std(rewards))
plt.show()
