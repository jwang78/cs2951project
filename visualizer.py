import gym
import script
import numpy as np

qtable = np.load("qvalues.npz")
print(list(qtable.keys())[0])
qtable = qtable[list(qtable.keys())[0]][1]
#qtable = np.random.normal(size=[16384, 4])
done = False
lunar_state_space = list(range(4**6 * 4))
max_steps = 500
lunar_discretization = script.Discretization(lunar_state_space, [np.array([-np.inf, -0.5, 0, 0.5, np.inf]) for i in range(6)] + [np.array([-0.1, 0.1, 1.5]) for i in range(2)])
env = gym.make("LunarLander-v2")
env._max_episode_steps = max_steps
rewards = []
for i in range(100):
    total_reward = 0
    the_state = env.reset()
    for j in range(max_steps):
        env.render()
        discrete_state = lunar_discretization(the_state)
        #action = agent.decide(discrete_state)
        action = list(range(4))[np.argmax(qtable[discrete_state, :])]   # Greedy
        next_state, reward, done, info = env.step(action)
        discrete_next_state = lunar_discretization(next_state)
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
