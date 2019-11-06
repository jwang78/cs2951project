import gym
import numpy as np
import matplotlib.pyplot as plt
import bisect
import multiprocessing
class Discretization:
    def __init__(self, space, intervals, is_continuous=False):
        self.space = space
        self.intervals = intervals
        self.is_continuous = is_continuous
    def __call__(self, state):
        if self.is_continuous:
            return state
        total = 0
        for i in range(len(state)-1, 0, -1):
            total *= len(self.intervals[i])
            idx = bisect.bisect_left(self.intervals[i].tolist(), state[i]) - 1
            if idx < 0:
                raise ValueError("State was below provided minimum %.2f < %.2f" % (state[i], self.intervals[i]))
            if state[i] > 1.5*self.intervals[i][-1]:
                print("Warning: value exceeded maximum given", state[i], self.intervals[i][-1])
            total += idx
        #print(total, self.space)
        return self.space[total]
class QRSOAgent:
    def __init__(self, state_space, action_space, state_discretization, learning_rate=0.1, discount_factor=1):
        self.action_space = list(action_space)
        self.state_space = list(state_space)
        self.Q = np.zeros((len(self.state_space), len(self.action_space)))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.state_discretization = state_discretization
    def decide(self, state):
        return self.action_space[np.argmax(self.Q[state, :])]
    def update(self, current_state, action, reward, next_state):
        beta = np.random.uniform(low=0.0, high=2.0)
        
        curr_state_idx = self.state_space.index(current_state)
        next_state_idx = self.state_space.index(next_state)
        action_idx = self.action_space.index(action)
        q_value = self.Q[curr_state_idx, action_idx]
        r_value = reward + self.discount_factor*np.amax(self.Q[next_state_idx, :]) - beta * (np.amax(self.Q[curr_state_idx, :]) - q_value)
        self.Q[curr_state_idx, action_idx] += self.learning_rate*(r_value - q_value)
    def reset(self):
        self.Q = self.Q * 0
class QAgent:
    def __init__(self, state_space, action_space, state_discretization, learning_rate=0.1, discount_factor=1):
        self.action_space = list(action_space)
        self.state_space = list(state_space)
        self.Q = np.zeros((len(self.state_space), len(self.action_space)))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.state_discretization = state_discretization
    def decide(self, state):
        return self.action_space[np.argmax(self.Q[state, :])]
    def update(self, current_state, action, reward, next_state):
        curr_state_idx = self.state_space.index(current_state)
        next_state_idx = self.state_space.index(next_state)
        action_idx = self.action_space.index(action)
        self.Q[curr_state_idx, action_idx] = ((1-self.learning_rate)*self.Q[curr_state_idx, action_idx] +
        self.learning_rate*(reward + self.discount_factor * np.amax(self.Q[next_state_idx, :] 
                                                                    )))
    def reset(self):
        self.Q = self.Q * 0
class QConsistentAgent:
    def __init__(self, state_space, action_space, state_discretization, learning_rate=0.1, discount_factor=1):
        self.action_space = list(action_space)
        self.state_space = list(state_space)
        self.Q = np.zeros((len(self.state_space), len(self.action_space)))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.state_discretization = state_discretization
        self.reset()
    def decide(self, state):
        return self.action_space[np.argmax(self.Q[state, :])]
    def update(self, current_state, action, reward, next_state):
        curr_state_idx = self.state_space.index(current_state)
        next_state_idx = self.state_space.index(next_state)
        action_idx = self.action_space.index(action)
        self.Q[curr_state_idx, action_idx] = ((1-self.learning_rate)*self.Q[curr_state_idx, action_idx] +
        self.learning_rate*(reward + self.discount_factor * (np.amax(self.Q[next_state_idx, :] if curr_state_idx != next_state_idx else self.Q[next_state_idx, action_idx])
                                                                    )))
    def reset(self):
        self.Q = self.Q*0
def run_agent(agent, environment, num_episodes=100, max_steps=1000, render=True):
    env = gym.make(environment)
    env._max_episode_steps = max_steps
    rewards = []
    agent.reset()
    for i in range(num_episodes):
        the_state = env.reset()
        total_reward = 0
        for j in range(max_steps):
            if render and i == num_episodes - 1:
                env.render()
            discrete_state = agent.state_discretization(the_state)
            action = agent.decide(discrete_state)
            next_state, reward, done, info = env.step(action)
            discrete_next_state = agent.state_discretization(next_state)
            total_reward += reward
            agent.update(discrete_state, action, reward, discrete_next_state)
            the_state = next_state
            if done:
                break
        agent.learning_rate *= 0.999
        rewards.append(total_reward)
    return np.array(rewards), agent.Q
def test_agent(agent, environment, num_episodes=100, max_steps=1000, render=False):
    env = gym.make(environment)
    env._max_episode_steps = max_steps
    rewards = []
    for i in range(num_episodes):
        the_state = env.reset()
        total_reward = 0
        for j in range(max_steps):
            if render:
                env.render()
            discrete_state = agent.state_discretization(the_state)
            action = agent.decide(discrete_state)
            next_state, reward, done, info = env.step(action)
            discrete_next_state = agent.state_discretization(next_state)
            total_reward += reward
            the_state = next_state
            if done:
                break
        rewards.append(total_reward)
    return np.array(rewards)
def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
def run_stuff_mountain_car():
    state_space = list(range(40*40))
    epsilon = 0.0000001
    max_steps = 200
    num_episodes = 10000
    num_test_episodes = 1000
    moving_average_length = 500
    num_experiments = 20
    learning_rate = 0.1
    discount_factor = 0.999
    mc_discretization = Discretization(state_space, [np.linspace(-1.2 - epsilon, 0.6, num=40, endpoint=False), np.linspace(-0.07 - epsilon, 0.07, num=40, endpoint=False)])
    for environment, discretization in [('MountainCar-v0', mc_discretization)]:
        args = [(QAgent(state_space, list(range(3)), discretization, learning_rate, discount_factor), environment, num_episodes, max_steps, False) for i in range(num_experiments)]
        all_rewards = pool.starmap(run_agent, args)
        for arg, (_, Q) in zip(args, all_rewards):
            arg[0].Q = Q
        q_table = all_rewards[0][1]
        all_rewards = [moving_average(reward, moving_average_length) for reward, _ in all_rewards]
        all_rewards = np.array(all_rewards)
        consistent_args = [(QConsistentAgent(state_space, list(range(3)), discretization, learning_rate, discount_factor), environment, num_episodes, max_steps, False) for i in range(num_experiments)]
        consistent_all_rewards = pool.starmap(run_agent, consistent_args)
        for arg, (_, Q) in zip(consistent_args, consistent_all_rewards):
            arg[0].Q = Q
        consistent_all_rewards = [moving_average(reward, moving_average_length) for reward, _ in consistent_all_rewards]
        rso_args = [(QRSOAgent(state_space, list(range(3)), discretization, learning_rate, discount_factor), environment, num_episodes, max_steps, False) for i in range(num_experiments)]
        rso_rewards = pool.starmap(run_agent, rso_args)
        for arg, (_, Q) in zip(rso_args, rso_rewards):
            arg[0].Q = Q
        rso_rewards = [moving_average(reward, moving_average_length) for reward, _ in rso_rewards]
        print("Testing...")

        rso_test_args = [arg[0:2] + (num_test_episodes, ) + arg[3:] for arg in rso_args]
        consistent_test_args = [arg[0:2] + (num_test_episodes, ) + arg[3:] for arg in consistent_args]
        q_test_args = [arg[0:2] + (num_test_episodes, ) + arg[3:] for arg in args]
        rso_test_rewards = np.array(pool.starmap(test_agent, rso_test_args))
        consistent_test_rewards = np.array(pool.starmap(test_agent, consistent_test_args))
        q_test_rewards = np.array(pool.starmap(test_agent, q_test_args))
        #print(np.mean(rso_test_rewards, axis=0))
        print(np.linalg.norm(q_test_args[0][0].Q), np.linalg.norm(q_table))
        print("RSO Test", np.mean(rso_test_rewards))
        print("Consistent Test", np.mean(consistent_test_rewards))
        print("Bellman Test", np.mean(q_test_rewards))
        x = np.arange(all_rewards.shape[1])
        plt.plot(x, -np.mean(all_rewards, axis=0), label="Bellman")
        plt.plot(x, -np.mean(consistent_all_rewards, axis=0), label="Consistent Bellman")
        plt.plot(x, -np.mean(rso_rewards, axis=0), label="RSO")
        plt.legend()
        
        plt.show()
    agent = QAgent(state_space, list(range(3)), discretization)
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = max_steps
    
    while True and False:
        total_reward = 0
        the_state = env.reset()
        agent.Q = np.array(q_table)
        for i in range(1000):
            env.render()
            discrete_state = agent.state_discretization(the_state)
            action = agent.decide(discrete_state)
            next_state, reward, done, info = env.step(action)
            discrete_next_state = agent.state_discretization(next_state)
            total_reward += reward
            agent.update(discrete_state, action, reward, discrete_next_state)
            the_state = next_state
            if done:
                break
def main():
    run_stuff_mountain_car()
    
##    state_space = list(range(40*40))
##    epsilon = 0.0000001
##    max_steps = 200
##    num_episodes = 3000
##    environment = 'MountainCar-v0'
##    discretization = Discretization(state_space, [np.linspace(-1.2 - epsilon, 0.6, num=40, endpoint=False), np.linspace(-0.07 - epsilon, 0.07, num=40, endpoint=False)])
##    args = (QConsistentAgent(state_space, list(range(3)), discretization, 0.5, 0.99), environment, num_episodes, max_steps, True)
##    run_agent(*args)
##    test_agent(*args)
##    print(args[0].Q)
    
if __name__ == "__main__":
    pool = multiprocessing.Pool(12)
    main()
