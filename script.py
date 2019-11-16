import gym
import numpy as np
import matplotlib.pyplot as plt
import bisect
import multiprocessing


def choose_action_softmax(state, actions, Q, temp=1):
    q_vals = Q[state,:]

    total = np.sum(np.exp(q_vals))
    softmax = np.exp(q_vals)/total

    return np.random.choice(actions, p=softmax)



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
        return self.space[total]



class Agent:
    def __init__(self, type, state_space, action_space, state_discretization, learning_rate, discount_factor, init_Q):
        assert type in ['B', 'C', 'R'], 'type should be B,C, or R'
        self.type = type
        self.action_space = list(action_space)
        self.state_space = list(state_space)
        self.init_Q = init_Q
        self.Q = np.zeros((len(self.state_space), len(self.action_space))) + self.init_Q
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.state_discretization = state_discretization

    def decide(self, state):
        return choose_action_softmax(state, self.action_space, self.Q)
    def update(self, current_state, action, reward, next_state):
        curr_state_idx = self.state_space.index(current_state)
        next_state_idx = self.state_space.index(next_state)
        action_idx = self.action_space.index(action)

        if self.type == 'B':
            self.Q[curr_state_idx, action_idx] = ((1-self.learning_rate)*self.Q[curr_state_idx, action_idx] +
            self.learning_rate*(reward + self.discount_factor * np.amax(self.Q[next_state_idx, :]
                                                                        )))
        elif self.type == 'C':
            self.Q[curr_state_idx, action_idx] = ((1-self.learning_rate)*self.Q[curr_state_idx, action_idx] +
            self.learning_rate*(reward + self.discount_factor * (np.amax(self.Q[next_state_idx, :] if curr_state_idx != next_state_idx else self.Q[next_state_idx, action_idx])
                                                                        )))

        elif self.type == 'R':
            beta = np.random.uniform(low=0.0, high=2.0)

            q_value = self.Q[curr_state_idx, action_idx]
            r_value = reward + self.discount_factor*np.amax(self.Q[next_state_idx, :]) - beta * (np.amax(self.Q[curr_state_idx, :]) - q_value)
            self.Q[curr_state_idx, action_idx] += self.learning_rate*(r_value - q_value)


    def reset(self):
        self.Q = self.Q * 0 + self.init_Q

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
        #agent.learning_rate *= 0.999
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
            #action = agent.decide(discrete_state)
            action = agent.action_space[np.argmax(agent.Q[discrete_state, :])]   # Greedy
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



def run(environment, state_space, discretization, test, alpha, gamma, init_Q):

    max_steps = 200
    num_episodes = 10000
    num_test_episodes = 1000
    moving_average_length = 1000
    num_experiments = 4


    for agent in ['Bellman', 'Consistent', 'RSO']:
        args = [(Agent(agent[0], state_space, list(range(3)), discretization, alpha, gamma, init_Q), environment, num_episodes, max_steps, False) for i in range(num_experiments)]
        all_rewards = pool.starmap(run_agent, args)
        for arg, (_, Q) in zip(args, all_rewards):
            arg[0].Q = Q
        q_table = all_rewards[0][1]
        all_rewards = [moving_average(reward, moving_average_length) for reward, _ in all_rewards]
        all_rewards = np.array(all_rewards)

        if test:
            test_args = [arg[0:2] + (num_test_episodes, ) + arg[3:] for arg in args]
            #test_args[-1] = test_args[-1][:-1] + (True,)
            test_rewards = np.array(pool.starmap(test_agent, test_args))
            print(agent, "Test", -np.mean(test_rewards), 'std:', np.std(test_rewards))

        x = np.arange(all_rewards.shape[1])
        plt.plot(x, -np.mean(all_rewards, axis=0), label=agent)

    plt.legend()
    plt.show()





def main():

    epsilon = 0.0000001

    # Mountain car args
    mc_env = 'MountainCar-v0'
    mc_state_space = list(range(40*40))
    mc_discretization = Discretization(mc_state_space, [np.linspace(-1.2 - epsilon, 0.6, num=40, endpoint=False), np.linspace(-0.07 - epsilon, 0.07, num=40, endpoint=False)])
    mc_test = True
    mc_alpha = 0.01
    mc_gamma = 0.99
    mc_init_Q = -50

    # Acrobot args
    ac_env = 'Acrobot-v1'
    ac_state_space = list(range(10*10*10*10*8*8))
    ac_discretization = Discretization(ac_state_space, [np.linspace(-1-epsilon,1,10,False),np.linspace(-1-epsilon,1,10,False),np.linspace(-1-epsilon,1,10,False),np.linspace(-1-epsilon,1,10,False),
                        np.linspace(-12.566371-epsilon,12.566371,8,False), np.linspace(-28.274334-epsilon,28.274334,8,False)])
    ac_test = False
    ac_alpha = 0.25
    ac_gamma = 0.999
    ac_init_Q = 0

    # Uncomment below to run Mountain Car experiment
    run(mc_env, mc_state_space, mc_discretization, mc_test, mc_alpha, mc_gamma, mc_init_Q)


    # Uncomment below to run Acrobot experiment
    #run(ac_env, ac_state_space, ac_discretization, ac_test, ac_alpha, ac_gamma, ac_init_Q)


if __name__ == "__main__":
    pool = multiprocessing.Pool(12)
    main()