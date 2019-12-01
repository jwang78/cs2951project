import gym
import numpy as np
import matplotlib.pyplot as plt
import bisect
import multiprocessing
import collections
import json
import re
def choose_action_softmax(state, actions, Q, temp):
    q_vals = Q[state,:]/temp

    total = np.sum(np.exp(q_vals))
    softmax = np.exp(q_vals)/total

    return np.random.choice(actions, p=softmax)

def choose_action_epsilon_greedy(epsilon,state,actions,Q):
    q_vals = Q[state,:]
    if np.random.random() < epsilon:
        choice = np.random.choice(actions)
    else:
        choice = actions[np.argmax(q_vals)]
    return choice

class Discretization:
    def __init__(self, space, intervals, is_continuous=False):
        self.space = list(space)
        self.intervals = [list(x) for x in intervals]
        self.is_continuous = is_continuous
    def __call__(self, state):
        if self.is_continuous:
            return state
        total = 0
        for i in range(len(state)-1, 0, -1):
            total *= len(self.intervals[i]) - 1
            idx = bisect.bisect_left(self.intervals[i], state[i]) - 1
            if idx < 0:
                raise ValueError("State was below provided minimum %.2f < %.2f" % (state[i], self.intervals[i][0]))
            if state[i] > 1.5*self.intervals[i][-1]:
                print("Warning: value exceeded maximum given", state[i], self.intervals[i][-1])
            total += idx
        return self.space[total]



class Agent:
    def __init__(self, type,
                 state_space,
                 action_space,
                 state_discretization,
                 learning_rate, discount_factor, init_Q, temperature, epsilon):
        #assert type in ['B', 'C', 'R'], 'type should be B,C, or R'
        self.type = type
        self.action_space = list(action_space)
        self.state_space = list(state_space)
        self.init_Q = init_Q
        self.Q = np.zeros((len(self.state_space), len(self.action_space))) + self.init_Q
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.state_discretization = state_discretization
        self.temperature = temperature
        self.epsilon = epsilon
        self.init_epsilon = epsilon
        self.init_learning_rate = learning_rate
    def anneal(self,episode,step):
        self.learning_rate = self.init_learning_rate / (1.0 + (step / 1000.0) * (1 + episode) / 2000.0)
        self.epsilon = self.init_epsilon / (1.0 + (step / 1000.0) * (1 + episode) / 2000.0)
    def decide(self, state):
        discrete_state = self.state_discretization(state)
        if self.type[1] == 'S':
            return choose_action_softmax(discrete_state, self.action_space, self.Q, temp=self.temperature)
        elif self.type[1] == 'E':
            return choose_action_epsilon_greedy(self.epsilon, discrete_state, self.action_space, self.Q)
        elif self.type[1] == 'G':
            return choose_action_epsilon_greedy(0, discrete_state, self.action_space, self.Q)
        else:
            raise ValueError("Type not found", self.type)
    def update(self, current_state, action, reward, next_state, episode, step):
        discrete_next_state = self.state_discretization(next_state)
        discrete_current_state = self.state_discretization(current_state)
        curr_state_idx = self.state_space.index(discrete_current_state)
        next_state_idx = self.state_space.index(discrete_next_state)
        action_idx = self.action_space.index(action)

        if self.type[0] == 'B':
            self.Q[curr_state_idx, action_idx] = ((1-self.learning_rate)*self.Q[curr_state_idx, action_idx] +
            self.learning_rate*(reward + self.discount_factor * np.amax(self.Q[next_state_idx, :]
                                                                        )))
        elif self.type[0] == 'C':
            self.Q[curr_state_idx, action_idx] = ((1-self.learning_rate)*self.Q[curr_state_idx, action_idx] +
            self.learning_rate*(reward + self.discount_factor * (np.amax(self.Q[next_state_idx, :] if curr_state_idx != next_state_idx else self.Q[next_state_idx, action_idx])
                                                                        )))

        elif self.type[0] == 'R':
            beta = np.random.uniform(low=0.0, high=2.0)

            q_value = self.Q[curr_state_idx, action_idx]
            r_value = reward + self.discount_factor*np.amax(self.Q[next_state_idx, :]) - beta * (np.amax(self.Q[curr_state_idx, :]) - q_value)
            self.Q[curr_state_idx, action_idx] += self.learning_rate*(r_value - q_value)
        else:
            raise ValueError("Type not found", self.type)
        if self.type[2] == 'A':
            self.anneal(episode, step)
        elif self.type[2] == 'C':
            pass
        else:
            raise ValueError("Type not found", self.type)

    def reset(self):
        self.Q = self.Q * 0 + self.init_Q
def run_agent(agent, environment, num_episodes, num_test_episodes, max_steps, test_render):
    return train_agent(agent, environment, num_episodes, max_steps, False) + (test_agent(agent, environment, num_test_episodes, max_steps, test_render),)
def train_agent(agent, environment, num_episodes, max_steps, render):
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

            action = agent.decide(the_state)
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            agent.update(the_state, action, reward, next_state, i, j)
            the_state = next_state
            if done:
                break
        #agent.learning_rate *= 0.999
        rewards.append(total_reward)
    return np.array(rewards), agent.Q

def test_agent(agent, environment, num_episodes, max_steps, render):
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

# hyperparams: agent, alpha, gamma, init_Q, temperature, epsilon, discretization
# experiment_params: num_episodes, num_test_episodes, max_steps, num_experiment
def run_multiple_hyperparameters_async(pool, params, callback):
    the_args = []
    params_tracker = []
    for hyperparameters, experiment_parameters in params:
        environment, discretization = experiment_parameters["environment"], hyperparameters["discretization"]
        sample_environment = gym.make(environment)
        state_space = discretization.space
        agent_params = (hyperparameters["agent"],
          state_space,
          list(range(sample_environment.action_space.n)),
          discretization,
          hyperparameters["alpha"],
          hyperparameters["gamma"],
          hyperparameters["init_Q"],
          hyperparameters["temperature"],
          hyperparameters["epsilon"])
        other_params = (environment,
                        experiment_parameters["num_episodes"],
                        experiment_parameters["num_test_episodes"],
                        experiment_parameters["max_steps"],
                        False)
        
        num_experiments = experiment_parameters['num_experiments']
        initialize_q_values = "checkpoint_Q" in experiment_parameters
        if not initialize_q_values:
            args = [(Agent(*agent_params),) + other_params for i in range(num_experiments)]
        else:
            args = [(Agent(
                *(agent_params[:6] + (experiment_parameters["checkpoint_Q"][i],) + agent_params[7:])
                     ),) + other_params for i in range(num_experiments)]
        params_tracker.extend([(hyperparameters, experiment_parameters) for i in range(num_experiments)])
        the_args.extend(args)
    def transform_results(results):
        for arg, (_, Q, _) in zip(args, results):
            arg[0].Q = Q
        raw_rewards = np.array([reward for reward, _, _ in results])
        q_tables = np.array([q_table for _, q_table, _ in results])
        raw_test_rewards = np.array([test_reward for _, _, test_reward in results])
        raw_rewards_dict = collections.defaultdict(list)
        q_tables_dict = collections.defaultdict(list)
        raw_test_rewards_dict = collections.defaultdict(list)
        for (hyperparams, experiment_params), reward, q_table, test_reward in zip(params_tracker, raw_rewards, q_tables, raw_test_rewards):
            parameters = {"environment": experiment_params["environment"],
                          "agent": hyperparams["agent"],
                          "alpha": hyperparams["alpha"],
                          "gamma": hyperparams["gamma"],
                          "init_Q": hyperparams["init_Q"],
                          "temperature": hyperparams["temperature"],
                          "epsilon": hyperparams["epsilon"]}
            name = re.sub(r"\s+", "", json.dumps(parameters, sort_keys=True))
            raw_rewards_dict[name].append(reward)
            q_tables_dict[name].append(q_table)
            raw_test_rewards_dict[name].append(test_reward)
        raw_train_rewards_result = {key: np.array(value) for key, value in raw_rewards_dict.items()}
        q_tables_result = {key: np.array(value) for key, value in q_tables_dict.items()}
        raw_test_rewards_result = {key: np.array(value) for key, value in raw_test_rewards_dict.items()}
        return raw_train_rewards_result, q_tables_result, raw_test_rewards_result
    def result_callback(results):
        callback(*transform_results(results))
    result = pool.starmap_async(run_agent, the_args, callback=result_callback)
    return lambda x: transform_results(result.get(x))
def run(pool, environment, state_space, discretization, alpha, gamma, init_Q, temperature, num_episodes, num_test_episodes, num_experiments):

    max_steps = 200
    q_tables_d = {}
    raw_rewards_d = {}
    raw_test_rewards_d = {}
    the_args = []
    envvv = gym.make(environment)
    for agent in ['B', 'C', 'R']:
        args = [(Agent(agent, state_space, list(range(envvv.action_space.n)), discretization, alpha, gamma, init_Q, temperature, None), environment, num_episodes, num_test_episodes, max_steps, False) for i in range(num_experiments)]
        the_args += args
    #run_agent(*the_args[0])
    all_rewards = pool.starmap(run_agent, the_args)
    for arg, (_, Q, _) in zip(args, all_rewards):
        arg[0].Q = Q
    raw_rewards = np.array([reward for reward, _, _ in all_rewards])
    q_tables = np.array([q_table for _, q_table, _ in all_rewards])
    raw_test_rewards = np.array([test_reward for _, _, test_reward in all_rewards])
    stuffs = collections.defaultdict(list)
    stuffs2 = collections.defaultdict(list)
    stuffs3 = collections.defaultdict(list)
    for arg, reward, q_table, test_reward in zip(the_args, raw_rewards, q_tables, raw_test_rewards):
        stuffs[arg[0].type].append(reward)
        stuffs2[arg[0].type].append(q_table)
        stuffs3[arg[0].type].append(test_reward)
    for agent_type in stuffs:
        parameters = {"environment": environment, "agent": agent_type, "alpha": alpha, "gamma": gamma, "init_Q": init_Q, "max_steps": max_steps, "temperature": temperature, "epsilon": None}
        name = re.sub(r"\s+", "", json.dumps(parameters, sort_keys=True))
        q_tables_d[name] = np.array(stuffs2[agent_type])
        raw_rewards_d[name] = np.array(stuffs[agent_type])
        t= np.array(stuffs3[agent_type])
        raw_test_rewards_d[name] = t
        print("Agent: %s Mean: %.4f Std: %.4f" % (agent_type, np.mean(t), np.std(t)))
    

    return raw_rewards_d, q_tables_d, raw_test_rewards_d


epsilon = 0.00001
mc_state_space = list(range(40*40))
MC_DISCRETIZATION = Discretization(mc_state_space, [np.linspace(-1.2 - epsilon, 0.6, num=41, endpoint=True), np.linspace(-0.07 - epsilon, 0.07, num=41, endpoint=True)])
MC_ENV_NAME = "MountainCar-v0"

ac_state_space = list(range(10*10*10*10*8*8))
AC_DISCRETIZATION = Discretization(ac_state_space, [np.linspace(-1-epsilon,1,11,True),
                                                    np.linspace(-1-epsilon,1,11,True),
                                                    np.linspace(-1-epsilon,1,11,True),
                                                    np.linspace(-1-epsilon,1,11,True),
                                                    np.linspace(-12.566371-epsilon,12.566371,9,True),
                                                    np.linspace(-28.274334-epsilon,28.274334,9,True)])
AC_ENV_NAME = "Acrobot-v1"

lunar_state_space = list(range(4**6 * 4))
# First 6 variables are continuous, discretized into four bins, and last two are binary 0, 1
LUNAR_DISCRETIZATION = Discretization(lunar_state_space, [np.array([-np.inf, -0.5, 0, 0.5, np.inf]) for i in range(6)] + [np.array([-0.1, 0.1, 1.5]) for i in range(2)])
LUNAR_ENV_NAME = "LunarLander-v2"

cartpole_state_space = list(range(8*8*10*10))
CP_DISCRETIZATION = Discretization(cartpole_state_space, [np.linspace(-2.6-epsilon, 2.6, 9, True),
                                                                np.hstack([np.array([-np.inf]), np.linspace(-4.0, 4.0, 9, True), np.array([np.inf])]),
                                                                np.linspace(-0.42, 0.42, 11, True),
                                                                np.linspace(-4.01, 4.01, 11, True)])
CP_ENV_NAME = "CartPole-v1"
def main():
    import time
    x = time.time()
    epsilon = 0.0000001
    num_experiments = 4
    # Mountain car args
    mc_env = MC_ENV_NAME
    mc_state_space = MC_DISCRETIZATION.space
    mc_discretization = MC_DISCRETIZATION
    mc_alpha = 0.01
    mc_gamma = 0.99
    mc_init_Q = -50
    mc_temperature = 1
    mc_num_episodes = 0
    mc_num_test_episodes = 100
    # Uncomment below to run Mountain Car experiment
    # params: agent, alpha, gamma, init_Q, temperature, epsilon, discretization
    # experiment_params: environment, num_episodes, num_test_episodes, max_steps, num_experiment
    params = [({"agent": agent, "alpha": mc_alpha, "gamma": mc_gamma, "init_Q": mc_init_Q, "temperature": mc_temperature, "epsilon": None, "discretization": mc_discretization},
               {"environment": mc_env, "num_episodes": mc_num_episodes, "num_test_episodes": mc_num_test_episodes, "max_steps": 200, "num_experiments": num_experiments}) for agent in ["B", "C", "R"]]
    """
    rewards, q_tables, test_rewards = run(pool,
                                          mc_env,
                                          mc_state_space,
                                          mc_discretization,
                                          mc_alpha,
                                          mc_gamma,
                                          mc_init_Q,
                                          mc_temperature,
                                          mc_num_episodes,
                                          mc_num_test_episodes,
                                          num_experiments)
    """
    promise = run_multiple_hyperparameters_async(pool, params, lambda x, y, z: None)
    rewards, q_tables, test_rewards = promise(None)
    print(np.mean(test_rewards[list(test_rewards.keys())[2]]))

    # Acrobot args
    ac_env = AC_ENV_NAME
    ac_state_space = AC_DISCRETIZATION.space
    ac_discretization = AC_DISCRETIZATION
    ac_alpha = 0.25
    ac_gamma = 0.999
    ac_init_Q = 0
    ac_temperature =1
    ac_num_episodes = 100000
    ac_num_test_episodes = 10000
    # Uncomment below to run Acrobot experiment
    """
    rewards, q_tables, test_rewards = run(pool,
                                          ac_env,
                                          ac_state_space,
                                          ac_discretization,
                                          ac_alpha,
                                          ac_gamma,
                                          ac_init_Q,
                                          ac_temperature,
                                          ac_num_episodes,
                                          ac_num_test_episodes,
                                          num_experiments)
    
    """
    # LunarLander args
    lunar_env = LUNAR_ENV_NAME
    lunar_state_space = LUNAR_DISCRETIZATION.space
    # First 6 variables are continuous, discretized into four bins, and last two are binary 0, 1
    lunar_discretization = LUNAR_DISCRETIZATION

    lunar_alpha = 0.1
    lunar_gamma = 0.99
    lunar_init_Q = 0
    lunar_temperature = 1.5
    lunar_num_episodes = 10000
    lunar_num_test_episodes = 1000
    """
    rewards, q_tables, test_rewards = run(pool,
                                          lunar_env,
                                          lunar_state_space,
                                          lunar_discretization,
                                          lunar_alpha,
                                          lunar_gamma,
                                          lunar_init_Q,
                                          lunar_temperature,
                                          lunar_num_episodes,
                                          lunar_num_test_episodes,
                                          num_experiments)
    """
    # CartPole args
    cartpole_env = CP_ENV_NAME
    cartpole_state_space = CP_DISCRETIZATION.space
    cartpole_discretization = CP_DISCRETIZATION
    cartpole_alpha = 0.1
    cartpole_gamma = 0.99
    cartpole_init_Q = 25
    cartpole_temperature = 2
    cartpole_num_episodes = 2000
    cartpole_num_test_episodes = 100

    # Run CartPole experiment
    """
    rewards, q_tables, test_rewards = run(pool,
                                          cartpole_env,
                                          cartpole_state_space,
                                          cartpole_discretization,
                                          cartpole_alpha,
                                          cartpole_gamma,
                                          cartpole_init_Q,
                                          cartpole_temperature,
                                          cartpole_num_episodes,
                                          cartpole_num_test_episodes,
                                          num_experiments)
    """
    
    print("Time", time.time() - x)
    np.savez_compressed("rewards.npz", **rewards)
    np.savez_compressed("qvalues.npz", **q_tables)
    np.savez_compressed("test_rewards.npz", **test_rewards)


if __name__ == "__main__":
    pool = multiprocessing.Pool(11)
    main()
