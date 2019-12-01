from script import Discretization, run
import numpy as np
import multiprocessing
def main():
    import time
    x = time.time()
    epsilon = 0.0000001
    num_experiments = 20
    
    # LunarLander args
    lunar_env = 'LunarLander-v2'
    lunar_state_space = list(range(4**6 * 4))
    
    # First 6 variables are continuous, discretized into four bins, and last two are binary 0, 1
    lunar_discretization = Discretization(lunar_state_space, [np.array([-np.inf, -0.5, 0, 0.5, np.inf]) for i in range(6)] + [np.array([-0.1, 0.1, 1.5]) for i in range(2)])

    lunar_alpha = 0.1
    lunar_gamma = 0.99
    lunar_init_Q = 0
    lunar_temperature = 1.5
    lunar_num_episodes = 10000
    lunar_num_test_episodes = 1000

    # Run LunarLander experiment
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

    print("Time", time.time() - x)
    
    np.savez_compressed("ll_rewards.npz", **rewards)
    np.savez_compressed("ll_qvalues.npz", **q_tables)
    np.savez_compressed("ll_test_rewards.npz", **test_rewards)
if __name__ == "__main__":
    # Change this to num_cpus * 2 on GCP
    pool = multiprocessing.Pool(1)
    main()
