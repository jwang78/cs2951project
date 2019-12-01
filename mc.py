from script import Discretization, run
import numpy as np
import multiprocessing
def main():
    import time
    x = time.time()
    epsilon = 0.0000001
    num_experiments = 20
    
    # Mountain car args
    mc_env = 'MountainCar-v0'
    mc_state_space = list(range(40*40))
    mc_discretization = Discretization(mc_state_space, [np.linspace(-1.2 - epsilon, 0.6, num=41, endpoint=True), np.linspace(-0.07 - epsilon, 0.07, num=41, endpoint=True)])
    mc_alpha = 0.01
    mc_gamma = 0.99
    mc_init_Q = -50
    mc_temperature = 1
    mc_num_episodes = 10000
    mc_num_test_episodes = 1000
    
    # Run Mountain Car experiment
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

    print("Time", time.time() - x)
    
    np.savez_compressed("mc_rewards.npz", **rewards)
    np.savez_compressed("mc_qvalues.npz", **q_tables)
    np.savez_compressed("mc_test_rewards.npz", **test_rewards)
if __name__ == "__main__":
    # Change this to num_cpus * 2 on GCP
    pool = multiprocessing.Pool(12)
    main()
