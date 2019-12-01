from script import Discretization, run
import numpy as np
import multiprocessing
def main():
    import time
    x = time.time()
    epsilon = 0.0000001
    num_experiments = 20
    
    # CartPole args
    cartpole_env = 'CartPole-v1'
    cartpole_state_space = list(range(8*8*10*10))
    cartpole_discretization = Discretization(cartpole_state_space, [np.linspace(-2.6-epsilon, 2.6, 9, True),
                                                                    np.linspace(-5.0, 5.0, 9, True),
                                                                    np.linspace(-0.42, 0.42, 11, True),
                                                                    np.linspace(-4.01, 4.01, 11, True)])
    cartpole_alpha = 0.1
    cartpole_gamma = 0.99
    cartpole_init_Q = 25
    cartpole_temperature = 2
    cartpole_num_episodes = 2000
    cartpole_num_test_episodes = 100

    # Run CartPole experiment
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

    print("Time", time.time() - x)
    
    np.savez_compressed("cp_rewards.npz", **rewards)
    np.savez_compressed("cp_qvalues.npz", **q_tables)
    np.savez_compressed("cp_test_rewards.npz", **test_rewards)
if __name__ == "__main__":
    # Change this to num_cpus * 2 on GCP
    pool = multiprocessing.Pool(1)
    main()
