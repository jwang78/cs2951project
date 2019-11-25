from script import Discretization, run
import numpy as np
import multiprocessing
def main():
    import time
    x = time.time()
    epsilon = 0.0000001
    num_experiments = 20
    
    # Acrobot args
    ac_env = 'Acrobot-v1'
    ac_state_space = list(range(10*10*10*10*8*8))
    ac_discretization = Discretization(ac_state_space, [np.linspace(-1-epsilon,1,11,True),np.linspace(-1-epsilon,1,11,True),
                                                        np.linspace(-1-epsilon,1,11,True),np.linspace(-1-epsilon,1,11,True),
                        np.linspace(-12.566371-epsilon,12.566371,9,True), np.linspace(-28.274334-epsilon,28.274334,9,True)])
    ac_alpha = 0.25
    ac_gamma = 0.999
    ac_init_Q = 0
    ac_temperature =1
    ac_num_episodes = 100000
    ac_num_test_episodes = 10000
    
    # Run Acrobot experiment
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

    print("Time", time.time() - x)
    
    np.savez("ac_rewards.npz", **rewards)
    np.savez("ac_qvalues.npz", **q_tables)
    np.savez("ac_test_rewards.npz", **test_rewards)
if __name__ == "__main__":
    # Change this to num_cpus * 2 on GCP
    pool = multiprocessing.Pool(1)
    main()
