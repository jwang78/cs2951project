from flask import Flask
from flask import request
from flask import jsonify
import json
import script
import re
import numpy as np
import base64
import io
import os
import sys
tmpdir = sys.argv[1]
app = Flask(__name__)
workers = {}
episode_count = {script.MC_ENV_NAME: 10000, script.AC_ENV_NAME: 100000, script.LUNAR_ENV_NAME: 20000, script.CP_ENV_NAME: 5000}
class WorkUnit:
    def __init__(self, work_id, environment, experiments, num_episodes, hyperparams, checkpoint_Q=None):
        self.hyperparams = hyperparams
        self.work_id = work_id
        self.experiments = list(experiments)
        self.num_episodes = list(num_episodes)
        self.environment = environment
        self.checkpoint_Q = checkpoint_Q
    def to_json(self):
        f = dict(self.hyperparams)
        f["environment"] = self.environment
        return re.sub(r"\s+", "", json.dumps(f, sort_keys=True))
class ExperimentalRunResult:
    def __init__(self, temporary_directory, work_unit):
        self.tmpdir = temporary_directory
        num = 0
        while os.path.exists(self.tmpdir):
            self.tmpdir = temporary_directory + "_"+str(num)
        [os.makedirs(self.tmpdir+"/"+x) for x in ["q", "r", "tr"]]
        self.q_file = self.tmpdir+"/q/%d.npz"
        self.r_file = self.tmpdir+"/r/%d.npz"
        self.tr_file = self.tmpdir+"/tr/%d.npz"
        self.experiments = work_unit.experiments
        self.q_files = []
        self.r_files = []
        self.tr_files = []
    def get_rewards(self):
        arr = [np.load(x)['r'] for x in self.r_files]
        x = np.concatenate(arr)
        return x
    def get_test_rewards(self):
        arr = [np.load(x)['tr'] for x in self.tr_files]
        x = np.concatenate(arr)
        return x
    def get_last_qtable(self):
        arr = np.load(self.q_files[-1])['q']
        return arr
    def add_qtable(self, stage, table, experiment_id):
        q_file = self.q_file % stage
        if os.path.exists(q_file):
            qtables = np.copy(np.load(q_file)["q"])
        else:
            qtables = np.full([len(self.experiments)] + list(table.shape), np.nan)
        if experiment_id not in self.experiments:
            print("Experiment id not found", experiment_id, self.experiments)
            return
        else:
            qtables[self.experiments.index(experiment_id)] = table
            np.savez_compressed(q_file, q=qtables)
            self.q_files.append(q_file)
    def add_rewards(self, stage, rewards, experiment_id):
        rewards_file = self.r_file % stage
        if os.path.exists(rewards_file):
            rewards_arr = np.copy(np.load(rewards_file)["r"])
        else:
            rewards_arr = np.full([len(self.experiments)] + list(rewards.shape), np.nan)
        if experiment_id not in self.experiments:
            print("Experiment id not found", experiment_id, self.experiments)
            return
        else:
            rewards_arr[self.experiments.index(experiment_id)] = rewards
            np.savez_compressed(rewards_file, r=rewards_arr)
            self.r_files.append(rewards_file)
    def add_test_rewards(self, stage, test_rewards, experiment_id):
        tr_file = self.tr_file % stage
        if os.path.exists(tr_file):
            test_rewards_arr = np.copy(np.load(tr_file)["tr"])
        else:
            test_rewards_arr = np.full([len(self.experiments)] + list(test_rewards.shape), np.nan)
        if experiment_id not in self.experiments:
            print("Experiment id not found", experiment_id, self.experiments)
            return
        else:
            test_rewards_arr[self.experiments.index(experiment_id)] = test_rewards
            np.savez_compressed(tr_file, tr=test_rewards_arr)
            self.tr_files.append(tr_file)
    def stage_complete(stage):
        pass
    
class ExperimentalRun:
    def __init__(self, run_id, work_unit, checkpoint_frequency):
        self.run_id = run_id
        work_units = self.split(work_unit, checkpoint_frequency)
        self.results = [[[0 for i in range(len(work_unit.experiments))] for k in range(3)] for j in range(len(work_units))]
        self.remaining = [set(work_unit.experiments)] * len(work_units)
        self.unreserved = [set(work_unit.experiments) for i in range(len(work_units))]*len(work_units)
        self.work_units = work_units
        self.run_results = ExperimentalRunResult(tmpdir+"/work"+str(hash(work_unit.to_json())), work_unit)
        self.job = work_unit
    def __eq__(self, other):
        return other.run_id == self.run_id
    def __str__(self):
        return str(self.run_id)
    def __repr__(self):
        return str(self)
    def get_next(self, num_cores):
        next_remaining = -1
        for i, item in enumerate(self.remaining):
            if len(item) != 0:
                next_remaining = i
                break
        if next_remaining == -1:
            return None, set()
        if len(self.unreserved[next_remaining]) == 0:
            self.unreserved[next_remaining] = set(self.job.experiments)
        wu = self.work_units[next_remaining]
        num_to_take = min(num_cores, len(self.unreserved[next_remaining]))
        to_take = list(self.unreserved[next_remaining])[:num_to_take]
        ret_wu = WorkUnit(wu.work_id + [to_take], wu.environment, to_take, wu.num_episodes, wu.hyperparams)
        if next_remaining >= 1:
            f = io.BytesIO()
            prev_q_tables = self.results[next_remaining - 1][1]
            prev_q_tables = prev_q_tables[to_take]
            assert type(prev_q_tables) != type(list), "Previous step unfinished"
            np.savez_compressed(f, **{"data": prev_q_tables})
            f.seek(0)
            ret_wu.checkpoint_Q = base64.b64encode(f.read()).decode("utf-8")
        retval = ret_wu, self.unreserved[next_remaining] - set(ret_wu.experiments)
        self.unreserved[next_remaining] = self.unreserved[next_remaining] - set(ret_wu.experiments)
        return retval
        
    def requires_work(self, work_id):
        assert work_id[0] == self.run_id
        return len(self.remaining[work_id[1]]) > 0
    def partial_complete(self, work_id, rewards, q_tables, test_rewards):
        #print(work_id, "asdf")
        if not self.requires_work(work_id):
            return False
        key = self.work_units[work_id[1]].to_json()
        assert key in rewards, str(list(rewards.keys())) + "|||" + key
        relevant_rewards, relevant_q_tables, relevant_test_rewards = rewards[key], q_tables[key], test_rewards[key]
        replace_indices = [x for x in work_id[2] if x in self.remaining[work_id[1]]]
        if len(replace_indices) == 0:
            return False
        # work_id: (something, stage_number, experiment_numbers)
        # index of experiment, id of experiment
        for i, idx in enumerate(work_id[2]):
            if idx not in self.remaining[work_id[1]]:
                continue
            # rewards
            self.run_results.add_qtable(stage=work_id[1], table=relevant_q_tables[i], experiment_id=idx)
            self.run_results.add_rewards(stage=work_id[1], rewards=relevant_rewards[i], experiment_id=idx)
            self.run_results.add_test_rewards(stage=work_id[1], test_rewards=relevant_test_rewards[i], experiment_id=idx)
            #self.results[work_id[1]][0][idx] = relevant_rewards[i]
            # qvalues
            #print(len(self.results[work_id[1]][1]), idx, work_id)
            #self.results[work_id[1]][1][idx] = relevant_q_tables[i]
            
            # test_rewards
            #self.results[work_id[1]][2][idx] = relevant_test_rewards[i]
        
        self.remaining[work_id[1]] = self.remaining[work_id[1]] - set(work_id[2])
        if not self.requires_work(work_id):
            if work_id[1] >= 1:
                self.run_results.stage_complete(work_id[1])
                #self.results[work_id[1] - 1][1] = [] # Delete previous qtables to save memory
        return True
    def is_complete(self):
        return all([len(x) == 0 for x in self.remaining])
    def gather_results(self):
        assert self.is_complete()
        key = self.job.to_json()
        test_rewards_dict = {}
        rewards_dict = {}
        q_tables_dict = {}
        f = self.run_results.get_rewards()
        g = self.run_results.get_test_rewards()
        rewards_dict[key] = f
        
        test_rewards_dict[key] = g
        q_tables_dict[key] = self.run_results.get_last_qtable()
        return rewards_dict, q_tables_dict, test_rewards_dict
            
    def split(self, work_unit, num_episodes_per_unit):
        total_episodes = sum(work_unit.num_episodes)
        num_episodes = list(work_unit.num_episodes)
        units = []
        idx = 0
        while total_episodes > 0:
            work_id = [self.run_id, idx]
            num_train_episodes = min(num_episodes[0], num_episodes_per_unit)
            num_test_episodes = min(num_episodes_per_unit - num_train_episodes, num_episodes[1])
            unit = WorkUnit(work_id, work_unit.environment, work_unit.experiments, [num_train_episodes, num_test_episodes], work_unit.hyperparams)
            num_episodes[0] -= num_train_episodes
            num_episodes[1] -= num_test_episodes
            units.append(unit)
            idx += 1
            total_episodes -= num_train_episodes + num_test_episodes
        assert total_episodes == 0
        return units
have_written = False
class Worker:
    def __init__(self, worker_id, num_cores):
        self.num_cores = num_cores
        self.worker_id = worker_id
    def __repr__(self):
        return str(self)
    def __str__(self):
        return "Worker %d:%d" % (self.worker_id, self.num_cores)
@app.route("/workers/<worker_id>", methods=['DELETE'])
def delete_worker(worker_id):
    del workers[worker_id]
@app.route("/workers", methods=['GET', 'POST'])
def add_worker():
    if request.method == "GET":
        return str(workers)
    else:
        try:
            num_cores = int(request.json["num_cores"])
            worker_id = len(workers)
            worker = Worker(worker_id, num_cores)
            workers[worker_id] = worker
            return jsonify({"worker_id": worker.worker_id}), 200
        except Exception as e:
            return str(e), 400
@app.route("/work", methods=['POST'])
def get_work():
    global experiment_queue, have_written
    # relies on experiments and experiment_queue global variable
    num_cores_left = request.json["num_cores"]
    incomplete = [experiment for experiment in experiment_queue if not experiment.is_complete()]
    num_experiments = len(incomplete)
    if num_experiments == 0:
        if not have_written:
            write_results()
            have_written = True
        return jsonify({"complete": True}), 200
    i = 0
    work_units = []
    processed = []
    while num_cores_left > 0 and num_experiments > 0:
        experiment = incomplete[i]
        wu, remaining = experiment.get_next(num_cores_left)
        
        
        work_units.append(wu)
        if len(remaining) == 0:
            processed.append(i)
        i += 1
        num_experiments -= 1
        num_cores_left -= len(wu.experiments)
    assert num_cores_left >= 0, "Number of cores left was negative"
    f = set(processed)
    # Reorder the queue
    experiment_queue_idx = [x for x in range(len(experiment_queue)) if x not in f] + list(f)
    experiment_queue = [experiment_queue[j] for j in experiment_queue_idx]
    #print(experiment_queue, f)
    assignment_id = [wu.work_id for wu in work_units]
    params = [{"hyperparameters": wu.hyperparams,
              "experiment_parameters": {"num_episodes": wu.num_episodes[0],
                                        "num_test_episodes": wu.num_episodes[1],
                                        "max_steps": 200,
                                        "num_experiments": len(wu.experiments),
                                        "environment": wu.environment
                                        }} for wu in work_units]
    for wu, param in zip(work_units, params):
        if wu.checkpoint_Q is not None:
            param["experiment_parameters"]["checkpoint_Q"] = wu.checkpoint_Q
    return jsonify({"work": params, "assignment_id": assignment_id}), 200
@app.route("/clearworkers")
def clear_workers():
    workers.clear()
@app.route("/write", methods=["POST"])
def write_results():
    f = request.json
    rewards_file, qtable_file, test_rewards_file = "rewards.npz", "qvalues.npz", "test_rewards.npz"
    if f is not None:
        if "rewards_file" in f:
            rewards_file = f["rewards_file"]
        if "qtable_file" in f:
            qtable_file = f["qtable_file"]
        if "test_rewards_file" in f:
            test_rewards_file = f["test_rewards_file"]
    completed_experiments = [experiment for eid, experiment in experiments.items() if experiment.is_complete()]
    rewards, q_tables, test_rewards = dict(), dict(), dict()
    results = [x.gather_results() for x in completed_experiments]
    for reward, q_table, test_reward in results:
        assert list(reward.keys())[0] not in rewards, "Duplicate key???"
        rewards.update(reward)
        q_tables.update(q_table)
        test_rewards.update(test_reward)
    np.savez_compressed(rewards_file, **rewards)
    np.savez_compressed(qtable_file, **q_tables)
    np.savez_compressed(test_rewards_file, **test_rewards)
    return "Success", 200
@app.route("/experiments", methods=["GET"])
def work_status():
    total_experiments = len(experiments)*len(experiment_queue[0].work_units)*len(experiment_queue[0].work_units[0].experiments)
    curr_progress = sum([sum([len(x) for x in experiment.remaining]) for experiment in experiment_queue])
    return jsonify({"queue": [{"id": experiment.run_id,
                               "remaining": [len(x) for x in experiment.remaining],
                               "complete": experiment.is_complete(),
                               "hyperparams": experiment.job.hyperparams,
                               "environment": experiment.job.environment
                               }
                              for experiment in experiment_queue], "current_progress": total_experiments-curr_progress,
                               "ending_progress": total_experiments})

@app.route("/complete", methods=["POST"])
def complete_work():
    f = request.json
    assignment_id = f["assignment_id"]
    rewards_string, q_string, test_rewards_string = base64.b64decode(f["rewards"].encode("utf-8")), base64.b64decode(f["qtable"].encode("utf-8")), base64.b64decode(f["test_rewards"].encode("utf-8"))
    x, y, z = io.BytesIO(rewards_string), io.BytesIO(q_string), io.BytesIO(test_rewards_string)
    rewards, q_tables, test_rewards = dict(np.load(x)), dict(np.load(y)), dict(np.load(z))
    complete_successes = []
    for work_id in assignment_id:
        c = experiments[work_id[0]].partial_complete(work_id, rewards, q_tables, test_rewards)
        complete_successes.append(c)
    if not any(complete_successes):
        return "All work duplicated", 200
    return "Success", 200

env = script.LUNAR_ENV_NAME
num_experiments = 20
num_episodes = [10000, 1000]
runs = [{"agent": agent, "alpha": alpha, "gamma": gamma, "init_Q": init_Q, "temperature": temperature, "epsilon": epsilon, "episode_annealing": episode_annealing, "step_annealing": step_annealing}
        for agent in ["BSC", "CSC", "RSC", "BEC", "CEC", "REC", "BSA", "CSA", "RSA"]
        for alpha in [0.01, 0.1, 0.5]
        for gamma in [0.995]
        for init_Q in [25]
        for temperature in [1]
        for epsilon in [0.2]
        for episode_annealing in [1000]
        for step_annealing in [2000]]

betas = [('BSC', "0"), ('CSC', "0"),
         ('RSC', 'lambda: np.random.uniform(0, 2)'),
         ('RSC', 'lambda: np.random.uniform(0, 1)'),
         ('RSC', 'lambda: 1'),
         ('RSC', 'lambda: 1.5 if np.random.uniform(0, 1) < 0.5 else 0.5'),
         ('RSC', 'lambda: np.random.uniform(0.5, 1.5)'),
         ('RSC', 'lambda: np.random.beta(2, 3)'),
         ('RSC', 'lambda: np.random.beta(2, 7)'),
         ('RSC', 'lambda: np.random.pareto(2)'),
         ('RSC', 'lambda: np.random.pareto(3)'),]
runs = []
m, l = script.MC_ENV_NAME, script.LUNAR_ENV_NAME
episodes = {m: [10000, 100], l: [10000, 1000]}
alphas = {m: 0.01, l: 0.1}
gammas = {m: 0.99, l: 0.995}
init_Qs = {m: -50, l: -25}
temps = {m: 1, l: 1}
for env in [script.MC_ENV_NAME, script.LUNAR_ENV_NAME]:
    num_episodes_f = episodes[env]
    runs.extend([({"agent": agent,
                  "alpha": alphas[env],
                  "gamma": gammas[env],
                  "init_Q": init_Qs[env],
                  "temperature": temps[env],
                  "epsilon": 0.2,
                  "episode_annealing": 1000,
                  "step_annealing": 2000,
                  "beta_function": function_string}, num_episodes_f) for agent, function_string in betas])


work_units = [WorkUnit(None, env, list(range(num_experiments)), n_episodes, run) for run, n_episodes in runs]
experiment_queue = [ExperimentalRun(i, wu, 2000) for i, wu in enumerate(work_units)]
experiments = {experiment.run_id: experiment for experiment in experiment_queue}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
