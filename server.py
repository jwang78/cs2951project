from flask import Flask
from flask import request
from flask import jsonify
import json
import script
import re
import numpy as np
import base64
import io
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
class ExperimentalRun:
    def __init__(self, run_id, work_unit, checkpoint_frequency):
        self.run_id = run_id
        work_units = self.split(work_unit, checkpoint_frequency)
        self.results = [[[0 for i in range(len(work_unit.experiments))] for k in range(3)] for j in range(len(work_units))]
        self.remaining = [set(work_unit.experiments)] * len(work_units)
        self.unreserved = [set(work_unit.experiments) for i in range(len(work_units))]*len(work_units)
        self.work_units = work_units
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
        if not self.requires_work(work_id):
            return False
        key = self.work_units[work_id[1]].to_json()
        assert key in rewards, str(list(rewards.keys())) + "|||" + key
        relevant_rewards, relevant_q_tables, relevant_test_rewards = rewards[key], q_tables[key], test_rewards[key]
        replace_indices = [x for x in work_id[2] if x in self.remaining[work_id[1]]]
        if len(replace_indices) == 0:
            return False
        for i, idx in enumerate(work_id[2]):
            if idx not in self.remaining[work_id[1]]:
                continue
            # rewards
            self.results[work_id[1]][0][idx] = relevant_rewards[i]
            # qvalues
            #print(len(self.results[work_id[1]][1]), idx, work_id)
            self.results[work_id[1]][1][idx] = relevant_q_tables[i]
            
            # test_rewards
            self.results[work_id[1]][2][idx] = relevant_test_rewards[i]
        self.remaining[work_id[1]] = self.remaining[work_id[1]] - set(work_id[2])
        if not self.requires_work(work_id):
            x = self.results[work_id[1]]
            x[0] = np.array(x[0])
            x[1] = np.array(x[1])
            x[2] = np.array(x[2])
        
            if work_id[1] >= 1:
                self.results[work_id[1] - 1][1] = [] # Delete previous qtables to save memory
        return True
    def is_complete(self):
        return all([len(x) == 0 for x in self.remaining])
    def gather_results(self):
        assert self.is_complete()
        key = self.job.to_json()
        test_rewards_dict = {}
        rewards_dict = {}
        q_tables_dict = {}
        f = np.concatenate([x for x, _, _ in self.results], axis=1)
        rewards_dict[key] = f
        
        g = np.concatenate([z for _, _, z in self.results], axis=1)
        test_rewards_dict[key] = g
        q_tables_dict[key] = np.array(self.results[-1][1])
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
    return jsonify({"queue": [{"id": experiment.run_id,
                               "remaining": [list(x) for x in experiment.remaining],
                               "complete": experiment.is_complete(),
                               "hyperparams": experiment.job.hyperparams,
                               "environment": experiment.job.environment
                               }
                              for experiment in experiment_queue]})

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
discretization = script.MC_DISCRETIZATION
env = script.MC_ENV_NAME
num_experiments = 5
num_episodes = [1000, 100]
runs = [{"agent": agent, "alpha": alpha, "gamma": gamma, "init_Q": init_Q, "temperature": temperature, "epsilon": epsilon}
        for agent in ["BSC", "CSC", "RSC", "BEA", "CEA", "REA"][0:3]
        for alpha in [0.01, 0.1]
        for gamma in [0.99]
        for init_Q in [-50]
        for temperature in [1]
        for epsilon in [0.1]]
work_units = [WorkUnit(None, env, list(range(num_experiments)), num_episodes, run) for run in runs]
experiment_queue = [ExperimentalRun(i, wu, 100) for i, wu in enumerate(work_units)]
experiments = {experiment.run_id: experiment for experiment in experiment_queue}

if __name__ == "__main__":
    app.run(debug=True)
