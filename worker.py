import sys
import requests
import json
import script
import io
import multiprocessing
import time
import base64
import numpy as np

discretizations = {script.MC_ENV_NAME: script.MC_DISCRETIZATION,
                   script.AC_ENV_NAME: script.AC_DISCRETIZATION,
                   script.LUNAR_ENV_NAME: script.LUNAR_DISCRETIZATION,
                   script.CP_ENV_NAME: script.CP_DISCRETIZATION}
def register(server, num_cores):
    response = ["", 0]
    try:
        r = requests.post("http://"+server+"/workers", json={"num_cores": num_cores})
        response[1] = r.status_code
        response[0] = r.text
        return json.loads(response[0])["worker_id"], None, None
    except Exception as e:
        return None, response, e
def main(server, num_cores_str):
    num_cores = int(num_cores_str)
    pool = multiprocessing.Pool(num_cores)
    while True:
        while True:
            worker_id, status, exception = register(server, num_cores)
        
            if worker_id is None:
                print("Could not register: ", server, status, exception)
                time.sleep(10)
            else:
                break
        url = "http://{}/".format(server)
        
        while True:
            try:
                req = requests.post(url+"work", json={"num_cores": num_cores})
                next_assignment = json.loads(req.text)
            except Exception as e:
                print("Exception occurred getting more work", e, req.text, req.status_code)
                time.sleep(10)
                continue
            if "complete" in next_assignment and next_assignment["complete"] == True:
                time.sleep(20)
                break
            params = []
            assignment_id = next_assignment["assignment_id"]
            for work_unit in next_assignment["work"]:
                h_params = dict(work_unit["hyperparameters"])
                e_params = dict(work_unit["experiment_parameters"])
                if "checkpoint_Q" in e_params and e_params["checkpoint_Q"] is not None:
                    f = io.BytesIO(base64.b64decode(e_params["checkpoint_Q"].encode("utf-8")))
                    e_params["checkpoint_Q"] = np.load(f)["data"]
                environment_name = e_params["environment"]
                discretization = discretizations[environment_name]
                h_params["discretization"] = discretization
                params.append((h_params, e_params))
            promise = script.run_multiple_hyperparameters_async(pool, params, lambda x, y, z: None)
            while True:
                try:
                    rewards, q_tables, test_rewards = promise(1)
                    rwds_io = io.BytesIO()
                    q_io = io.BytesIO()
                    test_rwds_io = io.BytesIO()
                    np.savez_compressed(rwds_io, **rewards)
                    np.savez_compressed(q_io, **q_tables)
                    np.savez_compressed(test_rwds_io, **test_rewards)
                    rwds_io.seek(0)
                    q_io.seek(0)
                    test_rwds_io.seek(0)
                    rewards_bytes = rwds_io.read()
                    q_bytes = q_io.read()
                    test_rewards_bytes = test_rwds_io.read()
                    rewards_string = base64.b64encode(rewards_bytes).decode("utf-8")
                    q_string = base64.b64encode(q_bytes).decode("utf-8")
                    test_rewards_string = base64.b64encode(test_rewards_bytes).decode("utf-8")
                    response = requests.post(url+"complete", json={"assignment_id": assignment_id, "rewards": rewards_string, "qtable": q_string, "test_rewards": test_rewards_string})
                    print("Finished work. Text:", response.text, "Status:", response.status_code)
                    break
                except multiprocessing.TimeoutError as e:
                    continue
        

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: worker.py <server> <num_cpu_cores>")
    main(sys.argv[1], sys.argv[2])
