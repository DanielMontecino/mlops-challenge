import numpy as np
from locust import HttpUser, TaskSet, task, between

N_FEATURES = 48


class RidgePredict(TaskSet):
    @task
    def predict(self):
        request_body = {"data": np.random.rand(1, N_FEATURES).tolist()}
        self.client.post('/ridge/predict', json=request_body)


class RidgeLoadTest(HttpUser):
    tasks = [RidgePredict]
    host = 'http://127.0.0.1'
    stop_timeout = 5
    wait_time = between(1, 5)
